"""Bayesian MMM specification and fitting (PyMC-Marketing).

``MixLabModel`` wraps ``pymc_marketing.mmm.multidimensional.MMM`` (the supported class as of
pymc-marketing 0.19; the older ``pymc_marketing.mmm.MMM`` is deprecated and removed in 0.20).

Model: revenue = intercept + linear trend + yearly Fourier seasonality + controls
       + sum over channels of beta_c * logistic_saturation(geometric_adstock(spend_c)) + noise.

Scaling: PyMC-Marketing divides revenue by its peak week and each channel's spend by its own
peak week before fitting. Every prior below is therefore in "share of peak week" units.
"""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Self

import arviz as az
import numpy as np
import pandas as pd
import pytensor.xtensor as ptx
import xarray as xr
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from pymc_extras.prior import Prior, register_tensor_transform
from pymc_marketing.mmm import (
    DelayedAdstock,
    GeometricAdstock,
    HillSaturation,
    LogisticSaturation,
)
from pymc_marketing.mmm.additive_effect import LinearTrendEffect
from pymc_marketing.mmm.linear_trend import LinearTrend
from pymc_marketing.mmm.multidimensional import MMM

from mixlab import config
from mixlab.config import ModelSettings
from mixlab.transforms import logistic_saturation
from mixlab.validate import channel_name, spend_columns

# --- Data preparation -----------------------------------------------------------------------


def default_control_columns(df: pd.DataFrame) -> list[str]:
    """Return the control columns used by default: holidays, promo flag and price index."""
    holidays = [c for c in df.columns if c.startswith(config.HOLIDAY_PREFIX)]
    extras = [c for c in (config.PROMO_COL, config.PRICE_COL) if c in df.columns]
    return [*holidays, *extras]


def design_matrix(
    df: pd.DataFrame, channels: list[str], controls: list[str], min_spend: float = 0.0
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Return the model inputs ``X`` and, if present, the target ``y``.

    The price index is re-centred on its base of 1.0 so that "no price change" is zero.
    PyMC-Marketing does not scale controls, and an uncentred index would be almost
    indistinguishable from the intercept.

    ``min_spend`` floors every spend value (see ``config.HILL_MIN_SPEND``).
    """
    X = df[[config.DATE_COL, *channels, *controls]].copy()
    X[config.DATE_COL] = pd.to_datetime(X[config.DATE_COL])
    if min_spend > 0:
        X[channels] = X[channels].clip(lower=min_spend)
    if config.PRICE_COL in controls:
        X[config.PRICE_COL] = X[config.PRICE_COL] - config.PRICE_INDEX_BASE
    y = df[config.TARGET_COL].astype(float) if config.TARGET_COL in df.columns else None
    return X, y


def spend_shares(df: pd.DataFrame, channels: list[str]) -> dict[str, float]:
    """Return each spend column's share of total spend (sums to 1)."""
    totals = df[channels].sum()
    return {c: float(totals[c] / totals.sum()) for c in channels}


# --- Priors ---------------------------------------------------------------------------------


def channel_prior_values(
    channels: list[str], settings: ModelSettings, shares: dict[str, float]
) -> dict[str, dict[str, list[float]]]:
    """Return per-channel prior parameters after applying YAML overrides and spend share.

    Order of precedence for the size of a channel's effect (``saturation_beta``):
    an explicit YAML override, then the spend-share rule if enabled, then the default.
    """
    defaults = settings.priors
    values: dict[str, dict[str, list[float]]] = {
        "adstock_alpha": {"alpha": [], "beta": []},
        "saturation_lam": {"alpha": [], "beta": []},
        "saturation_beta": {"sigma": []},
    }
    for column in channels:
        override = settings.channel_priors.get(channel_name(column))
        adstock = (override and override.adstock_alpha) or defaults.adstock_alpha
        lam = (override and override.saturation_lam) or defaults.saturation_lam
        values["adstock_alpha"]["alpha"].append(adstock.alpha)
        values["adstock_alpha"]["beta"].append(adstock.beta)
        values["saturation_lam"]["alpha"].append(lam.alpha)
        values["saturation_lam"]["beta"].append(lam.beta)

        sigma = defaults.saturation_beta.sigma
        if settings.spend_share_prior:
            sigma = defaults.saturation_beta.sigma * len(channels) * shares[column]
        if override and override.saturation_beta:
            sigma = override.saturation_beta.sigma
        values["saturation_beta"]["sigma"].append(sigma)
    return values


# Weibull adstock is not offered: its gradient is unavailable on the Numba backend this
# project runs on, so NUTS cannot sample it (see docs/model_log.md).
ADSTOCKS = {"geometric": GeometricAdstock, "delayed": DelayedAdstock}
SATURATIONS = {"logistic": LogisticSaturation, "hill": HillSaturation}


def roi_scale(df: pd.DataFrame, channels: list[str], settings: ModelSettings) -> np.ndarray:
    """Return, per channel, the model coefficient (beta) that corresponds to an ROI of 1.

    In the model, a channel's revenue is ``beta * peak_revenue * sum_t saturation(x_t)`` with
    spend scaled by its peak week, so ``ROI = beta * peak_revenue * S / total_spend``. ``S`` is
    evaluated at the prior-mean saturation speed, which makes this an approximate but
    well-scaled conversion: it is only used to place priors, never to report results.
    """
    lam = settings.priors.saturation_lam.alpha / settings.priors.saturation_lam.beta
    slope, kappa = settings.priors.hill_slope_median, settings.priors.hill_kappa_median
    peak_revenue = float(df[config.TARGET_COL].max())
    scale = []
    for column in channels:
        spend = df[column].to_numpy(dtype=float)
        x = spend / spend.max()
        if settings.saturation == "logistic":
            saturated = logistic_saturation(x, lam).sum()
        else:
            saturated = (x**slope / (kappa**slope + x**slope)).sum()
        scale.append(spend.sum() / (peak_revenue * saturated))
    return np.array(scale)


def register_roi_transform(scale: np.ndarray) -> str:
    """Register the "log ROI -> beta" transform for these channels and return its name.

    The transform is ``beta = exp(log_roi) * scale``. PyMC-Marketing looks transforms up by
    name when a saved model is reloaded, so the name encodes the scale values. A 2-D scale is
    one row per region (``geo`` x ``channel``), as used by the geo model.
    """
    values = np.asarray(scale, dtype=float)
    digest = hashlib.sha256(values.tobytes()).hexdigest()[:12]
    name = f"{config.ROI_TRANSFORM_PREFIX}{digest}"
    dims = ("channel",) if values.ndim == 1 else (config.GEO_COL, "channel")

    def to_beta(log_roi: Any) -> Any:
        return ptx.math.exp(log_roi) * ptx.as_xtensor(values, dims=dims)

    register_tensor_transform(name, to_beta)
    return name


def roi_beta_prior(
    channels: list[str], settings: ModelSettings, scale: np.ndarray
) -> tuple[Prior, np.ndarray]:
    """Return the effect-size prior implied by the ROI prior, and the scale it used.

    Pooled mode (partial pooling): ``log ROI_c = mu + tau * z_c`` with a shared typical ROI
    ``mu`` and a learned between-channel spread ``tau``. A small or noisy channel cannot be
    estimated on its own, so it is pulled toward the typical ROI instead of being left with a
    range of "anything from 0 to 34".

    Independent mode: each channel gets ``LogNormal(log(median_c), spread)``.

    Benchmarks shift a channel's centre: the pooled quantity becomes ROI relative to its
    benchmark.
    """
    roi = settings.roi_prior
    if roi is None:
        raise ValueError("settings.roi_prior is not set")
    centres = np.array([roi.benchmarks.get(channel_name(c), roi.median) for c in channels])
    if roi.mode == "independent":
        prior = Prior(
            "LogNormal", mu=np.log(centres * scale).tolist(), sigma=roi.spread, dims="channel"
        )
        return prior, scale
    relative = scale * centres / roi.median
    prior = Prior(
        "Normal",
        mu=Prior("Normal", mu=float(np.log(roi.median)), sigma=roi.median_sigma),
        sigma=Prior("HalfNormal", sigma=roi.spread),
        dims="channel",
        centered=False,
        transform=register_roi_transform(relative),
    )
    return prior, relative


def build_priors(
    channels: list[str], settings: ModelSettings, shares: dict[str, float]
) -> dict[str, Prior]:
    """Return the PyMC-Marketing ``model_config`` with one commented prior per parameter."""
    p = settings.priors
    per_channel = channel_prior_values(channels, settings, shares)
    return {
        # Baseline: revenue in a week with no marketing, no promo and no holiday. We expect
        # it to be roughly 10-90% of the best week ever, centred on half. A baseline above
        # the peak week or below zero is implausible, so the prior makes those unlikely.
        "intercept": Prior("Normal", mu=p.intercept_mu, sigma=p.intercept_sigma),
        # Carryover: the share of this week's advertising effect still felt next week.
        # Beta(1, 3) leans towards short memory (average 0.25) but allows anything from 0 to
        # about 0.7. Most digital channels fade within a week or two; channels believed to
        # linger (TV, video) get their own prior from the YAML file.
        **(
            {"adstock_alpha": Prior("Beta", **per_channel["adstock_alpha"], dims="channel")}
            if settings.adstock in ("geometric", "delayed")
            else {}
        ),
        # Saturation speed: how quickly extra spend stops paying off. Gamma(3, 1) centres on
        # a curve that is about 90% saturated at the channel's biggest-ever week, and allows
        # anything from nearly linear to saturating at a third of that. We stay vague here
        # because the data usually says little about the shape.
        **(
            {"saturation_lam": Prior("Gamma", **per_channel["saturation_lam"], dims="channel")}
            if settings.saturation == "logistic"
            else {
                # Hill shape. Slope: how sharply the curve bends (1 is a plain concave curve).
                # Kappa: spend, as a share of the channel's peak week, that reaches half the
                # channel's ceiling. Both are LogNormal so they stay away from zero.
                "saturation_slope": Prior(
                    "LogNormal",
                    mu=float(np.log(p.hill_slope_median)),
                    sigma=p.hill_slope_sigma,
                    dims="channel",
                ),
                "saturation_kappa": Prior(
                    "LogNormal",
                    mu=float(np.log(p.hill_kappa_median)),
                    sigma=p.hill_kappa_sigma,
                    dims="channel",
                ),
            }
        ),
        # Effect size: the most revenue a channel could add per week, as a share of the
        # peak week. HalfNormal keeps it non-negative (advertising does not reduce sales) and
        # sigma 0.3 says "probably under 30% of peak revenue, almost surely under 60%".
        # The library default (sigma 2) would let one channel add twice the best week ever.
        "saturation_beta": Prior("HalfNormal", **per_channel["saturation_beta"], dims="channel"),
        # Controls (holidays, promo, price): can push revenue up or down. Sigma 0.3 allows a
        # holiday week to add or remove up to roughly 60% of peak revenue, far wider than we
        # expect, so the data decides.
        "gamma_control": Prior("Normal", mu=0, sigma=p.control_sigma, dims="control"),
        # Seasonality: Laplace pulls small seasonal wiggles towards zero unless the data
        # insists, which stops seasonality from soaking up effects that belong to media.
        "gamma_fourier": Prior("Laplace", mu=0, b=p.fourier_scale, dims="fourier_mode"),
        # Noise: week-to-week revenue the model cannot explain, probably under 10% of peak.
        "likelihood": Prior(
            "Normal", sigma=Prior("HalfNormal", sigma=p.noise_sigma), dims=config.DATE_COL
        ),
    }


# SPEND-SHARE PRIOR: THE TRADE-OFF
#
# With ``spend_share_prior: true`` each channel's effect-size prior is scaled by its share of
# total spend (sigma_c = default_sigma * n_channels * share_c), a common industry practice.
#
# What you gain: it encodes "we spend more where we believe it works", stabilises the fit
# when channels move together, and stops a tiny channel from being handed an absurd share of
# revenue by chance.
#
# What you pay: it assumes every channel has a broadly similar ROI before seeing the data, so
# it nudges the answer towards "the current budget split is already right". A small channel
# with a genuinely high ROI (email here) is pulled down, and a large inefficient one is
# propped up. That is the opposite of what a budget optimizer needs to discover, so it is
# off by default; turn it on when channels are too correlated to separate any other way.


def build_trend(settings: ModelSettings) -> LinearTrendEffect:
    """Return the trend component.

    Trend is piecewise linear over the whole period. Laplace(0, 0.2) says underlying growth
    or decline over the full history is probably within 40% of peak revenue.
    """
    trend = LinearTrend(
        n_changepoints=settings.trend_changepoints,
        priors={"delta": Prior("Laplace", mu=0, b=settings.priors.trend_scale, dims="changepoint")},
    )
    return LinearTrendEffect(trend=trend, prefix=config.TREND_PREFIX)


# --- Prior predictive -----------------------------------------------------------------------


def interval_summary(draws: xr.DataArray) -> pd.DataFrame:
    """Return per-date quantiles (outer and inner interval plus median) of sampled revenue."""
    sample_dims = [d for d in draws.dims if d != config.DATE_COL]
    low, high = config.INTERVAL_OUTER
    mid_low, mid_high = config.INTERVAL_INNER
    quantiles = draws.quantile([low, mid_low, 0.5, mid_high, high], dim=sample_dims)
    frame = quantiles.transpose(config.DATE_COL, "quantile").to_pandas()
    frame.columns = ["lower", "inner_lower", "median", "inner_upper", "upper"]
    frame.insert(0, "mean", draws.mean(dim=sample_dims).to_pandas())
    return frame.rename_axis(config.DATE_COL).reset_index()


def prior_predictive_check(draws: xr.DataArray, observed: pd.Series) -> dict[str, Any]:
    """Judge whether the priors imply believable revenue, before seeing any fit.

    Believable means: the observed average week sits inside the prior's range, the top of
    that range is not many times the best week ever, and few simulated weeks are negative.
    """
    sample_dims = [d for d in draws.dims if d != config.DATE_COL]
    weekly_mean = draws.mean(dim=config.DATE_COL)
    low, high = (float(weekly_mean.quantile(q)) for q in config.INTERVAL_OUTER)
    negative_share = float((draws < 0).mean())
    observed_mean, observed_max = float(observed.mean()), float(observed.max())
    covers = low <= observed_mean <= high
    not_absurd = high <= config.PRIOR_MAX_REVENUE_MULTIPLE * observed_max
    few_negative = negative_share <= config.PRIOR_MAX_NEGATIVE_SHARE
    return {
        "observed_mean_weekly_revenue": observed_mean,
        "prior_mean_weekly_revenue_low": low,
        "prior_mean_weekly_revenue_median": float(weekly_mean.median()),
        "prior_mean_weekly_revenue_high": high,
        "share_of_negative_weeks": negative_share,
        "n_samples": int(np.prod([draws.sizes[d] for d in sample_dims])),
        "covers_observed": covers,
        "believable": bool(covers and not_absurd and few_negative),
    }


def _lakh(value: float, _position: int) -> str:
    """Format an INR amount as lakh for axis ticks."""
    return f"{value / config.INR_PER_LAKH:,.10g}"


def plot_prior_predictive(summary: pd.DataFrame, observed: pd.Series) -> Figure:
    """Return a figure of prior-simulated revenue bands against observed revenue."""
    dates = summary[config.DATE_COL]
    low, high = (round(100 * q) for q in config.INTERVAL_OUTER)
    fig = Figure(figsize=config.FIGURE_SIZE, facecolor=config.COLOR_SURFACE)
    ax = fig.subplots()
    ax.fill_between(
        dates,
        summary["lower"],
        summary["upper"],
        color=config.COLOR_SPEND,
        alpha=0.15,
        linewidth=0,
        label=f"Prior range, middle {high - low}%",
    )
    ax.fill_between(
        dates,
        summary["inner_lower"],
        summary["inner_upper"],
        color=config.COLOR_SPEND,
        alpha=0.35,
        linewidth=0,
        label="Prior range, middle 50%",
    )
    ax.plot(dates, observed.to_numpy(), color=config.COLOR_TEXT, linewidth=2, label="Observed")
    ax.axhline(0, color=config.COLOR_TEXT_MUTED, linewidth=0.8)
    ax.set_title(
        "Prior predictive check: revenue the model considers plausible before seeing the data",
        loc="left",
        fontsize=12,
        color=config.COLOR_TEXT,
    )
    ax.set_ylabel("Weekly revenue (INR lakh)", color=config.COLOR_TEXT_MUTED)
    ax.yaxis.set_major_formatter(FuncFormatter(_lakh))
    ax.set_facecolor(config.COLOR_SURFACE)
    ax.grid(axis="y", color=config.COLOR_GRID, linewidth=0.8)
    ax.tick_params(colors=config.COLOR_TEXT_MUTED, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(config.COLOR_GRID)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    return fig


# --- Model wrapper --------------------------------------------------------------------------


class MixLabModel:
    """Build, check, fit, save, load and predict with the MixLab MMM.

    Typical use::

        model = MixLabModel().build(df, ModelSettings.from_yaml("configs/default.yaml"))
        model.sample_prior_predictive()
        model.fit()
        model.save(config.MODELS_DIR)
        model = MixLabModel.load(config.MODELS_DIR)
    """

    def __init__(self) -> None:
        """Create an empty wrapper; call ``build`` or ``load`` next."""
        self.settings: ModelSettings = ModelSettings()
        self.mmm: MMM | None = None
        self.channels: list[str] = []
        self.controls: list[str] = []
        self.X: pd.DataFrame | None = None
        self.y: pd.Series | None = None
        self.fit_seconds: float | None = None
        self.roi_transform_scale: list[float] | None = None

    def _require_mmm(self) -> MMM:
        """Return the underlying model or fail with a clear message."""
        if self.mmm is None:
            raise RuntimeError("Model is not built. Call build(df, settings) or load(path).")
        return self.mmm

    @property
    def idata(self) -> az.InferenceData:
        """Return the inference data (priors, posterior, sampler statistics)."""
        idata = self._require_mmm().idata
        if idata is None:
            raise RuntimeError("No samples yet. Call sample_prior_predictive() or fit().")
        return idata

    @property
    def min_spend(self) -> float:
        """Return the spend floor this model's saturation curve needs."""
        return config.HILL_MIN_SPEND if self.settings.saturation == "hill" else 0.0

    @property
    def target_scale(self) -> float:
        """Return the INR value of 1.0 in the model's scaled revenue units (the peak week)."""
        return float(np.ravel(self.idata.constant_data["target_scale"].values)[0])

    @property
    def parameter_names(self) -> list[str]:
        """Return the names of the sampled parameters (for diagnostics)."""
        return [v.name for v in self._require_mmm().model.free_RVs]

    def build(self, df: pd.DataFrame, settings: ModelSettings | None = None) -> Self:
        """Define the model for a weekly frame that follows the data contract."""
        self.settings = settings or ModelSettings()
        self.channels = spend_columns(df)
        self.controls = self.settings.control_columns or default_control_columns(df)
        self.X, self.y = design_matrix(df, self.channels, self.controls, self.min_spend)
        if self.y is None:
            raise ValueError(f"Training data needs a '{config.TARGET_COL}' column.")

        sampler = self.settings.sampler
        priors = build_priors(self.channels, self.settings, spend_shares(df, self.channels))
        self.roi_transform_scale = None
        if self.settings.roi_prior is not None:
            scale = roi_scale(df, self.channels, self.settings)
            priors["saturation_beta"], used = roi_beta_prior(self.channels, self.settings, scale)
            if self.settings.roi_prior.mode == "pooled":
                self.roi_transform_scale = used.tolist()
        self.mmm = MMM(
            date_column=config.DATE_COL,
            channel_columns=self.channels,
            target_column=config.TARGET_COL,
            adstock=ADSTOCKS[self.settings.adstock](l_max=self.settings.adstock_l_max),
            saturation=SATURATIONS[self.settings.saturation](),
            time_varying_intercept=self.settings.time_varying_intercept,
            time_varying_media=self.settings.time_varying_media,
            control_columns=self.controls,
            yearly_seasonality=self.settings.yearly_seasonality_order,
            model_config=priors,
            sampler_config={
                "chains": sampler.chains,
                "draws": sampler.draws,
                "tune": sampler.tune,
                "target_accept": sampler.target_accept,
                "nuts_sampler": sampler.nuts_sampler,
            },
        )
        self.mmm.mu_effects.append(build_trend(self.settings))
        self.mmm.build_model(self.X, self.y)
        # Store channel contributions and fitted revenue in INR, not only in scaled units.
        self.mmm.add_original_scale_contribution_variable(["channel_contribution", "y"])
        return self

    def add_lift_tests(self, measurements: pd.DataFrame) -> Self:
        """Calibrate the model with experiment results before fitting.

        Each row says: moving a channel's weekly spend from ``x`` to ``x + delta_x`` changed
        weekly revenue by ``delta_y`` (give or take ``sigma``). The channel's response curve
        is then required to agree with that measurement as well as with the weekly data.
        See ``mixlab.calibration.to_lift_measurements`` for building this frame.
        """
        self._require_mmm().add_lift_test_measurements(
            measurements, name=config.LIFT_LIKELIHOOD_NAME
        )
        return self

    def sample_prior_predictive(self) -> xr.DataArray:
        """Simulate revenue (INR) from the priors alone, before the model sees the target."""
        mmm = self._require_mmm()
        mmm.sample_prior_predictive(
            self.X,
            self.y,
            samples=self.settings.prior_predictive_samples,
            random_seed=self.settings.sampler.seed,
        )
        return self.idata.prior_predictive["y"] * self.target_scale

    def fit(self, progressbar: bool = True) -> az.InferenceData:
        """Run MCMC with the configured sampler settings and report the time taken."""
        mmm = self._require_mmm()
        sampler = self.settings.sampler
        print(
            f"Fitting: {len(self.X)} weeks, {len(self.channels)} channels, "
            f"{len(self.controls)} controls | {sampler.chains} chains x "
            f"({sampler.tune} tune + {sampler.draws} draws), target_accept "
            f"{sampler.target_accept}, seed {sampler.seed}, sampler {sampler.nuts_sampler}"
        )
        start = time.perf_counter()
        mmm.fit(self.X, self.y, random_seed=sampler.seed, progressbar=progressbar)
        self.fit_seconds = time.perf_counter() - start
        divergences = int(self.idata.sample_stats["diverging"].sum())
        print(
            f"Fit finished in {self.fit_seconds:.1f}s: {sampler.chains * sampler.draws} "
            f"posterior draws, {divergences} divergences"
        )
        return self.idata

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return predicted revenue in INR (mean and interval) for each week in ``df``.

        For weeks after the training period, the last training weeks are prepended
        internally so advertising carryover is continuous.
        """
        mmm = self._require_mmm()
        X, _ = design_matrix(df, self.channels, self.controls, self.min_spend)
        training_end = pd.to_datetime(self.idata.fit_data[config.DATE_COL].values).max()
        is_future = bool(X[config.DATE_COL].min() > training_end)
        draws = mmm.sample_posterior_predictive(
            X,
            extend_idata=False,
            combined=True,
            include_last_observations=is_future,
            random_seed=self.settings.sampler.seed,
            progressbar=False,
        )
        summary = interval_summary(draws["y"] * self.target_scale)
        return summary[[config.DATE_COL, "mean", "lower", "upper"]]

    def save(
        self, directory: Path = config.MODELS_DIR, extra: dict[str, Any] | None = None
    ) -> Path:
        """Save the fitted model with its inference data, plus a small JSON of metadata.

        The ``.nc`` file holds the model definition and all samples, so the dashboard can
        load it without refitting.
        """
        mmm = self._require_mmm()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / config.MODEL_FILENAME
        # Write beside the target and swap it in, so a running app that still has the old
        # file open (and locked) cannot leave a half-written model behind.
        temporary = path.with_suffix(".tmp")
        mmm.save(str(temporary))
        os.replace(temporary, path)
        meta = {
            "saved_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "channels": self.channels,
            "controls": self.controls,
            "fit_seconds": self.fit_seconds,
            "settings": self.settings.model_dump(mode="json"),
            "roi_transform_scale": self.roi_transform_scale,
            **(extra or {}),
        }
        (directory / config.MODEL_META_FILENAME).write_text(json.dumps(meta, indent=2) + "\n")
        return path

    def save_slim(self, directory: Path = config.MODELS_DIR) -> Path:
        """Save a thinned copy of the fitted model that is small enough to commit and deploy.

        Keeps every ``n``-th posterior draw so that ``config.SLIM_DRAWS_PER_CHAIN`` remain per
        chain, and drops priors, warm-up draws and derived variables that can be recomputed.
        Estimates from the slim copy differ slightly from the full model (fewer draws).
        """
        idata = self.idata
        step = max(idata.posterior.sizes["draw"] // config.SLIM_DRAWS_PER_CHAIN, 1)
        groups = {name: idata[name] for name in config.SLIM_KEEP_GROUPS if name in idata.groups()}
        for name in ("posterior", "sample_stats"):
            groups[name] = groups[name].isel(draw=slice(None, None, step))
        drop = [v for v in config.SLIM_DROP_VARIABLES if v in groups["posterior"]]
        groups["posterior"] = groups["posterior"].drop_vars(drop).astype("float32")
        slim = az.InferenceData(**groups)
        slim.attrs.update(idata.attrs)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / config.MODEL_SLIM_FILENAME
        temporary = path.with_suffix(".tmp")
        slim.to_netcdf(str(temporary), compress=True)
        os.replace(temporary, path)
        return path

    @classmethod
    def load(cls, directory: Path = config.MODELS_DIR) -> Self:
        """Load a model saved with ``save``; no refitting needed.

        Falls back to the slim copy written by ``save_slim`` when the full file is absent,
        which is the case in a deployed copy of the app.
        """
        directory = Path(directory)
        meta = json.loads((directory / config.MODEL_META_FILENAME).read_text())
        model = cls()
        model.roi_transform_scale = meta.get("roi_transform_scale")
        if model.roi_transform_scale:
            # The saved model refers to its ROI transform by name; make it available again.
            register_roi_transform(np.array(model.roi_transform_scale))
        full = directory / config.MODEL_FILENAME
        path = full if full.exists() else directory / config.MODEL_SLIM_FILENAME
        model.mmm = MMM.load(str(path))
        model.settings = ModelSettings.model_validate(meta["settings"])
        model.channels = meta["channels"]
        model.controls = meta["controls"]
        model.fit_seconds = meta["fit_seconds"]
        return model
