"""Geo-level hierarchical MMM: one model across regions, channel ROI partially pooled.

The same model as ``mixlab.model`` (Hill saturation, geometric adstock, ROI priors), fitted
on a panel with one row per (week, region) using PyMC-Marketing's ``dims=("geo",)``:

- revenue and spend are scaled by each region's own peak week;
- each region has its own baseline; seasonality, trend, control effects, carryover and the
  curve's shape are shared, since they are relative effects;
- channel ROI is hierarchical: ``log ROI[g, c] = log ROI_national[c] + tau[c] * z[g, c]``.
  A region with little or noisy data is pulled toward the national ROI, and every region
  informs the national figure, which is why ranges get tighter than a national-only model.

``model_for`` picks this model when the data has more than one region and falls back to the
national ``MixLabModel`` otherwise.
"""

from typing import Any, Self

import numpy as np
import pandas as pd
import xarray as xr
from pymc_extras.prior import Prior
from pymc_marketing.mmm.multidimensional import MMM
from pymc_marketing.mmm.scaling import Scaling, VariableScaling

from mixlab import config
from mixlab.config import ModelSettings
from mixlab.insights import PosteriorDraws, control_group
from mixlab.model import (
    ADSTOCKS,
    SATURATIONS,
    MixLabModel,
    build_priors,
    build_trend,
    default_control_columns,
    design_matrix,
    register_roi_transform,
    roi_scale,
    spend_shares,
)
from mixlab.validate import channel_name, spend_columns


def geo_names(df: pd.DataFrame) -> list[str]:
    """Return the regions in a panel, sorted (the order PyMC-Marketing uses for coordinates)."""
    if config.GEO_COL not in df.columns:
        return []
    return sorted(df[config.GEO_COL].astype(str).unique())


def is_geo_panel(df: pd.DataFrame) -> bool:
    """Return whether the frame holds more than one region."""
    return len(geo_names(df)) > 1


def geo_roi_scale(
    df: pd.DataFrame, channels: list[str], geos: list[str], settings: ModelSettings
) -> np.ndarray:
    """Return the coefficient that means "ROI of 1", per region and channel (G x C).

    Each region is scaled by its own peak week, so the national conversion in
    ``model.roi_scale`` is applied to each region's rows separately.
    """
    rows = [
        roi_scale(df[df[config.GEO_COL] == geo].reset_index(drop=True), channels, settings)
        for geo in geos
    ]
    return np.vstack(rows)


def geo_beta_prior(
    channels: list[str], settings: ModelSettings, scale: np.ndarray
) -> tuple[Prior, np.ndarray]:
    """Return the hierarchical effect-size prior and the (G x C) scale it uses.

    ``log ROI_national[c]`` gets the national ROI prior (centred on the benchmark or the
    median, width ``spread``); each region's log ROI sits ``tau[c] * z[g, c]`` away from it,
    with ``tau[c] ~ HalfNormal(config.GEO_ROI_POOL_SIGMA)`` learned from the data.
    """
    roi = settings.roi_prior
    if roi is None:
        raise ValueError("The geo model needs settings.roi_prior (see configs/geo.yaml).")
    centres = np.array([roi.benchmarks.get(channel_name(c), roi.median) for c in channels])
    prior = Prior(
        "Normal",
        mu=Prior("Normal", mu=np.log(centres).tolist(), sigma=roi.spread, dims="channel"),
        sigma=Prior("HalfNormal", sigma=config.GEO_ROI_POOL_SIGMA, dims="channel"),
        dims=(config.GEO_COL, "channel"),
        centered=False,
        transform=register_roi_transform(scale),
    )
    return prior, scale


def geo_priors(
    df: pd.DataFrame, channels: list[str], geos: list[str], settings: ModelSettings
) -> tuple[dict[str, Prior], np.ndarray]:
    """Return the model configuration for the geo model and the ROI scale it uses."""
    priors = build_priors(channels, settings, spend_shares(df, channels))
    p = settings.priors
    # Each region has its own baseline (in its own peak-week units).
    priors["intercept"] = Prior(
        "Normal", mu=p.intercept_mu, sigma=p.intercept_sigma, dims=config.GEO_COL
    )
    # One noise level: every region is scaled by its own peak, so noise is comparable.
    priors["likelihood"] = Prior(
        "Normal",
        sigma=Prior("HalfNormal", sigma=p.noise_sigma),
        dims=(config.DATE_COL, config.GEO_COL),
    )
    scale = geo_roi_scale(df, channels, geos, settings)
    priors["saturation_beta"], used = geo_beta_prior(channels, settings, scale)
    return priors, used


class GeoMixLabModel(MixLabModel):
    """The MixLab MMM fitted on a regional panel with partially pooled channel ROI."""

    def __init__(self) -> None:
        """Create an empty wrapper; call ``build`` or ``load`` next."""
        super().__init__()
        self.geos: list[str] = []

    @property
    def target_scale(self) -> float:
        """Return the national peak-week scale (largest regional scale); see ``geo_scales``."""
        return float(np.max(self.idata.constant_data["target_scale"].values))

    @property
    def geo_scales(self) -> xr.DataArray:
        """Return each region's peak weekly revenue in INR (the model's unit per region)."""
        return self.idata.constant_data["target_scale"]

    def build(self, df: pd.DataFrame, settings: ModelSettings | None = None) -> Self:
        """Define the model for a long panel with a ``geo`` column."""
        self.settings = settings or ModelSettings()
        if self.settings.saturation != "hill":
            raise ValueError("The geo model supports Hill saturation only (configs/geo.yaml).")
        self.geos = geo_names(df)
        if not self.geos:
            raise ValueError(f"Geo data needs a '{config.GEO_COL}' column.")
        df = df.sort_values([config.DATE_COL, config.GEO_COL]).reset_index(drop=True)
        self.channels = spend_columns(df)
        self.controls = self.settings.control_columns or default_control_columns(df)
        X, self.y = design_matrix(df, self.channels, self.controls, self.min_spend)
        X.insert(1, config.GEO_COL, df[config.GEO_COL].astype(str).to_numpy())
        self.X = X
        if self.y is None:
            raise ValueError(f"Training data needs a '{config.TARGET_COL}' column.")

        priors, used = geo_priors(df, self.channels, self.geos, self.settings)
        self.roi_transform_scale = used.tolist()
        sampler = self.settings.sampler
        per_region = VariableScaling(method="max", dims=())
        self.mmm = MMM(
            date_column=config.DATE_COL,
            channel_columns=self.channels,
            target_column=config.TARGET_COL,
            dims=(config.GEO_COL,),
            scaling=Scaling(target=per_region, channel=per_region),
            adstock=ADSTOCKS[self.settings.adstock](l_max=self.settings.adstock_l_max),
            saturation=SATURATIONS[self.settings.saturation](),
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
        return self

    def sample_prior_predictive(self) -> xr.DataArray:
        """Simulate revenue (INR) per week and region from the priors alone."""
        mmm = self._require_mmm()
        mmm.sample_prior_predictive(
            self.X,
            self.y,
            samples=self.settings.prior_predictive_samples,
            random_seed=self.settings.sampler.seed,
        )
        return self.idata.prior_predictive["y"] * self.geo_scales

    def save(self, directory: Any = config.MODELS_DIR, extra: dict[str, Any] | None = None) -> Any:
        """Save the fitted model; the region list is stored with the metadata."""
        return super().save(directory, {"geos": self.geos, **(extra or {})})

    @classmethod
    def load(cls, directory: Any = config.MODELS_DIR) -> Self:
        """Load a geo model saved with ``save``; no refitting needed."""
        model = super().load(directory)
        model.geos = [str(g) for g in model.idata.constant_data[config.GEO_COL].values]
        return model


def model_for(df: pd.DataFrame) -> MixLabModel:
    """Return the right empty model: geo for a multi-region panel, national otherwise."""
    return GeoMixLabModel() if is_geo_panel(df) else MixLabModel()


def region_frame(df: pd.DataFrame, geo: str) -> pd.DataFrame:
    """Return one region's weekly rows, in date order."""
    rows = df[df[config.GEO_COL].astype(str) == geo]
    return rows.sort_values(config.DATE_COL).reset_index(drop=True)


def extract_geo_draws(model: GeoMixLabModel, df: pd.DataFrame) -> dict[str, PosteriorDraws]:
    """Return one ``PosteriorDraws`` per region, in INR, so every national tool applies.

    Each region's draws hold its own spend, revenue, scales and effect sizes. The shared
    parameters (carryover, curve shape) are the same in every region. Draws are aligned:
    draw ``s`` in every region comes from the same posterior sample, so regional results can
    be summed draw by draw.
    """
    post = model.idata.posterior.stack(sample=("chain", "draw"))
    constant = model.idata.constant_data
    n_weeks = post.sizes[config.DATE_COL]

    def shared(name: str) -> np.ndarray:
        return post[name].transpose("sample", "channel").to_numpy()

    def weekly(name: str, geo: str) -> np.ndarray:
        values = post[name]
        if config.GEO_COL in values.dims:
            values = values.sel({config.GEO_COL: geo})
        return values.transpose("sample", config.DATE_COL).to_numpy()

    result: dict[str, PosteriorDraws] = {}
    for geo in model.geos:
        scale = float(constant["target_scale"].sel(geo=geo))
        at = {config.GEO_COL: geo}
        organic: dict[str, np.ndarray] = {
            "baseline": np.repeat(
                post["intercept_contribution"].sel(at).to_numpy()[:, None], n_weeks, axis=1
            ),
            "trend": weekly(f"{config.TREND_PREFIX}_effect_contribution", geo),
            "seasonality": weekly("yearly_seasonality_contribution", geo),
        }
        controls = (
            post["control_contribution"].sel(at).transpose("sample", config.DATE_COL, "control")
        )
        for index, control in enumerate(controls["control"].to_numpy()):
            group = control_group(str(control))
            organic[group] = organic.get(group, 0.0) + controls.to_numpy()[:, :, index]
        frame = region_frame(df, geo)
        result[geo] = PosteriorDraws(
            dates=pd.DatetimeIndex(post[config.DATE_COL].to_numpy()),
            channels=[channel_name(c) for c in model.channels],
            spend=frame[model.channels].to_numpy(dtype=np.float64),
            revenue=frame[config.TARGET_COL].to_numpy(dtype=np.float64),
            channel_scale=constant["channel_scale"].sel(at).to_numpy().astype(np.float64),
            target_scale=scale,
            l_max=model.settings.adstock_l_max,
            alpha=shared("adstock_alpha"),
            lam=np.zeros_like(shared("adstock_alpha")),
            saturation="hill",
            min_spend=model.min_spend,
            noise_sigma=post["y_sigma"].to_numpy() * scale,
            slope=shared("saturation_slope"),
            kappa=shared("saturation_kappa"),
            beta=post["saturation_beta"].sel(at).transpose("sample", "channel").to_numpy(),
            organic={name: np.asarray(values) * scale for name, values in organic.items()},
            channel_contribution=post["channel_contribution"]
            .sel(at)
            .transpose("sample", config.DATE_COL, "channel")
            .to_numpy()
            * scale,
        )
    return result
