"""Synthetic weekly marketing data generator with known ground-truth channel effects.

Revenue is built additively::

    revenue = baseline + trend + seasonality + holiday lift + promo + price
              + sum over channels of beta_c * hill(adstock(spend_c))
              + noise

Because every term is generated here, the true contribution and ROI of each channel are
known exactly and can be compared with what a fitted MMM recovers.

Run ``python -m mixlab.data_gen --brand performance_heavy`` to write the dataset, the ground
truth file and one figure per channel.
"""

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from numpy.typing import NDArray

from mixlab import config
from mixlab.config import BrandConfig, ChannelConfig, EventConfig
from mixlab.transforms import (
    FloatArray,
    geometric_adstock,
    hill_saturation,
    logistic_saturation,
)


@dataclass(frozen=True)
class SyntheticDataset:
    """A generated brand: the modelling frame plus everything that is normally unobservable.

    Attributes:
        data: Weekly frame following the data contract (date, revenue, spend, controls).
        contributions: Weekly true revenue decomposition, one column per component.
        ground_truth: True parameters and summary metrics, ready to dump as JSON.

    """

    data: pd.DataFrame
    contributions: pd.DataFrame
    ground_truth: dict[str, Any]


# --- Calendar and controls ------------------------------------------------------------------


def week_starts(cfg: BrandConfig) -> pd.DatetimeIndex:
    """Return the start date of each week in the series."""
    return pd.date_range(cfg.start_date, periods=cfg.n_weeks, freq=f"{config.DAYS_PER_WEEK}D")


def years_elapsed(dates: pd.DatetimeIndex) -> FloatArray:
    """Return time since the first week, in years."""
    return np.arange(len(dates), dtype=np.float64) / config.WEEKS_PER_YEAR


def fourier_features(dates: pd.DatetimeIndex, order: int) -> pd.DataFrame:
    """Return yearly sin/cos seasonality columns ``season_sin_k`` and ``season_cos_k``."""
    phase = 2.0 * np.pi * dates.dayofyear.to_numpy(dtype=np.float64) / config.DAYS_PER_YEAR
    columns: dict[str, FloatArray] = {}
    for k in range(1, order + 1):
        columns[f"{config.SEASON_PREFIX}sin_{k}"] = np.sin(k * phase)
        columns[f"{config.SEASON_PREFIX}cos_{k}"] = np.cos(k * phase)
    return pd.DataFrame(columns, index=dates)


def event_week_offsets(dates: pd.DatetimeIndex, event: EventConfig) -> list[int]:
    """Return the week number (0-based, may fall outside the series) of each occurrence."""
    days = (pd.to_datetime(list(event.dates)) - dates[0]).days
    return [int(d // config.DAYS_PER_WEEK) for d in days]


def event_flags(dates: pd.DatetimeIndex, events: tuple[EventConfig, ...]) -> pd.DataFrame:
    """Return one 0/1 ``holiday_<event>`` column per event, set in the week of the event."""
    flags: dict[str, NDArray[np.int64]] = {}
    for event in events:
        flag = np.zeros(len(dates), dtype=np.int64)
        for offset in event_week_offsets(dates, event):
            if 0 <= offset < len(dates):
                flag[offset] = 1
        flags[f"{config.HOLIDAY_PREFIX}{event.name}"] = flag
    return pd.DataFrame(flags, index=dates)


def festive_index(dates: pd.DatetimeIndex, events: tuple[EventConfig, ...]) -> FloatArray:
    """Return a smooth index of marketing intensity that builds up around each event."""
    index = np.zeros(len(dates), dtype=np.float64)
    for event in events:
        for date in pd.to_datetime(list(event.dates)):
            weeks_away = (dates - date).days.to_numpy(dtype=np.float64) / config.DAYS_PER_WEEK
            index += event.spend_peak * np.exp(-0.5 * (weeks_away / event.ramp_weeks) ** 2)
    return index


def promo_flags(n_weeks: int, probability: float, rng: np.random.Generator) -> FloatArray:
    """Return a random 0/1 flag per week marking brand promotions."""
    return (rng.random(n_weeks) < probability).astype(np.float64)


def price_index(dates: pd.DatetimeIndex, cfg: BrandConfig, rng: np.random.Generator) -> FloatArray:
    """Return a price index near 1.0 that steps up each calendar year, with small noise."""
    years_since_start = (dates.year - dates.year[0]).to_numpy(dtype=np.float64)
    steps = (1.0 + cfg.price_annual_increase) ** years_since_start
    return np.round(steps + rng.normal(0.0, cfg.price_noise, len(dates)), 4)


# --- Organic revenue ------------------------------------------------------------------------


def organic_components(
    cfg: BrandConfig,
    dates: pd.DatetimeIndex,
    seasonal: pd.DataFrame,
    holidays: pd.DataFrame,
    promo: FloatArray,
    price: FloatArray,
) -> pd.DataFrame:
    """Return the non-media revenue components in INR, one column each."""
    base = cfg.baseline_revenue
    seasonality = np.zeros(len(dates), dtype=np.float64)
    for k, (sin_coef, cos_coef) in enumerate(cfg.seasonality, start=1):
        seasonality += sin_coef * seasonal[f"{config.SEASON_PREFIX}sin_{k}"].to_numpy()
        seasonality += cos_coef * seasonal[f"{config.SEASON_PREFIX}cos_{k}"].to_numpy()
    holiday = np.zeros(len(dates), dtype=np.float64)
    for event in cfg.events:
        flag = holidays[f"{config.HOLIDAY_PREFIX}{event.name}"].to_numpy(dtype=np.float64)
        holiday += event.revenue_lift * flag
    return pd.DataFrame(
        {
            "baseline": np.full(len(dates), base),
            "trend": base * cfg.annual_trend * years_elapsed(dates),
            "seasonality": base * seasonality,
            "holiday": base * holiday,
            "promo": base * cfg.promo_lift * promo,
            "price": base * cfg.price_sensitivity * (price - 1.0),
        },
        index=dates,
    )


def demand_index(organic: pd.DataFrame, cfg: BrandConfig, rng: np.random.Generator) -> FloatArray:
    """Return latent consumer demand relative to baseline (1.0 is a normal week).

    Demand rises with seasonality, holidays and promos. Channels with a non-zero
    ``demand_elasticity`` (search) spend more when demand is high, which is the confounding
    a real MMM has to untangle.
    """
    drivers = organic[["seasonality", "holiday", "promo"]].sum(axis=1).to_numpy()
    index = 1.0 + drivers / cfg.baseline_revenue + rng.normal(0.0, cfg.demand_noise, len(organic))
    return np.maximum(index, config.MIN_DEMAND_INDEX)


# --- Media ----------------------------------------------------------------------------------


def flight_mask(dates: pd.DatetimeIndex, channel: ChannelConfig, cfg: BrandConfig) -> FloatArray:
    """Return 1.0 for weeks the channel is on air, 0.0 otherwise (all ones if always on)."""
    if not channel.flight_events:
        return np.ones(len(dates), dtype=np.float64)
    mask = np.zeros(len(dates), dtype=np.float64)
    for event in cfg.events:
        if event.name not in channel.flight_events:
            continue
        for offset in event_week_offsets(dates, event):
            start = max(offset - channel.flight_weeks + 1, 0)
            stop = min(offset + 1, len(dates))
            if start < stop:
                mask[start:stop] = 1.0
    return mask


def channel_spend(
    channel: ChannelConfig,
    cfg: BrandConfig,
    dates: pd.DatetimeIndex,
    festive: FloatArray,
    demand: FloatArray,
    rng: np.random.Generator,
) -> FloatArray:
    """Return weekly spend in whole INR: base x growth x festive x demand x noise x flight."""
    growth = (1.0 + channel.annual_growth) ** years_elapsed(dates)
    festive_boost = 1.0 + channel.festive_sensitivity * festive
    demand_pull = demand**channel.demand_elasticity
    sigma = channel.spend_noise
    noise = rng.lognormal(mean=-0.5 * sigma**2, sigma=sigma, size=len(dates))
    spend = channel.base_spend * growth * festive_boost * demand_pull * noise
    return np.round(spend * flight_mask(dates, channel, cfg))


def channel_contribution(
    spend: FloatArray, channel: ChannelConfig, l_max: int, form: str = "hill"
) -> FloatArray:
    """Return the true weekly revenue in INR caused by a channel's spend.

    ``form`` selects the true response curve: Hill (default), or a logistic curve that reaches
    half its ceiling at the same spend.
    """
    adstocked = geometric_adstock(spend, channel.adstock_decay, l_max, normalize=True)
    if form == "logistic":
        speed = np.log(3.0) / channel.half_saturation
        return channel.beta * logistic_saturation(adstocked, speed)
    return channel.beta * hill_saturation(adstocked, channel.half_saturation, channel.hill_slope)


# --- Assembly -------------------------------------------------------------------------------


def build_ground_truth(
    cfg: BrandConfig, data: pd.DataFrame, contributions: pd.DataFrame
) -> dict[str, Any]:
    """Return the true parameters plus true ROI and contribution share for each channel."""
    total_revenue = float(data[config.TARGET_COL].sum())
    names = [c.name for c in cfg.channels]
    total_media = float(contributions[names].sum().sum())
    channels: dict[str, dict[str, float]] = {}
    for channel in cfg.channels:
        spend = float(data[f"{config.SPEND_PREFIX}{channel.name}"].sum())
        contribution = float(contributions[channel.name].sum())
        channels[channel.name] = {
            "adstock_decay": channel.adstock_decay,
            "half_saturation": channel.half_saturation,
            "hill_slope": channel.hill_slope,
            "beta": channel.beta,
            "total_spend": spend,
            "total_contribution": contribution,
            "true_roi": contribution / spend,
            "contribution_pct_of_revenue": 100.0 * contribution / total_revenue,
            "contribution_pct_of_media": 100.0 * contribution / total_media,
        }
    return {
        "brand": cfg.name,
        "seed": cfg.seed,
        "n_weeks": cfg.n_weeks,
        "start_date": cfg.start_date,
        "currency": config.CURRENCY,
        "adstock": {"type": "geometric", "l_max": cfg.adstock_l_max, "normalize": True},
        "saturation": {"type": "hill", "formula": "x^slope / (half_saturation^slope + x^slope)"},
        "organic": {
            "baseline_revenue": cfg.baseline_revenue,
            "annual_trend": cfg.annual_trend,
            "seasonality": [list(pair) for pair in cfg.seasonality],
            "holiday_lift": {e.name: e.revenue_lift for e in cfg.events},
            "promo_lift": cfg.promo_lift,
            "price_sensitivity": cfg.price_sensitivity,
            "revenue_noise": cfg.revenue_noise,
        },
        "totals": {
            "revenue": total_revenue,
            "media_contribution": total_media,
            "media_pct_of_revenue": 100.0 * total_media / total_revenue,
        },
        "channels": channels,
    }


def generate(cfg: BrandConfig) -> SyntheticDataset:
    """Generate one brand's weekly data, true decomposition and ground truth from ``cfg``."""
    rng = np.random.default_rng(cfg.seed)
    dates = week_starts(cfg)
    seasonal = fourier_features(dates, order=len(cfg.seasonality))
    holidays = event_flags(dates, cfg.events)
    promo = promo_flags(cfg.n_weeks, cfg.promo_probability, rng)
    price = price_index(dates, cfg, rng)
    organic = organic_components(cfg, dates, seasonal, holidays, promo, price)
    demand = demand_index(organic, cfg, rng)
    festive = festive_index(dates, cfg.events)

    spend: dict[str, FloatArray] = {}
    media: dict[str, FloatArray] = {}
    for channel in cfg.channels:
        spend[channel.name] = channel_spend(channel, cfg, dates, festive, demand, rng)
        media[channel.name] = channel_contribution(
            spend[channel.name], channel, cfg.adstock_l_max, cfg.true_saturation
        )

    contributions = organic.assign(**media)
    contributions["noise"] = rng.normal(0.0, cfg.revenue_noise * cfg.baseline_revenue, cfg.n_weeks)
    revenue = np.round(contributions.sum(axis=1).to_numpy())
    # Fold the rounding remainder into noise so the components sum exactly to revenue.
    contributions["noise"] += revenue - contributions.sum(axis=1).to_numpy()
    contributions.index.name = config.DATE_COL

    data = pd.DataFrame({config.DATE_COL: dates, config.TARGET_COL: revenue})
    for name, values in spend.items():
        data[f"{config.SPEND_PREFIX}{name}"] = values
    data[config.PROMO_COL] = promo.astype(np.int64)
    for column in holidays.columns:
        data[column] = holidays[column].to_numpy()
    data[config.PRICE_COL] = price
    for column in seasonal.columns:
        data[column] = seasonal[column].to_numpy()

    return SyntheticDataset(data, contributions, build_ground_truth(cfg, data, contributions))


def save_dataset(dataset: SyntheticDataset, out_dir: Path = config.SYNTHETIC_DIR) -> None:
    """Write the weekly CSV, the true decomposition CSV and the ground truth JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset.data.to_csv(out_dir / config.WEEKLY_DATA_FILENAME, index=False)
    dataset.contributions.to_csv(out_dir / config.TRUE_CONTRIBUTIONS_FILENAME)
    text = json.dumps(dataset.ground_truth, indent=2)
    (out_dir / config.GROUND_TRUTH_FILENAME).write_text(text + "\n")


# --- Figures --------------------------------------------------------------------------------


def _lakh(value: float, _position: int) -> str:
    """Format an INR amount as lakh for axis ticks."""
    return f"{value / config.INR_PER_LAKH:,.10g}"


def plot_channel(dataset: SyntheticDataset, channel: str) -> Figure:
    """Return a two-panel figure: weekly spend above, true revenue contribution below."""
    truth = dataset.ground_truth["channels"][channel]
    dates = dataset.data[config.DATE_COL]
    panels = (
        (dataset.data[f"{config.SPEND_PREFIX}{channel}"], "Spend (INR lakh)", config.COLOR_SPEND),
        (
            dataset.contributions[channel].to_numpy(),
            "True revenue contribution (INR lakh)",
            config.COLOR_CONTRIBUTION,
        ),
    )
    fig = Figure(figsize=config.FIGURE_SIZE, facecolor=config.COLOR_SURFACE)
    axes = fig.subplots(2, 1, sharex=True)
    for ax, (values, label, color) in zip(axes, panels, strict=True):
        ax.plot(dates, values, color=color, linewidth=2)
        ax.fill_between(dates, values, color=color, alpha=0.12, linewidth=0)
        ax.set_title(label, loc="left", fontsize=10, color=config.COLOR_TEXT_MUTED)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(FuncFormatter(_lakh))
        ax.set_facecolor(config.COLOR_SURFACE)
        ax.grid(axis="y", color=config.COLOR_GRID, linewidth=0.8)
        ax.tick_params(colors=config.COLOR_TEXT_MUTED, length=0, labelsize=9)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(config.COLOR_GRID)
    fig.suptitle(
        f"{channel}: weekly spend and the revenue it truly caused",
        x=0.06,
        ha="left",
        fontsize=13,
        color=config.COLOR_TEXT,
    )
    fig.text(
        0.06,
        0.915,
        f"True ROI {truth['true_roi']:.2f}  |  {truth['contribution_pct_of_revenue']:.1f}% of "
        f"revenue  |  adstock decay {truth['adstock_decay']:.2f}  |  "
        f"brand: {dataset.ground_truth['brand']}",
        fontsize=9,
        color=config.COLOR_TEXT_MUTED,
    )
    fig.subplots_adjust(left=0.06, right=0.98, top=0.84, bottom=0.07, hspace=0.3)
    return fig


def save_channel_plots(dataset: SyntheticDataset, out_dir: Path = config.FIGURES_DIR) -> list[Path]:
    """Save one spend-versus-true-contribution figure per channel and return the paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for channel in dataset.ground_truth["channels"]:
        path = out_dir / f"ground_truth_{channel}.png"
        plot_channel(dataset, channel).savefig(path, dpi=config.FIGURE_DPI)
        paths.append(path)
    return paths


def main() -> None:
    """Generate a brand from the command line and write data, ground truth and figures."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--brand",
        choices=sorted(config.BRAND_PRESETS),
        default=config.PERFORMANCE_HEAVY_BRAND.name,
    )
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()
    dataset = generate(replace(config.BRAND_PRESETS[args.brand], seed=args.seed))
    save_dataset(dataset)
    save_channel_plots(dataset)
    for name, truth in dataset.ground_truth["channels"].items():
        print(
            f"{name:<14} ROI {truth['true_roi']:.2f}  "
            f"{truth['contribution_pct_of_revenue']:5.1f}% of revenue"
        )


if __name__ == "__main__":
    main()
