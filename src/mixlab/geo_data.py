"""Synthetic geo-level panel: one national brand split across Indian regions.

Each region is generated with ``data_gen.generate`` from a regional copy of a national brand:

- organic revenue and spend scale with the region's size, and baseline demand varies;
- media prices differ, so a rupee buys more or less media (the half-saturation spend moves);
- the spend mix tilts away from the national mix, and weekly spend noise is independent;
- each channel's true effect is the national effect times a regional multiplier drawn from a
  shared LogNormal distribution (``config.GEO_EFFECT_SIGMA``);
- regional festivals (Onam, Durga Puja, Pongal, ...) lift revenue and pull spend in that
  region only, and TV also flights around the region's own festival.

The panel is long format: one row per (week, region). Summing it over regions gives the
national weekly frame a national model would see.

Run ``python -m mixlab.geo_data`` to write the panel and its ground truth.
"""

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mixlab import config
from mixlab.config import BrandConfig, ChannelConfig, EventConfig, RegionConfig
from mixlab.data_gen import generate
from mixlab.transforms import FloatArray


@dataclass(frozen=True)
class GeoDataset:
    """A generated geo panel plus everything that is normally unobservable.

    Attributes:
        data: Long weekly frame, one row per (date, geo), following the data contract.
        contributions: True revenue decomposition per (date, geo).
        ground_truth: True parameters and ROI per region and nationally, ready for JSON.

    """

    data: pd.DataFrame
    contributions: pd.DataFrame
    ground_truth: dict[str, Any]


def regional_multipliers(
    channels: tuple[ChannelConfig, ...],
    regions: tuple[RegionConfig, ...],
    seed: int,
    sigma: float,
) -> FloatArray:
    """Return LogNormal(0, sigma) multipliers (regions x channels) with a fixed seed."""
    rng = np.random.default_rng(seed)
    return rng.lognormal(mean=0.0, sigma=sigma, size=(len(regions), len(channels)))


def regional_events(region: RegionConfig) -> tuple[EventConfig, ...]:
    """Return the region's own festivals as calendar events."""
    return tuple(
        EventConfig(
            name=name,
            dates=dates,
            revenue_lift=config.REGIONAL_FESTIVAL_LIFT,
            spend_peak=config.REGIONAL_FESTIVAL_SPEND_PEAK,
            ramp_weeks=config.REGIONAL_FESTIVAL_RAMP_WEEKS,
        )
        for name, dates in region.festivals.items()
    )


def regional_channel(
    channel: ChannelConfig, region: RegionConfig, effect: float, tilt: float
) -> ChannelConfig:
    """Return the region's copy of a national channel.

    Spend scales with size and the regional tilt; the half-saturation spend scales with size
    and media price (a pricier market needs more rupees to reach the same audience); the
    ceiling scales with size, demand and the regional effect multiplier.
    """
    flights = channel.flight_events
    if flights:
        flights = (*flights, *region.festivals)
    return replace(
        channel,
        base_spend=channel.base_spend * region.size * tilt,
        half_saturation=channel.half_saturation * region.size * region.media_price,
        beta=channel.beta * region.size * region.demand * effect,
        flight_events=flights,
    )


def regional_brand(
    brand: BrandConfig,
    region: RegionConfig,
    index: int,
    effects: FloatArray,
    tilts: FloatArray,
) -> BrandConfig:
    """Return the regional brand: a scaled copy of ``brand`` with its own seed and festivals."""
    return replace(
        brand,
        name=region.name,
        seed=brand.seed + config.GEO_SEED_STRIDE * (index + 1),
        baseline_revenue=brand.baseline_revenue * region.size * region.demand,
        events=(*brand.events, *regional_events(region)),
        channels=tuple(
            regional_channel(channel, region, effects[c], tilts[c])
            for c, channel in enumerate(brand.channels)
        ),
    )


def regional_brands(
    brand: BrandConfig, regions: tuple[RegionConfig, ...] = config.INDIA_REGIONS
) -> list[BrandConfig]:
    """Return one regional brand per region, with effects and spend tilts drawn from the seed."""
    effects = regional_multipliers(brand.channels, regions, brand.seed, config.GEO_EFFECT_SIGMA)
    tilts = regional_multipliers(
        brand.channels, regions, brand.seed + 1, config.GEO_SPEND_TILT_SIGMA
    )
    return [
        regional_brand(brand, region, index, effects[index], tilts[index])
        for index, region in enumerate(regions)
    ]


def holiday_columns(frames: list[pd.DataFrame]) -> list[str]:
    """Return every holiday column used by any region, in first-seen order."""
    seen: dict[str, None] = {}
    for frame in frames:
        for column in frame.columns:
            if column.startswith(config.HOLIDAY_PREFIX):
                seen[column] = None
    return list(seen)


def channel_truth(
    data: pd.DataFrame, contributions: pd.DataFrame, channels: list[str]
) -> dict[str, dict[str, float]]:
    """Return total spend, total true contribution and true ROI per channel."""
    total_revenue = float(data[config.TARGET_COL].sum())
    truth: dict[str, dict[str, float]] = {}
    for name in channels:
        spend = float(data[f"{config.SPEND_PREFIX}{name}"].sum())
        contribution = float(contributions[name].sum())
        truth[name] = {
            "total_spend": spend,
            "total_contribution": contribution,
            "true_roi": contribution / spend if spend > 0 else 0.0,
            "contribution_pct_of_revenue": 100.0 * contribution / total_revenue,
        }
    return truth


def build_geo_ground_truth(
    brand: BrandConfig,
    regions: tuple[RegionConfig, ...],
    data: pd.DataFrame,
    contributions: pd.DataFrame,
    effects: FloatArray,
) -> dict[str, Any]:
    """Return true ROI per region and channel, and nationally (all regions summed)."""
    channels = [c.name for c in brand.channels]
    regional: dict[str, Any] = {}
    for index, region in enumerate(regions):
        rows = data[config.GEO_COL] == region.name
        regional[region.name] = {
            "label": region.label,
            "lat": region.lat,
            "lon": region.lon,
            "size": region.size,
            "demand": region.demand,
            "media_price": region.media_price,
            "festivals": list(region.festivals),
            "revenue": float(data.loc[rows, config.TARGET_COL].sum()),
            "channels": {
                name: {**values, "effect_multiplier": float(effects[index, c])}
                for c, (name, values) in enumerate(
                    channel_truth(data[rows], contributions[rows.to_numpy()], channels).items()
                )
            },
        }
    national = channel_truth(data, contributions, channels)
    media = sum(values["total_contribution"] for values in national.values())
    return {
        "brand": config.GEO_DEMO_BRAND,
        "national_brand": brand.name,
        "seed": brand.seed,
        "n_weeks": brand.n_weeks,
        "start_date": brand.start_date,
        "currency": config.CURRENCY,
        "geo_effect_sigma": config.GEO_EFFECT_SIGMA,
        "totals": {
            "revenue": float(data[config.TARGET_COL].sum()),
            "media_contribution": media,
            "media_pct_of_revenue": 100.0 * media / float(data[config.TARGET_COL].sum()),
        },
        "channels": national,
        "regions": regional,
    }


def generate_geo(
    brand: BrandConfig = config.PERFORMANCE_HEAVY_BRAND,
    regions: tuple[RegionConfig, ...] = config.INDIA_REGIONS,
) -> GeoDataset:
    """Generate the geo panel for ``brand`` split across ``regions``."""
    effects = regional_multipliers(brand.channels, regions, brand.seed, config.GEO_EFFECT_SIGMA)
    datasets = [generate(cfg) for cfg in regional_brands(brand, regions)]
    holidays = holiday_columns([d.data for d in datasets])
    frames, parts = [], []
    for region, dataset in zip(regions, datasets, strict=True):
        frame = dataset.data.copy()
        for column in holidays:
            if column not in frame:
                frame[column] = 0
        frame.insert(1, config.GEO_COL, region.name)
        frames.append(frame)
        part = dataset.contributions.reset_index()
        part.insert(1, config.GEO_COL, region.name)
        parts.append(part)
    data = pd.concat(frames, ignore_index=True)
    spend = [c for c in data.columns if c.startswith(config.SPEND_PREFIX)]
    others = [c for c in data.columns if c not in (config.DATE_COL, config.GEO_COL, *spend)]
    data = data[[config.DATE_COL, config.GEO_COL, *spend, *others]]
    contributions = pd.concat(parts, ignore_index=True).fillna(0.0)
    truth = build_geo_ground_truth(brand, regions, data, contributions, effects)
    return GeoDataset(data, contributions, truth)


def aggregate_national(data: pd.DataFrame) -> pd.DataFrame:
    """Sum a geo panel into the national weekly frame a national model would be given.

    Revenue and spend are summed. A holiday column is 1 if the festival falls that week in
    any region; the promo flag becomes the share of regions on promotion and the price index
    the average across regions. Seasonality columns are identical across regions.
    """
    spend = [c for c in data.columns if c.startswith(config.SPEND_PREFIX)]
    holidays = [c for c in data.columns if c.startswith(config.HOLIDAY_PREFIX)]
    seasonal = [c for c in data.columns if c.startswith(config.SEASON_PREFIX)]
    rules: dict[str, str] = {config.TARGET_COL: "sum", **dict.fromkeys(spend, "sum")}
    rules.update(dict.fromkeys(holidays, "max"))
    rules.update(dict.fromkeys(seasonal, "first"))
    for column in (config.PROMO_COL, config.PRICE_COL):
        if column in data:
            rules[column] = "mean"
    national = data.groupby(config.DATE_COL, sort=True).agg(rules).reset_index()
    ordered = [config.DATE_COL, config.TARGET_COL, *spend]
    return national[ordered + [c for c in national.columns if c not in ordered]]


def save_geo_dataset(dataset: GeoDataset, out_dir: Path = config.SYNTHETIC_DIR) -> None:
    """Write the panel CSV, its true decomposition and the ground truth JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset.data.to_csv(out_dir / config.GEO_WEEKLY_FILENAME, index=False)
    dataset.contributions.to_csv(out_dir / config.GEO_TRUE_CONTRIBUTIONS_FILENAME, index=False)
    text = json.dumps(dataset.ground_truth, indent=2)
    (out_dir / config.GEO_GROUND_TRUTH_FILENAME).write_text(text + "\n")


def main() -> None:
    """Generate the geo panel from the command line and print true ROI by region."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--brand", choices=sorted(config.BRAND_PRESETS), default="performance_heavy"
    )
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--out", type=Path, default=config.SYNTHETIC_DIR)
    args = parser.parse_args()
    dataset = generate_geo(replace(config.BRAND_PRESETS[args.brand], seed=args.seed))
    save_geo_dataset(dataset, args.out)
    for name, truth in dataset.ground_truth["channels"].items():
        print(f"{name:<14} national ROI {truth['true_roi']:.2f}")
    for region, values in dataset.ground_truth["regions"].items():
        rois = "  ".join(f"{v['true_roi']:.2f}" for v in values["channels"].values())
        print(f"{region:<14} {rois}")


if __name__ == "__main__":
    main()
