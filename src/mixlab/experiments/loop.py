"""The test-and-learn loop on synthetic data: simulate a geo test from the truth, then calibrate.

``simulate_geo_test`` changes a channel's spend in the test regions for the last weeks of the
panel and recomputes their revenue with the TRUE response (including carryover), as if the
test had been run. ``to_experiment`` turns the analysed result into the calibration input the
national model takes: national weekly spend before and during the test, and the measured lift.
"""

import dataclasses
from typing import Any

import numpy as np
import pandas as pd

from mixlab import config
from mixlab.calibration import Experiment
from mixlab.config import BrandConfig
from mixlab.data_gen import channel_contribution
from mixlab.geo_data import regional_brands


def truth_brands(truth: dict[str, Any]) -> dict[str, BrandConfig]:
    """Rebuild each region's true brand from a geo ground-truth file."""
    brand = dataclasses.replace(config.BRAND_PRESETS[truth["national_brand"]], seed=truth["seed"])
    regions = tuple(r for r in config.INDIA_REGIONS if r.name in truth["regions"])
    return {b.name: b for b in regional_brands(brand, regions)}


def trial_window(panel: pd.DataFrame, weeks: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the first and last week of a test run over the panel's last ``weeks`` weeks.

    Raises:
        ValueError: If the test would leave too little history to fit the synthetic control.

    """
    dates = pd.DatetimeIndex(sorted(panel[config.DATE_COL].unique()))
    if weeks < 1 or len(dates) - weeks < config.SYNTHETIC_CONTROL_MIN_WEEKS:
        raise ValueError(
            f"A {weeks}-week test leaves under {config.SYNTHETIC_CONTROL_MIN_WEEKS} weeks of "
            "history to build the comparison from."
        )
    return dates[-weeks], dates[-1]


def simulate_geo_test(
    panel: pd.DataFrame,
    brands: dict[str, BrandConfig],
    channel: str,
    regions: list[str],
    weeks: int,
    multiplier: float,
) -> tuple[pd.DataFrame, float]:
    """Return the panel as if the test had run, and the TRUE incremental revenue it caused.

    In each test region, the channel's spend is multiplied during the window and revenue
    changes by exactly what the true response curve says (carryover inside the window
    included). Everything else, including noise, is unchanged.
    """
    start, _ = trial_window(panel, weeks)
    column = f"{config.SPEND_PREFIX}{channel}"
    result = panel.copy()
    true_lift = 0.0
    for region in regions:
        rows = (result[config.GEO_COL] == region).to_numpy()
        frame = result.loc[rows].sort_values(config.DATE_COL)
        spend = frame[column].to_numpy(dtype=float)
        window = (frame[config.DATE_COL] >= start).to_numpy()
        tested = np.where(window, np.round(spend * multiplier), spend)
        brand = brands[region]
        spec = next(c for c in brand.channels if c.name == channel)
        before = channel_contribution(spend, spec, brand.adstock_l_max, brand.true_saturation)
        after = channel_contribution(tested, spec, brand.adstock_l_max, brand.true_saturation)
        lift = after - before
        true_lift += float(lift.sum())
        result.loc[frame.index, column] = tested
        result.loc[frame.index, config.TARGET_COL] = frame[config.TARGET_COL] + np.round(lift)
    return result, true_lift


def to_experiment(
    analysis: dict[str, Any], national: pd.DataFrame, channel: str, extra_weekly: float
) -> Experiment:
    """Return the calibration input: national spend moved from x to x + extra, lift measured.

    ``national`` is the national weekly frame before the test; its spend in the test window
    is the baseline, and the test added ``extra_weekly`` on top.
    """
    start = pd.Timestamp(analysis["start"])
    dates = pd.to_datetime(national[config.DATE_COL])
    window = dates >= start
    baseline = float(national.loc[window, f"{config.SPEND_PREFIX}{channel}"].mean())
    last_day = dates[window].max() + pd.Timedelta(days=config.DAYS_PER_WEEK - 1)
    return Experiment(
        channel=channel,
        start_date=start.date(),
        end_date=last_day.date(),
        incremental_revenue=max(analysis["effect"], 1.0),
        standard_error=analysis["standard_error"],
        baseline_weekly_spend=baseline,
        test_weekly_spend=baseline + extra_weekly,
    )


def simulate_holdout_study(
    panel: pd.DataFrame,
    brands: dict[str, BrandConfig],
    channel: str,
    weeks: int,
    relative_se: float = config.SIMULATED_TEST_RELATIVE_SE,
    seed: int = config.RANDOM_SEED,
) -> Experiment:
    """Simulate a user-level holdout (conversion-lift) study over the panel's last weeks.

    For channels too small to show up in regional revenue, a platform holdout measures the
    channel's incremental revenue directly: here, the TRUE revenue its spend caused in the
    window across all regions ("ads on" versus "ads off"), plus measurement noise.
    """
    start, end = trial_window(panel, weeks)
    column = f"{config.SPEND_PREFIX}{channel}"
    truth = 0.0
    for region, brand in brands.items():
        frame = panel[panel[config.GEO_COL] == region].sort_values(config.DATE_COL)
        spend = frame[column].to_numpy(dtype=float)
        window = (frame[config.DATE_COL] >= start).to_numpy()
        spec = next(c for c in brand.channels if c.name == channel)
        on = channel_contribution(spend, spec, brand.adstock_l_max, brand.true_saturation)
        off = channel_contribution(
            np.where(window, 0.0, spend), spec, brand.adstock_l_max, brand.true_saturation
        )
        truth += float((on - off)[window].sum())
    error = relative_se * truth
    measured = truth + float(np.random.default_rng(seed).normal(0.0, error))
    return Experiment(
        channel=channel,
        start_date=start.date(),
        end_date=(end + pd.Timedelta(days=config.DAYS_PER_WEEK - 1)).date(),
        incremental_revenue=measured,
        standard_error=error,
    )
