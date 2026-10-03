"""Lift-test calibration: use experiment results to pin down channel effects.

An MMM infers each channel's effect from correlations in weekly data, which leaves wide
ranges when channels move together. An experiment (a geo test or a platform conversion-lift
study) measures one channel's incremental revenue directly. Calibration adds that measurement
to the model, using PyMC-Marketing's ``add_lift_test_measurements``, so the channel's response
curve must agree with both the weekly data and the experiment.

The library compares points on the saturation curve and ignores carryover, so an experiment
is translated into average weekly spend and average weekly lift over its duration.
"""

import math
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from pydantic import BaseModel, Field, model_validator

from mixlab import config
from mixlab.config import BrandConfig, ModelSettings
from mixlab.data_gen import channel_contribution
from mixlab.model import MixLabModel


class Experiment(BaseModel):
    """One measured experiment on one channel.

    Attributes:
        channel: Channel name without the ``spend_`` prefix.
        start_date: First day of the test.
        end_date: Last day of the test.
        incremental_revenue: Revenue the test attributes to the spend change, in INR, for the
            whole test period.
        standard_error: Standard error of that measurement, in INR.
        baseline_weekly_spend: Weekly spend in the comparison condition. Defaults to 0, which
            is what a conversion-lift study or "ads on vs. ads off" geo test measures.
        test_weekly_spend: Weekly spend in the test condition. Defaults to the average spend
            recorded in the data during the test.

    """

    channel: str
    start_date: date
    end_date: date
    incremental_revenue: float = Field(gt=0)
    standard_error: float = Field(gt=0)
    baseline_weekly_spend: float | None = Field(default=None, ge=0)
    test_weekly_spend: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _dates_in_order(self) -> "Experiment":
        if self.end_date < self.start_date:
            raise ValueError("end_date is before start_date")
        return self

    @property
    def weeks(self) -> float:
        """Return the length of the test in weeks."""
        return ((self.end_date - self.start_date).days + 1) / config.DAYS_PER_WEEK


def load_experiments(path: Path) -> list[Experiment]:
    """Read experiments from a CSV with one row per experiment.

    Required columns: channel, start_date, end_date, incremental_revenue, standard_error.
    Optional: baseline_weekly_spend, test_weekly_spend.
    """
    frame = pd.read_csv(path)
    records = frame.astype(object).where(frame.notna(), None).to_dict(orient="records")
    return [Experiment.model_validate(record) for record in records]


def to_lift_measurements(experiments: list[Experiment], df: pd.DataFrame) -> pd.DataFrame:
    """Translate experiments into the per-week form PyMC-Marketing calibrates against.

    Returns a frame with ``channel`` (the spend column), ``x`` (baseline weekly spend),
    ``delta_x`` (change in weekly spend), ``delta_y`` (change in weekly revenue) and
    ``sigma`` (its standard error).

    Raises:
        ValueError: If a channel is not in the data or the test changed no spend.

    """
    dates = pd.to_datetime(df[config.DATE_COL])
    rows = []
    for experiment in experiments:
        column = f"{config.SPEND_PREFIX}{experiment.channel}"
        if column not in df.columns:
            raise ValueError(f"Experiment channel '{experiment.channel}' is not in the data.")
        window = (dates >= pd.Timestamp(experiment.start_date)) & (
            dates <= pd.Timestamp(experiment.end_date)
        )
        observed = float(df.loc[window, column].mean()) if window.any() else 0.0
        baseline = experiment.baseline_weekly_spend or 0.0
        tested = (
            experiment.test_weekly_spend if experiment.test_weekly_spend is not None else observed
        )
        if math.isclose(tested, baseline):
            raise ValueError(
                f"The '{experiment.channel}' experiment changes no spend: baseline and test "
                f"weekly spend are both {baseline:,.0f}. Give test_weekly_spend explicitly."
            )
        rows.append(
            {
                "channel": column,
                "x": baseline,
                "delta_x": tested - baseline,
                "delta_y": experiment.incremental_revenue / experiment.weeks,
                "sigma": experiment.standard_error / experiment.weeks,
            }
        )
    return pd.DataFrame(rows)


def simulate_lift_test(
    brand: BrandConfig,
    df: pd.DataFrame,
    channel: str,
    start: str,
    end: str,
    relative_se: float = config.SIMULATED_TEST_RELATIVE_SE,
    seed: int = config.RANDOM_SEED,
) -> Experiment:
    """Simulate an "ads off" test from the ground truth of a synthetic brand.

    The true incremental revenue is what the data generator says the channel's spend in the
    test window caused (including carryover): revenue with the spend minus revenue with the
    window switched off. Measurement noise with the given relative standard error is added.
    """
    spec = next(c for c in brand.channels if c.name == channel)
    spend = df[f"{config.SPEND_PREFIX}{channel}"].to_numpy(dtype=np.float64)
    dates = pd.to_datetime(df[config.DATE_COL])
    window = ((dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))).to_numpy()
    switched_off = np.where(window, 0.0, spend)
    truth = float(
        channel_contribution(spend, spec, brand.adstock_l_max).sum()
        - channel_contribution(switched_off, spec, brand.adstock_l_max).sum()
    )
    error = relative_se * truth
    measured = truth + float(np.random.default_rng(seed).normal(0.0, error))
    last_day = pd.Timestamp(end) + pd.Timedelta(days=config.DAYS_PER_WEEK - 1)
    return Experiment(
        channel=channel,
        start_date=pd.Timestamp(start).date(),
        end_date=last_day.date(),
        incremental_revenue=measured,
        standard_error=error,
    )


def fit_calibrated(
    df: pd.DataFrame, settings: ModelSettings, experiments: list[Experiment]
) -> MixLabModel:
    """Build a model, add the experiments as calibration, and fit it."""
    model = MixLabModel().build(df, settings)
    model.add_lift_tests(to_lift_measurements(experiments, df))
    model.fit(progressbar=False)
    return model


def compare_roi(
    before: dict[str, Any], after: dict[str, Any], ground_truth: dict[str, Any] | None = None
) -> pd.DataFrame:
    """Return ROI per channel before and after calibration, with ranges and their widths.

    ``before`` and ``after`` are insights summaries. ``range_shrink_pct`` is how much narrower
    the 94% range became.
    """
    rows = []
    for channel, metrics in before["channels"].items():
        old, new = metrics["roi"], after["channels"][channel]["roi"]
        old_width, new_width = old["hdi_high"] - old["hdi_low"], new["hdi_high"] - new["hdi_low"]
        row = {
            "channel": channel,
            "before": old["mean"],
            "before_low": old["hdi_low"],
            "before_high": old["hdi_high"],
            "after": new["mean"],
            "after_low": new["hdi_low"],
            "after_high": new["hdi_high"],
            "range_shrink_pct": 100 * (1 - new_width / old_width),
        }
        if ground_truth:
            row["true_roi"] = ground_truth["channels"][channel]["true_roi"]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_before_after(table: pd.DataFrame, tested: list[str]) -> Figure:
    """Return ROI ranges before and after calibration for every channel."""
    fig = Figure(figsize=(10, 0.8 * len(table) + 2.0), facecolor=config.COLOR_SURFACE)
    ax = fig.subplots()
    for row, record in enumerate(table.itertuples()):
        for offset, prefix, color, label in (
            (-0.16, "before", config.COLOR_TEXT_MUTED, "Before calibration"),
            (0.16, "after", config.COLOR_SPEND, "After calibration"),
        ):
            low = max(getattr(record, f"{prefix}_low"), config.ROI_AXIS_LIMITS[0])
            high = getattr(record, f"{prefix}_high")
            ax.plot(
                [low, high],
                [row + offset] * 2,
                color=color,
                linewidth=3,
                alpha=0.55,
                solid_capstyle="round",
            )
            ax.plot(
                getattr(record, prefix),
                row + offset,
                "o",
                color=color,
                markersize=8,
                markeredgecolor=config.COLOR_SURFACE,
                label=label if row == 0 else None,
            )
        if hasattr(record, "true_roi"):
            ax.plot(
                [record.true_roi] * 2,
                [row - 0.34, row + 0.34],
                color=config.COLOR_TEXT,
                linewidth=2,
                label="True ROI" if row == 0 else None,
            )
    names = [f"{c}  (tested)" if c in tested else c for c in table["channel"]]
    ax.set_yticks(range(len(table)), names)
    ax.invert_yaxis()
    ax.set_xlabel(
        "Revenue per rupee spent, log scale (best estimate and 94% range)",
        color=config.COLOR_TEXT_MUTED,
        fontsize=9,
    )
    ax.set_xscale("log")
    ax.set_xlim(*config.ROI_AXIS_LIMITS)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:g}"))
    ax.set_title(
        "ROI before and after calibrating with a lift test",
        loc="left",
        fontsize=13,
        color=config.COLOR_TEXT,
    )
    ax.set_facecolor(config.COLOR_SURFACE)
    ax.grid(axis="x", color=config.COLOR_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=config.COLOR_TEXT_MUTED, length=0, labelsize=9)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def plan_tests(insights: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    """Suggest which channel to test next and roughly how long the test needs to run.

    Priority is uncertainty times spend: the width of a channel's ROI range multiplied by
    what was spent on it, which is the revenue the business cannot currently account for.

    Duration is a rough power calculation for an "ads off in half the market" geo test:
    weeks needed for the expected weekly effect (ROI x weekly spend) to stand out from
    week-to-week revenue noise at 80% power. It is a planning figure, not a test design.
    """
    revenue = df[config.TARGET_COL].to_numpy(dtype=np.float64)
    noise = float(np.std(np.diff(revenue), ddof=1) / np.sqrt(2))
    rows = []
    for channel, metrics in insights["channels"].items():
        roi = metrics["roi"]
        width = roi["hdi_high"] - roi["hdi_low"]
        weekly_effect = roi["mean"] * metrics["current_weekly_spend"]
        raw_weeks = (config.POWER_Z * config.GEO_DESIGN_FACTOR * noise / weekly_effect) ** 2
        weeks = int(np.clip(math.ceil(raw_weeks), config.MIN_TEST_WEEKS, config.MAX_TEST_WEEKS))
        rows.append(
            {
                "channel": channel,
                "total_spend": metrics["total_spend"],
                "roi_low": roi["hdi_low"],
                "roi_high": roi["hdi_high"],
                "roi_range_width": width,
                "revenue_at_stake": width * metrics["total_spend"],
                "expected_weekly_effect": weekly_effect,
                "weeks_needed": weeks,
                "feasible_at_current_spend": raw_weeks <= config.MAX_TEST_WEEKS,
            }
        )
    plan = pd.DataFrame(rows).sort_values("revenue_at_stake", ascending=False)
    plan.insert(0, "priority", range(1, len(plan) + 1))
    return plan.reset_index(drop=True)
