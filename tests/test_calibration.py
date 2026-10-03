"""Tests for experiment handling, lift-test simulation, calibration and the test planner."""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from conftest import TINY
from pydantic import ValidationError

from mixlab import config
from mixlab.calibration import (
    Experiment,
    compare_roi,
    fit_calibrated,
    load_experiments,
    plan_tests,
    plot_before_after,
    simulate_lift_test,
    to_lift_measurements,
)
from mixlab.insights import build_summary, extract_draws
from mixlab.model import MixLabModel

BRAND = config.PERFORMANCE_HEAVY_BRAND
START, END = "2025-03-03", "2025-04-21"


def experiment(**overrides: object) -> Experiment:
    values = {
        "channel": "meta_ads",
        "start_date": date(2025, 3, 3),
        "end_date": date(2025, 4, 27),
        "incremental_revenue": 8_000_000.0,
        "standard_error": 800_000.0,
    }
    return Experiment(**{**values, **overrides})


def test_experiment_length_and_validation() -> None:
    assert experiment().weeks == 8
    with pytest.raises(ValidationError, match="before start_date"):
        experiment(end_date=date(2025, 1, 1))
    with pytest.raises(ValidationError):
        experiment(standard_error=0)


def test_experiments_load_from_the_shipped_template() -> None:
    (loaded,) = load_experiments(config.TEMPLATES_DIR / config.EXPERIMENTS_TEMPLATE_FILENAME)
    assert loaded.channel == "meta_ads" and loaded.baseline_weekly_spend is None
    assert loaded.weeks == 8


def test_lift_measurements_are_per_week_against_zero_spend(df: pd.DataFrame) -> None:
    (row,) = to_lift_measurements([experiment()], df).to_dict("records")
    window = (df["date"] >= "2025-03-03") & (df["date"] <= "2025-04-27")
    assert row["channel"] == "spend_meta_ads"
    assert row["x"] == 0
    assert row["delta_x"] == pytest.approx(df.loc[window, "spend_meta_ads"].mean())
    assert row["delta_y"] == pytest.approx(1_000_000.0)
    assert row["sigma"] == pytest.approx(100_000.0)


def test_lift_measurements_accept_explicit_spend_levels_and_reject_bad_input(
    df: pd.DataFrame,
) -> None:
    scale_up = experiment(baseline_weekly_spend=1_000_000.0, test_weekly_spend=1_500_000.0)
    (row,) = to_lift_measurements([scale_up], df).to_dict("records")
    assert (row["x"], row["delta_x"]) == (1_000_000.0, 500_000.0)
    with pytest.raises(ValueError, match="not in the data"):
        to_lift_measurements([experiment(channel="radio")], df)
    with pytest.raises(ValueError, match="changes no spend"):
        to_lift_measurements([experiment(baseline_weekly_spend=5.0, test_weekly_spend=5.0)], df)


def test_simulated_lift_test_matches_ground_truth(df: pd.DataFrame) -> None:
    exact = simulate_lift_test(BRAND, df, "google_search", START, END, relative_se=1e-9)
    noisy = simulate_lift_test(BRAND, df, "google_search", START, END, relative_se=0.1)
    window = (df["date"] >= START) & (df["date"] <= END)
    spend = df.loc[window, "spend_google_search"].sum()
    assert exact.weeks == 8
    assert 0.5 < exact.incremental_revenue / spend < 4  # a plausible ROI for search
    assert noisy.standard_error == pytest.approx(0.1 * exact.incremental_revenue, rel=1e-6)
    assert abs(noisy.incremental_revenue - exact.incremental_revenue) < 4 * noisy.standard_error
    assert simulate_lift_test(BRAND, df, "google_search", START, END) == noisy  # seeded


def test_planner_ranks_by_uncertainty_times_spend(df: pd.DataFrame, fitted: MixLabModel) -> None:
    insights = build_summary(extract_draws(fitted, df))
    plan = plan_tests(insights, df)
    assert plan["priority"].tolist() == list(range(1, 7))
    assert plan["revenue_at_stake"].is_monotonic_decreasing
    first = plan.iloc[0]
    assert first["revenue_at_stake"] == pytest.approx(
        first["roi_range_width"] * first["total_spend"]
    )
    assert plan["weeks_needed"].between(config.MIN_TEST_WEEKS, config.MAX_TEST_WEEKS).all()


def test_planner_needs_longer_tests_for_smaller_effects(
    df: pd.DataFrame, fitted: MixLabModel
) -> None:
    insights = build_summary(extract_draws(fitted, df))
    plan = plan_tests(insights, df).sort_values("expected_weekly_effect")
    assert plan["weeks_needed"].is_monotonic_decreasing


def test_calibration_adds_the_experiment_to_the_model_and_compares(
    df: pd.DataFrame, fitted: MixLabModel
) -> None:
    lift = simulate_lift_test(BRAND, df, "google_search", START, END)
    calibrated = fit_calibrated(df, TINY, [lift])
    assert config.LIFT_LIKELIHOOD_NAME in calibrated.mmm.model.named_vars
    assert config.LIFT_LIKELIHOOD_NAME not in fitted.mmm.model.named_vars

    before = build_summary(extract_draws(fitted, df))
    after = build_summary(extract_draws(calibrated, df))
    truth = {"channels": {c.name: {"true_roi": 1.0} for c in BRAND.channels}}
    table = compare_roi(before, after, truth)
    assert table["channel"].tolist() == [c.name for c in BRAND.channels]
    assert {"before_low", "after_high", "range_shrink_pct", "true_roi"} <= set(table.columns)
    search = table.set_index("channel").loc["google_search"]
    assert search["after_high"] - search["after_low"] < search["before_high"] - search["before_low"]
    assert plot_before_after(table, ["google_search"]).axes


def test_experiment_file_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "experiments.csv"
    pd.DataFrame([experiment().model_dump()]).to_csv(path, index=False)
    assert load_experiments(path) == [experiment()]
