"""Tests for value of information, the geo-lift designer and analyzer, and the test loop."""

import numpy as np
import pandas as pd
import pytest

from mixlab import config
from mixlab.calibration import to_lift_measurements
from mixlab.experiments.analyze import (
    analyze_test,
    difference_in_differences,
    revenue_matrix,
    synthetic_control,
)
from mixlab.experiments.design import design_test, plan_text, rank_regions, weeks_needed
from mixlab.experiments.loop import (
    simulate_geo_test,
    simulate_holdout_study,
    to_experiment,
    trial_window,
    truth_brands,
)
from mixlab.experiments.value_of_information import (
    expected_loss,
    loss_after_test,
    shift,
    value_of_information,
)
from mixlab.geo_data import GeoDataset, aggregate_national
from mixlab.insights import PosteriorDraws, extract_draws
from mixlab.model import MixLabModel

RNG = np.random.default_rng(config.RANDOM_SEED)


def toy_panel(effect: float = 0.0, weeks: int = 80, test_weeks: int = 10) -> pd.DataFrame:
    """Five regions sharing one demand pattern; region 'a' gets ``effect`` a week at the end."""
    dates = pd.date_range("2024-01-01", periods=weeks, freq="7D")
    common = 100_000 + 10_000 * np.sin(np.arange(weeks) / 5)
    rows = []
    for index, region in enumerate("abcde"):
        revenue = common * (1 + 0.2 * index) + RNG.normal(0, 500, weeks)
        if region == "a":
            revenue[-test_weeks:] += effect
        for date, value in zip(dates, revenue, strict=True):
            rows.append({config.DATE_COL: date, config.GEO_COL: region, config.TARGET_COL: value})
    return pd.DataFrame(rows)


# --- Value of information -------------------------------------------------------------------


def test_shift_keeps_the_budget() -> None:
    plan = {"a": 100.0, "b": 300.0, "c": 600.0}
    moved = shift(plan, "a", 0.2)
    assert sum(moved.values()) == pytest.approx(1000.0)
    assert moved["a"] == pytest.approx(120.0)
    assert moved["b"] / moved["c"] == pytest.approx(0.5)


def test_expected_loss_is_zero_when_every_draw_agrees() -> None:
    values = np.array([[1.0, 2.0, 3.0], [0.5, 1.0, 4.0]])
    assert expected_loss(values) == 0.0
    disagree = np.array([[3.0, 0.0], [0.0, 1.0]])
    assert expected_loss(disagree) == pytest.approx(0.5)  # picks action 0, loses 1 in draw 2


def test_a_precise_test_removes_the_loss_and_a_useless_one_does_not() -> None:
    signal = RNG.normal(1.0, 0.5, 300)
    values = np.stack([np.zeros_like(signal), signal - 1.0], axis=1)  # raise only if ROI > 1
    before = expected_loss(values)
    assert loss_after_test(values, signal, 1e-3, 1) < 0.05 * before
    assert loss_after_test(values, signal, 1e3, 1) == pytest.approx(before, rel=0.2)


@pytest.fixture(scope="module")
def draws(df: pd.DataFrame, fitted: MixLabModel) -> PosteriorDraws:
    return extract_draws(fitted, df)


def test_value_of_information_ranks_every_channel(draws: PosteriorDraws) -> None:
    ranking = value_of_information(draws)
    assert [row["priority"] for row in ranking] == list(range(1, len(draws.channels) + 1))
    values = [row["value_of_test"] for row in ranking]
    assert values == sorted(values, reverse=True) and min(values) >= 0
    for row in ranking:
        assert row["expected_loss_after_test"] <= row["expected_loss_now"] + 1e-6
        assert row["decision_now"] in config.VOI_ACTIONS


# --- Analyzer -------------------------------------------------------------------------------


def test_synthetic_control_recovers_a_known_effect() -> None:
    panel = toy_panel(effect=5_000.0)
    start = sorted(panel[config.DATE_COL].unique())[-10]
    result = analyze_test(panel, ["a"], list("bcde"), start)
    assert result["effect"] == pytest.approx(50_000.0, rel=0.15)
    assert result["range_95"][0] < 50_000.0 < result["range_95"][1]
    assert result["did_effect"] == pytest.approx(50_000.0, rel=0.3)
    assert result["placebo_p_value"] == pytest.approx(0.2)  # 1 / (4 placebos + 1)
    assert result["pre_period_mape_pct"] < 2


def test_no_effect_gives_a_range_around_zero() -> None:
    panel = toy_panel(effect=0.0)
    wide = revenue_matrix(panel)
    start = wide.index[-10]
    result = synthetic_control(wide, ["a"], list("bcde"), start)
    assert abs(result["effect"]) < 3 * result["standard_error"]
    assert abs(difference_in_differences(wide, ["a"], list("bcde"), start)) < 20_000


# --- Designer -------------------------------------------------------------------------------


def test_weeks_needed_follows_the_power_formula() -> None:
    assert weeks_needed(100.0, 280.0) == pytest.approx(1.0)
    assert weeks_needed(100.0, 140.0) == pytest.approx(4.0)
    assert weeks_needed(100.0, 0.0) == float("inf")


def test_design_picks_matched_regions_and_the_smallest_feasible_change(
    geo_dataset: GeoDataset,
) -> None:
    panel = geo_dataset.data
    ranking = rank_regions(panel)
    assert [r["fit_error_pct"] for r in ranking] == sorted(r["fit_error_pct"] for r in ranking)
    plan = design_test(panel, "meta_ads", marginal_roi=1.0, n_test=1)
    assert len(plan["test_regions"]) == 1 and len(plan["control_regions"]) == 2
    feasible = [o for o in plan["options"] if o["feasible"]]
    assert plan["chosen"] == (feasible[0] if feasible else plan["options"][-1])
    tiny = design_test(panel, "email", marginal_roi=1e-6, n_test=1)
    assert not tiny["feasible"]
    text = plan_text(plan, {}, "2025-10-06", "2025-12-22")
    assert plan["test_regions"][0] in text and "Decision rule" in text


# --- Loop -----------------------------------------------------------------------------------


def test_simulated_test_changes_only_the_test_regions_in_the_window(
    geo_dataset: GeoDataset,
) -> None:
    panel = geo_dataset.data
    brands = truth_brands(geo_dataset.ground_truth)
    tested, true_lift = simulate_geo_test(panel, brands, "meta_ads", ["karnataka"], 8, 2.0)
    start, _ = trial_window(panel, 8)
    changed = tested[config.TARGET_COL] != panel[config.TARGET_COL]
    assert set(tested.loc[changed, config.GEO_COL]) == {"karnataka"}
    assert (tested.loc[changed, config.DATE_COL] >= start).all()
    assert true_lift > 0
    gain = (tested[config.TARGET_COL] - panel[config.TARGET_COL]).sum()
    assert gain == pytest.approx(true_lift, abs=10.0)


def test_window_needs_enough_history(geo_dataset: GeoDataset) -> None:
    with pytest.raises(ValueError, match="history"):
        trial_window(geo_dataset.data, config.OPTIMIZER_WEEKS * 20)


def test_holdout_and_geo_results_become_calibration_input(geo_dataset: GeoDataset) -> None:
    panel = geo_dataset.data
    national = aggregate_national(panel)
    brands = truth_brands(geo_dataset.ground_truth)
    holdout = simulate_holdout_study(panel, brands, "email", 13)
    assert holdout.incremental_revenue > 0 and holdout.baseline_weekly_spend is None
    analysis = {
        "start": str(trial_window(panel, 8)[0].date()),
        "effect": 1e6,
        "standard_error": 2e5,
    }
    geo = to_experiment(analysis, national, "tv", extra_weekly=50_000.0)
    assert geo.test_weekly_spend - geo.baseline_weekly_spend == pytest.approx(50_000.0)
    lift = to_lift_measurements([holdout, geo], national)
    assert lift["delta_x"].gt(0).all() and lift["sigma"].gt(0).all()
