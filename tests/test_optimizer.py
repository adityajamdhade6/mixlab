"""Tests for the budget optimizer and scenario simulator (uses the shared tiny fitted model)."""

import numpy as np
import pandas as pd
import pytest

from mixlab import config
from mixlab.insights import PosteriorDraws, extract_draws
from mixlab.model import MixLabModel
from mixlab.optimizer import (
    BudgetAllocator,
    budget_curve,
    build_bounds,
    compare_scenarios,
    last_quarter_spend,
    optimize_budget,
    payback_budget,
    plan_revenue_draws,
    plot_allocation,
    plot_budget_curve,
    prebuilt_scenarios,
    true_plan_revenue,
    validate_against_truth,
    what_if,
)

WEEKS = config.OPTIMIZER_WEEKS
BRAND = config.PERFORMANCE_HEAVY_BRAND
TOLERANCE = 1.0  # rupees


@pytest.fixture(scope="module")
def draws(df: pd.DataFrame, fitted: MixLabModel) -> PosteriorDraws:
    return extract_draws(fitted, df)


@pytest.fixture(scope="module")
def allocator(fitted: MixLabModel) -> BudgetAllocator:
    return BudgetAllocator(fitted, WEEKS)


@pytest.fixture(scope="module")
def current(draws: PosteriorDraws) -> dict[str, float]:
    return last_quarter_spend(draws, WEEKS)


def test_last_quarter_spend_matches_data(df: pd.DataFrame, current: dict[str, float]) -> None:
    assert current["meta_ads"] == pytest.approx(df["spend_meta_ads"].tail(WEEKS).sum())
    assert set(current) == {c.name for c in BRAND.channels}


def test_bounds_from_max_change_and_floor(current: dict[str, float]) -> None:
    budget = sum(current.values())
    bounds = build_bounds(current, budget, minimum={"tv": 2_000_000.0}, max_change=0.3)
    assert bounds["meta_ads"] == pytest.approx(
        (0.7 * current["meta_ads"], 1.3 * current["meta_ads"])
    )
    assert bounds["tv"][0] == 2_000_000.0
    assert build_bounds(current, budget)["email"] == (0.0, budget)


def test_infeasible_bounds_raise(current: dict[str, float]) -> None:
    budget = sum(current.values())
    with pytest.raises(ValueError, match="cannot be met"):
        build_bounds(current, budget * 2, max_change=0.3)
    with pytest.raises(ValueError, match="above maximum"):
        build_bounds(current, budget, minimum={"tv": 5e6}, maximum={"tv": 1e6})


def test_what_if_zero_spend_gives_zero_incremental_revenue(draws: PosteriorDraws) -> None:
    outcome = what_if(draws, {}, WEEKS)
    assert outcome.incremental_revenue.mean == 0
    assert outcome.total_revenue.mean > 0  # organic revenue remains


def test_what_if_rejects_bad_input(draws: PosteriorDraws) -> None:
    with pytest.raises(ValueError, match="Unknown channel"):
        what_if(draws, {"radio": 1e6})
    with pytest.raises(ValueError, match="negative"):
        what_if(draws, {"tv": -1.0})


def test_more_spend_on_a_channel_never_reduces_revenue(
    draws: PosteriorDraws, current: dict[str, float]
) -> None:
    base = plan_revenue_draws(draws, current, WEEKS)
    more = plan_revenue_draws(draws, {**current, "youtube": current["youtube"] * 2}, WEEKS)
    assert (more.sum(axis=1) >= base.sum(axis=1)).all()


def test_prebuilt_scenarios(current: dict[str, float]) -> None:
    scenarios = prebuilt_scenarios(current)
    total = sum(current.values())
    assert sum(scenarios["budget cut 20%"].values()) == pytest.approx(0.8 * total)
    assert sum(scenarios["budget increase 20%"].values()) == pytest.approx(1.2 * total)
    shifted = scenarios["shift 15% of meta_ads to youtube"]
    assert sum(shifted.values()) == pytest.approx(total)
    assert shifted["meta_ads"] == pytest.approx(0.85 * current["meta_ads"])
    assert scenarios["pause tv"]["tv"] == 0


def test_scenario_table_is_consistent(draws: PosteriorDraws, current: dict[str, float]) -> None:
    table = compare_scenarios(draws, prebuilt_scenarios(current), WEEKS)
    assert table.loc["current", "revenue_change_mean"] == 0
    assert table.loc["budget cut 20%", "revenue_change_mean"] < 0
    assert table.loc["budget increase 20%", "revenue_change_mean"] > 0
    assert table.loc["budget increase 20%", "prob_revenue_up"] == 1.0
    assert (table["revenue_hdi_low"] <= table["revenue_hdi_high"]).all()


def test_allocation_respects_bounds_and_total_budget(
    allocator: BudgetAllocator, draws: PosteriorDraws, current: dict[str, float]
) -> None:
    result = optimize_budget(allocator, draws, minimum={"tv": 2_000_000.0}, max_change=0.3)
    plan = result.recommended.spend
    assert result.converged
    assert sum(plan.values()) == pytest.approx(sum(current.values()), abs=TOLERANCE)
    for channel, (low, high) in result.bounds.items():
        assert low - TOLERANCE <= plan[channel] <= high + TOLERANCE
    assert plan["tv"] >= 2_000_000.0 - TOLERANCE


def test_recommended_is_at_least_as_good_as_current(
    allocator: BudgetAllocator, draws: PosteriorDraws
) -> None:
    result = optimize_budget(allocator, draws, max_change=0.3)
    assert result.uplift.mean >= -TOLERANCE
    assert (
        result.recommended.incremental_revenue.mean
        >= result.current.incremental_revenue.mean - TOLERANCE
    )


def test_more_budget_never_gives_less_revenue(
    allocator: BudgetAllocator, draws: PosteriorDraws, current: dict[str, float]
) -> None:
    total = sum(current.values())
    curve = budget_curve(allocator, draws, total * np.array([0.5, 1.0, 1.5, 2.0]))
    assert (np.diff(curve["revenue_mean"]) >= -TOLERANCE).all()
    spend_columns = [c for c in curve.columns if c.startswith("spend_")]
    np.testing.assert_allclose(curve[spend_columns].sum(axis=1), curve["budget"], rtol=1e-6)
    assert plot_budget_curve(curve, total, payback_budget(curve)).axes


def test_conservative_objective_respects_budget(
    allocator: BudgetAllocator, draws: PosteriorDraws, current: dict[str, float]
) -> None:
    result = optimize_budget(allocator, draws, max_change=0.3, objective="percentile")
    assert "percentile" in result.objective
    assert result.recommended.total_spend == pytest.approx(sum(current.values()), abs=TOLERANCE)
    assert plot_allocation(result).axes


def test_payback_budget_interpolates() -> None:
    curve = pd.DataFrame(
        {"budget": [0.0, 10.0, 20.0, 30.0], "revenue_mean": [0.0, 20.0, 32.0, 36.0]}
    )
    curve["marginal_return"] = curve["revenue_mean"].diff() / curve["budget"].diff()
    # marginal returns 2.0, 1.2, 0.4 at midpoints 5, 15, 25 -> crosses 1.0 at 17.5
    assert payback_budget(curve) == pytest.approx(17.5)
    curve["marginal_return"] = [np.nan, 3.0, 2.0, 1.5]
    assert payback_budget(curve) is None


def test_true_plan_revenue_uses_the_data_generator(current: dict[str, float]) -> None:
    assert sum(true_plan_revenue(BRAND, {}, WEEKS).values()) == 0
    base = true_plan_revenue(BRAND, current, WEEKS)
    double = true_plan_revenue(BRAND, {c: 2 * v for c, v in current.items()}, WEEKS)
    assert all(double[c] > base[c] > 0 for c in base)
    assert all(double[c] < 2 * base[c] for c in base if c != "tv")  # diminishing returns


def test_truth_check_reports_real_uplift(allocator: BudgetAllocator, draws: PosteriorDraws) -> None:
    result = optimize_budget(allocator, draws, max_change=0.3)
    check = validate_against_truth(BRAND, result)
    assert check["true_uplift"] == pytest.approx(
        check["true_revenue_recommended"] - check["true_revenue_current"]
    )
    assert check["model_expected_uplift"] == pytest.approx(result.uplift.mean)


def test_library_optimizer_objective_matches_our_plan_revenue(
    allocator: BudgetAllocator, draws: PosteriorDraws, current: dict[str, float]
) -> None:
    """Guard against API drift: the library's budget is per week and its objective is ours."""
    import xarray as xr

    weekly_total = sum(current.values()) / WEEKS
    bounds = xr.DataArray(
        np.array([[0.0, weekly_total]] * len(current)),
        dims=["channel", "bound"],
        coords={"channel": allocator.columns, "bound": ["lower", "upper"]},
    )
    allocation, result = allocator._optimizer("mean", 0.0).allocate_budget(
        total_budget=weekly_total, budget_bounds=bounds, return_if_fail=True
    )
    plan = dict(zip(current, allocation.to_numpy() * WEEKS, strict=True))
    ours = plan_revenue_draws(draws, plan, WEEKS).sum(axis=1).mean()
    assert -result.fun == pytest.approx(ours, rel=1e-6)
