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


def test_confidence_gate_holds_flagged_channels_to_ten_percent_with_caveats(
    allocator: BudgetAllocator, draws: PosteriorDraws, current: dict[str, float]
) -> None:
    """The live-demo situation: TV (bursts) and Email (tiny) must not get the full +30%."""
    from mixlab.ai_explainer import build_facts, template_brief
    from mixlab.evaluate import trust_notes
    from mixlab.insights import build_summary
    from mixlab.optimizer import build_optimizer_summary, change_lines, channel_caveats

    caveats = channel_caveats(draws)
    assert set(caveats) == {"tv", "email"}
    assert "bursts" in caveats["tv"] and "too small" in caveats["email"]

    ungated = optimize_budget(allocator, draws, max_change=0.3)
    gated = optimize_budget(allocator, draws, max_change=0.3, gate=True)
    peak = dict(zip(draws.channels, draws.spend.max(axis=0) * WEEKS, strict=True))
    for channel in current:
        change = gated.recommended.spend[channel] / current[channel] - 1
        limit = gated.limits[channel]["max_change"]
        assert limit == config.GATED_MAX_CHANGE if channel in caveats else limit <= 0.3
        assert gated.limits[channel]["reason"]
        assert abs(change) <= limit + 1e-6
        low, high = gated.bounds[channel]
        assert low == pytest.approx(current[channel] * (1 - limit))
        assert high <= current[channel] * (1 + limit) + TOLERANCE
    for channel in caveats:
        assert gated.recommended.spend[channel] <= peak[channel] + TOLERANCE
        assert channel not in gated.extrapolated_channels
    assert ungated.bounds["tv"][1] == pytest.approx(current["tv"] * 1.3)
    assert gated.caveats == caveats == ungated.caveats
    assert gated.realistic_uplift == pytest.approx(config.UPLIFT_SHRINKAGE * gated.uplift.mean)

    # The caveat appears next to the change wherever the recommendation is described.
    lines = change_lines(gated, name=str.upper)
    for line in lines:
        channel = line.split()[0].lower()
        assert (channel in caveats) == (caveats.get(channel, "~") in line)

    insights = build_summary(draws)
    summary = build_optimizer_summary(allocator, draws, BRAND, curve_points=3)
    assert summary["expected_revenue"]["caveats"] == caveats
    brief = template_brief(build_facts(insights, summary))
    shift = next(line for line in brief.splitlines() if line.startswith("Increase"))
    for channel, caveat in caveats.items():
        if f"{channel} (" in shift:
            assert f"{caveat})" in shift.split(f"{channel} (")[1].split(")")[0] + ")"
    assert "+30%" not in "".join(
        part for part in shift.split(",") if "tv" in part or "email" in part
    )

    # Health notes and the gate agree on which channels are hard to measure.
    notes = " ".join(
        n["detail"]
        for n in trust_notes(insights, summary)
        if "bursts" in n["title"] or "small" in n["title"]
    )
    assert "tv" in notes and "email" in notes


def test_measured_shrinkage_averages_true_over_expected_uplift() -> None:
    from mixlab.optimizer import measured_shrinkage

    def brand(expected: float, true: float) -> dict:
        return {
            "expected_revenue": {
                "truth_check": {"model_expected_uplift": expected, "true_uplift": true}
            }
        }

    assert measured_shrinkage([brand(10, 4), brand(20, 12)]) == pytest.approx(0.5)
    assert measured_shrinkage([{"expected_revenue": {}}]) is None
    assert measured_shrinkage([brand(-1, 5)]) is None


def test_optimizer_works_on_a_hill_model_with_roi_priors(df: pd.DataFrame) -> None:
    from conftest import TINY

    from mixlab.config import RoiPriorSettings

    settings = TINY.model_copy(
        update={"saturation": "hill", "roi_prior": RoiPriorSettings(mode="independent")}
    )
    model = MixLabModel().build(df, settings)
    model.fit(progressbar=False)
    hill_draws = extract_draws(model, df)
    result = optimize_budget(BudgetAllocator(model, WEEKS), hill_draws, max_change=0.3, gate=True)
    spend = result.recommended.spend
    assert sum(spend.values()) == pytest.approx(result.current.total_spend, abs=TOLERANCE)
    assert all(np.isfinite(value) for value in spend.values())
    assert result.uplift.mean >= -TOLERANCE


def test_unconstrained_recommendation_is_never_rated_below_the_current_plan(
    allocator: BudgetAllocator, draws: PosteriorDraws
) -> None:
    for objective in ("mean", "percentile"):
        result = optimize_budget(allocator, draws, objective=objective)
        current = plan_revenue_draws(draws, result.current.spend, WEEKS).sum(axis=1)
        recommended = plan_revenue_draws(draws, result.recommended.spend, WEEKS).sum(axis=1)
        if objective == "mean":
            assert recommended.mean() >= current.mean() - TOLERANCE
        else:
            level = config.RISK_PERCENTILE
            assert np.percentile(recommended, level) >= np.percentile(current, level) - TOLERANCE


# --- Phase 3: robust optimizer ---------------------------------------------------------------


def test_objectives_score_as_documented() -> None:
    from mixlab.optimizer import objective_value

    revenue = np.array([80.0, 100.0, 120.0])
    assert objective_value(revenue, 50.0, "mean") == 100.0
    assert objective_value(revenue, 50.0, "profit", margin=0.4) == pytest.approx(-10.0)
    assert objective_value(revenue, 50.0, "risk_adjusted", risk_lambda=1.0) == pytest.approx(
        100.0 - revenue.std()
    )
    assert objective_value(revenue, 50.0, "percentile", percentile=0.0) == 80.0


def test_thin_keeps_an_even_subset_of_draws(draws: PosteriorDraws) -> None:
    from mixlab.optimizer import thin

    small = thin(draws, 10)
    assert small.alpha.shape[0] <= 20 < draws.alpha.shape[0]
    assert small.channel_contribution.shape[1:] == draws.channel_contribution.shape[1:]
    assert thin(draws, 10_000) is draws


def test_solver_matches_or_beats_the_library_optimizer(
    allocator: BudgetAllocator, draws: PosteriorDraws, current: dict[str, float]
) -> None:
    from mixlab.optimizer import solve_allocation

    budget = sum(current.values())
    bounds = {channel: (0.7 * v, 1.3 * v) for channel, v in current.items()}
    ours, converged, _ = solve_allocation(draws, budget, bounds, WEEKS, starts=[current])
    theirs, _, _ = allocator.allocate(budget, bounds)

    def revenue(plan: dict[str, float]) -> float:
        return float(plan_revenue_draws(draws, plan, WEEKS).sum(axis=1).mean())

    assert converged and sum(ours.values()) == pytest.approx(budget, rel=1e-6)
    assert revenue(ours) >= revenue(theirs) * (1 - 1e-3)


def test_risk_averse_objectives_never_take_more_risk_than_the_mean_plan(
    draws: PosteriorDraws,
) -> None:
    mean_plan = optimize_budget(WEEKS, draws, max_change=0.3)
    for objective in ("risk_adjusted", "percentile", "profit"):
        result = optimize_budget(WEEKS, draws, max_change=0.3, objective=objective)
        assert result.recommended.total_spend == pytest.approx(
            mean_plan.current.total_spend, rel=1e-6
        )
        assert (
            result.objective
            == {
                "risk_adjusted": "risk-adjusted revenue",
                "percentile": "10th percentile revenue",
                "profit": "expected profit",
            }[objective]
        )


def test_limits_tighten_with_uncertainty_and_explain_themselves(draws: PosteriorDraws) -> None:
    from mixlab.optimizer import channel_limits

    limits = channel_limits(draws, WEEKS)
    assert (
        limits["tv"]["max_change"] == config.GATED_MAX_CHANGE and "bursts" in limits["tv"]["reason"]
    )
    assert limits["email"]["max_change"] == config.GATED_MAX_CHANGE
    for info in limits.values():
        assert info["max_change"] in (
            config.GATED_MAX_CHANGE,
            config.UNCERTAIN_MAX_CHANGE,
            config.DEFAULT_MAX_CHANGE,
        )
        assert info["reason"] and info["ceiling"] > 0
    uncertain = [i for i in limits.values() if i["max_change"] == config.UNCERTAIN_MAX_CHANGE]
    assert all("uncertain" in info["reason"] for info in uncertain)


def test_corner_solutions_are_flagged_and_explained() -> None:
    from mixlab.optimizer import corner_check

    bounds = {"a": (70.0, 130.0), "b": (70.0, 130.0), "c": (70.0, 130.0), "d": (70.0, 130.0)}
    flagged, note = corner_check({"a": 130.0, "b": 70.0, "c": 130.0, "d": 100.0}, bounds)
    assert flagged and "3 of 4 channels are at a limit" in note
    assert corner_check({"a": 100.0, "b": 95.0, "c": 110.0, "d": 100.0}, bounds) == (False, "")


def test_revenue_target_goal_finds_a_cheaper_plan_that_still_hits_it(draws: PosteriorDraws) -> None:
    from mixlab.optimizer import optimize_goal

    current = last_quarter_spend(draws, WEEKS)
    now = float(plan_revenue_draws(draws, current, WEEKS).sum(axis=1).mean())
    result = optimize_goal(draws, WEEKS, revenue_target=0.9 * now, max_change=0.5)
    assert result.recommended.total_spend < result.current.total_spend
    assert result.recommended.incremental_revenue.mean >= 0.9 * now * 0.98
    with pytest.raises(ValueError, match="below the target"):
        optimize_goal(draws, WEEKS, revenue_target=50 * now, max_change=0.1)
    with pytest.raises(ValueError, match="exactly one"):
        optimize_goal(draws, WEEKS)


def test_roi_target_goal_keeps_roi_at_or_above_the_target(draws: PosteriorDraws) -> None:
    from mixlab.optimizer import optimize_goal

    current = last_quarter_spend(draws, WEEKS)
    roi_now = float(plan_revenue_draws(draws, current, WEEKS).sum(axis=1).mean()) / sum(
        current.values()
    )
    result = optimize_goal(draws, WEEKS, roi_target=roi_now, max_change=0.5)
    assert result.recommended.roi.mean >= roi_now * 0.97
    assert "ROI of at least" in result.objective


def test_weekly_plan_keeps_totals_and_respects_the_burst_minimum(draws: PosteriorDraws) -> None:
    from mixlab.optimizer import weekly_plan

    plan = last_quarter_spend(draws, WEEKS)
    floor = plan["tv"] / 5  # at most five weeks on air
    schedule = weekly_plan(draws, plan, WEEKS, {"tv": floor})
    assert len(schedule) == WEEKS and schedule["week"].tolist() == list(range(1, WEEKS + 1))
    for channel, total in plan.items():
        assert schedule[channel].sum() == pytest.approx(total)
    on_air = schedule.loc[schedule["tv"] > 0, "tv"]
    assert config.MIN_BURST_WEEKS <= len(on_air) <= 5 and (on_air >= floor - 1).all()


def test_rollout_steps_reach_the_recommendation_with_checkpoints(draws: PosteriorDraws) -> None:
    from mixlab.optimizer import rollout_plan

    result = optimize_budget(WEEKS, draws, max_change=0.3, gate=True)
    steps = rollout_plan(draws, result)
    assert [s["step"] for s in steps] == [1, 2, 3]
    assert steps[-1]["spend"] == pytest.approx(result.recommended.spend)
    assert steps[0]["weeks"] == "1 to 4" and steps[-1]["share_of_change"] == 1.0
    for step in steps:
        assert step["confirm"] and step["stop"] and step["weekly_noise"] > 0
        assert set(step["expected_weekly_revenue_change"]) == {
            "mean",
            "median",
            "hdi_low",
            "hdi_high",
        }


def test_optimism_bootstrap_returns_a_haircut_between_zero_and_one(
    df: pd.DataFrame, fitted: MixLabModel, draws: PosteriorDraws, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mixlab.optimizer import estimate_optimism

    monkeypatch.setattr(config, "OPTIMISM_DRAWS", 30)
    result = estimate_optimism(df, fitted.settings, draws, WEEKS, n_boot=2)
    assert 0.0 <= result["shrinkage"] <= 1.0
    assert 0.0 <= result["shrinkage_low"] <= result["shrinkage_high"] <= 1.0
    assert len(result["replicates"]) == 2
    assert all({"expected", "delivered"} == set(pair) for pair in result["replicates"])


def test_summary_carries_rollout_weekly_plan_and_limits(draws: PosteriorDraws) -> None:
    from mixlab.optimizer import build_optimizer_summary

    summary = build_optimizer_summary(
        WEEKS, draws, BRAND, curve_points=3, optimism={"shrinkage": 0.5}
    )
    main = summary["expected_revenue"]
    assert main["shrinkage"] == 0.5
    assert main["realistic_uplift"] == pytest.approx(0.5 * main["uplift"]["mean"])
    assert set(main["limits"]) == set(draws.channels)
    assert len(summary["rollout"]) == config.ROLLOUT_STEPS
    assert len(summary["weekly_plan"]) == WEEKS and "tv" in summary["burst_minimum"]


def test_goal_result_reports_the_chance_of_meeting_it(draws: PosteriorDraws) -> None:
    from mixlab.optimizer import optimize_goal

    current = last_quarter_spend(draws, WEEKS)
    now = float(plan_revenue_draws(draws, current, WEEKS).sum(axis=1).mean())
    result = optimize_goal(draws, WEEKS, revenue_target=0.9 * now, max_change=0.5)
    assert result.goal_probability is not None and 0.0 <= result.goal_probability <= 1.0
    assert optimize_budget(WEEKS, draws, max_change=0.3).goal_probability is None
