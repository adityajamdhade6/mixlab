"""Tests for insights: known-answer checks on hand-built draws, plus one real fitted model."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mixlab import config
from mixlab.insights import (
    PosteriorDraws,
    build_summary,
    carryover_weeks_draws,
    channel_metrics,
    decomposition_weekly,
    export_summary,
    extract_draws,
    hdi,
    marginal_roi_draws,
    metrics_table,
    naive_last_click_revenue,
    response_curve_draws,
    saturation_spend_draws,
    save_figures,
    simulate_contributions,
    summarize,
)
from mixlab.model import MixLabModel

N_DRAWS, N_WEEKS = 200, 60
ESTIMATE_KEYS = {"mean", "median", "hdi_low", "hdi_high"}


@pytest.fixture(scope="module")
def draws() -> PosteriorDraws:
    """Two channels: 'strong' is far from saturated, 'weak' is below break-even throughout."""
    rng = np.random.default_rng(0)
    spend = np.column_stack([rng.uniform(4e5, 8e5, N_WEEKS), rng.uniform(1e5, 2e5, N_WEEKS)])
    target_scale = 1e7
    base = PosteriorDraws(
        dates=pd.date_range("2024-01-01", periods=N_WEEKS, freq="7D"),
        channels=["strong", "weak"],
        spend=spend,
        revenue=np.full(N_WEEKS, 8e6),
        channel_scale=spend.max(axis=0),
        target_scale=target_scale,
        l_max=8,
        alpha=np.column_stack([rng.uniform(0.5, 0.7, N_DRAWS), np.zeros(N_DRAWS) + 1e-9]),
        lam=np.column_stack([rng.uniform(1.5, 2.5, N_DRAWS), rng.uniform(1.5, 2.5, N_DRAWS)]),
        beta=np.column_stack([rng.uniform(0.3, 0.4, N_DRAWS), rng.uniform(0.001, 0.002, N_DRAWS)]),
        organic={"baseline": np.full((N_DRAWS, N_WEEKS), 5e6)},
        channel_contribution=np.zeros((N_DRAWS, N_WEEKS, 2)),
    )
    contribution = simulate_contributions(base, spend)
    return PosteriorDraws(**{**base.__dict__, "channel_contribution": contribution})


def test_hdi_is_narrowest_interval() -> None:
    low, high = hdi(np.arange(100.0), prob=0.9)
    assert high - low == pytest.approx(89.0)
    skewed = np.random.default_rng(1).exponential(size=5000)
    low, high = hdi(skewed)
    assert low < np.quantile(skewed, 0.03)  # HDI hugs the mode, unlike equal-tailed quantiles
    assert np.mean((skewed >= low) & (skewed <= high)) == pytest.approx(config.HDI_PROB, abs=0.01)


def test_hdi_works_column_wise() -> None:
    low, high = hdi(np.random.default_rng(2).normal(size=(1000, 3)))
    assert low.shape == high.shape == (3,)
    assert (low < 0).all() and (high > 0).all()


def test_summarize_orders_fields() -> None:
    estimate = summarize(np.random.default_rng(3).normal(10, 1, 2000))
    assert estimate.hdi_low < estimate.mean < estimate.hdi_high
    assert estimate.median == pytest.approx(10, abs=0.1)


def test_zero_spend_gives_zero_contribution(draws: PosteriorDraws) -> None:
    assert simulate_contributions(draws, np.zeros_like(draws.spend)).sum() == 0


def test_roi_is_revenue_over_spend(draws: PosteriorDraws) -> None:
    metrics = channel_metrics(draws)["strong"]
    expected = draws.channel_contribution[:, :, 0].sum(axis=1).mean() / draws.spend[:, 0].sum()
    assert metrics["roi"]["mean"] == pytest.approx(expected)
    assert metrics["roas"] == metrics["roi"]
    assert metrics["cost_per_incremental_revenue"]["median"] == pytest.approx(
        1 / metrics["roi"]["median"], rel=0.02
    )


def test_marginal_roi_is_below_average_roi_under_diminishing_returns(draws: PosteriorDraws) -> None:
    marginal = marginal_roi_draws(draws)
    average = draws.channel_contribution.sum(axis=1) / draws.spend.sum(axis=0)
    assert (marginal > 0).all()
    assert (marginal < average).all()


def test_saturation_point_is_where_slope_equals_one(draws: PosteriorDraws) -> None:
    saturation = saturation_spend_draws(draws)
    assert (saturation[:, 1] == 0).all()  # weak channel never clears break-even
    point = saturation[0, 0]
    step = 1.0
    curve = response_curve_draws(draws, 0, np.array([point - step, point + step]))[0]
    assert (curve[1] - curve[0]) / (2 * step) == pytest.approx(1.0, abs=1e-4)


def test_response_curve_starts_at_zero_and_rises(draws: PosteriorDraws) -> None:
    curve = response_curve_draws(draws, 0, np.linspace(0, 1e6, 20))
    assert (curve[:, 0] == 0).all()
    assert (np.diff(curve, axis=1) > 0).all()


def test_carryover_weeks(draws: PosteriorDraws) -> None:
    weeks = carryover_weeks_draws(draws)
    assert weeks[:, 1] == pytest.approx(1.0)  # no carryover: everything lands in week one
    assert (weeks[:, 0] > 2).all() and (weeks[:, 0] <= draws.l_max).all()


def test_naive_last_click_credits_all_revenue(draws: PosteriorDraws) -> None:
    credited = naive_last_click_revenue(draws.spend, draws.revenue)
    assert credited.sum() == pytest.approx(draws.revenue.sum())
    assert credited[0] > credited[1]  # bigger spender gets more credit, regardless of effect


def test_summary_metrics_are_distributions(draws: PosteriorDraws) -> None:
    summary = build_summary(draws)
    for metrics in summary["channels"].values():
        for key in (
            "attributed_revenue",
            "contribution_pct",
            "roi",
            "roas",
            "marginal_roi",
            "cost_per_incremental_revenue",
            "saturation_weekly_spend",
            "weeks_to_90pct_effect",
        ):
            assert set(metrics[key]) == ESTIMATE_KEYS
    assert summary["channels"]["weak"]["last_click"]["naive_over_mmm_ratio"]["median"] > 1
    assert set(summary["decomposition"]) == {"baseline", "strong", "weak"}
    assert summary["hdi_prob"] == config.HDI_PROB


def test_export_and_table(draws: PosteriorDraws, tmp_path: Path) -> None:
    summary = build_summary(draws)
    path = export_summary(summary, tmp_path)
    assert json.loads(path.read_text())["channels"].keys() == summary["channels"].keys()
    assert list(metrics_table(summary).index) == ["strong", "weak"]


def test_numpy_response_reproduces_the_fitted_model(df: pd.DataFrame, fitted: MixLabModel) -> None:
    real = extract_draws(fitted, df)
    simulated = simulate_contributions(real, real.spend)
    np.testing.assert_allclose(simulated, real.channel_contribution, rtol=1e-4, atol=1.0)


def test_real_model_decomposition_and_figures(
    df: pd.DataFrame, fitted: MixLabModel, tmp_path: Path
) -> None:
    real = extract_draws(fitted, df)
    weekly = decomposition_weekly(real)
    assert {"baseline", "trend", "seasonality", "holidays", "promos", "price"} <= set(weekly)
    explained = weekly.drop(columns="observed_revenue").sum(axis=1)
    assert abs(explained.sum() / weekly["observed_revenue"].sum() - 1) < 0.05
    paths = save_figures(real, build_summary(real), tmp_path)
    assert len(paths) == 5 and all(p.stat().st_size > 0 for p in paths)


def test_chance_next_rupee_profitable_tightens_with_lower_margin(draws: PosteriorDraws) -> None:
    from mixlab.insights import chance_next_rupee_profitable

    marginal = marginal_roi_draws(draws)
    full = chance_next_rupee_profitable(marginal, 1.0)
    thin = chance_next_rupee_profitable(marginal, 0.2)
    np.testing.assert_allclose(full, (marginal > 1).mean(axis=0))
    assert (thin <= full).all() and thin.shape == (2,)


def test_hill_saturation_point_matches_slope_of_one() -> None:
    from mixlab.insights import saturate

    rng = np.random.default_rng(1)
    n_draws, spend = 50, np.column_stack([np.linspace(1e5, 9e5, 40)])
    hill = PosteriorDraws(
        dates=pd.date_range("2024-01-01", periods=40, freq="7D"),
        channels=["only"],
        spend=spend,
        revenue=np.full(40, 5e6),
        channel_scale=spend.max(axis=0),
        target_scale=1e7,
        l_max=4,
        alpha=np.full((n_draws, 1), 0.3),
        lam=np.zeros((n_draws, 1)),
        beta=rng.uniform(0.3, 0.5, (n_draws, 1)),
        organic={"baseline": np.full((n_draws, 40), 4e6)},
        channel_contribution=np.zeros((n_draws, 40, 1)),
        saturation="hill",
        slope=np.full((n_draws, 1), 1.0),
        kappa=rng.uniform(0.4, 0.8, (n_draws, 1)),
    )
    half = saturate(hill, hill.kappa[:, :, None] * np.ones((n_draws, 1, 1)))
    np.testing.assert_allclose(half, 0.5)  # a Hill curve is at half its ceiling at kappa
    assert simulate_contributions(hill, np.zeros_like(spend)).sum() == 0

    point = saturation_spend_draws(hill)[:, 0]
    # With slope 1 the curve is B*x/(k+x): its gradient is 1 where x = sqrt(B*k*scale) - k*scale.
    b = hill.beta[:, 0] * hill.target_scale
    k = hill.kappa[:, 0] * hill.channel_scale[0]
    np.testing.assert_allclose(point, np.sqrt(b * k) - k, rtol=0.02)
