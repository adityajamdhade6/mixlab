"""Tests for regional insights, the regional optimizer and the national-versus-geo comparison."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from mixlab import config
from mixlab.geo_data import GeoDataset
from mixlab.geo_insights import (
    build_geo_outputs,
    compare_national_geo,
    comparison_headline,
    current_geo_plan,
    geo_bounds,
    geo_plan_revenue,
    investment_status,
    load_geo_draws,
    national_roi_draws,
    regional_metrics,
    regional_optimizer,
    roi_draws,
    roi_hdi_table,
    save_geo_draws,
    spread_national_plan,
)
from mixlab.geo_model import GeoMixLabModel, extract_geo_draws
from mixlab.insights import PosteriorDraws
from mixlab.optimizer import plan_revenue_draws, thin

WEEKS = config.OPTIMIZER_WEEKS


@pytest.fixture(scope="module")
def by_geo(geo_fitted: GeoMixLabModel, geo_dataset: GeoDataset) -> dict[str, PosteriorDraws]:
    return extract_geo_draws(geo_fitted, geo_dataset.data)


def test_national_roi_is_spend_weighted_regional_roi(by_geo: dict[str, PosteriorDraws]) -> None:
    national = national_roi_draws(by_geo)
    spend = np.array([d.spend.sum(axis=0) for d in by_geo.values()])
    regional = np.array([roi_draws(d) for d in by_geo.values()])
    expected = (regional * spend[:, None, :]).sum(axis=0) / spend.sum(axis=0)
    np.testing.assert_allclose(national, expected)


def test_investment_status_thresholds() -> None:
    ratio = config.GEO_INVESTMENT_RATIO
    sure = config.GEO_STATUS_CONFIDENCE
    assert investment_status(1.0 * ratio + 0.01, 1.0, sure) == "under-invested"
    assert investment_status(1.0 / ratio - 0.01, 1.0, 1 - sure) == "over-invested"
    assert investment_status(1.0, 1.0, 0.5) == "about right"
    # A large gap the model is unsure about is not flagged.
    assert investment_status(2.0, 1.0, 0.6) == "about right"
    assert investment_status(0.5, 1.0, 0.4) == "about right"


def test_regional_metrics_carry_ranges_status_and_truth(
    by_geo: dict[str, PosteriorDraws], geo_dataset: GeoDataset
) -> None:
    metrics = regional_metrics(by_geo, geo_dataset.ground_truth)
    regions = metrics["regions"]
    assert set(regions) == set(by_geo)
    assert sum(r["spend_share_pct"] for r in regions.values()) == pytest.approx(100.0)
    for region in regions.values():
        assert region["status"] in {"under-invested", "over-invested", "about right"}
        assert region["lat"] is not None
        roi = region["channels"]["tv"]["roi"]
        assert roi["hdi_low"] <= roi["mean"] <= roi["hdi_high"]
        assert isinstance(region["channels"]["tv"]["truth_inside_range"], bool)
    blind = regional_metrics(by_geo)["regions"][next(iter(by_geo))]
    assert "truth_inside_range" not in blind["channels"]["tv"]


def test_comparison_counts_and_width_ratio(
    by_geo: dict[str, PosteriorDraws], geo_dataset: GeoDataset
) -> None:
    truth = geo_dataset.ground_truth
    # A "national model" whose draws are twice as spread out as the geo model's.
    geo_national = next(iter(by_geo.values()))
    widened = dataclasses.replace(
        geo_national,
        spend=sum(d.spend for d in by_geo.values()),
        channel_contribution=sum(d.channel_contribution for d in by_geo.values()),
    )
    mean = widened.channel_contribution.mean(axis=0, keepdims=True)
    widened = dataclasses.replace(
        widened, channel_contribution=mean + 2.0 * (widened.channel_contribution - mean)
    )
    comparison = compare_national_geo(by_geo, widened, truth)
    assert comparison["n_channels"] == 6
    assert comparison["median_width_ratio"] == pytest.approx(0.5, rel=0.05)
    assert comparison["regional_cells"] == len(by_geo) * 6
    lines = comparison_headline(comparison)
    assert "narrower" in lines[1] and "region-channel pairs" in lines[2]


def test_spread_national_plan_keeps_totals_and_regional_split(
    by_geo: dict[str, PosteriorDraws],
) -> None:
    current = current_geo_plan(by_geo, WEEKS)
    totals = {"tv": 2 * sum(r["tv"] for r in current.values())}
    plan = spread_national_plan(totals, current)
    assert sum(r["tv"] for r in plan.values()) == pytest.approx(totals["tv"])
    first = next(iter(current))
    assert plan[first]["tv"] == pytest.approx(2 * current[first]["tv"])
    assert plan[first]["meta_ads"] == pytest.approx(current[first]["meta_ads"])


def test_regional_optimizer_keeps_budget_and_bounds(
    by_geo: dict[str, PosteriorDraws], geo_dataset: GeoDataset
) -> None:
    current = current_geo_plan(by_geo, WEEKS)
    national_plan = {
        channel: sum(r[channel] for r in current.values())
        for channel in next(iter(current.values()))
    }
    result = regional_optimizer(by_geo, geo_dataset.ground_truth, national_plan=national_plan)
    recommended = sum(r["recommended_spend"] for r in result["regions"].values())
    assert recommended == pytest.approx(result["budget"], rel=1e-6)
    bounds = geo_bounds(current, by_geo, config.DEFAULT_MAX_CHANGE, WEEKS)
    for geo, region in result["regions"].items():
        for channel, cell in region["channels"].items():
            low, high = bounds[(geo, channel)]
            assert low - 1.0 <= cell["recommended"] <= high + 1.0
    assert result["uplift"]["mean"] >= -1.0
    assert 0.0 <= result["prob_beats_current"] <= 1.0
    check = result["truth_check"]
    assert check["national_plan_true_uplift"] == pytest.approx(0.0, abs=1.0)  # unchanged plan
    assert np.isfinite(check["true_uplift"])


def test_bounds_never_exceed_the_highest_week_on_record(by_geo: dict[str, PosteriorDraws]) -> None:
    current = current_geo_plan(by_geo, WEEKS)
    bounds = geo_bounds(current, by_geo, 5.0, WEEKS)
    geo = next(iter(by_geo))
    peak = by_geo[geo].spend.max(axis=0) * WEEKS
    for index, channel in enumerate(by_geo[geo].channels):
        assert bounds[(geo, channel)][1] <= max(peak[index], current[geo][channel]) + 1e-6


def test_compact_draws_round_trip(by_geo: dict[str, PosteriorDraws], tmp_path: Path) -> None:
    path = save_geo_draws(by_geo, tmp_path / config.GEO_DRAWS_FILENAME)
    loaded = load_geo_draws(path)
    assert list(loaded) == list(by_geo)
    geo = next(iter(by_geo))
    plan = current_geo_plan(by_geo, WEEKS)[geo]
    original = plan_revenue_draws(thin(by_geo[geo]), plan, WEEKS).sum(axis=1)
    again = plan_revenue_draws(loaded[geo], plan, WEEKS).sum(axis=1)
    np.testing.assert_allclose(again, original, rtol=1e-4)
    np.testing.assert_allclose(roi_draws(loaded[geo]), roi_draws(thin(by_geo[geo])), rtol=1e-4)
    total = geo_plan_revenue(loaded, current_geo_plan(loaded, WEEKS), WEEKS)
    assert total.shape == (loaded[geo].alpha.shape[0],)


def test_build_outputs_with_and_without_truth(
    by_geo: dict[str, PosteriorDraws], geo_dataset: GeoDataset
) -> None:
    national = dataclasses.replace(
        next(iter(by_geo.values())),
        spend=sum(d.spend for d in by_geo.values()),
        revenue=sum(d.revenue for d in by_geo.values()),
    )
    summary, comparison = build_geo_outputs(by_geo, national, geo_dataset.ground_truth)
    assert summary["n_regions"] == len(by_geo)
    assert set(summary["national_roi"]) == set(national.channels)
    assert "truth_check" in summary["optimizer"]
    assert len(comparison["headline"]) == 3
    _, blind = build_geo_outputs(by_geo, national, None)
    assert "No ground truth" in blind["headline"][0]
    table = roi_hdi_table(by_geo)
    assert all(low <= high for row in table.values() for low, high in row.values())


def test_poorly_measured_cells_get_tighter_limits(by_geo: dict[str, PosteriorDraws]) -> None:
    from mixlab.optimizer import channel_limits

    current = current_geo_plan(by_geo, WEEKS)
    bounds = geo_bounds(current, by_geo, config.DEFAULT_MAX_CHANGE, WEEKS)
    for geo, draws in by_geo.items():
        limits = channel_limits(draws, WEEKS, config.DEFAULT_MAX_CHANGE)
        for channel, spend in current[geo].items():
            low, _ = bounds[(geo, channel)]
            assert low == pytest.approx(spend * (1 - limits[channel]["max_change"]))
    changes = {
        limit["max_change"]
        for draws in by_geo.values()
        for limit in channel_limits(draws, WEEKS, config.DEFAULT_MAX_CHANGE).values()
    }
    assert min(changes) < config.DEFAULT_MAX_CHANGE  # the tiny test fit is uncertain somewhere
