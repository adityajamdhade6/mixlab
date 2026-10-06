"""Tests for the synthetic geo panel and its national aggregate."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mixlab import config
from mixlab.geo_data import (
    GeoDataset,
    aggregate_national,
    generate_geo,
    regional_brands,
    regional_channel,
    regional_multipliers,
    save_geo_dataset,
)

BRAND = config.PERFORMANCE_HEAVY_BRAND
REGIONS = config.INDIA_REGIONS[:3]


def test_panel_has_one_row_per_week_and_region(geo_dataset: GeoDataset) -> None:
    data = geo_dataset.data
    assert len(data) == BRAND.n_weeks * len(REGIONS)
    assert list(data.columns[:2]) == [config.DATE_COL, config.GEO_COL]
    assert set(data[config.GEO_COL]) == {r.name for r in REGIONS}
    assert not data.duplicated([config.DATE_COL, config.GEO_COL]).any()


def test_regional_festival_only_flags_its_own_region(geo_dataset: GeoDataset) -> None:
    data = geo_dataset.data
    flags = data.groupby(config.GEO_COL)["holiday_ganesh_chaturthi"].sum()
    assert flags["maharashtra"] > 0
    assert flags.drop("maharashtra").sum() == 0
    assert (data.groupby(config.GEO_COL)["holiday_diwali"].sum() > 0).all()


def test_ground_truth_adds_up_across_regions(geo_dataset: GeoDataset) -> None:
    truth = geo_dataset.ground_truth
    for channel, national in truth["channels"].items():
        regional = [r["channels"][channel] for r in truth["regions"].values()]
        spend = sum(r["total_spend"] for r in regional)
        contribution = sum(r["total_contribution"] for r in regional)
        assert national["total_spend"] == pytest.approx(spend)
        assert national["true_roi"] == pytest.approx(contribution / spend)
    contributions = geo_dataset.contributions
    assert contributions[config.GEO_COL].tolist() == geo_dataset.data[config.GEO_COL].tolist()


def test_regions_differ_in_true_roi_and_size(geo_dataset: GeoDataset) -> None:
    truth = geo_dataset.ground_truth["regions"]
    meta = [truth[r.name]["channels"]["meta_ads"]["true_roi"] for r in REGIONS]
    assert len(set(np.round(meta, 3))) == len(REGIONS)
    revenue = geo_dataset.data.groupby(config.GEO_COL)[config.TARGET_COL].sum()
    assert revenue["maharashtra"] > revenue["karnataka"]


def test_generation_is_reproducible_and_seeded() -> None:
    first = generate_geo(BRAND, REGIONS).data
    again = generate_geo(BRAND, REGIONS).data
    other = generate_geo(replace(BRAND, seed=7), REGIONS).data
    pd.testing.assert_frame_equal(first, again)
    assert not first[config.TARGET_COL].equals(other[config.TARGET_COL])


def test_regional_channel_scales_spend_price_and_effect() -> None:
    tv = next(c for c in BRAND.channels if c.name == "tv")
    region = REGIONS[0]
    copy = regional_channel(tv, region, effect=1.5, tilt=0.8)
    assert copy.base_spend == pytest.approx(tv.base_spend * region.size * 0.8)
    assert copy.half_saturation == pytest.approx(
        tv.half_saturation * region.size * region.media_price
    )
    assert copy.beta == pytest.approx(tv.beta * region.size * region.demand * 1.5)
    assert set(region.festivals) <= set(copy.flight_events)


def test_multipliers_are_seeded_and_subsets_keep_their_draws() -> None:
    full = regional_multipliers(BRAND.channels, config.INDIA_REGIONS, 1, 0.3)
    subset = regional_multipliers(BRAND.channels, REGIONS, 1, 0.3)
    np.testing.assert_allclose(full[: len(REGIONS)], subset)
    assert (full > 0).all()
    assert len(regional_brands(BRAND, REGIONS)) == len(REGIONS)


def test_national_aggregate_sums_money_and_averages_rates(geo_dataset: GeoDataset) -> None:
    data = geo_dataset.data
    national = aggregate_national(data)
    assert len(national) == BRAND.n_weeks
    assert national[config.TARGET_COL].sum() == pytest.approx(data[config.TARGET_COL].sum())
    assert national["spend_tv"].sum() == pytest.approx(data["spend_tv"].sum())
    assert national["holiday_ganesh_chaturthi"].max() == 1
    assert national[config.PROMO_COL].between(0, 1).all()
    assert config.GEO_COL not in national.columns


def test_save_writes_panel_and_truth(geo_dataset: GeoDataset, tmp_path: Path) -> None:
    save_geo_dataset(geo_dataset, tmp_path)
    for name in (
        config.GEO_WEEKLY_FILENAME,
        config.GEO_GROUND_TRUTH_FILENAME,
        config.GEO_TRUE_CONTRIBUTIONS_FILENAME,
    ):
        assert (tmp_path / name).exists()
