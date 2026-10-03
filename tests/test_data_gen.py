"""Tests for the synthetic data generator."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mixlab import config
from mixlab.data_gen import SyntheticDataset, generate, save_channel_plots, save_dataset

CFG = config.PERFORMANCE_HEAVY_BRAND
CHANNELS = [c.name for c in CFG.channels]
SPEND_COLS = [f"{config.SPEND_PREFIX}{name}" for name in CHANNELS]


@pytest.fixture(scope="module")
def dataset() -> SyntheticDataset:
    return generate(CFG)


def test_shape_and_weekly_dates(dataset: SyntheticDataset) -> None:
    data = dataset.data
    assert len(data) == 156
    assert data[config.DATE_COL].iloc[0] == pd.Timestamp("2023-01-02")
    assert (data[config.DATE_COL].diff().dropna() == pd.Timedelta(days=7)).all()
    assert not data.isna().any().any()


def test_follows_data_contract(dataset: SyntheticDataset) -> None:
    columns = list(dataset.data.columns)
    assert columns[:2] == [config.DATE_COL, config.TARGET_COL]
    assert set(SPEND_COLS) <= set(columns)
    assert config.PROMO_COL in columns and config.PRICE_COL in columns
    assert any(c.startswith(config.HOLIDAY_PREFIX) for c in columns)
    assert any(c.startswith(config.SEASON_PREFIX) for c in columns)


def test_no_negative_spend_or_revenue(dataset: SyntheticDataset) -> None:
    assert (dataset.data[SPEND_COLS] >= 0).all().all()
    assert (dataset.data[config.TARGET_COL] > 0).all()


def test_tv_runs_only_in_flights(dataset: SyntheticDataset) -> None:
    on_air = dataset.data["spend_tv"] > 0
    tv = next(c for c in CFG.channels if c.name == "tv")
    assert on_air.sum() == tv.flight_weeks * 3
    assert dataset.data.loc[dataset.data["holiday_diwali"] == 1, "spend_tv"].gt(0).all()


def test_search_spend_is_confounded_with_demand(dataset: SyntheticDataset) -> None:
    organic = dataset.contributions[["seasonality", "holiday", "promo"]].sum(axis=1).to_numpy()
    detrended = (
        dataset.data["spend_google_search"]
        / dataset.data["spend_google_search"].rolling(13, center=True, min_periods=1).mean()
    )
    assert np.corrcoef(organic, detrended)[0, 1] > 0.4


def test_components_sum_to_revenue(dataset: SyntheticDataset) -> None:
    np.testing.assert_allclose(
        dataset.contributions.sum(axis=1).to_numpy(), dataset.data[config.TARGET_COL].to_numpy()
    )


def test_ground_truth_file_matches_config(dataset: SyntheticDataset, tmp_path: Path) -> None:
    save_dataset(dataset, tmp_path)
    truth = json.loads((tmp_path / config.GROUND_TRUTH_FILENAME).read_text())
    saved = pd.read_csv(tmp_path / config.WEEKLY_DATA_FILENAME)
    contributions = pd.read_csv(tmp_path / config.TRUE_CONTRIBUTIONS_FILENAME)

    assert truth["seed"] == CFG.seed and truth["brand"] == CFG.name
    assert list(truth["channels"]) == CHANNELS
    for channel in CFG.channels:
        entry = truth["channels"][channel.name]
        assert entry["adstock_decay"] == channel.adstock_decay
        assert entry["half_saturation"] == channel.half_saturation
        assert entry["hill_slope"] == channel.hill_slope
        assert entry["beta"] == channel.beta
        spend = saved[f"{config.SPEND_PREFIX}{channel.name}"].sum()
        contribution = contributions[channel.name].sum()
        assert entry["true_roi"] == pytest.approx(contribution / spend)
        assert entry["contribution_pct_of_revenue"] == pytest.approx(
            100 * contribution / saved[config.TARGET_COL].sum()
        )
    assert sum(c["contribution_pct_of_media"] for c in truth["channels"].values()) == pytest.approx(
        100.0
    )


def test_same_seed_is_reproducible(dataset: SyntheticDataset) -> None:
    again = generate(CFG)
    pd.testing.assert_frame_equal(dataset.data, again.data)
    assert dataset.ground_truth == again.ground_truth


def test_different_seed_changes_data(dataset: SyntheticDataset) -> None:
    other = generate(replace(CFG, seed=CFG.seed + 1))
    assert not dataset.data[config.TARGET_COL].equals(other.data[config.TARGET_COL])


def test_tv_heavy_brand_leans_on_tv() -> None:
    share = {
        name: generate(cfg).ground_truth["channels"]["tv"]["contribution_pct_of_media"]
        for name, cfg in config.BRAND_PRESETS.items()
    }
    assert share["tv_heavy"] > share["performance_heavy"]


def test_one_plot_per_channel(dataset: SyntheticDataset, tmp_path: Path) -> None:
    paths = save_channel_plots(dataset, tmp_path)
    assert len(paths) == len(CHANNELS)
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)
