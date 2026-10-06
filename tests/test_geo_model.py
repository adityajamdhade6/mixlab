"""Tests for the hierarchical geo model (uses the shared tiny geo fit)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import GEO_TINY

from mixlab import config
from mixlab.config import ModelSettings
from mixlab.geo_data import GeoDataset, aggregate_national
from mixlab.geo_model import (
    GeoMixLabModel,
    extract_geo_draws,
    geo_names,
    geo_roi_scale,
    is_geo_panel,
    model_for,
    region_frame,
)
from mixlab.model import MixLabModel, roi_scale
from mixlab.validate import spend_columns


def test_model_for_falls_back_to_national(geo_dataset: GeoDataset) -> None:
    national = aggregate_national(geo_dataset.data)
    assert isinstance(model_for(geo_dataset.data), GeoMixLabModel)
    assert type(model_for(national)) is MixLabModel
    assert not is_geo_panel(national) and geo_names(national) == []
    single = geo_dataset.data[geo_dataset.data[config.GEO_COL] == "maharashtra"]
    assert not is_geo_panel(single)


def test_roi_scale_is_computed_per_region(geo_dataset: GeoDataset) -> None:
    data = geo_dataset.data
    channels = spend_columns(data)
    geos = geo_names(data)
    scale = geo_roi_scale(data, channels, geos, GEO_TINY)
    assert scale.shape == (len(geos), len(channels)) and (scale > 0).all()
    expected = roi_scale(region_frame(data, geos[1]), channels, GEO_TINY)
    np.testing.assert_allclose(scale[1], expected)


def test_build_rejects_unsupported_inputs(geo_dataset: GeoDataset) -> None:
    with pytest.raises(ValueError, match="Hill"):
        GeoMixLabModel().build(geo_dataset.data, ModelSettings())
    with pytest.raises(ValueError, match="geo"):
        GeoMixLabModel().build(aggregate_national(geo_dataset.data), GEO_TINY)
    no_roi = GEO_TINY.model_copy(update={"roi_prior": None})
    with pytest.raises(ValueError, match="roi_prior"):
        GeoMixLabModel().build(geo_dataset.data, no_roi)


def test_channel_roi_is_partially_pooled_across_regions(geo_fitted: GeoMixLabModel) -> None:
    names = {v.name for v in geo_fitted.mmm.model.free_RVs}
    assert {"saturation_beta_raw_mu", "saturation_beta_raw_sigma"} <= names
    beta = geo_fitted.idata.posterior["saturation_beta"]
    assert beta.dims[-2:] == (config.GEO_COL, "channel")
    assert (beta > 0).all()
    assert geo_fitted.idata.posterior["adstock_alpha"].dims[-1] == "channel"  # shared shape


def test_each_region_is_scaled_by_its_own_peak(
    geo_fitted: GeoMixLabModel, geo_dataset: GeoDataset
) -> None:
    peaks = geo_dataset.data.groupby(config.GEO_COL)[config.TARGET_COL].max()
    scales = geo_fitted.geo_scales.to_series()
    pd.testing.assert_series_equal(
        scales.sort_index(), peaks.sort_index().astype(float), check_names=False
    )
    assert geo_fitted.target_scale == pytest.approx(peaks.max())


def test_extracted_regions_match_the_model(
    geo_fitted: GeoMixLabModel, geo_dataset: GeoDataset
) -> None:
    by_geo = extract_geo_draws(geo_fitted, geo_dataset.data)
    assert list(by_geo) == geo_fitted.geos
    geo = "karnataka"
    draws = by_geo[geo]
    frame = region_frame(geo_dataset.data, geo)
    np.testing.assert_allclose(draws.spend, frame[geo_fitted.channels].to_numpy())
    model_total = float(
        geo_fitted.idata.posterior["channel_contribution"]
        .sel(geo=geo)
        .sum(("date", "channel"))
        .mean()
    )
    assert draws.channel_contribution.sum(axis=(1, 2)).mean() == pytest.approx(
        model_total * draws.target_scale
    )
    assert set(draws.organic) >= {"baseline", "trend", "seasonality", "holidays"}


def test_simulated_contributions_reproduce_the_fitted_model(
    geo_fitted: GeoMixLabModel, geo_dataset: GeoDataset
) -> None:
    from mixlab.insights import simulate_contributions

    by_geo = extract_geo_draws(geo_fitted, geo_dataset.data)
    for draws in by_geo.values():
        simulated = simulate_contributions(draws, draws.spend)
        np.testing.assert_allclose(simulated, draws.channel_contribution, rtol=1e-6, atol=1.0)


def test_prior_predictive_is_in_rupees_per_region(geo_dataset: GeoDataset) -> None:
    model = GeoMixLabModel().build(geo_dataset.data, GEO_TINY)
    prior = model.sample_prior_predictive()
    assert config.GEO_COL in prior.dims
    observed = geo_dataset.data[config.TARGET_COL].mean()
    assert 0.1 * observed < float(prior.mean()) < 10 * observed


def test_save_and_load_in_a_fresh_registry(geo_fitted: GeoMixLabModel, tmp_path: Path) -> None:
    import pymc_extras.prior as prior_module

    geo_fitted.save(tmp_path)
    for name in [
        n for n in prior_module.CUSTOM_TRANSFORMS if n.startswith(config.ROI_TRANSFORM_PREFIX)
    ]:
        del prior_module.CUSTOM_TRANSFORMS[name]
    loaded = GeoMixLabModel.load(tmp_path)
    assert loaded.geos == geo_fitted.geos
    assert np.asarray(loaded.roi_transform_scale).shape == (len(loaded.geos), 6)
    np.testing.assert_allclose(
        loaded.idata.posterior["saturation_beta"], geo_fitted.idata.posterior["saturation_beta"]
    )
