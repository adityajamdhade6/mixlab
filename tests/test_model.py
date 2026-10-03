"""Tests for the model wrapper. One tiny fit is shared by the sampling tests."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import TINY
from pydantic import ValidationError

from mixlab import config
from mixlab.config import ChannelPriorOverride, HalfNormalParams, ModelSettings
from mixlab.model import (
    MixLabModel,
    build_priors,
    channel_prior_values,
    default_control_columns,
    design_matrix,
    interval_summary,
    plot_prior_predictive,
    prior_predictive_check,
    spend_shares,
)
from mixlab.validate import spend_columns

DEFAULT_YAML = config.PROJECT_ROOT / "configs" / "default.yaml"


def test_default_yaml_loads_with_spec_defaults() -> None:
    settings = ModelSettings.from_yaml(DEFAULT_YAML)
    assert settings.adstock_l_max == 8
    assert settings.sampler.chains == 4
    assert settings.sampler.target_accept == 0.9
    assert settings.sampler.seed == config.RANDOM_SEED
    assert settings.channel_priors["tv"].adstock_alpha.alpha == 4


def test_yaml_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("adstock_lmax: 8\n")
    with pytest.raises(ValidationError):
        ModelSettings.from_yaml(path)


def test_design_matrix_centres_price_and_drops_seasonality(df: pd.DataFrame) -> None:
    channels, controls = spend_columns(df), default_control_columns(df)
    X, y = design_matrix(df, channels, controls)
    assert list(X.columns) == [config.DATE_COL, *channels, *controls]
    assert not any(c.startswith(config.SEASON_PREFIX) for c in X.columns)
    np.testing.assert_allclose(X[config.PRICE_COL], df[config.PRICE_COL] - 1.0)
    assert y is not None and len(y) == len(df)
    assert design_matrix(df.drop(columns=[config.TARGET_COL]), channels, controls)[1] is None


def test_channel_override_changes_only_that_channel(df: pd.DataFrame) -> None:
    channels = spend_columns(df)
    settings = ModelSettings.from_yaml(DEFAULT_YAML)
    values = channel_prior_values(channels, settings, spend_shares(df, channels))
    alpha = dict(zip(channels, values["adstock_alpha"]["alpha"], strict=True))
    assert alpha["spend_tv"] == 4 and alpha["spend_youtube"] == 3
    assert alpha["spend_meta_ads"] == settings.priors.adstock_alpha.alpha
    assert set(values["saturation_beta"]["sigma"]) == {settings.priors.saturation_beta.sigma}


def test_spend_share_prior_scales_with_share(df: pd.DataFrame) -> None:
    channels = spend_columns(df)
    shares = spend_shares(df, channels)
    settings = ModelSettings(
        spend_share_prior=True,
        channel_priors={"email": ChannelPriorOverride(saturation_beta=HalfNormalParams(sigma=0.5))},
    )
    sigma = dict(
        zip(
            channels,
            channel_prior_values(channels, settings, shares)["saturation_beta"]["sigma"],
            strict=True,
        )
    )
    base = settings.priors.saturation_beta.sigma
    assert sigma["spend_meta_ads"] == pytest.approx(base * len(channels) * shares["spend_meta_ads"])
    assert sigma["spend_meta_ads"] > sigma["spend_tv"]
    assert sigma["spend_email"] == 0.5  # explicit override beats the spend-share rule


def test_build_priors_covers_every_component(df: pd.DataFrame) -> None:
    channels = spend_columns(df)
    priors = build_priors(channels, ModelSettings(), spend_shares(df, channels))
    assert set(priors) == {
        "intercept",
        "adstock_alpha",
        "saturation_lam",
        "saturation_beta",
        "gamma_control",
        "gamma_fourier",
        "likelihood",
    }


def test_build_defines_expected_model(df: pd.DataFrame) -> None:
    model = MixLabModel().build(df, TINY)
    assert model.channels == spend_columns(df)
    assert config.PROMO_COL in model.controls and config.PRICE_COL in model.controls
    assert model.mmm.adstock.l_max == 8
    assert f"{config.TREND_PREFIX}_effect_contribution" in model.mmm.model.named_vars
    assert "yearly_seasonality_contribution" in model.mmm.model.named_vars


def test_unbuilt_model_fails_clearly() -> None:
    with pytest.raises(RuntimeError, match="not built"):
        MixLabModel().fit()


def test_prior_predictive_is_in_inr_and_plots(df: pd.DataFrame, fitted: MixLabModel) -> None:
    prior = fitted.idata.prior_predictive["y"] * fitted.target_scale
    assert fitted.target_scale == pytest.approx(df[config.TARGET_COL].max())
    check = prior_predictive_check(prior, df[config.TARGET_COL])
    assert check["n_samples"] == TINY.prior_predictive_samples
    assert check["believable"]
    summary = interval_summary(prior)
    assert len(summary) == len(df)
    assert (summary["lower"] <= summary["upper"]).all()
    assert plot_prior_predictive(summary, df[config.TARGET_COL]).axes


def test_fit_produces_posterior_with_fixed_seed(fitted: MixLabModel) -> None:
    posterior = fitted.idata.posterior
    assert posterior.sizes["chain"] == 2 and posterior.sizes["draw"] == 40
    assert "channel_contribution_original_scale" in posterior
    assert fitted.fit_seconds is not None and fitted.fit_seconds > 0
    assert "adstock_alpha" in fitted.parameter_names


def test_predict_returns_revenue_in_inr(df: pd.DataFrame, fitted: MixLabModel) -> None:
    prediction = fitted.predict(df)
    assert list(prediction.columns) == [config.DATE_COL, "mean", "lower", "upper"]
    assert len(prediction) == len(df)
    assert (prediction["lower"] <= prediction["upper"]).all()
    error = (prediction["mean"] - df[config.TARGET_COL]).abs().mean() / df[config.TARGET_COL].mean()
    assert error < 0.25


def test_save_and_load_roundtrip(df: pd.DataFrame, fitted: MixLabModel, tmp_path: Path) -> None:
    path = fitted.save(tmp_path, extra={"note": "test"})
    assert path.exists() and (tmp_path / config.MODEL_META_FILENAME).exists()
    loaded = MixLabModel.load(tmp_path)
    assert loaded.channels == fitted.channels
    assert loaded.settings == fitted.settings
    assert loaded.idata.posterior.sizes == fitted.idata.posterior.sizes
    np.testing.assert_allclose(loaded.predict(df)["mean"], fitted.predict(df)["mean"], rtol=1e-6)


def test_slim_copy_is_smaller_and_loads_without_the_full_file(
    df: pd.DataFrame, fitted: MixLabModel, tmp_path: Path
) -> None:
    full = fitted.save(tmp_path)
    slim = fitted.save_slim(tmp_path)
    assert slim.stat().st_size < full.stat().st_size
    full.unlink()
    loaded = MixLabModel.load(tmp_path)
    assert "channel_contribution" in loaded.idata.posterior
    assert "prior" not in loaded.idata.groups()
    prediction = loaded.predict(df)
    assert len(prediction) == len(df) and (prediction["mean"] > 0).all()


def test_predicting_weeks_after_training_carries_over_adstock(df: pd.DataFrame) -> None:
    from mixlab.evaluate import prediction_error

    train, test = df.iloc[:-8], df.iloc[-8:]
    model = MixLabModel().build(train, TINY)
    model.fit(progressbar=False)
    prediction = model.predict(test)
    assert prediction[config.DATE_COL].tolist() == test[config.DATE_COL].tolist()
    error = prediction_error(prediction, test[config.TARGET_COL])
    assert 0 <= error["mape_pct"] < 30 and 0 <= error["coverage_pct"] <= 100
