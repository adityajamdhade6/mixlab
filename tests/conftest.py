"""Shared fixtures: the synthetic frame and one tiny fitted model reused across test modules."""

import pandas as pd
import pytest

from mixlab import config
from mixlab.config import ModelSettings, SamplerSettings
from mixlab.data_gen import generate
from mixlab.geo_data import GeoDataset, generate_geo
from mixlab.geo_model import GeoMixLabModel
from mixlab.model import MixLabModel

TINY = ModelSettings(
    prior_predictive_samples=50,
    sampler=SamplerSettings(chains=2, draws=40, tune=40),
)


@pytest.fixture(scope="session")
def df() -> pd.DataFrame:
    return generate(config.PERFORMANCE_HEAVY_BRAND).data


@pytest.fixture(scope="session")
def fitted(df: pd.DataFrame) -> MixLabModel:
    model = MixLabModel().build(df, TINY)
    model.sample_prior_predictive()
    model.fit(progressbar=False)
    return model


GEO_REGIONS = config.INDIA_REGIONS[:3]
GEO_TINY = ModelSettings.from_yaml(config.PROJECT_ROOT / "configs" / "geo.yaml").model_copy(
    update={
        "prior_predictive_samples": 20,
        "sampler": SamplerSettings(chains=2, draws=30, tune=30),
    }
)


@pytest.fixture(scope="session")
def geo_dataset() -> GeoDataset:
    return generate_geo(config.PERFORMANCE_HEAVY_BRAND, GEO_REGIONS)


@pytest.fixture(scope="session")
def geo_fitted(geo_dataset: GeoDataset) -> GeoMixLabModel:
    model = GeoMixLabModel().build(geo_dataset.data, GEO_TINY)
    model.fit(progressbar=False)
    return model
