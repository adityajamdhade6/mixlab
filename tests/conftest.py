"""Shared fixtures: the synthetic frame and one tiny fitted model reused across test modules."""

import pandas as pd
import pytest

from mixlab import config
from mixlab.config import ModelSettings, SamplerSettings
from mixlab.data_gen import generate
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
