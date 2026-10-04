"""Tests for the random brand generator."""

from mixlab import config
from mixlab.benchmark import random_brand
from mixlab.data_gen import generate


def test_random_brands_are_reproducible_and_distinct() -> None:
    first, again, other = random_brand(3), random_brand(3), random_brand(4)
    assert first == again and first != other
    assert first.name == "random_003" and first.seed != other.seed
    base = config.PERFORMANCE_HEAVY_BRAND
    assert [c.name for c in first.channels] == [c.name for c in base.channels]
    assert first.channels[0].beta != base.channels[0].beta
    assert all(0.05 <= c.adstock_decay <= 0.85 for c in first.channels)


def test_random_brand_generates_valid_data() -> None:
    dataset = generate(random_brand(0))
    assert len(dataset.data) == 156 and (dataset.data[config.TARGET_COL] > 0).all()
    assert set(dataset.ground_truth["channels"]) == {c.name for c in random_brand(0).channels}
