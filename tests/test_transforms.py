"""Tests for adstock and saturation transforms."""

import numpy as np
import pytest

from mixlab.transforms import adstock_weights, geometric_adstock, hill_saturation


def test_adstock_weights_normalized_sum_to_one() -> None:
    assert adstock_weights(0.6, 12).sum() == pytest.approx(1.0)


def test_adstock_of_single_pulse_decays_geometrically() -> None:
    pulse = np.array([100.0, 0.0, 0.0, 0.0])
    out = geometric_adstock(pulse, decay=0.5, l_max=4, normalize=False)
    np.testing.assert_allclose(out, [100.0, 50.0, 25.0, 12.5])


def test_zero_decay_leaves_spend_unchanged() -> None:
    spend = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(geometric_adstock(spend, decay=0.0, l_max=3), spend)


def test_adstock_rejects_invalid_decay() -> None:
    with pytest.raises(ValueError):
        adstock_weights(1.0, 4)


def test_hill_is_half_at_half_saturation_and_bounded() -> None:
    out = hill_saturation(np.array([0.0, 500.0, 1e12]), half_saturation=500.0, slope=1.3)
    assert out[0] == 0.0
    assert out[1] == pytest.approx(0.5)
    assert out[2] < 1.0


def test_hill_is_monotonic() -> None:
    out = hill_saturation(np.linspace(0, 1000, 50), half_saturation=300.0, slope=1.0)
    assert np.all(np.diff(out) > 0)
