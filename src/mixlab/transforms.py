"""Adstock (carryover) and saturation (diminishing returns) transforms."""

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def adstock_weights(decay: float, l_max: int, normalize: bool = True) -> FloatArray:
    """Return geometric carryover weights ``decay**lag`` for lags ``0..l_max-1``.

    With ``normalize=True`` the weights sum to 1, so adstocked spend stays in spend units.
    """
    if not 0.0 <= decay < 1.0:
        raise ValueError(f"decay must be in [0, 1), got {decay}")
    if l_max < 1:
        raise ValueError(f"l_max must be at least 1, got {l_max}")
    weights = np.power(decay, np.arange(l_max, dtype=np.float64))
    return weights / weights.sum() if normalize else weights


def geometric_adstock(
    spend: FloatArray, decay: float, l_max: int, normalize: bool = True
) -> FloatArray:
    """Spread each week's spend over the following weeks with geometric decay.

    Args:
        spend: Weekly spend, oldest first.
        decay: Share of the effect that carries into the next week.
        l_max: Number of weeks (including the current one) the effect lasts.
        normalize: If true, weights sum to 1 so the output stays in spend units.

    Returns:
        Adstocked spend, the same length as ``spend``.

    """
    weights = adstock_weights(decay, l_max, normalize)
    return np.convolve(np.asarray(spend, dtype=np.float64), weights)[: len(spend)]


def hill_saturation(x: FloatArray, half_saturation: float, slope: float) -> FloatArray:
    """Map spend to a 0-1 response with diminishing returns (Hill curve).

    Args:
        x: Non-negative (adstocked) spend.
        half_saturation: Spend level at which the response is 0.5.
        slope: Steepness; 1 is concave, above 1 is S-shaped.

    Returns:
        Response in [0, 1), the same shape as ``x``.

    """
    if half_saturation <= 0.0 or slope <= 0.0:
        raise ValueError("half_saturation and slope must be positive")
    powered = np.power(np.asarray(x, dtype=np.float64), slope)
    return powered / (half_saturation**slope + powered)


def logistic_saturation(x: FloatArray, lam: FloatArray | float) -> FloatArray:
    """Map spend to a 0-1 response with the logistic curve used by PyMC-Marketing.

    ``(1 - exp(-lam * x)) / (1 + exp(-lam * x))``: zero at zero spend, approaching 1 as
    spend grows, with ``lam`` controlling how quickly returns diminish.
    """
    decay = np.exp(-np.asarray(lam, dtype=np.float64) * np.asarray(x, dtype=np.float64))
    return (1.0 - decay) / (1.0 + decay)
