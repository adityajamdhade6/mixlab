"""Analyse a geo-lift test: synthetic control, difference-in-differences and placebo checks.

**Synthetic control.** Before the test, find the non-negative mix of control regions whose
weekly revenue best tracks the test regions. During the test, that mix is the estimate of what
the test regions would have earned without the change; the gap is the incremental revenue.
Its uncertainty comes from how closely the mix tracked the test regions before the test.

**Difference-in-differences.** A simpler cross-check: the change in the test regions' average
weekly revenue minus the change in the control regions', scaled to the test regions' size.

**Placebo checks.** Pretend each control region was the test region and repeat the analysis.
If many fake tests show effects as large as the real one, the result is not convincing.
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from mixlab import config
from mixlab.transforms import FloatArray


def revenue_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    """Return weekly revenue with one column per region (dates as the index)."""
    return panel.pivot(index=config.DATE_COL, columns=config.GEO_COL, values=config.TARGET_COL)


def synthetic_weights(target: FloatArray, donors: FloatArray) -> FloatArray:
    """Return non-negative donor weights that best reproduce ``target`` (least squares)."""
    weights, _ = nnls(donors, target)
    return weights


def synthetic_control(
    wide: pd.DataFrame, test: list[str], control: list[str], start: pd.Timestamp
) -> dict[str, Any]:
    """Estimate the effect on the summed ``test`` regions from ``start`` onwards.

    Returns the weekly gap (actual minus synthetic), the total effect, its standard error
    (pre-period residual spread times the square root of the test length) and the fit quality.
    """
    pre, post = wide.index < start, wide.index >= start
    target = wide[test].sum(axis=1).to_numpy(dtype=float)
    donors = wide[control].to_numpy(dtype=float)
    weights = synthetic_weights(target[pre], donors[pre])
    synthetic = donors @ weights
    gap = target - synthetic
    residual_sd = float(np.std(gap[pre], ddof=1))
    weeks = int(post.sum())
    return {
        "weights": dict(zip(control, weights.tolist(), strict=True)),
        "weekly_gap": gap[post].tolist(),
        "effect": float(gap[post].sum()),
        "standard_error": residual_sd * float(np.sqrt(weeks)),
        "pre_period_mape_pct": float(100 * np.mean(np.abs(gap[pre]) / target[pre])),
        "residual_sd": residual_sd,
        "weeks": weeks,
    }


def difference_in_differences(
    wide: pd.DataFrame, test: list[str], control: list[str], start: pd.Timestamp
) -> float:
    """Return the total effect by difference-in-differences, in the test regions' rupees."""
    pre, post = wide.index < start, wide.index >= start
    test_series, control_series = wide[test].sum(axis=1), wide[control].sum(axis=1)
    ratio = test_series[pre].mean() / control_series[pre].mean()
    change_test = test_series[post].mean() - test_series[pre].mean()
    change_control = (control_series[post].mean() - control_series[pre].mean()) * ratio
    return float((change_test - change_control) * post.sum())


def placebo_effects(wide: pd.DataFrame, control: list[str], start: pd.Timestamp) -> list[float]:
    """Return the standardised effect of a fake test on each control region in turn."""
    scores = []
    for fake in control:
        donors = [c for c in control if c != fake]
        if not donors:
            continue
        result = synthetic_control(wide, [fake], donors, start)
        scores.append(result["effect"] / max(result["standard_error"], 1e-9))
    return scores


def analyze_test(
    panel: pd.DataFrame, test: list[str], control: list[str], start: str | pd.Timestamp
) -> dict[str, Any]:
    """Return the test's incremental revenue with uncertainty, a cross-check and placebos."""
    wide = revenue_matrix(panel)
    start = pd.Timestamp(start)
    result = synthetic_control(wide, test, control, start)
    z = result["effect"] / max(result["standard_error"], 1e-9)
    placebos = placebo_effects(wide, control, start)
    as_extreme = sum(abs(score) >= abs(z) for score in placebos)
    p_value = (as_extreme + 1) / (len(placebos) + 1)
    low = result["effect"] - config.Z_95 * result["standard_error"]
    high = result["effect"] + config.Z_95 * result["standard_error"]
    return {
        **result,
        "z_score": float(z),
        "range_95": [float(low), float(high)],
        "did_effect": difference_in_differences(wide, test, control, start),
        "placebo_z_scores": placebos,
        "placebo_p_value": float(p_value),
        "passes_placebo": bool(p_value <= config.PLACEBO_ALPHA),
        "test_regions": test,
        "control_regions": control,
        "start": str(start.date()),
    }
