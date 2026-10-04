"""Tests for convergence diagnostics."""

import arviz as az
import numpy as np
import pytest

from mixlab.evaluate import convergence_diagnostics, explain_convergence


def make_idata(shift: float = 0.0, divergences: int = 0) -> az.InferenceData:
    rng = np.random.default_rng(0)
    draws = rng.normal(size=(4, 500))
    draws[0] += shift
    diverging = np.zeros((4, 500), dtype=bool)
    diverging[0, :divergences] = True
    return az.from_dict(posterior={"beta": draws}, sample_stats={"diverging": diverging})


def test_healthy_chains_converge() -> None:
    diagnostics = convergence_diagnostics(make_idata(), ["beta"])
    assert diagnostics["converged"]
    assert diagnostics["chains"] == 4 and diagnostics["draws_per_chain"] == 500
    assert all(line.startswith("PASS") for line in explain_convergence(diagnostics))


def test_disagreeing_chains_fail_rhat() -> None:
    diagnostics = convergence_diagnostics(make_idata(shift=5.0), ["beta"])
    assert not diagnostics["converged"]
    assert diagnostics["max_r_hat"] > 1.1
    assert explain_convergence(diagnostics)[0].startswith("FAIL")


def test_divergences_fail() -> None:
    diagnostics = convergence_diagnostics(make_idata(divergences=7), ["beta"])
    assert diagnostics["divergences"] == 7
    assert not diagnostics["converged"]
    assert explain_convergence(diagnostics)[2].startswith("FAIL")


def test_roi_recovery_and_grouped_trust_notes() -> None:
    from mixlab.evaluate import roi_recovery, trust_notes

    def channel(low: float, mean: float, high: float, weeks: int, spend: float) -> dict:
        return {
            "roi": {"mean": mean, "hdi_low": low, "hdi_high": high},
            "active_weeks": weeks,
            "total_spend": spend,
        }

    insights = {
        "period": {"n_weeks": 100},
        "totals": {"media_spend": 1000.0},
        "channels": {
            "meta_ads": channel(0.1, 1.0, 2.0, 100, 600.0),
            "tv": channel(0.6, 0.7, 0.8, 12, 390.0),
            "email": channel(0.0, 5.0, 20.0, 100, 10.0),
        },
    }
    truth = {
        "channels": {
            "meta_ads": {"true_roi": 1.2},
            "tv": {"true_roi": 1.5},
            "email": {"true_roi": 3.0},
        }
    }
    optimizer = {"expected_revenue": {"extrapolated_channels": ["email"]}}

    recovery = roi_recovery(insights, truth)
    assert recovery.set_index("channel")["truth_inside_range"].to_dict() == {
        "meta_ads": True,
        "tv": False,
        "email": True,
    }

    notes = trust_notes(insights, optimizer, None, recovery, name=str.upper)
    titles = [note["title"] for note in notes]
    assert len(titles) == len(set(titles))  # one note per kind of problem, not per channel
    by_title = {note["title"]: note["detail"] for note in notes}
    assert "META_ADS" in by_title["Some ROI ranges are too wide to act on alone"]
    assert "EMAIL" in by_title["Some ROI ranges are too wide to act on alone"]
    assert (
        "TV (12 of 100 weeks)"
        in by_title["Channels that ran in bursts are hard to separate from the season"]
    )
    assert "EMAIL (1.0% of spend)" in by_title["Very small channels cannot be measured precisely"]
    assert "TV (true 1.50" in by_title["The model got some channels wrong"]


def test_measurement_flags_mark_burst_and_tiny_channels() -> None:
    from mixlab import config
    from mixlab.evaluate import measurement_flags

    steady = np.full(100, 100.0)
    bursts = np.where(np.arange(100) < 20, 100.0, 0.0)
    tiny = np.full(100, 1.0)
    flags = measurement_flags(np.column_stack([steady, bursts, tiny]), ["steady", "bursts", "tiny"])
    assert flags == {"steady": [], "bursts": [config.CAVEAT_BURSTS], "tiny": [config.CAVEAT_SMALL]}


def test_rolling_backtest_uses_non_overlapping_windows_and_scores_each() -> None:
    import pandas as pd

    from mixlab.evaluate import rolling_backtest

    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=60, freq="7D"),
            "revenue": np.arange(100.0, 160.0),
        }
    )
    seen: list[tuple[int, int]] = []

    def perfect(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
        seen.append((len(train), len(test)))
        return pd.DataFrame(
            {
                "date": test["date"],
                "mean": test["revenue"] * 1.1,
                "lower": test["revenue"] * 0.9,
                "upper": test["revenue"] * 1.3,
            }
        )

    result = rolling_backtest(frame, perfect, horizon=10, folds=3)
    assert seen == [(30, 10), (40, 10), (50, 10)]
    assert result["holdout"] == result["folds"][-1]
    assert result["holdout"]["test_end"] == "2025-02-17"
    for fold in result["folds"]:
        assert fold["mape_pct"] == pytest.approx(10.0)
        assert fold["coverage_pct"] == 100.0 and fold["r2"] < 1
        assert len(fold["weeks"]) == 10


def test_prediction_error_r2_is_one_for_a_perfect_prediction() -> None:
    import pandas as pd

    from mixlab.evaluate import prediction_error

    actual = pd.Series([10.0, 20.0, 30.0])
    perfect = pd.DataFrame({"mean": actual, "lower": actual - 1, "upper": actual + 1})
    assert prediction_error(perfect, actual) == {"mape_pct": 0.0, "r2": 1.0, "coverage_pct": 100.0}
