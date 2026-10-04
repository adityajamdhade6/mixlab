"""Model diagnostics and fit metrics: convergence, out-of-sample error, parameter recovery."""

from collections.abc import Callable
from typing import Any

import arviz as az
import numpy as np
import pandas as pd

from mixlab import config


def convergence_diagnostics(idata: az.InferenceData, var_names: list[str]) -> dict[str, Any]:
    """Return the headline MCMC health numbers for the sampled parameters.

    Args:
        idata: Inference data with ``posterior`` and ``sample_stats`` groups.
        var_names: Sampled parameters to check (not derived quantities).

    Returns:
        Worst r_hat, smallest bulk and tail effective sample size, the parameter that owns
        each, the number of divergences, and an overall ``converged`` flag.

    """
    summary = az.summary(idata, var_names=var_names, kind="diagnostics")
    posterior = idata.posterior
    divergences = int(idata.sample_stats["diverging"].sum())
    max_rhat = float(summary["r_hat"].max())
    min_bulk = float(summary["ess_bulk"].min())
    min_tail = float(summary["ess_tail"].min())
    diagnostics: dict[str, Any] = {
        "chains": int(posterior.sizes["chain"]),
        "draws_per_chain": int(posterior.sizes["draw"]),
        "n_parameters": int(len(summary)),
        "max_r_hat": max_rhat,
        "max_r_hat_parameter": str(summary["r_hat"].idxmax()),
        "min_ess_bulk": min_bulk,
        "min_ess_bulk_parameter": str(summary["ess_bulk"].idxmin()),
        "min_ess_tail": min_tail,
        "min_ess_tail_parameter": str(summary["ess_tail"].idxmin()),
        "divergences": divergences,
    }
    diagnostics["converged"] = is_converged(diagnostics)
    return diagnostics


def divergence_limit(diagnostics: dict[str, Any]) -> int:
    """Return how many divergent draws are tolerated for this number of draws."""
    draws = diagnostics["chains"] * diagnostics["draws_per_chain"]
    return int(config.MAX_DIVERGENCE_SHARE * draws)


def is_converged(diagnostics: dict[str, Any]) -> bool:
    """Return whether r_hat, effective sample size and divergences are all within limits."""
    ess = min(diagnostics["min_ess_bulk"], diagnostics["min_ess_tail"])
    return bool(
        diagnostics["max_r_hat"] <= config.MAX_RHAT
        and ess >= config.MIN_ESS
        and diagnostics["divergences"] <= divergence_limit(diagnostics)
    )


def explain_convergence(diagnostics: dict[str, Any]) -> list[str]:
    """Return one plain-English line per diagnostic, each starting with PASS or FAIL."""
    rhat_ok = diagnostics["max_r_hat"] <= config.MAX_RHAT
    ess = min(diagnostics["min_ess_bulk"], diagnostics["min_ess_tail"])
    ess_ok = ess >= config.MIN_ESS
    limit = divergence_limit(diagnostics)
    div_ok = diagnostics["divergences"] <= limit
    flag = {True: "PASS", False: "FAIL"}
    return [
        f"{flag[rhat_ok]} r_hat: worst {diagnostics['max_r_hat']:.3f} "
        f"({diagnostics['max_r_hat_parameter']}), limit {config.MAX_RHAT}. "
        + (
            "The independent chains agree with each other."
            if rhat_ok
            else "The chains disagree, so the answer depends on where the sampler started."
        ),
        f"{flag[ess_ok]} effective sample size: smallest {ess:.0f} "
        f"({diagnostics['min_ess_bulk_parameter']}), minimum {config.MIN_ESS}. "
        + (
            "There are enough independent draws for stable averages and ranges."
            if ess_ok
            else "Too few independent draws; averages and ranges will wobble between runs."
        ),
        f"{flag[div_ok]} divergences: {diagnostics['divergences']} (tolerated: {limit}). "
        + (
            "The sampler explored the space without meaningful trouble."
            if div_ok
            else "The sampler hit regions it could not explore, so results may be biased."
        ),
    ]


def roi_recovery(insights: dict[str, Any], ground_truth: dict[str, Any]) -> pd.DataFrame:
    """Compare each channel's estimated ROI with the true ROI from the data generator.

    Returns one row per channel with the true ROI, the estimate and its 94% range, and
    whether the truth falls inside that range.
    """
    rows = []
    for channel, metrics in insights["channels"].items():
        true_roi = ground_truth["channels"][channel]["true_roi"]
        roi = metrics["roi"]
        rows.append(
            {
                "channel": channel,
                "true_roi": true_roi,
                "estimated_roi": roi["mean"],
                "low": roi["hdi_low"],
                "high": roi["hdi_high"],
                "truth_inside_range": roi["hdi_low"] <= true_roi <= roi["hdi_high"],
            }
        )
    return pd.DataFrame(rows)


def trust_notes(
    insights: dict[str, Any],
    optimizer: dict[str, Any],
    diagnostics: dict[str, Any] | None = None,
    recovery: pd.DataFrame | None = None,
    name: Callable[[str], str] = str,
    backtest: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Return plain-English notes on when not to trust the model, specific to these results.

    Each note has a ``title`` and a ``detail``. General limits come first, then anything the
    results themselves reveal: poor sampling, wide ranges, burst channels, tiny channels,
    recommendations beyond the spend range seen, and misses against the ground truth.
    Channels sharing a problem are grouped into one note. ``name`` maps a channel id to the
    label shown to the reader.
    """
    notes = [
        {
            "title": "It measures revenue, not profit",
            "detail": "ROI here is revenue per rupee spent. A channel with ROI above 1 can "
            "still lose money once product margin is applied.",
        },
        {
            "title": "It only knows the spend levels it has seen",
            "detail": "Predictions for spend far above or below the historical range are "
            "extrapolations from an assumed curve shape, not evidence.",
        },
    ]
    if diagnostics and not is_converged(diagnostics):
        notes.append(
            {
                "title": "The sampler reported problems",
                "detail": f"{diagnostics['divergences']} divergences, worst r-hat "
                f"{diagnostics['max_r_hat']:.3f}. Treat ranges as approximate until refitted.",
            }
        )
    if backtest:
        errors = [fold["mape_pct"] for fold in backtest["folds"]]
        fits = [fold["r2"] for fold in backtest["folds"]]
        if max(errors) > config.UNEVEN_BACKTEST_RATIO * min(errors) or min(fits) < config.MIN_R2:
            notes.append(
                {
                    "title": "Forecast accuracy is uneven across periods",
                    "detail": f"Across the rolling backtest, error ranged from {min(errors):.1f}% "
                    f"to {max(errors):.1f}% and R-squared from {min(fits):.2f} to "
                    f"{max(fits):.2f}. The latest window is not representative of all of them.",
                }
            )
    n_weeks = insights["period"]["n_weeks"]
    total_spend = insights["totals"]["media_spend"]
    channels = insights["channels"]
    wide = [
        f"{name(c)} ({m['roi']['hdi_low']:.2f} to {m['roi']['hdi_high']:.2f})"
        for c, m in channels.items()
        if m["roi"]["hdi_low"] < config.WIDE_INTERVAL_RATIO * m["roi"]["mean"]
    ]
    if wide:
        notes.append(
            {
                "title": "Some ROI ranges are too wide to act on alone",
                "detail": f"Likely ROI: {'; '.join(wide)}. Run a holdout or regional test "
                "before a large budget move on these channels.",
            }
        )
    bursts = [
        f"{name(c)} ({m['active_weeks']} of {n_weeks} weeks)"
        for c, m in channels.items()
        if m["active_weeks"] < config.FLIGHTED_ZERO_SHARE * n_weeks
    ]
    if bursts:
        notes.append(
            {
                "title": "Channels that ran in bursts are hard to separate from the season",
                "detail": f"{'; '.join(bursts)}. If those weeks coincide with festivals, the "
                "effect is tangled with the seasonal lift.",
            }
        )
    small = [
        f"{name(c)} ({100 * m['total_spend'] / total_spend:.1f}% of spend)"
        for c, m in channels.items()
        if m["total_spend"] < config.MIN_SPEND_SHARE * total_spend
    ]
    if small:
        notes.append(
            {
                "title": "Very small channels cannot be measured precisely",
                "detail": f"{'; '.join(small)}. Their effect is smaller than week-to-week noise.",
            }
        )
    extrapolated = optimizer["expected_revenue"]["extrapolated_channels"]
    if extrapolated:
        notes.append(
            {
                "title": "The recommendation goes beyond past spend",
                "detail": f"It pushes {', '.join(name(c) for c in extrapolated)} above any "
                "weekly spend in the data. Step up gradually and watch the results.",
            }
        )
    check = optimizer["expected_revenue"].get("truth_check")
    if check:
        notes.append(
            {
                "title": "Expected uplift is optimistic",
                "detail": f"The model expected {check['model_expected_uplift_pct']:+.1f}% "
                f"from its recommendation; scored against the known truth it delivers "
                f"{check['true_uplift_pct']:+.1f}%. Optimizers favour channels the model "
                "happens to overestimate.",
            }
        )
    if recovery is not None and not recovery["truth_inside_range"].all():
        missed = [
            f"{name(row.channel)} (true {row.true_roi:.2f}, model {row.low:.2f} to {row.high:.2f})"
            for row in recovery.loc[~recovery["truth_inside_range"]].itertuples()
        ]
        notes.append(
            {
                "title": "The model got some channels wrong",
                "detail": f"{'; '.join(missed)}. Only visible because this brand is synthetic.",
            }
        )
    return notes


def prediction_error(prediction: pd.DataFrame, actual: pd.Series) -> dict[str, float]:
    """Return MAPE (%), R-squared and the share of weeks (%) inside the 94% predictive range."""
    truth = actual.to_numpy(dtype=float)
    mean = prediction["mean"].to_numpy(dtype=float)
    inside = (truth >= prediction["lower"].to_numpy()) & (truth <= prediction["upper"].to_numpy())
    total = float(((truth - truth.mean()) ** 2).sum())
    return {
        "mape_pct": float((np.abs(mean - truth) / truth).mean() * 100),
        "r2": float(1 - ((truth - mean) ** 2).sum() / total) if total > 0 else float("nan"),
        "coverage_pct": float(inside.mean() * 100),
    }


def measurement_flags(spend: np.ndarray, channels: list[str]) -> dict[str, list[str]]:
    """Return, per channel, the reasons its effect is hard to measure (empty if none).

    A channel is flagged if it ran in bursts (active in under half the weeks) or is a very
    small share of total spend. These are the same conditions the trust notes report, and
    the optimizer's confidence gate uses them to hold such channels to smaller moves.
    """
    active_share = (spend > 0).mean(axis=0)
    spend_share = spend.sum(axis=0) / spend.sum()
    flags: dict[str, list[str]] = {}
    for index, channel in enumerate(channels):
        reasons = []
        if active_share[index] < config.FLIGHTED_ZERO_SHARE:
            reasons.append(config.CAVEAT_BURSTS)
        if spend_share[index] < config.MIN_SPEND_SHARE:
            reasons.append(config.CAVEAT_SMALL)
        flags[channel] = reasons
    return flags


def rolling_backtest(
    df: pd.DataFrame,
    fit_and_predict: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame],
    horizon: int = config.BACKTEST_HORIZON_WEEKS,
    folds: int = config.BACKTEST_FOLDS,
) -> dict[str, Any]:
    """Score the model on weeks it has not seen, several times over.

    Each fold trains on everything before a cut-off and predicts the next ``horizon`` weeks;
    cut-offs step back by ``horizon`` so the test windows do not overlap. The last fold is the
    most recent ``horizon`` weeks (the holdout).

    Args:
        df: Weekly contract frame.
        fit_and_predict: Function taking (train, test) and returning a prediction frame with
            date, mean, lower and upper.
        horizon: Weeks per test window.
        folds: Number of windows.

    Returns:
        ``folds`` (oldest first, each with metrics and week-level predictions) and ``holdout``
        (the last fold).

    """
    results = []
    for fold in range(folds, 0, -1):
        stop = len(df) - (fold - 1) * horizon
        train, test = df.iloc[: stop - horizon], df.iloc[stop - horizon : stop]
        prediction = fit_and_predict(train, test)
        actual = test[config.TARGET_COL]
        results.append(
            {
                "train_weeks": len(train),
                "test_start": str(pd.Timestamp(test[config.DATE_COL].iloc[0]).date()),
                "test_end": str(pd.Timestamp(test[config.DATE_COL].iloc[-1]).date()),
                **prediction_error(prediction, actual),
                "weeks": [
                    {
                        "date": str(pd.Timestamp(date).date()),
                        "actual": float(observed),
                        "predicted": float(row.mean),
                        "lower": float(row.lower),
                        "upper": float(row.upper),
                    }
                    for date, observed, row in zip(
                        test[config.DATE_COL], actual, prediction.itertuples(), strict=True
                    )
                ],
            }
        )
    return {"horizon_weeks": horizon, "folds": results, "holdout": results[-1]}
