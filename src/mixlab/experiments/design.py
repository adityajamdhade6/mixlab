"""Design a geo-lift test: pick matched regions, size the test, and write a one-page plan.

**Regions.** A region makes a good test region when the other regions can reproduce its past
weekly revenue closely (a tight synthetic control), because then a change during the test
stands out. Regions are ranked by that fit; the best ``config.TEST_REGIONS`` are tested and
the rest form the control pool.

**Power.** With weekly residual spread ``sd`` between the test regions and their synthetic
control, a test of ``W`` weeks has a standard error of ``sd x sqrt(W)`` on the total effect.
For 80% power at 5% significance the expected effect must reach ``config.POWER_Z`` standard
errors, so ``W >= (POWER_Z x sd / weekly effect)^2``. The weekly effect is the extra spend
times the channel's marginal ROI from the model. The designer tries larger spend changes until
the test fits within ``config.MAX_TEST_WEEKS``.
"""

import math
from typing import Any

import numpy as np
import pandas as pd

from mixlab import config
from mixlab.experiments.analyze import revenue_matrix, synthetic_weights


def fit_error(wide: pd.DataFrame, region: str, donors: list[str]) -> tuple[float, float]:
    """Return (relative error, residual sd) of a synthetic control for ``region`` over history."""
    target = wide[region].to_numpy(dtype=float)
    pool = wide[donors].to_numpy(dtype=float)
    gap = target - pool @ synthetic_weights(target, pool)
    return float(np.mean(np.abs(gap)) / np.mean(target)), float(np.std(gap, ddof=1))


def rank_regions(panel: pd.DataFrame) -> list[dict[str, Any]]:
    """Rank regions by how well the others reproduce their weekly revenue (best first)."""
    wide = revenue_matrix(panel)
    rows = []
    for region in wide.columns:
        donors = [c for c in wide.columns if c != region]
        error, sd = fit_error(wide, region, donors)
        rows.append({"region": region, "fit_error_pct": 100 * error, "residual_sd": sd})
    return sorted(rows, key=lambda row: row["fit_error_pct"])


def weeks_needed(residual_sd: float, weekly_effect: float) -> float:
    """Return the test length in weeks for 80% power (may be fractional or huge)."""
    if weekly_effect <= 0:
        return math.inf
    return (config.POWER_Z * residual_sd / weekly_effect) ** 2


def design_test(
    panel: pd.DataFrame,
    channel: str,
    marginal_roi: float,
    n_test: int = config.TEST_REGIONS,
    recent_weeks: int = config.OPTIMIZER_WEEKS,
) -> dict[str, Any]:
    """Return a geo-lift test plan for raising ``channel`` spend in matched test regions.

    Args:
        panel: Long weekly panel with a ``geo`` column.
        channel: Channel name without the ``spend_`` prefix.
        marginal_roi: Revenue per extra rupee expected by the model (posterior median).
        n_test: Number of test regions.
        recent_weeks: Weeks of recent history used for the regions' current weekly spend.

    """
    ranking = rank_regions(panel)
    test = [row["region"] for row in ranking[:n_test]]
    control = [row["region"] for row in ranking[n_test:]]
    wide = revenue_matrix(panel)
    target = wide[test].sum(axis=1).to_numpy(dtype=float)
    pool = wide[control].to_numpy(dtype=float)
    residual_sd = float(np.std(target - pool @ synthetic_weights(target, pool), ddof=1))
    column = f"{config.SPEND_PREFIX}{channel}"
    rows = panel[panel[config.GEO_COL].isin(test)]
    weekly_spend = float(rows.groupby(config.DATE_COL)[column].sum().tail(recent_weeks).mean())

    options = []
    for multiplier in config.TEST_SPEND_MULTIPLIERS:
        extra = (multiplier - 1) * weekly_spend
        raw = weeks_needed(residual_sd, extra * marginal_roi)
        weeks = max(config.MIN_TEST_WEEKS, math.ceil(raw)) if math.isfinite(raw) else None
        options.append(
            {
                "multiplier": multiplier,
                "extra_weekly_spend": extra,
                "expected_weekly_effect": extra * marginal_roi,
                "weeks_needed": weeks,
                "feasible": weeks is not None and weeks <= config.MAX_TEST_WEEKS,
                "test_cost": extra * (weeks or config.MAX_TEST_WEEKS),
            }
        )
    chosen = next((o for o in options if o["feasible"]), options[-1])
    return {
        "channel": channel,
        "test_regions": test,
        "control_regions": control,
        "region_ranking": ranking,
        "residual_sd": residual_sd,
        "current_weekly_spend": weekly_spend,
        "marginal_roi": marginal_roi,
        "options": options,
        "chosen": chosen,
        "feasible": chosen["feasible"],
    }


def plan_text(plan: dict[str, Any], labels: dict[str, str], start: str, end: str) -> str:
    """Return the one-page test plan in plain English (Markdown)."""
    chosen = plan["chosen"]
    test = ", ".join(labels.get(r, r) for r in plan["test_regions"])
    control = ", ".join(labels.get(r, r) for r in plan["control_regions"])
    lakh = config.INR_PER_LAKH
    weeks = chosen["weeks_needed"]
    return "\n".join(
        [
            f"### Test plan: {plan['channel'].replace('_', ' ')}",
            f"- **Regions:** raise spend in {test}. Compare with a weighted mix of {control}.",
            f"- **Dates:** {start} to {end} ({weeks} weeks).",
            f"- **Spend change:** {chosen['multiplier']:g}x current spend in the test regions, "
            f"about ₹{chosen['extra_weekly_spend'] / lakh:,.1f} L extra a week "
            f"(₹{chosen['test_cost'] / lakh:,.1f} L in total).",
            f"- **Success metric:** incremental revenue in the test regions versus their "
            f"synthetic control; the model expects about ₹"
            f"{chosen['expected_weekly_effect'] / lakh:,.1f} L a week.",
            "- **Decision rule:** if the measured return per extra rupee is clearly above "
            "the model's current estimate, raise the budget in the next quarter; if it is "
            "clearly below, cut it; if the range still straddles the estimate, keep spend "
            "and extend the test. Results only count if the placebo check passes.",
            f"- **Sized for:** 80% power at 5% significance, given week-to-week noise of ₹"
            f"{plan['residual_sd'] / lakh:,.1f} L between the test regions and their control.",
        ]
    )
