"""Budget allocation optimizer over fitted channel response curves.

Optimization uses PyMC-Marketing's built-in ``BudgetOptimizer`` (present in 0.19). Note its
``total_budget`` is a per-week amount applied to every week of the window; this module works
in period totals and converts.

A plan is a total spend per channel over ``n_weeks``, spread evenly across the weeks. Its
revenue is the incremental (media-driven) revenue it causes, including carryover that lands
after the window, evaluated for every posterior draw. Current and recommended plans are
evaluated the same way, so comparisons are like for like.

Run ``python -m mixlab.optimizer`` to write the summary JSON and figures.
"""

import argparse
import dataclasses
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from pydantic import BaseModel, Field
from pymc_marketing.mmm.budget_optimizer import BudgetOptimizer
from pymc_marketing.mmm.multidimensional import MultiDimensionalBudgetOptimizerWrapper
from pymc_marketing.mmm.utility import average_response, value_at_risk
from scipy.optimize import minimize

from mixlab import config
from mixlab.config import BrandConfig
from mixlab.data_gen import channel_contribution
from mixlab.evaluate import measurement_flags
from mixlab.insights import (
    Estimate,
    PosteriorDraws,
    channel_colors,
    extract_draws,
    hdi,
    simulate_contributions,
    summarize,
)
from mixlab.model import MixLabModel
from mixlab.transforms import FloatArray

Objective = Literal["mean", "profit", "risk_adjusted", "percentile"]
Plan = dict[str, float]
Bounds = dict[str, tuple[float, float]]


class PlanOutcome(BaseModel):
    """Predicted result of one spend plan.

    Attributes:
        name: Plan label.
        n_weeks: Length of the planning window.
        spend: Total spend per channel over the window (INR).
        total_spend: Sum of ``spend``.
        incremental_revenue: Media-driven revenue caused by the plan.
        total_revenue: Incremental revenue plus organic revenue, assuming organic revenue
            repeats the last ``n_weeks`` of history.
        revenue_at_risk_percentile: Incremental revenue at the conservative percentile.
        roi: Incremental revenue per rupee spent.
        channel_revenue: Posterior-mean incremental revenue per channel.

    """

    name: str
    n_weeks: int
    spend: Plan
    total_spend: float
    incremental_revenue: Estimate
    total_revenue: Estimate
    revenue_at_risk_percentile: float
    roi: Estimate
    channel_revenue: Plan


class OptimizationResult(BaseModel):
    """Recommended allocation compared with the current one."""

    objective: str
    converged: bool
    message: str
    bounds: dict[str, tuple[float, float]]
    current: PlanOutcome
    recommended: PlanOutcome
    uplift: Estimate
    uplift_pct: Estimate
    prob_recommended_beats_current: float
    extrapolated_channels: list[str]
    caveats: dict[str, str] = Field(default_factory=dict)
    realistic_uplift: float = 0.0
    realistic_uplift_pct: float = 0.0
    limits: dict[str, dict[str, Any]] = Field(default_factory=dict)
    goal_probability: float | None = None
    corner_solution: bool = False
    corner_note: str = ""
    shrinkage: float = config.UPLIFT_SHRINKAGE
    margin: float = config.DEFAULT_MARGIN


# --- Plans and their revenue ----------------------------------------------------------------


def last_quarter_spend(draws: PosteriorDraws, n_weeks: int = config.OPTIMIZER_WEEKS) -> Plan:
    """Return total spend per channel over the last ``n_weeks`` of history."""
    totals = draws.spend[-n_weeks:].sum(axis=0)
    return {channel: float(total) for channel, total in zip(draws.channels, totals, strict=True)}


def plan_matrix(draws: PosteriorDraws, spend: Plan, n_weeks: int) -> FloatArray:
    """Return weekly spend (n_weeks + l_max, C): the plan spread evenly, then zero spend.

    The trailing zero weeks capture carryover that lands after the window ends.
    """
    weekly = np.array([spend.get(channel, 0.0) / n_weeks for channel in draws.channels])
    return np.vstack([np.tile(weekly, (n_weeks, 1)), np.zeros((draws.l_max, len(weekly)))])


def plan_revenue_draws(draws: PosteriorDraws, spend: Plan, n_weeks: int) -> FloatArray:
    """Return incremental revenue (S, C) of a plan, per posterior draw and channel."""
    return simulate_contributions(draws, plan_matrix(draws, spend, n_weeks)).sum(axis=1)


def organic_revenue_draws(draws: PosteriorDraws, n_weeks: int) -> FloatArray:
    """Return organic revenue (S) over the window, assuming it repeats the last ``n_weeks``."""
    return sum(values[:, -n_weeks:].sum(axis=1) for values in draws.organic.values())


def what_if(
    draws: PosteriorDraws, spend: Plan, n_weeks: int = config.OPTIMIZER_WEEKS, name: str = "plan"
) -> PlanOutcome:
    """Predict the revenue of any spend plan, with uncertainty.

    Args:
        draws: Posterior draws from the fitted model.
        spend: Total spend per channel over the window; missing channels spend nothing.
        n_weeks: Length of the window.
        name: Label for the plan.

    """
    unknown = set(spend) - set(draws.channels)
    if unknown:
        raise ValueError(f"Unknown channels: {sorted(unknown)}")
    if any(value < 0 for value in spend.values()):
        raise ValueError("Spend cannot be negative.")
    by_channel = plan_revenue_draws(draws, spend, n_weeks)
    incremental = by_channel.sum(axis=1)
    total_spend = float(sum(spend.values()))
    roi = incremental / total_spend if total_spend > 0 else np.zeros_like(incremental)
    return PlanOutcome(
        name=name,
        n_weeks=n_weeks,
        spend={channel: float(spend.get(channel, 0.0)) for channel in draws.channels},
        total_spend=total_spend,
        incremental_revenue=summarize(incremental),
        total_revenue=summarize(incremental + organic_revenue_draws(draws, n_weeks)),
        revenue_at_risk_percentile=float(np.percentile(incremental, config.RISK_PERCENTILE)),
        roi=summarize(roi),
        channel_revenue=dict(zip(draws.channels, by_channel.mean(axis=0).tolist(), strict=True)),
    )


# --- Bounds ---------------------------------------------------------------------------------


def build_bounds(
    reference: Plan,
    total_budget: float,
    minimum: Plan | None = None,
    maximum: Plan | None = None,
    max_change: float | dict[str, float] | None = None,
) -> Bounds:
    """Return (min, max) total spend per channel.

    Args:
        reference: Current plan, used for the ``max_change`` rule.
        total_budget: Budget to allocate.
        minimum: Explicit floors, e.g. ``{"tv": 2_000_000}`` for "TV at least 20 lakh".
        maximum: Explicit ceilings.
        max_change: If set, each channel stays within plus or minus this share of its
            reference spend (0.3 means "no channel changes more than 30%").

    Explicit floors and ceilings take precedence over ``max_change``.

    Raises:
        ValueError: If the bounds contradict each other or cannot add up to the budget.

    """
    bounds: Bounds = {}
    for channel, current in reference.items():
        low, high = 0.0, total_budget
        change = max_change.get(channel) if isinstance(max_change, dict) else max_change
        if change is not None:
            low, high = current * (1 - change), current * (1 + change)
        if minimum and channel in minimum:
            low = minimum[channel]
            high = max(high, low)
        if maximum and channel in maximum:
            high = maximum[channel]
        if low > high:
            raise ValueError(f"'{channel}': minimum {low:,.0f} is above maximum {high:,.0f}.")
        bounds[channel] = (float(low), float(high))
    floor = sum(low for low, _ in bounds.values())
    ceiling = sum(high for _, high in bounds.values())
    if not floor - config.BOUND_TOLERANCE <= total_budget <= ceiling + config.BOUND_TOLERANCE:
        raise ValueError(
            f"Budget {total_budget:,.0f} cannot be met: channel bounds allow between "
            f"{floor:,.0f} and {ceiling:,.0f}. Loosen the bounds or change the budget."
        )
    return bounds


# --- Optimization ---------------------------------------------------------------------------


class BudgetAllocator:
    """Reusable wrapper around PyMC-Marketing's ``BudgetOptimizer`` for one window length.

    Compiling the optimizer takes a few seconds; allocations afterwards take about a second,
    so one allocator is reused across budgets and scenarios.
    """

    def __init__(self, model: MixLabModel, n_weeks: int = config.OPTIMIZER_WEEKS) -> None:
        """Prepare an optimization window of ``n_weeks`` starting after the training data."""
        week = pd.Timedelta(days=config.DAYS_PER_WEEK)
        start = pd.to_datetime(model.idata.fit_data[config.DATE_COL].to_numpy()).max() + week
        end = start + week * (n_weeks - 1)
        self.n_weeks = n_weeks
        self.columns = model.channels
        self._wrapper = MultiDimensionalBudgetOptimizerWrapper(
            model=model.mmm, start_date=str(start.date()), end_date=str(end.date())
        )
        self._optimizers: dict[tuple[str, float], BudgetOptimizer] = {}

    def _optimizer(self, objective: Objective, percentile: float) -> BudgetOptimizer:
        """Return a compiled optimizer for the objective, building it on first use."""
        key = (objective, percentile if objective == "percentile" else 0.0)
        if key not in self._optimizers:
            # "mean" maximises expected revenue. "percentile" maximises the revenue we would
            # still get in a bad case (e.g. the 10th percentile), which favours channels
            # whose effect is well established over ones that only might be good.
            utility = (
                average_response
                if objective == "mean"
                else value_at_risk(confidence_level=1 - percentile / 100)
            )
            self._optimizers[key] = BudgetOptimizer(
                num_periods=self._wrapper.num_periods,
                utility_function=utility,
                response_variable="total_media_contribution_original_scale",
                model=self._wrapper,
            )
        return self._optimizers[key]

    def allocate(
        self,
        total_budget: float,
        bounds: Bounds,
        objective: Objective = "mean",
        percentile: float = config.RISK_PERCENTILE,
        start: Plan | None = None,
    ) -> tuple[Plan, bool, str]:
        """Return the best plan (period totals), whether the solver converged, and its message.

        ``start`` is an optional plan to begin the search from; it must satisfy the bounds
        and add up to the budget. Starting near the answer makes the solver much faster.
        """
        names = list(bounds)
        weekly_bounds = xr.DataArray(
            np.array([bounds[name] for name in names]) / self.n_weeks,
            dims=["channel", "bound"],
            coords={"channel": self.columns, "bound": ["lower", "upper"]},
        )
        # Start from a point that already satisfies the bounds and the budget: the library's
        # default start (an equal split) can sit outside tight bounds and stall the solver.
        low, high = weekly_bounds.to_numpy().T
        room = high - low
        weekly_budget = total_budget / self.n_weeks
        share = (weekly_budget - low.sum()) / room.sum() if room.sum() > 0 else 0.0
        x0 = (
            np.array([start[name] for name in names]) / self.n_weeks
            if start is not None
            else low + share * room
        )
        allocation, result = self._optimizer(objective, percentile).allocate_budget(
            total_budget=weekly_budget,
            budget_bounds=weekly_bounds,
            x0=x0,
            minimize_kwargs={
                "options": {
                    "ftol": config.OPTIMIZER_FTOL,
                    "maxiter": config.OPTIMIZER_MAX_ITERATIONS,
                }
            },
            return_if_fail=True,
        )
        totals = allocation.to_numpy() * self.n_weeks
        plan = {
            name: float(np.clip(total, *bounds[name]))
            for name, total in zip(names, totals, strict=True)
        }
        return plan, bool(result.success), str(result.message)


OBJECTIVE_LABELS: dict[str, str] = {
    "mean": "expected revenue",
    "profit": "expected profit",
    "risk_adjusted": "risk-adjusted revenue",
    "percentile": f"{config.RISK_PERCENTILE:g}th percentile revenue",
}


def thin(draws: PosteriorDraws, n: int = config.OPTIMIZER_DRAWS) -> PosteriorDraws:
    """Return an evenly spaced subset of the draws, to keep the solver fast."""
    step = max(draws.alpha.shape[0] // n, 1)
    if step == 1:
        return draws
    pick = slice(None, None, step)

    def cut(values: FloatArray | None) -> FloatArray | None:
        return None if values is None else values[pick]

    return dataclasses.replace(
        draws,
        alpha=draws.alpha[pick],
        lam=draws.lam[pick],
        beta=draws.beta[pick],
        slope=cut(draws.slope),
        kappa=cut(draws.kappa),
        noise_sigma=cut(draws.noise_sigma),
        organic={name: values[pick] for name, values in draws.organic.items()},
        channel_contribution=draws.channel_contribution[pick],
    )


def objective_value(
    revenue: FloatArray,
    total_spend: float,
    objective: Objective = "mean",
    margin: float = config.DEFAULT_MARGIN,
    risk_lambda: float = config.RISK_LAMBDA,
    percentile: float = config.RISK_PERCENTILE,
) -> float:
    """Score a plan from its incremental-revenue draws (one value per posterior draw).

    - ``mean``: expected revenue.
    - ``profit``: expected gross profit, ``margin * revenue - spend``.
    - ``risk_adjusted``: mean minus ``risk_lambda`` standard deviations, which penalises plans
      whose payoff the model is unsure about.
    - ``percentile``: the revenue still achieved in a bad case (e.g. the 10th percentile).
    """
    if objective == "profit":
        return float(margin * revenue.mean() - total_spend)
    if objective == "risk_adjusted":
        return float(revenue.mean() - risk_lambda * revenue.std())
    if objective == "percentile":
        return float(np.percentile(revenue, percentile))
    return float(revenue.mean())


def solve_allocation(
    draws: PosteriorDraws,
    budget: float,
    bounds: Bounds,
    n_weeks: int,
    objective: Objective = "mean",
    margin: float = config.DEFAULT_MARGIN,
    risk_lambda: float = config.RISK_LAMBDA,
    percentile: float = config.RISK_PERCENTILE,
    starts: list[Plan] | None = None,
) -> tuple[Plan, bool, str]:
    """Split ``budget`` across channels to maximise the objective over the posterior.

    Every candidate plan is evaluated on (a thinned set of) posterior draws, so the objective
    reflects uncertainty rather than a single best-guess curve. The response surface is not
    concave for S-shaped curves, so the search starts from several points and keeps the best.

    Returns the plan (period totals), whether the solver converged, and its message.
    """
    names = list(bounds)
    low = np.array([bounds[name][0] for name in names])
    high = np.array([bounds[name][1] for name in names])
    if budget <= 0 or np.allclose(low, high):
        return dict(zip(names, low.tolist(), strict=True)), True, "Bounds leave no freedom."
    sample = thin(draws)

    def score(shares: FloatArray) -> float:
        plan = dict(zip(names, (shares * budget).tolist(), strict=True))
        revenue = plan_revenue_draws(sample, plan, n_weeks).sum(axis=1)
        return objective_value(revenue, budget, objective, margin, risk_lambda, percentile)

    room = high - low
    spread = (budget - low.sum()) / room.sum() if room.sum() > 0 else 0.0
    candidates = [low + spread * room]
    for start in starts or []:
        values = np.array([start.get(name, 0.0) for name in names])
        if abs(values.sum() - budget) <= config.BOUND_TOLERANCE * budget and np.all(
            (values >= low - 1.0) & (values <= high + 1.0)
        ):
            candidates.append(np.clip(values, low, high))
    reference = abs(score(candidates[0] / budget)) or 1.0
    best: tuple[float, FloatArray, bool, str] | None = None
    for start in candidates:
        result = minimize(
            lambda shares: -score(shares) / reference,
            start / budget,
            method="SLSQP",
            bounds=list(zip(low / budget, high / budget, strict=True)),
            constraints=[{"type": "eq", "fun": lambda shares: shares.sum() - 1.0}],
            options={"ftol": config.OPTIMIZER_FTOL, "maxiter": config.OPTIMIZER_MAX_ITERATIONS},
        )
        shares = np.clip(result.x, low / budget, high / budget)
        value = score(shares)
        if best is None or value > best[0]:
            best = (value, shares, bool(result.success), str(result.message))
    assert best is not None
    plan = dict(zip(names, (best[1] * budget).tolist(), strict=True))
    return plan, best[2], best[3]


def channel_limits(
    draws: PosteriorDraws,
    n_weeks: int,
    base_change: float = config.DEFAULT_MAX_CHANGE,
    measured: frozenset[str] = frozenset(),
) -> dict[str, dict[str, Any]]:
    """Return how far each channel may move by default, and why.

    Limits tighten with the evidence against trusting a channel's curve:

    - ran in bursts, or is a very small share of spend: ``config.GATED_MAX_CHANGE``;
    - its ROI range is wide relative to its estimate: ``config.UNCERTAIN_MAX_CHANGE``;
    - otherwise the standard ``base_change``.

    No channel is pushed above its highest weekly spend on record, where the curve is an
    extrapolation.

    Channels in ``measured`` have been calibrated with an experiment, which answers the
    "ran in bursts" and "too small to measure" caveats directly; only the width of their
    (calibrated) ROI range can still tighten their limit.
    """
    flags = measurement_flags(draws.spend, draws.channels)
    flags = {c: ([] if c in measured else f) for c, f in flags.items()}
    roi = draws.channel_contribution.sum(axis=1) / draws.spend.sum(axis=0)
    low, high = hdi(roi)
    peak = draws.spend.max(axis=0) * n_weeks
    limits: dict[str, dict[str, Any]] = {}
    for index, channel in enumerate(draws.channels):
        width = (high[index] - low[index]) / max(float(roi[:, index].mean()), 1e-9)
        if flags[channel]:
            change, reason = config.GATED_MAX_CHANGE, "; ".join(flags[channel])
        elif width > config.UNCERTAIN_RELATIVE_WIDTH:
            change = config.UNCERTAIN_MAX_CHANGE
            reason = (
                f"its ROI is uncertain (94% range {low[index]:.2f} to {high[index]:.2f}), "
                "so moves are kept small until a test narrows it"
            )
        else:
            change, reason = base_change, "measured well enough for the standard limit"
        limits[channel] = {
            "max_change": min(change, base_change),
            "reason": reason,
            "ceiling": float(peak[index]),
        }
    return limits


def corner_check(plan: Plan, bounds: Bounds) -> tuple[bool, str]:
    """Flag a plan where most channels sit on a limit, and say what that means."""
    at_limit = [
        channel
        for channel, (low, high) in bounds.items()
        if high > low
        and (
            abs(plan[channel] - low) <= 0.005 * max(high, 1.0)
            or abs(plan[channel] - high) <= 0.005 * max(high, 1.0)
        )
    ]
    if len(at_limit) < config.CORNER_SHARE * len(bounds):
        return False, ""
    return True, (
        f"{len(at_limit)} of {len(bounds)} channels are at a limit, so the limits, not the "
        "response curves, are deciding this plan. Read it as the direction to move in, not as "
        "a precise optimum; loosening a limit would change the answer."
    )


def _window(allocator_or_weeks: Any) -> int:
    """Return the planning window from an allocator or a plain number of weeks."""
    return int(getattr(allocator_or_weeks, "n_weeks", allocator_or_weeks))


def build_result(
    draws: PosteriorDraws,
    n_weeks: int,
    current: Plan,
    plan: Plan,
    bounds: Bounds,
    label: str,
    converged: bool,
    message: str,
    limits: dict[str, dict[str, Any]] | None = None,
    shrinkage: float = config.UPLIFT_SHRINKAGE,
    margin: float = config.DEFAULT_MARGIN,
) -> OptimizationResult:
    """Assemble the comparison of a recommended plan with the current one."""
    current_draws = plan_revenue_draws(draws, current, n_weeks).sum(axis=1)
    plan_draws = plan_revenue_draws(draws, plan, n_weeks).sum(axis=1)
    uplift = plan_draws - current_draws
    peak = draws.spend.max(axis=0)
    extrapolated = [
        channel
        for channel, highest in zip(draws.channels, peak, strict=True)
        if plan[channel] / n_weeks > highest * (1 + config.BOUND_TOLERANCE)
    ]
    corner, note = corner_check(plan, bounds)
    return OptimizationResult(
        objective=label,
        converged=converged,
        message=message,
        bounds=bounds,
        current=what_if(draws, current, n_weeks, "current"),
        recommended=what_if(draws, plan, n_weeks, "recommended"),
        uplift=summarize(uplift),
        uplift_pct=summarize(100 * uplift / current_draws),
        prob_recommended_beats_current=float((uplift > 0).mean()),
        extrapolated_channels=extrapolated,
        caveats=channel_caveats(draws),
        realistic_uplift=shrinkage * float(uplift.mean()),
        realistic_uplift_pct=shrinkage * float((100 * uplift / current_draws).mean()),
        limits=limits or {},
        corner_solution=corner,
        corner_note=note,
        shrinkage=shrinkage,
        margin=margin,
    )


def optimize_budget(
    allocator: Any,
    draws: PosteriorDraws,
    total_budget: float | None = None,
    minimum: Plan | None = None,
    maximum: Plan | None = None,
    max_change: float | None = None,
    objective: Objective = "mean",
    percentile: float = config.RISK_PERCENTILE,
    gate: bool = False,
    margin: float = config.DEFAULT_MARGIN,
    risk_lambda: float = config.RISK_LAMBDA,
    shrinkage: float = config.UPLIFT_SHRINKAGE,
    measured: frozenset[str] = frozenset(),
) -> OptimizationResult:
    """Find the best allocation of a budget and compare it with the current allocation.

    Candidate plans are scored across posterior draws (``solve_allocation``). With
    ``gate=True`` each channel's limit comes from ``channel_limits``: tighter where the
    evidence is weak, and never above the channel's highest week on record.

    Args:
        allocator: A ``BudgetAllocator`` or simply the number of weeks in the window.
        draws: Posterior draws from the fitted model.
        total_budget: Budget for the window; defaults to what was spent last quarter.
        minimum: Per-channel floors (period totals).
        maximum: Per-channel ceilings (period totals).
        max_change: Maximum relative change per channel versus last quarter.
        objective: ``mean``, ``profit``, ``risk_adjusted`` or ``percentile``.
        percentile: Percentile used by the ``percentile`` objective.
        gate: Apply uncertainty-aware limits.
        margin: Product margin, used by the ``profit`` objective.
        risk_lambda: Weight on the standard deviation in the ``risk_adjusted`` objective.
        shrinkage: Share of the expected uplift to report as the realistic uplift.
        measured: Channels calibrated with an experiment (see ``channel_limits``).

    """
    n_weeks = _window(allocator)
    current = last_quarter_spend(draws, n_weeks)
    budget = float(total_budget if total_budget is not None else sum(current.values()))
    change: float | dict[str, float] | None = max_change
    explained: dict[str, dict[str, Any]] = {}
    if gate and max_change is not None:
        explained = channel_limits(draws, n_weeks, max_change, measured)
        change = {channel: info["max_change"] for channel, info in explained.items()}
        ceilings = {
            channel: max(min(current[channel] * (1 + change[channel]), info["ceiling"]), 0.0)
            for channel, info in explained.items()
        }
        maximum = {**ceilings, **(maximum or {})}
    bounds = build_bounds(current, budget, minimum, maximum, change)
    plan, converged, message = solve_allocation(
        draws, budget, bounds, n_weeks, objective, margin, risk_lambda, percentile, [current]
    )

    def score(candidate: Plan) -> float:
        revenue = plan_revenue_draws(draws, candidate, n_weeks).sum(axis=1)
        total = sum(candidate.values())
        return objective_value(revenue, total, objective, margin, risk_lambda, percentile)

    same_budget = abs(budget - sum(current.values())) <= config.BOUND_TOLERANCE * max(budget, 1)
    if same_budget and score(plan) < score(current):
        # The solver works on a subset of draws; on the full posterior the current plan can
        # still come out ahead. Never recommend a plan the model itself rates below it.
        plan, message = dict(current), "No better plan found; keeping the current allocation."
    return build_result(
        draws,
        n_weeks,
        current,
        plan,
        bounds,
        OBJECTIVE_LABELS[objective],
        converged,
        message,
        explained,
        shrinkage,
        margin,
    )


def optimize_goal(
    draws: PosteriorDraws,
    n_weeks: int,
    revenue_target: float | None = None,
    roi_target: float | None = None,
    max_change: float | None = None,
) -> OptimizationResult:
    """Find the plan that meets a business goal other than "spend this budget".

    - ``revenue_target``: the cheapest plan whose expected incremental revenue reaches the
      target ("minimise spend to hit a revenue number").
    - ``roi_target``: the largest plan whose expected revenue per rupee stays at or above the
      target ("grow as far as a target ROI allows").

    Exactly one target must be given. Customer acquisition cost is not supported: the data
    has revenue, not customers.

    Raises:
        ValueError: If no plan within the limits can meet the target.

    """
    if (revenue_target is None) == (roi_target is None):
        raise ValueError("Give exactly one of revenue_target or roi_target.")
    current = last_quarter_spend(draws, n_weeks)
    names = list(current)
    values = np.array([current[name] for name in names])
    if max_change is None:
        low, high = np.zeros_like(values), draws.spend.max(axis=0) * n_weeks
    else:
        low, high = values * (1 - max_change), values * (1 + max_change)
    scale = float(values.sum())
    sample = thin(draws)

    def revenue(spend: FloatArray) -> float:
        plan = dict(zip(names, spend.tolist(), strict=True))
        return float(plan_revenue_draws(sample, plan, n_weeks).sum(axis=1).mean())

    if revenue_target is not None:
        if revenue(high) < revenue_target:
            raise ValueError(
                f"Even at the upper limits expected revenue is {revenue(high):,.0f}, below the "
                f"target of {revenue_target:,.0f}. Raise the limits or lower the target."
            )
        label = f"lowest spend for {revenue_target / config.INR_PER_CRORE:.2f} Cr of revenue"
        result = minimize(
            lambda x: x.sum(),
            high / scale,
            method="SLSQP",
            bounds=list(zip(low / scale, high / scale, strict=True)),
            constraints=[
                {"type": "ineq", "fun": lambda x: (revenue(x * scale) - revenue_target) / scale}
            ],
            options={"ftol": config.OPTIMIZER_FTOL, "maxiter": config.OPTIMIZER_MAX_ITERATIONS},
        )
    else:
        if revenue(low) < roi_target * low.sum() and low.sum() > 0:
            raise ValueError(
                f"Even at the lower limits the ROI is {revenue(low) / low.sum():.2f}, below the "
                f"target of {roi_target:.2f}. Widen the limits or lower the target."
            )
        label = f"largest plan with ROI of at least {roi_target:.2f}"
        result = minimize(
            lambda x: -revenue(x * scale) / scale,
            np.maximum(low, 1.0) / scale,
            method="SLSQP",
            bounds=list(zip(low / scale, high / scale, strict=True)),
            constraints=[
                {
                    "type": "ineq",
                    "fun": lambda x: (revenue(x * scale) - roi_target * x.sum() * scale) / scale,
                }
            ],
            options={"ftol": config.OPTIMIZER_FTOL, "maxiter": config.OPTIMIZER_MAX_ITERATIONS},
        )
    spend = np.clip(result.x * scale, low, high)
    plan = dict(zip(names, spend.tolist(), strict=True))
    bounds = {name: (float(a), float(b)) for name, a, b in zip(names, low, high, strict=True)}
    outcome = build_result(
        draws, n_weeks, current, plan, bounds, label, bool(result.success), str(result.message)
    )
    # The search targets the expected value; say how likely the goal is to be met at all.
    achieved = plan_revenue_draws(draws, plan, n_weeks).sum(axis=1)
    threshold = revenue_target if revenue_target is not None else roi_target * spend.sum()
    outcome.goal_probability = float((achieved >= threshold).mean())
    return outcome


def _score(revenue_draws: FloatArray, objective: Objective, percentile: float) -> float:
    """Return the value the optimizer is maximising for a plan's revenue draws."""
    return objective_value(revenue_draws, 0.0, objective, percentile=percentile)


def channel_caveats(draws: PosteriorDraws) -> dict[str, str]:
    """Return a one-line caveat for each channel the health checks cannot measure well."""
    flags = measurement_flags(draws.spend, draws.channels)
    return {channel: "; ".join(reasons) for channel, reasons in flags.items() if reasons}


def change_lines(result: OptimizationResult, name: Callable[[str], str] = str) -> list[str]:
    """Return one line per channel whose spend changes, with any caveat next to the change.

    Example: ``"TV +10% (ran in bursts, so its effect is tangled with the season)"``.
    """
    lines = []
    for channel, current in result.current.spend.items():
        if not current:
            continue
        change = 100 * (result.recommended.spend[channel] / current - 1)
        if abs(change) < 0.5:
            continue
        caveat = f" ({result.caveats[channel]})" if channel in result.caveats else ""
        lines.append(f"{name(channel)} {change:+.0f}%{caveat}")
    return lines


def measured_shrinkage(summaries: list[dict[str, Any]]) -> float | None:
    """Return true uplift as a share of expected uplift, averaged over synthetic brands.

    This is the optimizer's-curse correction: an optimizer moves money to wherever the model's
    estimate is highest, and the highest estimates are disproportionately overestimates, so
    the delivered uplift is systematically below the expected one. Only brands with a ground
    truth check and a positive expected uplift contribute. Returns ``None`` if there are none.
    """
    ratios = []
    for summary in summaries:
        check = summary["expected_revenue"].get("truth_check")
        if check and check["model_expected_uplift"] > 0:
            ratios.append(check["true_uplift"] / check["model_expected_uplift"])
    return float(np.mean(ratios)) if ratios else None


# --- Timing, rollout and optimism -----------------------------------------------------------


def weekly_plan(
    draws: PosteriorDraws,
    plan: Plan,
    n_weeks: int,
    burst_minimum: Plan | None = None,
    min_burst_weeks: int = config.MIN_BURST_WEEKS,
) -> pd.DataFrame:
    """Turn period totals into a week-by-week spend plan, respecting carryover and flighting.

    For each channel the total is either spread evenly or concentrated into one burst of
    ``k`` consecutive weeks; the option with the highest expected revenue (carryover included)
    wins. A channel listed in ``burst_minimum`` must spend at least that much in any week it is
    on, which rules out thin bursts ("TV at no less than 20 lakh a week").

    The model is additive, so a channel's effect does not depend on which calendar weeks it
    runs in; bursts are therefore placed at the start of the window, and festival timing
    should be decided by the planner, not read off this output.

    Returns a frame with one row per week and one column per channel.
    """
    sample = thin(draws)
    schedule = np.zeros((n_weeks, len(draws.channels)))
    for index, channel in enumerate(draws.channels):
        total = plan.get(channel, 0.0)
        if total <= 0:
            continue
        floor = (burst_minimum or {}).get(channel, 0.0)
        best_value, best_weeks = -np.inf, n_weeks
        for weeks in range(min(min_burst_weeks, n_weeks), n_weeks + 1):
            if total / weeks < floor:
                break
            matrix = np.zeros((n_weeks + draws.l_max, len(draws.channels)))
            matrix[:weeks, index] = total / weeks
            value = float(simulate_contributions(sample, matrix)[:, :, index].sum(axis=1).mean())
            if value > best_value + 1e-6:
                best_value, best_weeks = value, weeks
        schedule[:best_weeks, index] = total / best_weeks
    frame = pd.DataFrame(schedule, columns=draws.channels)
    frame.insert(0, "week", range(1, n_weeks + 1))
    return frame


def rollout_plan(
    draws: PosteriorDraws,
    result: OptimizationResult,
    steps: int = config.ROLLOUT_STEPS,
    weeks_per_step: int = config.ROLLOUT_WEEKS_PER_STEP,
) -> list[dict[str, Any]]:
    """Turn a recommendation into stepped changes with a checkpoint after each.

    Each step moves a further ``1 / steps`` of the way from the current plan to the
    recommended one. The checkpoint states what the model expects, how that compares with
    ordinary week-to-week noise, and the result that should stop the rollout.
    """
    n_weeks = result.current.n_weeks
    current, target = result.current.spend, result.recommended.spend
    base = plan_revenue_draws(draws, current, n_weeks).sum(axis=1)
    noise = float(np.median(draws.noise_sigma)) if draws.noise_sigma is not None else 0.0
    # Average weekly revenue over a step is known to within this much from noise alone.
    step_noise = 2 * noise / np.sqrt(weeks_per_step)
    plan_steps = []
    for step in range(1, steps + 1):
        share = step / steps
        spend = {c: current[c] + share * (target[c] - current[c]) for c in current}
        weekly = (plan_revenue_draws(draws, spend, n_weeks).sum(axis=1) - base) / n_weeks
        estimate = summarize(weekly)
        visible = abs(estimate.mean) > step_noise
        plan_steps.append(
            {
                "step": step,
                "weeks": f"{(step - 1) * weeks_per_step + 1} to {step * weeks_per_step}",
                "share_of_change": share,
                "spend": spend,
                "expected_weekly_revenue_change": estimate.model_dump(),
                "weekly_noise": noise,
                "visible_in_topline": bool(visible),
                "confirm": (
                    "Average weekly revenue over the step is at or above the forecast."
                    if visible
                    else "The expected change is smaller than normal week-to-week noise, so "
                    "revenue alone cannot confirm it. Confirm with a holdout test, or proceed "
                    "if nothing below triggers."
                ),
                "stop": (
                    f"Average weekly revenue over the step is more than {step_noise:,.0f} below "
                    "the forecast for the current plan."
                ),
            }
        )
    return plan_steps


def estimate_optimism(
    df: pd.DataFrame,
    settings: Any,
    draws: PosteriorDraws,
    n_weeks: int,
    max_change: float = config.DEFAULT_MAX_CHANGE,
    n_boot: int = config.OPTIMISM_BOOTSTRAPS,
    seed: int = config.RANDOM_SEED,
) -> dict[str, Any]:
    """Estimate how inflated the optimizer's expected uplift is, without using any truth.

    The optimizer's curse: a plan is chosen because the model rates it highly, and plans rated
    highly are disproportionately ones the model overrates. To measure the effect, each
    replicate treats one posterior draw as the real world, simulates revenue from it, refits
    the model on that simulated history, optimizes, and then scores the chosen plan under the
    world that generated the data. The ratio of uplift delivered to uplift expected, pooled
    over replicates, is the haircut to apply to this brand's expected uplift.
    """
    rng = np.random.default_rng(seed)
    quick = settings.model_copy(
        update={
            "sampler": settings.sampler.model_copy(
                update={"draws": config.OPTIMISM_DRAWS, "tune": config.OPTIMISM_DRAWS, "chains": 2}
            )
        }
    )
    mean = sum(draws.organic.values()) + draws.channel_contribution.sum(axis=2)
    pairs = []
    for replicate in rng.choice(mean.shape[0], size=n_boot, replace=False):
        noise = rng.normal(0.0, draws.noise_sigma[replicate], mean.shape[1])
        simulated = df.assign(**{config.TARGET_COL: np.maximum(mean[replicate] + noise, 1.0)})
        model = MixLabModel().build(simulated, quick)
        model.fit(progressbar=False)
        refit = extract_draws(model, simulated)
        result = optimize_budget(n_weeks, refit, max_change=max_change, gate=True)
        world = _single_draw(draws, int(replicate))
        delivered = float(
            plan_revenue_draws(world, result.recommended.spend, n_weeks).sum()
            - plan_revenue_draws(world, result.current.spend, n_weeks).sum()
        )
        pairs.append({"expected": result.uplift.mean, "delivered": delivered})
    expected = sum(pair["expected"] for pair in pairs)
    delivered = sum(pair["delivered"] for pair in pairs)
    ratio = delivered / expected if expected > 0 else 1.0
    # With few replicates the haircut is itself uncertain; resample them to show how much.
    resampled = []
    for _ in range(config.OPTIMISM_RESAMPLES):
        chosen = rng.integers(0, len(pairs), len(pairs))
        total = sum(pairs[i]["expected"] for i in chosen)
        if total > 0:
            resampled.append(sum(pairs[i]["delivered"] for i in chosen) / total)
    low, high = (
        (float(np.clip(q, 0.0, 1.0)) for q in np.percentile(resampled, [10, 90]))
        if resampled
        else (0.0, 1.0)
    )
    return {
        "shrinkage": float(np.clip(ratio, 0.0, 1.0)),
        "shrinkage_low": low,
        "shrinkage_high": high,
        "raw_ratio": float(ratio),
        "replicates": pairs,
        "method": "parametric bootstrap from the posterior, refit and re-optimize",
    }


def _single_draw(draws: PosteriorDraws, index: int) -> PosteriorDraws:
    """Return a copy of ``draws`` holding only one posterior draw (a possible real world)."""
    pick = slice(index, index + 1)

    def cut(values: FloatArray | None) -> FloatArray | None:
        return None if values is None else values[pick]

    return dataclasses.replace(
        draws,
        alpha=draws.alpha[pick],
        lam=draws.lam[pick],
        beta=draws.beta[pick],
        slope=cut(draws.slope),
        kappa=cut(draws.kappa),
        noise_sigma=cut(draws.noise_sigma),
        organic={name: values[pick] for name, values in draws.organic.items()},
        channel_contribution=draws.channel_contribution[pick],
    )


# --- Scenarios ------------------------------------------------------------------------------


def prebuilt_scenarios(current: Plan) -> dict[str, Plan]:
    """Return the standard what-if scenarios, each as a full spend plan."""
    change = config.SCENARIO_BUDGET_CHANGE
    source, target = config.SCENARIO_SHIFT_FROM, config.SCENARIO_SHIFT_TO
    moved = config.SCENARIO_SHIFT_SHARE * current[source]
    shifted = {**current, source: current[source] - moved, target: current[target] + moved}
    return {
        "current": dict(current),
        f"budget cut {change:.0%}": {c: v * (1 - change) for c, v in current.items()},
        f"budget increase {change:.0%}": {c: v * (1 + change) for c, v in current.items()},
        f"shift {config.SCENARIO_SHIFT_SHARE:.0%} of {source} to {target}": shifted,
        f"pause {config.SCENARIO_PAUSE_CHANNEL}": {**current, config.SCENARIO_PAUSE_CHANNEL: 0.0},
    }


def compare_scenarios(
    draws: PosteriorDraws,
    scenarios: dict[str, Plan],
    n_weeks: int = config.OPTIMIZER_WEEKS,
    baseline: str = "current",
) -> pd.DataFrame:
    """Return a side-by-side table of scenarios, with change measured against ``baseline``.

    The change in revenue is computed draw by draw, so its range reflects uncertainty about
    the difference between two plans, which is much narrower than either plan's own range.
    """
    base = plan_revenue_draws(draws, scenarios[baseline], n_weeks).sum(axis=1)
    base_spend = sum(scenarios[baseline].values())
    rows: dict[str, dict[str, float]] = {}
    for name, plan in scenarios.items():
        revenue = plan_revenue_draws(draws, plan, n_weeks).sum(axis=1)
        delta = summarize(revenue - base)
        estimate = summarize(revenue)
        spend = float(sum(plan.values()))
        rows[name] = {
            "spend": spend,
            "spend_change": spend - base_spend,
            "revenue_mean": estimate.mean,
            "revenue_hdi_low": estimate.hdi_low,
            "revenue_hdi_high": estimate.hdi_high,
            "revenue_change_mean": delta.mean,
            "revenue_change_hdi_low": delta.hdi_low,
            "revenue_change_hdi_high": delta.hdi_high,
            "net_change_mean": delta.mean - (spend - base_spend),
            "prob_revenue_up": float((revenue > base).mean()),
            "roi_mean": estimate.mean / spend if spend else 0.0,
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("scenario")


# --- Budget-level curve ---------------------------------------------------------------------


def budget_curve(allocator: Any, draws: PosteriorDraws, budgets: FloatArray) -> pd.DataFrame:
    """Return the best achievable incremental revenue at each total budget.

    Each budget is optimally allocated with no channel bounds. ``marginal_return`` is the
    extra revenue per extra rupee between consecutive budgets.
    """
    channels = draws.channels
    n_weeks = _window(allocator)
    rows = []
    plan: Plan | None = None
    for budget in np.sort(np.asarray(budgets, dtype=np.float64)):
        bounds = {channel: (0.0, float(budget)) for channel in channels}
        # Start each budget from the previous optimum, scaled up to the new total.
        scale = budget / sum(plan.values()) if plan else 0.0
        start = {channel: spend * scale for channel, spend in plan.items()} if plan else None
        plan, _, _ = solve_allocation(
            draws, float(budget), bounds, n_weeks, starts=[start] if start else None
        )
        estimate = summarize(plan_revenue_draws(draws, plan, n_weeks).sum(axis=1))
        rows.append(
            {
                "budget": float(budget),
                "revenue_mean": estimate.mean,
                "revenue_hdi_low": estimate.hdi_low,
                "revenue_hdi_high": estimate.hdi_high,
                **{f"spend_{channel}": plan[channel] for channel in channels},
            }
        )
    curve = pd.DataFrame(rows)
    curve["marginal_return"] = curve["revenue_mean"].diff() / curve["budget"].diff()
    return curve


def payback_budget(curve: pd.DataFrame) -> float | None:
    """Return the budget beyond which an extra rupee returns less than one rupee of revenue.

    Marginal return between two budgets is assigned to their midpoint and interpolated.
    Returns ``None`` if extra budget still pays back at the largest budget examined, and the
    smallest budget if it never pays back.
    """
    midpoints = ((curve["budget"] + curve["budget"].shift()) / 2).to_numpy()[1:]
    marginal = curve["marginal_return"].to_numpy()[1:]
    below = np.flatnonzero(marginal < config.MARGINAL_ROI_BREAKEVEN)
    if len(below) == 0:
        return None
    first = int(below[0])
    if first == 0:
        return float(curve["budget"].iloc[0])
    x0, x1 = midpoints[first - 1], midpoints[first]
    y0, y1 = marginal[first - 1], marginal[first]
    return float(x0 + (y0 - config.MARGINAL_ROI_BREAKEVEN) * (x1 - x0) / (y0 - y1))


# --- Ground-truth validation ----------------------------------------------------------------


def true_plan_revenue(brand: BrandConfig, spend: Plan, n_weeks: int) -> Plan:
    """Return the TRUE incremental revenue per channel of a plan, from the data generator.

    Uses the real adstock, saturation and effect size that produced the synthetic data,
    with the plan spread evenly over the window and carryover counted after it.
    """
    revenue: Plan = {}
    for channel in brand.channels:
        weekly = np.concatenate(
            [
                np.full(n_weeks, spend.get(channel.name, 0.0) / n_weeks),
                np.zeros(brand.adstock_l_max),
            ]
        )
        revenue[channel.name] = float(
            channel_contribution(weekly, channel, brand.adstock_l_max, brand.true_saturation).sum()
        )
    return revenue


def validate_against_truth(brand: BrandConfig, result: OptimizationResult) -> dict[str, Any]:
    """Score a recommendation with the true data-generating process.

    Returns the uplift the model expected and the uplift the recommendation would really
    have produced, so over-confident recommendations are visible.
    """
    n_weeks = result.current.n_weeks
    true_current = true_plan_revenue(brand, result.current.spend, n_weeks)
    true_recommended = true_plan_revenue(brand, result.recommended.spend, n_weeks)
    current_total, recommended_total = sum(true_current.values()), sum(true_recommended.values())
    return {
        "true_revenue_current": current_total,
        "true_revenue_recommended": recommended_total,
        "true_uplift": recommended_total - current_total,
        "true_uplift_pct": 100 * (recommended_total - current_total) / current_total,
        "model_expected_uplift": result.uplift.mean,
        "model_expected_uplift_pct": result.uplift_pct.mean,
        "model_uplift_hdi": [result.uplift.hdi_low, result.uplift.hdi_high],
        "true_uplift_inside_model_hdi": bool(
            result.uplift.hdi_low <= recommended_total - current_total <= result.uplift.hdi_high
        ),
        "true_revenue_by_channel": {"current": true_current, "recommended": true_recommended},
    }


# --- Figures and tables ---------------------------------------------------------------------


def _style(ax: Any, grid_axis: str = "y") -> None:
    """Apply the shared recessive chart styling to an axis."""
    ax.set_facecolor(config.COLOR_SURFACE)
    ax.grid(axis=grid_axis, color=config.COLOR_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=config.COLOR_TEXT_MUTED, length=0, labelsize=9)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)


def _crore(value: float, _position: int) -> str:
    """Format an INR amount as crore for axis ticks."""
    return f"{value / config.INR_PER_CRORE:,.10g}"


def plot_budget_curve(curve: pd.DataFrame, current_budget: float, payback: float | None) -> Figure:
    """Return the budget-level curve with the current budget and payback point marked."""
    fig = Figure(figsize=(10, 5.4), facecolor=config.COLOR_SURFACE)
    ax = fig.subplots()
    ax.fill_between(
        curve["budget"],
        curve["revenue_hdi_low"],
        curve["revenue_hdi_high"],
        color=config.COLOR_SPEND,
        alpha=0.18,
        linewidth=0,
        label="94% HDI",
    )
    ax.plot(
        curve["budget"],
        curve["revenue_mean"],
        color=config.COLOR_SPEND,
        linewidth=2,
        marker="o",
        markersize=5,
        label="Best achievable incremental revenue (mean)",
    )
    ax.plot(
        curve["budget"],
        curve["budget"],
        color=config.COLOR_TEXT_MUTED,
        linewidth=0.9,
        linestyle=(0, (1, 3)),
        label="Revenue = budget",
    )
    ax.axvline(current_budget, color=config.COLOR_TEXT, linewidth=1.2, label="Current budget")
    if payback is not None:
        ax.axvline(
            payback,
            color=config.COLOR_TEXT,
            linewidth=1.2,
            linestyle=(0, (4, 3)),
            label="Extra budget stops paying back",
        )
    ax.set_title(
        "Budget-level curve: best achievable incremental revenue at each total budget",
        loc="left",
        fontsize=13,
        color=config.COLOR_TEXT,
    )
    ax.set_xlabel("Total media budget for the period (INR crore)", color=config.COLOR_TEXT_MUTED)
    ax.set_ylabel("Incremental revenue (INR crore)", color=config.COLOR_TEXT_MUTED)
    ax.xaxis.set_major_formatter(FuncFormatter(_crore))
    ax.yaxis.set_major_formatter(FuncFormatter(_crore))
    ax.set_ylim(bottom=0)
    _style(ax, grid_axis="both")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def plot_allocation(result: OptimizationResult) -> Figure:
    """Return current versus recommended spend per channel as paired bars."""
    channels = list(result.current.spend)
    colors = channel_colors(channels)
    lakh = config.INR_PER_LAKH
    fig = Figure(figsize=(10, 0.75 * len(channels) + 1.9), facecolor=config.COLOR_SURFACE)
    ax = fig.subplots()
    for row, channel in enumerate(channels):
        current, recommended = result.current.spend[channel], result.recommended.spend[channel]
        ax.barh(row - 0.19, current, height=0.34, color=colors[channel], alpha=0.4)
        ax.barh(row + 0.19, recommended, height=0.34, color=colors[channel])
        change = 100 * (recommended / current - 1) if current else float("nan")
        ax.text(
            max(current, recommended),
            row,
            f"  {current / lakh:,.0f} to {recommended / lakh:,.0f} lakh ({change:+.0f}%)",
            va="center",
            fontsize=9,
            color=config.COLOR_TEXT_MUTED,
        )
    ax.set_yticks(range(len(channels)), channels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.5 * max(*result.current.spend.values(), *result.recommended.spend.values()))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v / lakh:,.0f}"))
    ax.set_xlabel(
        f"Spend over {result.current.n_weeks} weeks (INR lakh)",
        color=config.COLOR_TEXT_MUTED,
        fontsize=9,
    )
    ax.set_title(
        f"Recommended reallocation, maximising {result.objective}",
        loc="left",
        fontsize=13,
        color=config.COLOR_TEXT,
    )
    fig.text(
        0.012,
        0.012,
        "Pale bar: current (last quarter). Solid bar: recommended.",
        fontsize=8,
        color=config.COLOR_TEXT_MUTED,
    )
    _style(ax, grid_axis="x")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def allocation_table(result: OptimizationResult) -> pd.DataFrame:
    """Return current and recommended spend per channel in INR lakh, with the change."""
    lakh = config.INR_PER_LAKH
    table = pd.DataFrame(
        {
            "current (lakh)": pd.Series(result.current.spend) / lakh,
            "recommended (lakh)": pd.Series(result.recommended.spend) / lakh,
        }
    )
    table["change %"] = 100 * (table["recommended (lakh)"] / table["current (lakh)"] - 1)
    return table.round(1)


def describe(result: OptimizationResult) -> str:
    """Return a short plain-text summary of an optimization result."""
    crore = config.INR_PER_CRORE
    rec, cur = result.recommended.incremental_revenue, result.current.incremental_revenue
    return (
        f"Objective: {result.objective} | converged: {result.converged}\n"
        f"Budget {result.recommended.total_spend / crore:.2f} Cr over "
        f"{result.recommended.n_weeks} weeks\n"
        f"Incremental revenue: current {cur.mean / crore:.2f} Cr "
        f"[{cur.hdi_low / crore:.2f}, {cur.hdi_high / crore:.2f}] -> recommended "
        f"{rec.mean / crore:.2f} Cr [{rec.hdi_low / crore:.2f}, {rec.hdi_high / crore:.2f}]\n"
        f"Uplift: {result.uplift.mean / crore:+.2f} Cr "
        f"[{result.uplift.hdi_low / crore:+.2f}, {result.uplift.hdi_high / crore:+.2f}] "
        f"({result.uplift_pct.mean:+.1f}%), probability of beating current "
        f"{result.prob_recommended_beats_current:.0%}"
    )


def build_optimizer_summary(
    allocator: Any,
    draws: PosteriorDraws,
    brand: BrandConfig | None = None,
    max_change: float = config.DEFAULT_MAX_CHANGE,
    curve_points: int = config.BUDGET_CURVE_POINTS,
    optimism: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run every optimizer analysis and return one JSON-ready summary.

    Contains three recommendations (expected revenue and conservative, both with
    uncertainty-aware limits; and unconstrained), a rollout plan and week-by-week schedule for
    the main one, the pre-built scenarios, and the budget-level curve. ``optimism`` is the
    output of ``estimate_optimism``; its haircut replaces the default for "realistic uplift".
    If ``brand`` is given (synthetic data), each result is also scored against the truth.
    """
    n_weeks = _window(allocator)
    current = last_quarter_spend(draws, n_weeks)
    shrinkage = optimism["shrinkage"] if optimism else config.UPLIFT_SHRINKAGE
    runs = {
        "expected_revenue": optimize_budget(
            n_weeks, draws, max_change=max_change, gate=True, shrinkage=shrinkage
        ),
        "conservative": optimize_budget(
            n_weeks,
            draws,
            max_change=max_change,
            objective="percentile",
            gate=True,
            shrinkage=shrinkage,
        ),
        "unconstrained": optimize_budget(n_weeks, draws, shrinkage=shrinkage),
    }
    summary: dict[str, Any] = {"n_weeks": n_weeks, "max_change": max_change}
    if optimism:
        summary["optimism"] = optimism
    for name, result in runs.items():
        summary[name] = result.model_dump()
        if brand is not None:
            summary[name]["truth_check"] = validate_against_truth(brand, result)

    main_result = runs["expected_revenue"]
    summary["rollout"] = rollout_plan(draws, main_result)
    bursts = {
        channel: float(draws.spend[draws.spend[:, index] > 0, index].min())
        for index, channel in enumerate(draws.channels)
        if config.CAVEAT_BURSTS in main_result.caveats.get(channel, "")
    }
    summary["weekly_plan"] = weekly_plan(
        draws, main_result.recommended.spend, n_weeks, bursts
    ).to_dict(orient="records")
    summary["burst_minimum"] = bursts

    scenarios = prebuilt_scenarios(current)
    scenarios["recommended (same budget)"] = main_result.recommended.spend
    table = compare_scenarios(draws, scenarios, n_weeks)
    if brand is not None:
        true_base = sum(true_plan_revenue(brand, current, n_weeks).values())
        table["true_revenue_change"] = [
            sum(true_plan_revenue(brand, plan, n_weeks).values()) - true_base
            for plan in scenarios.values()
        ]
    summary["scenarios"] = table.reset_index().to_dict(orient="records")

    total = sum(current.values())
    budgets = total * np.linspace(*config.BUDGET_CURVE_MULTIPLES, curve_points)
    curve = budget_curve(n_weeks, draws, budgets)
    summary["budget_curve"] = curve.astype(object).where(curve.notna(), None).to_dict("records")
    summary["payback_budget"] = payback_budget(curve)
    summary["current_budget"] = total
    return summary


def main() -> None:
    """Optimize, simulate scenarios, trace the budget curve and validate against the truth."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, default=config.MODELS_DIR)
    parser.add_argument(
        "--data", type=Path, default=config.SYNTHETIC_DIR / config.WEEKLY_DATA_FILENAME
    )
    parser.add_argument("--weeks", type=int, default=config.OPTIMIZER_WEEKS)
    parser.add_argument("--max-change", type=float, default=config.DEFAULT_MAX_CHANGE)
    args = parser.parse_args()

    df = pd.read_csv(args.data, parse_dates=[config.DATE_COL])
    model = MixLabModel.load(args.model)
    draws = extract_draws(model, df)
    truth = json.loads((config.SYNTHETIC_DIR / config.GROUND_TRUTH_FILENAME).read_text())
    summary = build_optimizer_summary(
        BudgetAllocator(model, args.weeks),
        draws,
        config.BRAND_PRESETS[truth["brand"]],
        args.max_change,
    )

    crore = config.INR_PER_CRORE
    for name in ("expected_revenue", "conservative", "unconstrained"):
        result = OptimizationResult.model_validate(summary[name])
        check = summary[name]["truth_check"]
        print(f"\n== {name} ==\n{describe(result)}")
        print(allocation_table(result).to_string())
        print(
            f"TRUE uplift: {check['true_uplift'] / crore:+.2f} Cr "
            f"({check['true_uplift_pct']:+.1f}%) | model expected "
            f"{check['model_expected_uplift'] / crore:+.2f} Cr "
            f"({check['model_expected_uplift_pct']:+.1f}%)"
        )

    table = pd.DataFrame(summary["scenarios"]).set_index("scenario")
    money = [c for c in table.columns if c not in ("prob_revenue_up", "roi_mean")]
    table[money] = table[money] / crore
    print("\n== Scenarios (INR crore) ==")
    print(table.round(2).to_string())

    curve = pd.DataFrame(summary["budget_curve"]).astype(float)
    payback, total = summary["payback_budget"], summary["current_budget"]
    print("\n== Budget curve (INR crore) ==")
    print(
        (curve[["budget", "revenue_mean"]] / crore)
        .round(2)
        .join(curve["marginal_return"].round(2))
        .to_string(index=False)
    )
    print(
        f"Extra budget stops paying back at: "
        f"{'beyond the range examined' if payback is None else f'{payback / crore:.2f} Cr'}"
    )

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_budget_curve(curve, total, payback).savefig(
        config.FIGURES_DIR / config.BUDGET_CURVE_FIGURE, dpi=config.FIGURE_DPI
    )
    plot_allocation(OptimizationResult.model_validate(summary["expected_revenue"])).savefig(
        config.FIGURES_DIR / config.ALLOCATION_FIGURE, dpi=config.FIGURE_DPI
    )
    path = config.REPORTS_DIR / config.OPTIMIZER_SUMMARY_FILENAME
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
