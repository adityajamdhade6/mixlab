"""Regional insights, the regional budget optimizer, and the national-versus-geo comparison.

Everything here works on one ``PosteriorDraws`` per region (``geo_model.extract_geo_draws``).
Draws are aligned across regions, so national figures are regional figures summed draw by
draw, and every range is a proper posterior range, not a sum of separate ranges.
"""

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from mixlab import config
from mixlab.config import BrandConfig
from mixlab.geo_data import regional_brands
from mixlab.insights import PosteriorDraws, hdi, marginal_roi_draws, summarize
from mixlab.optimizer import (
    build_bounds,
    last_quarter_spend,
    plan_revenue_draws,
    solve_allocation,
    thin,
    true_plan_revenue,
)
from mixlab.transforms import FloatArray

GeoPlan = dict[str, dict[str, float]]


# --- ROI by region and nationally -----------------------------------------------------------


def roi_draws(draws: PosteriorDraws) -> FloatArray:
    """Return ROI draws (S, C) over the whole history: attributed revenue / spend."""
    spend = draws.spend.sum(axis=0)
    return draws.channel_contribution.sum(axis=1) / np.where(spend > 0, spend, np.nan)


def national_roi_draws(by_geo: dict[str, PosteriorDraws]) -> FloatArray:
    """Return national ROI draws (S, C): all regions' attributed revenue / all their spend."""
    revenue = sum(d.channel_contribution.sum(axis=1) for d in by_geo.values())
    spend = sum(d.spend.sum(axis=0) for d in by_geo.values())
    return revenue / spend


def blended_marginal_roi(marginal: FloatArray, draws: PosteriorDraws) -> FloatArray:
    """Return revenue per rupee (S) of one more rupee spread across channels like today's mix."""
    share = draws.spend.sum(axis=0) / draws.spend.sum()
    return (marginal * share).sum(axis=1)


def investment_status(region_marginal: float, national_marginal: float) -> str:
    """Classify a region by how its next rupee compares with the national next rupee."""
    ratio = config.GEO_INVESTMENT_RATIO
    if region_marginal > ratio * national_marginal:
        return "under-invested"
    if region_marginal < national_marginal / ratio:
        return "over-invested"
    return "about right"


def regional_metrics(
    by_geo: dict[str, PosteriorDraws], truth: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return per-region metrics: spend, ROI by channel, next-rupee return and a status.

    With ground truth (synthetic data only) each channel also carries its true ROI and
    whether the truth lies inside the 94% range.
    """
    marginals = {geo: marginal_roi_draws(d) for geo, d in by_geo.items()}
    weights = {geo: d.spend.sum() for geo, d in by_geo.items()}
    national_marginal = sum(
        blended_marginal_roi(marginals[geo], d) * weights[geo] for geo, d in by_geo.items()
    ) / sum(weights.values())
    national_median = float(np.median(national_marginal))
    regions: dict[str, Any] = {}
    for geo, draws in by_geo.items():
        roi = roi_draws(draws)
        spend = draws.spend.sum(axis=0)
        media = draws.channel_contribution.sum(axis=(1, 2))
        blended = blended_marginal_roi(marginals[geo], draws)
        region_truth = (truth or {}).get("regions", {}).get(geo, {})
        channels: dict[str, Any] = {}
        for index, channel in enumerate(draws.channels):
            estimate = summarize(roi[:, index])
            entry: dict[str, Any] = {
                "total_spend": float(spend[index]),
                "roi": estimate.model_dump(),
                "marginal_roi": summarize(marginals[geo][:, index]).model_dump(),
            }
            true_roi = region_truth.get("channels", {}).get(channel, {}).get("true_roi")
            if true_roi is not None:
                entry["true_roi"] = true_roi
                entry["truth_inside_range"] = bool(
                    estimate.hdi_low <= true_roi <= estimate.hdi_high
                )
            channels[channel] = entry
        regions[geo] = {
            "label": region_truth.get("label", geo),
            "lat": region_truth.get("lat"),
            "lon": region_truth.get("lon"),
            "revenue": float(draws.revenue.sum()),
            "media_spend": float(spend.sum()),
            "spend_share_pct": 100.0 * float(spend.sum()) / sum(weights.values()),
            "media_revenue": summarize(media).model_dump(),
            "blended_roi": summarize(media / spend.sum()).model_dump(),
            "blended_marginal_roi": summarize(blended).model_dump(),
            "status": investment_status(float(np.median(blended)), national_median),
            "channels": channels,
        }
    return {"national_marginal_roi": summarize(national_marginal).model_dump(), "regions": regions}


# --- National versus geo --------------------------------------------------------------------


def recovery_row(true_roi: float, draws: FloatArray) -> dict[str, Any]:
    """Return estimate, 94% range, width and whether the truth is inside, for one channel."""
    estimate = summarize(draws)
    return {
        "estimate": estimate.mean,
        "low": estimate.hdi_low,
        "high": estimate.hdi_high,
        "width": estimate.hdi_high - estimate.hdi_low,
        "truth_inside_range": bool(estimate.hdi_low <= true_roi <= estimate.hdi_high),
        "error": estimate.mean - true_roi,
    }


def compare_national_geo(
    by_geo: dict[str, PosteriorDraws], national: PosteriorDraws, truth: dict[str, Any]
) -> dict[str, Any]:
    """Compare national ROI from the national model and from the geo model, against truth.

    Range width ratio below 1 means the geo model's range is tighter. Regional recovery
    counts how many (region, channel) truths sit inside the geo model's regional ranges.
    """
    geo_roi = national_roi_draws(by_geo)
    national_roi = roi_draws(national)
    channels: dict[str, Any] = {}
    for index, channel in enumerate(national.channels):
        true_roi = truth["channels"][channel]["true_roi"]
        nat = recovery_row(true_roi, national_roi[:, index])
        geo = recovery_row(true_roi, geo_roi[:, index])
        channels[channel] = {
            "true_roi": true_roi,
            "national": nat,
            "geo": geo,
            "width_ratio": geo["width"] / nat["width"] if nat["width"] > 0 else None,
        }
    ratios = [c["width_ratio"] for c in channels.values() if c["width_ratio"] is not None]
    inside = [
        cell["truth_inside_range"]
        for region in regional_metrics(by_geo, truth)["regions"].values()
        for cell in region["channels"].values()
        if "truth_inside_range" in cell
    ]
    return {
        "channels": channels,
        "national_recovered": sum(c["national"]["truth_inside_range"] for c in channels.values()),
        "geo_recovered": sum(c["geo"]["truth_inside_range"] for c in channels.values()),
        "n_channels": len(channels),
        "median_width_ratio": float(np.median(ratios)) if ratios else None,
        "national_mean_abs_error": float(
            np.mean([abs(c["national"]["error"]) for c in channels.values()])
        ),
        "geo_mean_abs_error": float(np.mean([abs(c["geo"]["error"]) for c in channels.values()])),
        "regional_recovered": int(sum(inside)),
        "regional_cells": len(inside),
    }


def comparison_headline(comparison: dict[str, Any]) -> list[str]:
    """Return plain-English lines summarising the comparison."""
    n = comparison["n_channels"]
    lines = [
        f"National model: true ROI inside the 94% range for {comparison['national_recovered']} "
        f"of {n} channels. Geo model: {comparison['geo_recovered']} of {n}.",
    ]
    ratio = comparison["median_width_ratio"]
    if ratio is not None:
        lines.append(
            f"Geo ROI ranges are {100 * (1 - ratio):.0f}% narrower than the national model's "
            f"(median width ratio {ratio:.2f})."
        )
    lines.append(
        f"Regional ranges contain the truth for {comparison['regional_recovered']} of "
        f"{comparison['regional_cells']} region-channel pairs."
    )
    return lines


# --- Regional budget optimizer --------------------------------------------------------------


def current_geo_plan(
    by_geo: dict[str, PosteriorDraws], n_weeks: int = config.OPTIMIZER_WEEKS
) -> GeoPlan:
    """Return each region's spend per channel over the last ``n_weeks`` of history."""
    return {
        geo: dict(zip(d.channels, d.spend[-n_weeks:].sum(axis=0).tolist(), strict=True))
        for geo, d in by_geo.items()
    }


def geo_plan_revenue(by_geo: dict[str, PosteriorDraws], plan: GeoPlan, n_weeks: int) -> FloatArray:
    """Return total incremental revenue draws (S) of a regional plan, summed over regions."""
    return sum(plan_revenue_draws(d, plan[geo], n_weeks).sum(axis=1) for geo, d in by_geo.items())


def geo_bounds(
    current: GeoPlan, by_geo: dict[str, PosteriorDraws], max_change: float, n_weeks: int
) -> dict[tuple[str, str], tuple[float, float]]:
    """Return (low, high) per (region, channel): ``max_change`` either side of today.

    The upper bound never exceeds the cell's highest weekly spend on record times the window,
    so the plan stays where the regional curve was observed. Cells with no spend stay at zero.
    """
    bounds: dict[tuple[str, str], tuple[float, float]] = {}
    for geo, plan in current.items():
        peak = by_geo[geo].spend.max(axis=0) * n_weeks
        for index, (channel, spend) in enumerate(plan.items()):
            high = min(spend * (1 + max_change), max(float(peak[index]), spend))
            bounds[(geo, channel)] = (spend * (1 - max_change), high)
    return bounds


def solve_geo_allocation(
    by_geo: dict[str, PosteriorDraws],
    budget: float,
    bounds: dict[tuple[str, str], tuple[float, float]],
    n_weeks: int,
) -> tuple[GeoPlan, bool, str]:
    """Split ``budget`` across (region, channel) cells to maximise expected revenue.

    The same approach as ``optimizer.solve_allocation``, with one variable per cell. Draws
    are thinned identically in every region, so they stay aligned.
    """
    sample = {geo: thin(d) for geo, d in by_geo.items()}
    cells = list(bounds)
    low = np.array([bounds[c][0] for c in cells])
    high = np.array([bounds[c][1] for c in cells])

    def to_plan(values: FloatArray) -> GeoPlan:
        plan: GeoPlan = {geo: {} for geo in by_geo}
        for (geo, channel), value in zip(cells, values, strict=True):
            plan[geo][channel] = float(value)
        return plan

    room = high - low
    spread = (budget - low.sum()) / room.sum() if room.sum() > 0 else 0.0
    start = low + np.clip(spread, 0.0, 1.0) * room
    reference = abs(float(geo_plan_revenue(sample, to_plan(start), n_weeks).mean())) or 1.0

    def loss(shares: FloatArray) -> float:
        revenue = geo_plan_revenue(sample, to_plan(shares * budget), n_weeks)
        return -float(revenue.mean()) / reference

    result = minimize(
        loss,
        start / budget,
        method="SLSQP",
        bounds=list(zip(low / budget, high / budget, strict=True)),
        constraints=[{"type": "eq", "fun": lambda shares: shares.sum() - 1.0}],
        options={"ftol": config.OPTIMIZER_FTOL, "maxiter": config.OPTIMIZER_MAX_ITERATIONS},
    )
    shares = np.clip(result.x, low / budget, high / budget)
    return to_plan(shares * budget), bool(result.success), str(result.message)


def true_geo_revenue(brands: dict[str, BrandConfig], plan: GeoPlan, n_weeks: int) -> float:
    """Return the TRUE incremental revenue of a regional plan (synthetic data only)."""
    return sum(
        sum(true_plan_revenue(brands[geo], spend, n_weeks).values()) for geo, spend in plan.items()
    )


def spread_national_plan(channel_totals: dict[str, float], current: GeoPlan) -> GeoPlan:
    """Split national channel totals across regions in today's regional proportions.

    This is what a national model's recommendation becomes in practice: it says how much per
    channel, not where.
    """
    plan: GeoPlan = {}
    for geo, spend in current.items():
        plan[geo] = {}
        for channel, value in spend.items():
            national = sum(region[channel] for region in current.values())
            share = value / national if national > 0 else 0.0
            plan[geo][channel] = channel_totals.get(channel, national) * share
    return plan


def regional_optimizer(
    by_geo: dict[str, PosteriorDraws],
    truth: dict[str, Any] | None = None,
    max_change: float = config.DEFAULT_MAX_CHANGE,
    n_weeks: int = config.OPTIMIZER_WEEKS,
    national_plan: dict[str, float] | None = None,
    budget: float | None = None,
) -> dict[str, Any]:
    """Recommend a regional allocation at the same budget and compare it with today.

    With ground truth the recommendation is also scored on the real data-generating process,
    alongside a national plan (channel totals only) spread across regions as today.
    """
    current = current_geo_plan(by_geo, n_weeks)
    total = budget or sum(sum(p.values()) for p in current.values())
    bounds = geo_bounds(current, by_geo, max_change, n_weeks)
    plan, converged, message = solve_geo_allocation(by_geo, total, bounds, n_weeks)
    now = geo_plan_revenue(by_geo, current, n_weeks)
    after = geo_plan_revenue(by_geo, plan, n_weeks)
    uplift = after - now
    regions = {
        geo: {
            "current_spend": float(sum(current[geo].values())),
            "recommended_spend": float(sum(plan[geo].values())),
            "change_pct": 100.0
            * (sum(plan[geo].values()) / max(sum(current[geo].values()), 1.0) - 1.0),
            "channels": {
                channel: {"current": current[geo][channel], "recommended": plan[geo][channel]}
                for channel in current[geo]
            },
        }
        for geo in by_geo
    }
    result: dict[str, Any] = {
        "n_weeks": n_weeks,
        "budget": float(total),
        "max_change": max_change,
        "converged": converged,
        "message": message,
        "current_revenue": summarize(now).model_dump(),
        "recommended_revenue": summarize(after).model_dump(),
        "uplift": summarize(uplift).model_dump(),
        "uplift_pct": summarize(100 * uplift / now).model_dump(),
        "prob_beats_current": float((uplift > 0).mean()),
        "regions": regions,
    }
    if truth is not None:
        brand = config.BRAND_PRESETS[truth["national_brand"]]
        brand = dataclasses.replace(brand, seed=truth["seed"])
        # Rebuild the regions in the order they were generated (a subset keeps config order).
        regions = tuple(r for r in config.INDIA_REGIONS if r.name in truth["regions"])
        brands = {b.name: b for b in regional_brands(brand, regions)}
        true_now = true_geo_revenue(brands, current, n_weeks)
        check: dict[str, Any] = {
            "true_uplift": true_geo_revenue(brands, plan, n_weeks) - true_now,
            "true_current_revenue": true_now,
        }
        check["true_uplift_pct"] = 100 * check["true_uplift"] / true_now
        if national_plan is not None:
            spread = spread_national_plan(national_plan, current)
            check["national_plan_true_uplift"] = (
                true_geo_revenue(brands, spread, n_weeks) - true_now
            )
            check["national_plan_true_uplift_pct"] = (
                100 * check["national_plan_true_uplift"] / true_now
            )
        result["truth_check"] = check
    return result


# --- Compact draws for the app --------------------------------------------------------------


def save_geo_draws(by_geo: dict[str, PosteriorDraws], path: Path) -> Path:
    """Save the response parameters per region (small), enough to re-run the optimizer.

    Holds a thinned set of draws of carryover, curve shape and effect size, plus each
    region's spend history and scales. Weekly contributions are not stored.
    """
    geos = list(by_geo)
    sample = {geo: thin(d) for geo, d in by_geo.items()}
    first = sample[geos[0]]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        geos=np.array(geos),
        channels=np.array(first.channels),
        dates=first.dates.to_numpy().astype("datetime64[D]"),
        l_max=first.l_max,
        min_spend=first.min_spend,
        alpha=first.alpha.astype(np.float32),
        slope=first.slope.astype(np.float32),
        kappa=first.kappa.astype(np.float32),
        beta=np.stack([sample[g].beta for g in geos]).astype(np.float32),
        spend=np.stack([sample[g].spend for g in geos]),
        revenue=np.stack([sample[g].revenue for g in geos]),
        channel_scale=np.stack([sample[g].channel_scale for g in geos]),
        target_scale=np.array([sample[g].target_scale for g in geos]),
        total_contribution=np.stack(
            [sample[g].channel_contribution.sum(axis=1) for g in geos]
        ).astype(np.float32),
    )
    return path


def load_geo_draws(path: Path) -> dict[str, PosteriorDraws]:
    """Load draws saved by ``save_geo_draws``.

    ``channel_contribution`` holds one row per draw with the whole-history total (S, 1, C),
    so ROI over the history still works; weekly decompositions are not available.
    """
    data = np.load(path)
    dates = pd.DatetimeIndex(data["dates"])
    alpha = data["alpha"].astype(np.float64)
    result: dict[str, PosteriorDraws] = {}
    for index, geo in enumerate(data["geos"].tolist()):
        result[geo] = PosteriorDraws(
            dates=dates,
            channels=data["channels"].tolist(),
            spend=data["spend"][index],
            revenue=data["revenue"][index],
            channel_scale=data["channel_scale"][index],
            target_scale=float(data["target_scale"][index]),
            l_max=int(data["l_max"]),
            alpha=alpha,
            lam=np.zeros_like(alpha),
            beta=data["beta"][index].astype(np.float64),
            organic={},
            channel_contribution=data["total_contribution"][index][:, None, :].astype(np.float64),
            saturation="hill",
            slope=data["slope"].astype(np.float64),
            kappa=data["kappa"].astype(np.float64),
            min_spend=float(data["min_spend"]),
        )
    return result


# --- Assembly -------------------------------------------------------------------------------


def roi_hdi_table(by_geo: dict[str, PosteriorDraws]) -> dict[str, dict[str, list[float]]]:
    """Return the 94% ROI range per region and channel, for maps and tables."""
    table: dict[str, dict[str, list[float]]] = {}
    for geo, draws in by_geo.items():
        low, high = hdi(roi_draws(draws))
        table[geo] = {
            channel: [float(low[i]), float(high[i])] for i, channel in enumerate(draws.channels)
        }
    return table


def build_geo_outputs(
    by_geo: dict[str, PosteriorDraws],
    national: PosteriorDraws,
    truth: dict[str, Any] | None,
    diagnostics: dict[str, Any] | None = None,
    national_diagnostics: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the geo summary (regions, optimizer) and the national-versus-geo comparison."""
    current = last_quarter_spend(national, config.OPTIMIZER_WEEKS)
    budget = sum(current.values())
    bounds = build_bounds(current, budget, max_change=config.DEFAULT_MAX_CHANGE)
    national_plan, _, _ = solve_allocation(national, budget, bounds, config.OPTIMIZER_WEEKS)
    metrics = regional_metrics(by_geo, truth)
    summary = {
        "currency": config.CURRENCY,
        "hdi_prob": config.HDI_PROB,
        "n_regions": len(by_geo),
        "channels": national.channels,
        "national_roi": {
            channel: summarize(values).model_dump()
            for channel, values in zip(national.channels, national_roi_draws(by_geo).T, strict=True)
        },
        **metrics,
        "optimizer": regional_optimizer(by_geo, truth, national_plan=national_plan),
        "national_plan": national_plan,
        "diagnostics": diagnostics,
    }
    comparison: dict[str, Any] = {
        "diagnostics": {"geo": diagnostics, "national": national_diagnostics}
    }
    if truth is not None:
        comparison.update(compare_national_geo(by_geo, national, truth))
        comparison["headline"] = comparison_headline(comparison)
    else:
        comparison["headline"] = ["No ground truth: comparison against truth is not available."]
    return summary, comparison
