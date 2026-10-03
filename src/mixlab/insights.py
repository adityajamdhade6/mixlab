"""Channel contributions, ROI, and response curves derived from the fitted posterior.

Every metric is computed once per posterior draw and reported as a distribution (mean,
median and 94% highest-density interval), never as a single number.

WHY MARGINAL ROI MATTERS MORE THAN AVERAGE ROI
Average ROI answers "was the money we already spent worth it?". Marginal ROI answers "what
does the NEXT rupee earn?", which is the only question a budget decision can act on. Because
of diminishing returns the two differ: a channel can have a healthy average ROI of 2 and a
marginal ROI of 0.5, meaning the early rupees worked hard but the channel is now saturated
and extra budget loses money. Moving budget from low-marginal to high-marginal channels
raises revenue without spending more, even when the average ROIs suggest the opposite.

Run ``python -m mixlab.insights`` to write the summary JSON and all figures.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from pydantic import BaseModel

from mixlab import config
from mixlab.model import MixLabModel
from mixlab.transforms import FloatArray, logistic_saturation
from mixlab.validate import channel_name

# --- Posterior summaries --------------------------------------------------------------------


class Estimate(BaseModel):
    """A posterior distribution summarised as mean, median and highest-density interval."""

    mean: float
    median: float
    hdi_low: float
    hdi_high: float


def hdi(draws: FloatArray, prob: float = config.HDI_PROB) -> tuple[FloatArray, FloatArray]:
    """Return the narrowest interval holding ``prob`` of the draws, along the first axis."""
    ordered = np.sort(np.asarray(draws, dtype=np.float64), axis=0)
    n = ordered.shape[0]
    window = int(np.ceil(prob * n))
    widths = ordered[window - 1 :] - ordered[: n - window + 1]
    start = np.expand_dims(np.argmin(widths, axis=0), 0)
    low = np.take_along_axis(ordered, start, axis=0)[0]
    high = np.take_along_axis(ordered, start + window - 1, axis=0)[0]
    return low, high


def summarize(draws: FloatArray) -> Estimate:
    """Summarise one-dimensional posterior draws."""
    low, high = hdi(draws)
    return Estimate(
        mean=float(np.mean(draws)),
        median=float(np.median(draws)),
        hdi_low=float(low),
        hdi_high=float(high),
    )


# --- Posterior container --------------------------------------------------------------------


@dataclass(frozen=True)
class PosteriorDraws:
    """Everything the insights need, with posterior draws flattened to one ``sample`` axis.

    Shapes: S draws, T weeks, C channels. Money is in INR.

    Attributes:
        dates: Week start dates (T).
        channels: Channel names without the ``spend_`` prefix (C).
        spend: Weekly spend (T, C).
        revenue: Observed weekly revenue (T).
        channel_scale: Peak weekly spend per channel used by the model for scaling (C).
        target_scale: Peak weekly revenue used by the model for scaling.
        l_max: Adstock window in weeks.
        alpha: Adstock decay draws (S, C).
        lam: Saturation speed draws (S, C).
        beta: Effect size draws in scaled units (S, C).
        organic: Non-media component draws, name -> (S, T).
        channel_contribution: Channel contribution draws (S, T, C).

    """

    dates: pd.DatetimeIndex
    channels: list[str]
    spend: FloatArray
    revenue: FloatArray
    channel_scale: FloatArray
    target_scale: float
    l_max: int
    alpha: FloatArray
    lam: FloatArray
    beta: FloatArray
    organic: dict[str, FloatArray]
    channel_contribution: FloatArray


def control_group(control: str) -> str:
    """Map a control column to its decomposition component."""
    if control.startswith(config.HOLIDAY_PREFIX):
        return "holidays"
    return {config.PROMO_COL: "promos", config.PRICE_COL: "price"}.get(control, "other_controls")


def extract_draws(model: MixLabModel, df: pd.DataFrame) -> PosteriorDraws:
    """Pull parameter and component draws out of a fitted model, converted to INR."""
    idata = model.idata
    post = idata.posterior.stack(sample=("chain", "draw"))
    scale = model.target_scale
    n_weeks = post.sizes[config.DATE_COL]

    def by_channel(name: str) -> FloatArray:
        return post[name].transpose("sample", "channel").to_numpy()

    organic: dict[str, FloatArray] = {
        "baseline": np.repeat(post["intercept_contribution"].to_numpy()[:, None], n_weeks, axis=1),
        "trend": post[f"{config.TREND_PREFIX}_effect_contribution"]
        .transpose("sample", config.DATE_COL)
        .to_numpy(),
        "seasonality": post["yearly_seasonality_contribution"]
        .transpose("sample", config.DATE_COL)
        .to_numpy(),
    }
    controls = post["control_contribution"].transpose("sample", config.DATE_COL, "control")
    for index, control in enumerate(controls["control"].to_numpy()):
        group = control_group(str(control))
        organic[group] = organic.get(group, 0.0) + controls.to_numpy()[:, :, index]

    return PosteriorDraws(
        dates=pd.DatetimeIndex(post[config.DATE_COL].to_numpy()),
        channels=[channel_name(c) for c in model.channels],
        spend=df[model.channels].to_numpy(dtype=np.float64),
        revenue=df[config.TARGET_COL].to_numpy(dtype=np.float64),
        channel_scale=np.ravel(idata.constant_data["channel_scale"].to_numpy()).astype(np.float64),
        target_scale=scale,
        l_max=model.settings.adstock_l_max,
        alpha=by_channel("adstock_alpha"),
        lam=by_channel("saturation_lam"),
        beta=by_channel("saturation_beta"),
        organic={name: values * scale for name, values in organic.items()},
        channel_contribution=post["channel_contribution"]
        .transpose("sample", config.DATE_COL, "channel")
        .to_numpy()
        * scale,
    )


# --- Channel response -----------------------------------------------------------------------


def simulate_contributions(draws: PosteriorDraws, spend: FloatArray) -> FloatArray:
    """Return channel revenue (S, T, C) in INR for any spend plan, for every posterior draw.

    Applies exactly what the fitted model does: scale spend by the channel's peak week,
    normalised geometric adstock over ``l_max`` weeks, logistic saturation, times effect size.
    """
    scaled = np.asarray(spend, dtype=np.float64) / draws.channel_scale
    n_weeks = scaled.shape[0]
    weights = draws.alpha[:, None, :] ** np.arange(draws.l_max)[None, :, None]
    weights = weights / weights.sum(axis=1, keepdims=True)
    adstocked = np.zeros((draws.alpha.shape[0], n_weeks, scaled.shape[1]))
    for lag in range(min(draws.l_max, n_weeks)):
        adstocked[:, lag:, :] += weights[:, lag, None, :] * scaled[None, : n_weeks - lag, :]
    saturated = logistic_saturation(adstocked, draws.lam[:, None, :])
    return draws.beta[:, None, :] * saturated * draws.target_scale


def marginal_roi_draws(draws: PosteriorDraws) -> FloatArray:
    """Return, per draw and channel (S, C), the revenue earned per rupee of the NEXT 1 lakh.

    The extra 1 lakh is spread over the weeks the channel already runs, in proportion to its
    current spend, and the channel response is re-simulated.
    """
    base = simulate_contributions(draws, draws.spend).sum(axis=1)
    totals = draws.spend.sum(axis=0)
    result = np.zeros_like(base)
    for index in range(len(draws.channels)):
        bumped = draws.spend.copy()
        bumped[:, index] *= 1.0 + config.MARGINAL_STEP_INR / totals[index]
        extra = simulate_contributions(draws, bumped).sum(axis=1)[:, index] - base[:, index]
        result[:, index] = extra / config.MARGINAL_STEP_INR
    return result


def response_curve_draws(draws: PosteriorDraws, index: int, weekly_spend: FloatArray) -> FloatArray:
    """Return weekly incremental revenue (S, G) if the channel spent a steady amount each week."""
    scaled = np.asarray(weekly_spend, dtype=np.float64)[None, :] / draws.channel_scale[index]
    saturated = logistic_saturation(scaled, draws.lam[:, index, None])
    return draws.beta[:, index, None] * saturated * draws.target_scale


def saturation_spend_draws(draws: PosteriorDraws) -> FloatArray:
    """Return the steady weekly spend (S, C) beyond which the next rupee earns under 1 rupee.

    With response ``B * (1 - e^{-ks}) / (1 + e^{-ks})`` the slope is 1 where
    ``u^2 + (2 - 2Bk)u + 1 = 0`` for ``u = e^{-ks}``. If even the first rupee returns less
    than 1 (``Bk < 2``), the saturation point is zero.
    """
    k = draws.lam / draws.channel_scale
    slope_at_zero_x2 = draws.beta * draws.target_scale * k / config.MARGINAL_ROI_BREAKEVEN
    reachable = slope_at_zero_x2 >= 2.0
    a = np.where(reachable, slope_at_zero_x2 - 1.0, 1.0)
    u = a - np.sqrt(np.maximum(a**2 - 1.0, 0.0))
    return np.where(reachable, -np.log(u) / k, 0.0)


def carryover_weeks_draws(draws: PosteriorDraws) -> FloatArray:
    """Return weeks (S, C), counting the week of spend, until 90% of the effect has landed."""
    alpha = np.clip(draws.alpha, 1e-12, 1 - 1e-12)
    remaining = 1.0 - config.CARRYOVER_SHARE * (1.0 - alpha**draws.l_max)
    return np.maximum(np.log(remaining) / np.log(alpha), 1.0)


def naive_last_click_revenue(spend: FloatArray, revenue: FloatArray) -> FloatArray:
    """Return revenue (C) credited to each channel by a naive same-week attribution.

    A stand-in for platform or last-click reporting: all of each week's revenue is credited
    to the channels running that week, in proportion to spend. No baseline, no carryover.
    """
    weekly_total = spend.sum(axis=1, keepdims=True)
    share = np.divide(spend, weekly_total, out=np.zeros_like(spend), where=weekly_total > 0)
    return (share * revenue[:, None]).sum(axis=0)


# --- Metrics --------------------------------------------------------------------------------


def component_draws(draws: PosteriorDraws) -> dict[str, FloatArray]:
    """Return every revenue component as (S, T) draws: organic components, then channels."""
    components = {n: draws.organic[n] for n in config.ORGANIC_COMPONENTS if n in draws.organic}
    for index, channel in enumerate(draws.channels):
        components[channel] = draws.channel_contribution[:, :, index]
    return components


def decomposition_weekly(draws: PosteriorDraws) -> pd.DataFrame:
    """Return posterior-mean weekly revenue by component, with observed revenue alongside."""
    frame = pd.DataFrame(
        {name: values.mean(axis=0) for name, values in component_draws(draws).items()},
        index=draws.dates,
    )
    frame["observed_revenue"] = draws.revenue
    return frame.rename_axis(config.DATE_COL)


def decomposition_totals(draws: PosteriorDraws) -> dict[str, dict[str, Any]]:
    """Return each component's total revenue and share of observed revenue, as estimates."""
    total_revenue = draws.revenue.sum()
    return {
        name: {
            "revenue": summarize(values.sum(axis=1)).model_dump(),
            "pct_of_revenue": summarize(100 * values.sum(axis=1) / total_revenue).model_dump(),
        }
        for name, values in component_draws(draws).items()
    }


def channel_metrics(draws: PosteriorDraws) -> dict[str, dict[str, Any]]:
    """Return the per-channel metrics table; every modelled metric is an ``Estimate``."""
    spend = draws.spend.sum(axis=0)
    attributed = draws.channel_contribution.sum(axis=1)
    roi = attributed / spend
    marginal = marginal_roi_draws(draws)
    saturation = saturation_spend_draws(draws)
    carryover = carryover_weeks_draws(draws)
    naive_revenue = naive_last_click_revenue(draws.spend, draws.revenue)
    naive_roi = naive_revenue / spend
    media_share = 100 * attributed / attributed.sum(axis=1, keepdims=True)
    naive_share = 100 * naive_revenue / naive_revenue.sum()

    table: dict[str, dict[str, Any]] = {}
    for i, channel in enumerate(draws.channels):
        active = draws.spend[:, i] > 0
        current = float(draws.spend[active, i].mean())
        table[channel] = {
            "total_spend": float(spend[i]),
            "active_weeks": int(active.sum()),
            "current_weekly_spend": current,
            "attributed_revenue": summarize(attributed[:, i]).model_dump(),
            "contribution_pct": summarize(
                100 * attributed[:, i] / draws.revenue.sum()
            ).model_dump(),
            "roi": summarize(roi[:, i]).model_dump(),
            "roas": summarize(roi[:, i]).model_dump(),
            "marginal_roi": summarize(marginal[:, i]).model_dump(),
            "prob_marginal_roi_above_1": float(
                (marginal[:, i] > config.MARGINAL_ROI_BREAKEVEN).mean()
            ),
            "cost_per_incremental_revenue": summarize(spend[i] / attributed[:, i]).model_dump(),
            "saturation_weekly_spend": summarize(saturation[:, i]).model_dump(),
            "prob_first_rupee_below_breakeven": float((saturation[:, i] == 0).mean()),
            "adstock_decay": summarize(draws.alpha[:, i]).model_dump(),
            "weeks_to_90pct_effect": summarize(carryover[:, i]).model_dump(),
            "last_click": {
                "naive_roi": float(naive_roi[i]),
                "naive_over_mmm_ratio": summarize(naive_roi[i] / roi[:, i]).model_dump(),
                "naive_share_of_media_credit_pct": float(naive_share[i]),
                "mmm_share_of_media_credit_pct": summarize(media_share[:, i]).model_dump(),
            },
        }
    return table


DEFINITIONS: dict[str, str] = {
    "estimate": "Every modelled number is a posterior distribution: mean, median and the "
    "94% highest-density interval (the narrowest range holding 94% of plausible values).",
    "roi": "Attributed revenue divided by spend. Revenue-based, not profit-based: no product "
    "margin is applied, so break-even on profit needs an ROI well above 1.",
    "roas": "Identical to ROI here, because ROI is defined as revenue / spend.",
    "marginal_roi": "Revenue earned per rupee of the NEXT 1 lakh, spread over the weeks the "
    "channel already runs. This, not average ROI, should drive budget moves: average ROI "
    "describes money already spent, marginal ROI describes the money you are about to move.",
    "cost_per_incremental_revenue": "Rupees spent per rupee of revenue the channel caused "
    "(the inverse of ROI). Use the median; the mean is unstable when ROI can be near zero.",
    "saturation_weekly_spend": "Steady weekly spend beyond which the next rupee returns less "
    "than 1 rupee of revenue. Zero means even the first rupee is below break-even.",
    "weeks_to_90pct_effect": "Weeks, counting the week of spend, until 90% of the effect of "
    "that spend has been realised.",
    "last_click": "A naive same-week attribution used as a stand-in for platform reporting: "
    "each week's revenue is split across channels in proportion to that week's spend, with "
    "no baseline and no carryover. No click data exists in this dataset.",
}


def build_summary(draws: PosteriorDraws) -> dict[str, Any]:
    """Return the full insights summary used by the AI explainer and the dashboard."""
    decomposition = decomposition_totals(draws)
    media = draws.channel_contribution.sum(axis=(1, 2))
    explained = sum(v.sum(axis=1) for v in component_draws(draws).values())
    return {
        "currency": config.CURRENCY,
        "hdi_prob": config.HDI_PROB,
        "n_posterior_draws": int(draws.alpha.shape[0]),
        "period": {
            "start": str(draws.dates[0].date()),
            "end": str(draws.dates[-1].date()),
            "n_weeks": len(draws.dates),
        },
        "totals": {
            "revenue": float(draws.revenue.sum()),
            "media_spend": float(draws.spend.sum()),
            "media_revenue": summarize(media).model_dump(),
            "media_pct_of_revenue": summarize(100 * media / draws.revenue.sum()).model_dump(),
            "blended_media_roi": summarize(media / draws.spend.sum()).model_dump(),
            "unexplained_revenue": summarize(draws.revenue.sum() - explained).model_dump(),
        },
        "decomposition": decomposition,
        "channels": channel_metrics(draws),
        "definitions": DEFINITIONS,
    }


def metrics_table(summary: dict[str, Any]) -> pd.DataFrame:
    """Return a readable per-channel table (means with 94% HDI) from the summary."""

    def fmt(estimate: dict[str, float], scale: float = 1.0, digits: int = 2) -> str:
        return (
            f"{estimate['mean'] / scale:.{digits}f} "
            f"[{estimate['hdi_low'] / scale:.{digits}f}, {estimate['hdi_high'] / scale:.{digits}f}]"
        )

    crore = config.INR_PER_CRORE
    rows = {
        channel: {
            "spend (Cr)": f"{m['total_spend'] / crore:.2f}",
            "revenue (Cr)": fmt(m["attributed_revenue"], crore),
            "contribution %": fmt(m["contribution_pct"], digits=1),
            "ROI / ROAS": fmt(m["roi"]),
            "marginal ROI": fmt(m["marginal_roi"]),
            "cost per Rs 1 (median)": f"{m['cost_per_incremental_revenue']['median']:.2f}",
            "weeks to 90%": fmt(m["weeks_to_90pct_effect"], digits=1),
            "last-click ROI": f"{m['last_click']['naive_roi']:.2f}",
        }
        for channel, m in summary["channels"].items()
    }
    return pd.DataFrame.from_dict(rows, orient="index")


def export_summary(summary: dict[str, Any], directory: Path = config.REPORTS_DIR) -> Path:
    """Write the summary JSON and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / config.INSIGHTS_SUMMARY_FILENAME
    path.write_text(json.dumps(summary, indent=2) + "\n")
    return path


# --- Figures --------------------------------------------------------------------------------


def channel_colors(channels: list[str]) -> dict[str, str]:
    """Return a fixed colour per channel, assigned in channel order."""
    return dict(zip(channels, config.CHANNEL_COLORS, strict=False))


def _style(ax: Any, grid_axis: str = "y") -> None:
    """Apply the shared recessive chart styling to an axis."""
    ax.set_facecolor(config.COLOR_SURFACE)
    ax.grid(axis=grid_axis, color=config.COLOR_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=config.COLOR_TEXT_MUTED, length=0, labelsize=9)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)


def _money_formatter(unit: float) -> FuncFormatter:
    """Return a tick formatter that shows INR in the given unit (lakh or crore)."""
    return FuncFormatter(lambda value, _pos: f"{value / unit:,.10g}")


def plot_decomposition(weekly: pd.DataFrame, channels: list[str]) -> Figure:
    """Return a stacked area of weekly revenue by component (posterior means).

    Components that are negative in a week (price, seasonality troughs) stack below zero.
    """
    components = [c for c in weekly.columns if c != "observed_revenue"]
    colors = {**config.ORGANIC_COLORS, **channel_colors(channels)}
    fig = Figure(figsize=(11, 5.8), facecolor=config.COLOR_SURFACE)
    ax = fig.subplots()
    top = np.zeros(len(weekly))
    bottom = np.zeros(len(weekly))
    for name in components:
        values = weekly[name].to_numpy()
        positive, negative = np.clip(values, 0, None), np.clip(values, None, 0)
        ax.fill_between(
            weekly.index,
            top,
            top + positive,
            color=colors[name],
            edgecolor=config.COLOR_SURFACE,
            linewidth=0.6,
            label=name,
        )
        if negative.any():
            ax.fill_between(
                weekly.index, bottom + negative, bottom, color=colors[name], linewidth=0
            )
        top, bottom = top + positive, bottom + negative
    ax.plot(
        weekly.index,
        weekly["observed_revenue"],
        color=config.COLOR_TEXT,
        linewidth=1.4,
        label="observed revenue",
    )
    ax.axhline(0, color=config.COLOR_TEXT_MUTED, linewidth=0.8)
    ax.set_title(
        "Where weekly revenue comes from (posterior mean)",
        loc="left",
        fontsize=13,
        color=config.COLOR_TEXT,
    )
    ax.set_ylabel("INR lakh per week", color=config.COLOR_TEXT_MUTED)
    ax.yaxis.set_major_formatter(_money_formatter(config.INR_PER_LAKH))
    _style(ax)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[::-1],
        labels[::-1],
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
        title="Top to bottom",
        alignment="left",
    )
    fig.tight_layout()
    return fig


def plot_waterfall(summary: dict[str, Any]) -> Figure:
    """Return a waterfall from baseline through each driver to total modelled revenue.

    Channel rows carry their spend, so the chart reads from media spend to revenue.
    """
    crore = config.INR_PER_CRORE
    channels = list(summary["channels"])
    colors = {**config.ORGANIC_COLORS, **channel_colors(channels)}
    names = list(summary["decomposition"])
    values = [summary["decomposition"][n]["revenue"]["mean"] for n in names]
    starts = np.concatenate([[0.0], np.cumsum(values)[:-1]])
    total = float(np.sum(values))

    fig = Figure(figsize=(11, 0.48 * (len(names) + 1) + 1.6), facecolor=config.COLOR_SURFACE)
    ax = fig.subplots()
    rows = np.arange(len(names) + 1)
    for row, (name, value, start) in enumerate(zip(names, values, starts, strict=True)):
        ax.barh(row, value, left=start, color=colors[name], height=0.62)
        label = f"{value / crore:+,.1f} Cr"
        if name in summary["channels"]:
            m = summary["channels"][name]
            revenue = m["attributed_revenue"]
            ax.plot(
                [start + revenue["hdi_low"], start + revenue["hdi_high"]],
                [row, row],
                color=config.COLOR_TEXT,
                linewidth=1.2,
            )
            label = (
                f"{value / crore:+,.1f} Cr from {m['total_spend'] / crore:,.1f} Cr spend "
                f"(ROI {m['roi']['mean']:.1f})"
            )
            anchor = start + max(revenue["hdi_high"], value)
        else:
            anchor = start + max(value, 0.0)
        ax.text(
            anchor + 0.01 * total,
            row,
            label,
            va="center",
            fontsize=9,
            color=config.COLOR_TEXT_MUTED,
        )
    ax.barh(rows[-1], total, color=config.COLOR_TEXT, height=0.62)
    ax.text(
        total * 1.01,
        rows[-1],
        f"{total / crore:,.1f} Cr",
        va="center",
        fontsize=9,
        color=config.COLOR_TEXT_MUTED,
    )
    ax.set_yticks(rows, [*names, "modelled revenue"])
    ax.invert_yaxis()
    ax.set_xlim(0, total * 1.45)
    ax.xaxis.set_major_formatter(_money_formatter(crore))
    ax.set_xlabel("INR crore", color=config.COLOR_TEXT_MUTED, fontsize=9)
    ax.set_title(
        f"From {summary['totals']['media_spend'] / crore:,.0f} Cr of media spend to "
        f"{summary['totals']['revenue'] / crore:,.0f} Cr of revenue",
        loc="left",
        fontsize=13,
        color=config.COLOR_TEXT,
    )
    fig.text(
        0.012,
        0.012,
        "Bars are posterior means. Thin lines on channel bars show the 94% HDI.",
        fontsize=8,
        color=config.COLOR_TEXT_MUTED,
    )
    _style(ax, grid_axis="x")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def plot_response_curves(draws: PosteriorDraws, summary: dict[str, Any]) -> Figure:
    """Return one response curve per channel with band, current spend and saturation point."""
    lakh = config.INR_PER_LAKH
    colors = channel_colors(draws.channels)
    n_cols = 3
    n_rows = int(np.ceil(len(draws.channels) / n_cols))
    fig = Figure(figsize=(12, 3.7 * n_rows + 0.6), facecolor=config.COLOR_SURFACE)
    axes = np.atleast_1d(fig.subplots(n_rows, n_cols)).ravel()
    for index, (ax, channel) in enumerate(zip(axes, draws.channels, strict=False)):
        m = summary["channels"][channel]
        grid = np.linspace(
            0,
            config.RESPONSE_CURVE_MAX_MULTIPLE * draws.spend[:, index].max(),
            config.RESPONSE_CURVE_POINTS,
        )
        curves = response_curve_draws(draws, index, grid)
        low, high = hdi(curves)
        ax.fill_between(
            grid / lakh, low / lakh, high / lakh, color=colors[channel], alpha=0.18, linewidth=0
        )
        ax.plot(grid / lakh, curves.mean(axis=0) / lakh, color=colors[channel], linewidth=2)
        ax.plot(
            grid / lakh,
            grid / lakh,
            color=config.COLOR_TEXT_MUTED,
            linewidth=0.8,
            linestyle=(0, (1, 3)),
        )
        current = m["current_weekly_spend"]
        ax.axvline(current / lakh, color=config.COLOR_TEXT, linewidth=1.2)
        saturation = m["saturation_weekly_spend"]["median"]
        note = f"current {current / lakh:,.1f}"
        if saturation > 0:
            ax.axvline(
                saturation / lakh, color=config.COLOR_TEXT, linewidth=1.2, linestyle=(0, (4, 3))
            )
            note += f"  |  saturation point {saturation / lakh:,.1f}"
        else:
            note += "  |  below break-even from the first rupee"
        ax.set_title(f"{channel}\n{note}", loc="left", fontsize=9.5, color=config.COLOR_TEXT)
        ax.set_xlim(0, grid[-1] / lakh)
        ax.set_ylim(bottom=0)
        _style(ax, grid_axis="both")
    for ax in axes[len(draws.channels) :]:
        ax.set_visible(False)
    fig.suptitle(
        "Response curves: weekly incremental revenue vs. steady weekly spend (INR lakh)",
        x=0.01,
        ha="left",
        fontsize=13,
        color=config.COLOR_TEXT,
    )
    fig.text(
        0.01,
        0.008,
        "Line: posterior mean. Band: 94% HDI. Dotted diagonal: revenue = spend. Solid vertical: "
        "current average weekly spend (active weeks).\nDashed vertical: saturation point "
        "(median draw), where the next rupee earns under 1 rupee of revenue.",
        fontsize=8,
        color=config.COLOR_TEXT_MUTED,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    return fig


def plot_carryover(summary: dict[str, Any]) -> Figure:
    """Return a dot-and-interval chart of weeks until 90% of each channel's effect lands."""
    channels = sorted(
        summary["channels"],
        key=lambda c: summary["channels"][c]["weeks_to_90pct_effect"]["mean"],
    )
    fig = Figure(figsize=(9, 0.55 * len(channels) + 1.8), facecolor=config.COLOR_SURFACE)
    ax = fig.subplots()
    for row, channel in enumerate(channels):
        weeks = summary["channels"][channel]["weeks_to_90pct_effect"]
        ax.plot(
            [weeks["hdi_low"], weeks["hdi_high"]],
            [row, row],
            color=config.COLOR_SPEND,
            linewidth=2,
            alpha=0.45,
            solid_capstyle="round",
        )
        ax.plot(
            weeks["mean"],
            row,
            "o",
            color=config.COLOR_SPEND,
            markersize=9,
            markeredgecolor=config.COLOR_SURFACE,
            markeredgewidth=1.5,
        )
        ax.text(
            weeks["hdi_high"] + 0.15,
            row,
            f"{weeks['mean']:.1f} wk",
            va="center",
            fontsize=9,
            color=config.COLOR_TEXT_MUTED,
        )
    ax.set_yticks(range(len(channels)), channels)
    ax.set_xlim(0, None)
    ax.set_xlabel("Weeks, counting the week of spend", color=config.COLOR_TEXT_MUTED, fontsize=9)
    ax.set_title(
        f"Carryover: weeks until {config.CARRYOVER_SHARE:.0%} of a channel's effect is realised",
        loc="left",
        fontsize=13,
        color=config.COLOR_TEXT,
    )
    fig.text(
        0.012,
        0.012,
        "Dot: posterior mean. Line: 94% HDI.",
        fontsize=8,
        color=config.COLOR_TEXT_MUTED,
    )
    _style(ax, grid_axis="x")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def plot_roi_vs_last_click(summary: dict[str, Any]) -> Figure:
    """Return MMM ROI (with 94% HDI) against naive same-week ROI for each channel."""
    channels = sorted(
        summary["channels"],
        key=lambda c: summary["channels"][c]["last_click"]["naive_over_mmm_ratio"]["median"],
        reverse=True,
    )
    fig = Figure(figsize=(10, 0.62 * len(channels) + 2.2), facecolor=config.COLOR_SURFACE)
    ax = fig.subplots()
    for row, channel in enumerate(channels):
        m = summary["channels"][channel]
        roi, naive = m["roi"], m["last_click"]["naive_roi"]
        ax.plot(
            [roi["hdi_low"], roi["hdi_high"]],
            [row, row],
            color=config.COLOR_SPEND,
            linewidth=2,
            alpha=0.45,
            solid_capstyle="round",
        )
        ax.plot(
            roi["mean"],
            row,
            "o",
            color=config.COLOR_SPEND,
            markersize=9,
            markeredgecolor=config.COLOR_SURFACE,
            markeredgewidth=1.5,
            label="MMM ROI (mean, 94% HDI)" if row == 0 else None,
        )
        ax.plot(
            naive,
            row,
            "D",
            color=config.COLOR_NAIVE,
            markersize=8,
            markeredgecolor=config.COLOR_SURFACE,
            markeredgewidth=1.5,
            label="Naive same-week ROI" if row == 0 else None,
        )
        ratio = m["last_click"]["naive_over_mmm_ratio"]["median"]
        verdict = (
            f"over-credited {ratio:.1f}x" if ratio >= 1 else f"under-credited {1 / ratio:.1f}x"
        )
        ax.annotate(
            verdict,
            (1.01, row),
            xycoords=("axes fraction", "data"),
            va="center",
            fontsize=9,
            color=config.COLOR_TEXT_MUTED,
        )
    ax.axvline(config.MARGINAL_ROI_BREAKEVEN, color=config.COLOR_TEXT_MUTED, linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlim(*config.ROI_AXIS_LIMITS)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:g}"))
    ax.set_yticks(range(len(channels)), channels)
    ax.invert_yaxis()
    ax.set_xlabel("Revenue per rupee spent (log scale)", color=config.COLOR_TEXT_MUTED, fontsize=9)
    ax.set_title(
        "What naive same-week attribution claims vs. what the MMM estimates",
        loc="left",
        fontsize=13,
        color=config.COLOR_TEXT,
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=2, frameon=False, fontsize=9)
    _style(ax, grid_axis="x")
    fig.tight_layout(rect=(0, 0, 0.86, 1))
    return fig


def save_figures(
    draws: PosteriorDraws, summary: dict[str, Any], directory: Path = config.FIGURES_DIR
) -> list[Path]:
    """Save all insight figures and return their paths."""
    directory.mkdir(parents=True, exist_ok=True)
    figures = {
        "insights_decomposition": plot_decomposition(decomposition_weekly(draws), draws.channels),
        "insights_waterfall": plot_waterfall(summary),
        "insights_response_curves": plot_response_curves(draws, summary),
        "insights_carryover": plot_carryover(summary),
        "insights_roi_vs_last_click": plot_roi_vs_last_click(summary),
    }
    paths = []
    for name, figure in figures.items():
        path = directory / f"{name}.png"
        figure.savefig(path, dpi=config.FIGURE_DPI)
        paths.append(path)
    return paths


def main() -> None:
    """Load the saved model and write the summary JSON, weekly decomposition and figures."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, default=config.MODELS_DIR)
    parser.add_argument(
        "--data", type=Path, default=config.SYNTHETIC_DIR / config.WEEKLY_DATA_FILENAME
    )
    args = parser.parse_args()
    df = pd.read_csv(args.data, parse_dates=[config.DATE_COL])
    draws = extract_draws(MixLabModel.load(args.model), df)
    summary = build_summary(draws)
    print(f"Saved {export_summary(summary)}")
    decomposition_weekly(draws).to_csv(config.REPORTS_DIR / config.DECOMPOSITION_FILENAME)
    for path in save_figures(draws, summary):
        print(f"Saved {path}")
    print(metrics_table(summary).to_string())


if __name__ == "__main__":
    main()
