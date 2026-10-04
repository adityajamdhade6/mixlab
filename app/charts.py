"""Plotly figures for the dashboard, all drawn on one shared theme."""

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from common import label

from mixlab import config
from mixlab.insights import PosteriorDraws, channel_colors, hdi, response_curve_draws

LAKH = config.INR_PER_LAKH
CRORE = config.INR_PER_CRORE
COLOR_CURRENT = "#a5a39b"
ROI_TICKS = [0.03, 0.1, 0.3, 1, 3, 10, 30]
FONT = "Inter, -apple-system, 'Segoe UI', sans-serif"


def themed(figure: go.Figure, title: str, height: int = 400) -> go.Figure:
    """Apply the shared look: quiet grid, left-aligned title, legend above the plot."""
    figure.update_layout(
        title={"text": title, "x": 0, "xanchor": "left", "font": {"size": 16}},
        height=height,
        font={"family": FONT, "size": 13, "color": config.COLOR_TEXT},
        paper_bgcolor=config.COLOR_SURFACE,
        plot_bgcolor=config.COLOR_SURFACE,
        margin={"l": 10, "r": 10, "t": 70 if title else 20, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0, "title": None},
        hoverlabel={"font": {"family": FONT}},
    )
    figure.update_xaxes(gridcolor=config.COLOR_GRID, zeroline=False, linecolor=config.COLOR_GRID)
    figure.update_yaxes(gridcolor=config.COLOR_GRID, zerolinecolor=config.COLOR_TEXT_MUTED)
    return figure


def rgba(hex_color: str, alpha: float) -> str:
    """Return a hex colour in CSS rgba form with the given opacity."""
    red, green, blue = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({red},{green},{blue},{alpha})"


def decomposition(weekly: pd.DataFrame, channels: list[str]) -> go.Figure:
    """Stacked area of weekly revenue by driver, with observed revenue on top."""
    colors = {**config.ORGANIC_COLORS, **channel_colors(channels)}
    figure = go.Figure()
    for name in [c for c in weekly.columns if c != "observed_revenue"]:
        values = weekly[name] / LAKH
        figure.add_trace(
            go.Scatter(
                x=weekly.index,
                y=values.clip(lower=0),
                name=label(name),
                stackgroup="positive",
                mode="lines",
                line={"width": 0.5, "color": config.COLOR_SURFACE},
                fillcolor=colors[name],
                customdata=values,
                hovertemplate="%{customdata:,.1f} L",
            )
        )
        if (values < 0).any():
            figure.add_trace(
                go.Scatter(
                    x=weekly.index,
                    y=values.clip(upper=0),
                    stackgroup="negative",
                    mode="lines",
                    line={"width": 0},
                    fillcolor=colors[name],
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
    figure.add_trace(
        go.Scatter(
            x=weekly.index,
            y=weekly["observed_revenue"] / LAKH,
            name="Observed revenue",
            mode="lines",
            line={"color": config.COLOR_TEXT, "width": 1.6},
            hovertemplate="%{y:,.1f} L",
        )
    )
    figure.update_yaxes(title="₹ lakh per week")
    themed(figure, "", height=480)
    # Thirteen series: the legend goes under the plot so it cannot collide with anything.
    figure.update_layout(
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "top", "y": -0.12, "x": 0, "traceorder": "normal"},
    )
    return figure


def response_curve(draws: PosteriorDraws, channel: str, metrics: dict[str, Any]) -> go.Figure:
    """Response curve for one channel with its range, current spend and saturation point."""
    index = draws.channels.index(channel)
    color = channel_colors(draws.channels)[channel]
    grid = np.linspace(
        0,
        config.RESPONSE_CURVE_MAX_MULTIPLE * draws.spend[:, index].max(),
        config.RESPONSE_CURVE_POINTS,
    )
    curves = response_curve_draws(draws, index, grid)
    low, high = hdi(curves)
    x = grid / LAKH
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=x, y=low / LAKH, mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip"
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x,
            y=high / LAKH,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor=rgba(color, 0.18),
            name="94% range",
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x,
            y=curves.mean(axis=0) / LAKH,
            mode="lines",
            name="Best estimate",
            line={"color": color, "width": 2.5},
            customdata=np.column_stack([low / LAKH, high / LAKH]),
            hovertemplate="Spend %{x:,.1f} L/week<br>Revenue %{y:,.1f} L/week"
            "<br>likely %{customdata[0]:,.1f} to %{customdata[1]:,.1f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x,
            y=x,
            mode="lines",
            name="Revenue = spend",
            line={"color": config.COLOR_TEXT_MUTED, "width": 1, "dash": "dot"},
            hoverinfo="skip",
        )
    )
    current = metrics["current_weekly_spend"] / LAKH
    figure.add_vline(
        x=current,
        line={"color": config.COLOR_TEXT, "width": 1.5},
        annotation_text=f"Current {current:,.1f} L",
        annotation_position="top",
    )
    saturation = metrics["saturation_weekly_spend"]["median"] / LAKH
    if saturation > 0:
        figure.add_vline(
            x=saturation,
            line={"color": config.COLOR_TEXT, "width": 1.5, "dash": "dash"},
            annotation_text=f"Saturation {saturation:,.1f} L",
            annotation_position="bottom right",
        )
    figure.update_xaxes(title="Steady weekly spend (₹ lakh)", rangemode="tozero")
    figure.update_yaxes(title="Weekly incremental revenue (₹ lakh)", rangemode="tozero")
    return themed(figure, f"{label(channel)}: what each extra rupee buys", height=430)


def roi_vs_naive(insights: dict[str, Any]) -> go.Figure:
    """MMM ROI with its range against the ROI a naive same-week attribution would report."""
    channels = sorted(
        insights["channels"],
        key=lambda c: insights["channels"][c]["last_click"]["naive_over_mmm_ratio"]["median"],
    )
    rows = [insights["channels"][c] for c in channels]
    names = [label(c) for c in channels]
    floor = config.ROI_AXIS_LIMITS[0]
    mean = [max(r["roi"]["mean"], floor) for r in rows]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=mean,
            y=names,
            mode="markers",
            name="MMM estimate (94% range)",
            marker={"color": config.COLOR_SPEND, "size": 12},
            error_x={
                "type": "data",
                "symmetric": False,
                "thickness": 2,
                "width": 0,
                "color": rgba(config.COLOR_SPEND, 0.5),
                "array": [r["roi"]["hdi_high"] - m for r, m in zip(rows, mean, strict=True)],
                "arrayminus": [
                    m - max(r["roi"]["hdi_low"], floor) for r, m in zip(rows, mean, strict=True)
                ],
            },
            customdata=[[r["roi"]["hdi_low"], r["roi"]["hdi_high"]] for r in rows],
            hovertemplate="MMM ROI %{x:.2f}<br>likely %{customdata[0]:.2f} to "
            "%{customdata[1]:.2f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[r["last_click"]["naive_roi"] for r in rows],
            y=names,
            mode="markers",
            name="Naive same-week attribution",
            marker={"color": config.COLOR_NAIVE, "size": 11, "symbol": "diamond"},
            hovertemplate="Naive ROI %{x:.2f}<extra></extra>",
        )
    )
    figure.add_vline(x=1, line={"color": config.COLOR_TEXT_MUTED, "width": 1})
    figure.update_xaxes(
        type="log",
        title="Revenue per rupee spent (log scale)",
        tickvals=ROI_TICKS,
        ticktext=[f"{v:g}" for v in ROI_TICKS],
        range=[np.log10(v) for v in config.ROI_AXIS_LIMITS],
    )
    return themed(
        figure,
        "What naive attribution claims vs. what the model estimates",
        height=110 + 48 * len(channels),
    )


def allocation(current: dict[str, float], recommended: dict[str, float], weeks: int) -> go.Figure:
    """Return side-by-side bars of current and recommended spend per channel."""
    names = [label(c) for c in current]
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=names,
            y=[v / LAKH for v in current.values()],
            name="Current",
            marker={"color": COLOR_CURRENT, "cornerradius": 4},
            hovertemplate="%{y:,.0f} L<extra>Current</extra>",
        )
    )
    figure.add_trace(
        go.Bar(
            x=names,
            y=[v / LAKH for v in recommended.values()],
            name="Recommended",
            marker={"color": config.COLOR_SPEND, "cornerradius": 4},
            text=[
                f"{100 * (recommended[c] / current[c] - 1):+.0f}%" if current[c] else ""
                for c in current
            ],
            textposition="outside",
            hovertemplate="%{y:,.0f} L<extra>Recommended</extra>",
        )
    )
    figure.update_layout(barmode="group", bargap=0.28, bargroupgap=0.08)
    figure.update_yaxes(title=f"Spend over {weeks} weeks (₹ lakh)")
    return themed(figure, "Current vs. recommended allocation", height=420)


def recovery(table: pd.DataFrame) -> go.Figure:
    """Estimated ROI with its range against the true ROI, per channel (log scale)."""
    names = [label(c) for c in table["channel"]]
    floor = config.ROI_AXIS_LIMITS[0]
    estimate = table["estimated_roi"].clip(lower=floor)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=estimate,
            y=names,
            mode="markers",
            name="Model estimate (94% range)",
            marker={"color": config.COLOR_SPEND, "size": 12},
            error_x={
                "type": "data",
                "symmetric": False,
                "thickness": 2,
                "width": 0,
                "color": rgba(config.COLOR_SPEND, 0.5),
                "array": table["high"] - estimate,
                "arrayminus": estimate - table["low"].clip(lower=floor),
            },
            customdata=table[["estimated_roi", "low", "high"]],
            hovertemplate="Estimate %{customdata[0]:.2f}<br>likely %{customdata[1]:.2f} to "
            "%{customdata[2]:.2f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=table["true_roi"],
            y=names,
            mode="markers",
            name="True ROI",
            marker={
                "color": config.COLOR_TEXT,
                "size": 14,
                "symbol": "line-ns",
                "line": {"width": 3, "color": config.COLOR_TEXT},
            },
            hovertemplate="True ROI %{x:.2f}<extra></extra>",
        )
    )
    figure.update_xaxes(
        type="log",
        title="Revenue per rupee spent (log scale)",
        range=[np.log10(v) for v in config.ROI_AXIS_LIMITS],
        tickvals=ROI_TICKS,
        ticktext=[f"{v:g}" for v in ROI_TICKS],
    )
    return themed(figure, "Does the model recover the true ROI?", height=110 + 48 * len(table))


def scenario_changes(table: pd.DataFrame) -> go.Figure:
    """Revenue change versus the current plan for each saved scenario, with ranges."""
    rows = table.drop(index="current", errors="ignore")
    figure = go.Figure(
        go.Bar(
            x=rows["revenue_change_mean"] / CRORE,
            y=list(rows.index),
            orientation="h",
            marker={"color": config.COLOR_SPEND, "cornerradius": 4},
            error_x={
                "type": "data",
                "symmetric": False,
                "color": config.COLOR_TEXT,
                "thickness": 1.5,
                "array": (rows["revenue_change_hdi_high"] - rows["revenue_change_mean"]) / CRORE,
                "arrayminus": (rows["revenue_change_mean"] - rows["revenue_change_hdi_low"])
                / CRORE,
            },
            hovertemplate="%{x:+.2f} Cr<extra></extra>",
        )
    )
    figure.update_xaxes(title="Change in incremental revenue vs. current plan (₹ crore)")
    figure.update_yaxes(autorange="reversed")
    return themed(figure, "Saved scenarios compared", height=130 + 44 * len(rows))


def budget_curve(optimizer: dict[str, Any]) -> go.Figure:
    """Best achievable incremental revenue at each total budget."""
    curve = pd.DataFrame(optimizer["budget_curve"]).astype(float)
    x = curve["budget"] / CRORE
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=x,
            y=curve["revenue_hdi_low"] / CRORE,
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x,
            y=curve["revenue_hdi_high"] / CRORE,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor=rgba(config.COLOR_SPEND, 0.18),
            name="94% range",
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x,
            y=curve["revenue_mean"] / CRORE,
            mode="lines+markers",
            name="Best achievable revenue",
            line={"color": config.COLOR_SPEND, "width": 2.5},
            hovertemplate="Budget %{x:.2f} Cr<br>Revenue %{y:.2f} Cr<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x,
            y=x,
            mode="lines",
            name="Revenue = budget",
            line={"color": config.COLOR_TEXT_MUTED, "width": 1, "dash": "dot"},
            hoverinfo="skip",
        )
    )
    figure.add_vline(
        x=optimizer["current_budget"] / CRORE,
        line={"color": config.COLOR_TEXT, "width": 1.5},
        annotation_text="Last quarter",
        annotation_position="top",
    )
    if optimizer["payback_budget"] is not None:
        figure.add_vline(
            x=optimizer["payback_budget"] / CRORE,
            line={"color": config.COLOR_TEXT, "width": 1.5, "dash": "dash"},
            annotation_text="Extra budget stops paying back",
            annotation_position="bottom right",
        )
    figure.update_xaxes(title="Total quarterly media budget (₹ crore)", rangemode="tozero")
    figure.update_yaxes(title="Incremental revenue (₹ crore)", rangemode="tozero")
    return themed(figure, "How much is the next crore of budget worth?", height=420)


def actual_vs_predicted(weeks: list[dict[str, Any]]) -> go.Figure:
    """Return held-out weeks: actual revenue against the prediction and its 94% range."""
    frame = pd.DataFrame(weeks)
    x = pd.to_datetime(frame["date"])
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=x,
            y=frame["lower"] / LAKH,
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x,
            y=frame["upper"] / LAKH,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor=rgba(config.COLOR_SPEND, 0.18),
            name="94% range",
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x,
            y=frame["predicted"] / LAKH,
            mode="lines+markers",
            name="Predicted",
            line={"color": config.COLOR_SPEND, "width": 2.5},
            hovertemplate="Predicted ₹%{y:,.1f} L<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x,
            y=frame["actual"] / LAKH,
            mode="lines+markers",
            name="Actual",
            line={"color": config.COLOR_TEXT, "width": 2},
            hovertemplate="Actual ₹%{y:,.1f} L<extra></extra>",
        )
    )
    figure.update_layout(hovermode="x unified")
    figure.update_yaxes(title="Weekly revenue (₹ lakh)")
    return themed(figure, "Weeks the model never saw: actual vs. predicted", height=380)
