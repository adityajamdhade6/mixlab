"""Channel performance: metrics with ranges, response curves and the last-click comparison."""

import charts
import pandas as pd
import streamlit as st
from common import label, lakh, load_runtime, margin, range_text, setup, show

from mixlab import config

brand, results = setup(
    "Channel performance",
    "What each channel returned, how sure the model is, and what the next rupee is worth.",
)
insights = results["insights"]
channels = insights["channels"]

table = pd.DataFrame(
    [
        {
            "Channel": label(name),
            "Spend (₹ Cr)": m["total_spend"] / config.INR_PER_CRORE,
            "Revenue (₹ Cr)": m["attributed_revenue"]["mean"] / config.INR_PER_CRORE,
            "Revenue low": m["attributed_revenue"]["hdi_low"] / config.INR_PER_CRORE,
            "Revenue high": m["attributed_revenue"]["hdi_high"] / config.INR_PER_CRORE,
            "ROI": m["roi"]["mean"],
            "ROI low": m["roi"]["hdi_low"],
            "ROI high": m["roi"]["hdi_high"],
            "Profit ROI": m["roi"]["mean"] * margin(),
            "Profit ROI low": m["roi"]["hdi_low"] * margin(),
            "Profit ROI high": m["roi"]["hdi_high"] * margin(),
            "Marginal ROI": m["marginal_roi"]["mean"],
            "Marginal ROI low": m["marginal_roi"]["hdi_low"],
            "Marginal ROI high": m["marginal_roi"]["hdi_high"],
            "Chance next rupee pays back (%)": 100 * m["prob_marginal_roi_above_1"],
            "Weeks to 90% of effect": m["weeks_to_90pct_effect"]["mean"],
        }
        for name, m in channels.items()
    ]
)


def with_range(column: str, decimals: int) -> list[str]:
    """Return 'estimate (low to high)' strings so each value travels with its range."""
    return [
        f"{value:.{decimals}f}  ({low:.{decimals}f} to {high:.{decimals}f})"
        for value, low, high in zip(
            table[column],
            table[f"{column.split(' (')[0]} low"],
            table[f"{column.split(' (')[0]} high"],
            strict=True,
        )
    ]


display = pd.DataFrame(
    {
        "Channel": table["Channel"],
        "Spend (₹ Cr)": table["Spend (₹ Cr)"],
        "Revenue (₹ Cr)": with_range("Revenue (₹ Cr)", 1),
        "ROI (₹ per ₹1)": with_range("ROI", 2),
        "Profit ROI (₹ per ₹1)": with_range("Profit ROI", 2),
        "Next ₹1 returns (₹)": with_range("Marginal ROI", 2),
        "Chance next ₹1 returns over ₹1": table["Chance next rupee pays back (%)"],
        "Weeks to 90% of effect": table["Weeks to 90% of effect"],
    }
)
st.dataframe(
    display,
    hide_index=True,
    width="stretch",
    column_config={
        "Channel": st.column_config.TextColumn(pinned=True),
        "Spend (₹ Cr)": st.column_config.NumberColumn(format="%.2f"),
        "ROI (₹ per ₹1)": st.column_config.TextColumn(
            help="Revenue per ₹1 spent, with its 94% range."
        ),
        "Profit ROI (₹ per ₹1)": st.column_config.TextColumn(
            help="Gross profit per ₹1 spent at the margin set in the sidebar. Above 1.00 the "
            "channel pays for itself."
        ),
        "Next ₹1 returns (₹)": st.column_config.TextColumn(
            help="Marginal ROI: revenue from the next rupee. This is what budget moves change."
        ),
        "Chance next ₹1 returns over ₹1": st.column_config.ProgressColumn(
            format="%.0f%%", min_value=0, max_value=100
        ),
        "Weeks to 90% of effect": st.column_config.NumberColumn(format="%.1f wk"),
    },
)
st.caption(
    "Values show the best estimate with its 94% range in brackets. ROI is revenue per ₹1 of "
    f"spend; profit ROI applies the {margin():.0%} margin set in the sidebar."
)
st.download_button(
    "Download metrics as CSV",
    table.to_csv(index=False).encode(),
    file_name=f"mixlab_channel_metrics_{brand}.csv",
    mime="text/csv",
    icon=":material/download:",
)

st.subheader("Response curve")
picker, _ = st.columns([1, 3])
channel = picker.selectbox("Channel", list(channels), format_func=label)
metrics = channels[channel]
chart, facts = st.columns([3, 1])
with chart:
    show(charts.response_curve(load_runtime(brand).draws, channel, metrics))
with facts:
    marginal = metrics["marginal_roi"]
    st.metric("Next ₹1 returns", f"₹{marginal['mean']:.2f} revenue")
    st.caption(
        range_text(f"₹{marginal['hdi_low']:.2f}", f"₹{marginal['hdi_high']:.2f}")
        + f" · ₹{marginal['mean'] * margin():.2f} of gross profit"
    )
    st.metric("Current weekly spend", lakh(metrics["current_weekly_spend"], 1))
    saturation = metrics["saturation_weekly_spend"]["median"]
    if saturation > 0:
        st.metric("Saturation point", lakh(saturation, 1))
        st.caption("Weekly spend beyond which the next rupee returns less than a rupee.")
    else:
        st.metric("Saturation point", "None")
        st.caption("The next rupee likely returns less than a rupee at every spend level.")

st.subheader("Model vs. naive attribution")
show(charts.roi_vs_naive(insights))
st.caption(
    "Naive attribution credits each week's revenue to whatever was spent that week. It ignores "
    "the baseline and carryover, so it overstates most channels. This dataset has no click "
    "data, so this stands in for last-click reporting."
)
