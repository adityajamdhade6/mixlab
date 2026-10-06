"""Regions: the geo model's ROI by region, an India map, and a regional budget optimizer."""

import charts
import pandas as pd
import streamlit as st
from common import (
    GEO_BUILD_COMMAND,
    chance,
    crore,
    has_geo,
    header,
    label,
    lakh,
    load_geo,
    load_geo_runtime,
    per_rupee,
    range_text,
    show,
    use_hosted_secret,
)

from mixlab import config
from mixlab.geo_insights import regional_optimizer

use_hosted_secret()
header(config.GEO_DEMO_BRAND)
st.title("Regions")
st.caption(
    "A separate demo brand sold in ten Indian regions, fitted with the geo-level model: one "
    "model across all regions, where each region's channel ROI is pulled toward the national "
    "figure unless its own data disagrees."
)

if not has_geo():
    st.error(
        "The regional demo has not been built yet, so there is nothing to show. Build it once "
        "with the command below (about 30 minutes), then reload this page."
    )
    st.code(GEO_BUILD_COMMAND, language="bash")
    st.stop()

geo = load_geo()
summary, comparison, truth = geo["summary"], geo["comparison"], geo["truth"]
regions = summary["regions"]
channels = summary["channels"]
optimizer = summary["optimizer"]

first, second, third, fourth = st.columns(4)
first.metric("Regions", f"{summary['n_regions']}")
first.caption(f"{crore(sum(r['media_spend'] for r in regions.values()))} media spend in total")
if comparison.get("median_width_ratio") is not None:
    narrower = 100 * (1 - comparison["median_width_ratio"])
    second.metric("ROI ranges vs. national model", f"{narrower:.0f}% narrower")
    second.caption("Median across channels, same data summed to national weeks")
    third.metric(
        "True ROI recovered",
        f"{comparison['geo_recovered']} of {comparison['n_channels']}",
        f"national model: {comparison['national_recovered']} of {comparison['n_channels']}",
        delta_color="off",
    )
    third.caption("Channels with the truth inside the 94% range")
uplift = optimizer["uplift"]
fourth.metric("Regional reallocation", f"{uplift['mean'] / config.INR_PER_CRORE:+.2f} Cr")
fourth.caption(
    range_text(crore(uplift["hdi_low"], 2), crore(uplift["hdi_high"], 2))
    + f" · same budget, {optimizer['n_weeks']} weeks"
)

st.subheader("India map")
left, right = st.columns([1, 3])
with left:
    colour_by = st.radio(
        "Colour regions by",
        ["Where to invest", *[label(c) for c in channels]],
        help="'Where to invest' compares what the next rupee earns in each region with the "
        "national figure. A channel colours each region by that channel's ROI.",
    )
    st.caption(
        f"**Under-invested:** the next rupee earns over {config.GEO_INVESTMENT_RATIO:g}x the "
        "national figure. **Over-invested:** under 1/"
        f"{config.GEO_INVESTMENT_RATIO:g} of it. Bubble size is media spend."
    )
chosen = (
    None
    if colour_by == "Where to invest"
    else channels[[label(c) for c in channels].index(colour_by)]
)
with right:
    show(charts.india_map(summary, chosen))

st.subheader("One region in detail")
geo_names = sorted(regions, key=lambda g: regions[g]["label"])
selected = st.selectbox(
    "Region", geo_names, format_func=lambda g: regions[g]["label"], key="region"
)
region = regions[selected]
marginal = region["blended_marginal_roi"]
national_marginal = summary["national_marginal_roi"]["median"]
st.markdown(
    f"**{region['label']}** takes {region['spend_share_pct']:.0f}% of media spend. Its next "
    f"rupee, spread like today's mix, likely returns between {per_rupee(marginal['hdi_low'])} "
    f"and {per_rupee(marginal['hdi_high'])} (median {per_rupee(marginal['median'])}, national "
    f"median {per_rupee(national_marginal)}): **{region['status']}**."
)
rows = []
for channel in channels:
    cell = region["channels"][channel]
    row = {
        "Channel": label(channel),
        "Spend": lakh(cell["total_spend"]),
        "ROI (94% range)": f"{cell['roi']['mean']:.2f} ({cell['roi']['hdi_low']:.2f} to "
        f"{cell['roi']['hdi_high']:.2f})",
        "Next ₹1 returns": f"{per_rupee(cell['marginal_roi']['median'])}",
    }
    if "true_roi" in cell:
        row["True ROI"] = f"{cell['true_roi']:.2f}"
        row["Inside range"] = "Yes" if cell["truth_inside_range"] else "No"
    rows.append(row)
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

channel = st.selectbox("Compare one channel across regions", channels, format_func=label)
show(charts.regional_roi(summary, channel))

st.subheader("Why a geo model")
for line in comparison.get("headline", []):
    st.markdown(f"- {line}")
if "channels" in comparison:
    show(charts.geo_vs_national(comparison))
    st.caption(
        "Both models see the same weeks. The national model only sees the regions summed; "
        "the geo model sees ten regions whose spend moved differently, which separates the "
        "channels' effects far better. Synthetic data: the true ROI is known."
    )

st.subheader("Regional budget optimizer")
st.caption(
    "Moves money between regions as well as channels, at the same total budget. Each "
    "region-channel pair stays within a set share of today's spend and never above its "
    "highest week on record."
)
max_change = st.slider(
    "Largest change for any region-channel pair",
    min_value=10,
    max_value=50,
    value=round(100 * optimizer["max_change"]),
    step=5,
    format="%d%%",
)
result = optimizer
if round(100 * optimizer["max_change"]) != max_change:
    if st.button("Re-run the regional optimizer", type="primary"):
        with st.spinner("Searching across every region and channel (up to a minute)…"):
            st.session_state["geo_result"] = regional_optimizer(
                load_geo_runtime(), truth, max_change=max_change / 100
            )
    rerun = st.session_state.get("geo_result")
    if rerun and abs(rerun["max_change"] - max_change / 100) < 1e-9:
        result = rerun
    else:
        st.info("Showing the saved plan at the default limit. Re-run to use your limit.")

uplift = result["uplift"]
left, middle, right = st.columns(3)
left.metric("Expected extra revenue", crore(uplift["mean"], 2))
left.caption(range_text(crore(uplift["hdi_low"], 2), crore(uplift["hdi_high"], 2)))
middle.metric("Chance it beats today's split", chance(result["prob_beats_current"]))
check = result.get("truth_check")
if check:
    right.metric("True extra revenue", crore(check["true_uplift"], 2))
    if "national_plan_true_uplift" in check:
        right.caption(
            f"A national-model plan, spread across regions as today: "
            f"{crore(check['national_plan_true_uplift'], 2)}"
        )
show(charts.regional_shift(result, summary))

table = pd.DataFrame(
    [
        {
            "region": regions[g]["label"],
            "channel": c,
            "current_spend_inr": cell["current"],
            "recommended_spend_inr": cell["recommended"],
            "roi_mean": regions[g]["channels"][c]["roi"]["mean"],
            "roi_low": regions[g]["channels"][c]["roi"]["hdi_low"],
            "roi_high": regions[g]["channels"][c]["roi"]["hdi_high"],
        }
        for g, r in result["regions"].items()
        for c, cell in r["channels"].items()
    ]
)
st.download_button(
    "Download the regional plan (CSV)",
    table.to_csv(index=False).encode(),
    file_name="mixlab_regional_plan.csv",
    mime="text/csv",
)
