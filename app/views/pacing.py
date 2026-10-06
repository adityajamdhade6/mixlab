"""Pacing: the quarter in progress against plan and forecast, drift, refresh and versions."""

import charts
import pandas as pd
import streamlit as st
from common import (
    crore,
    label,
    lakh,
    load_monitoring,
    load_versions,
    plural,
    setup,
    show,
)

PACING_COMMAND = "uv run python scripts/simulate_weeks.py"

brand, results = setup(
    "Pacing and forecast",
    "Week to week: is spend going to plan, is revenue landing where the forecast said, and "
    "is the model still right?",
)
story = load_monitoring(brand)
if story is None:
    st.info(
        "No quarter has been simulated for this brand yet. The performance-heavy brand has "
        "one; to build it, run the command below (about five minutes)."
    )
    st.code(PACING_COMMAND, language="bash")
    st.stop()

pacing, drift = story["pacing"], story["drift"]
accuracy = pacing["accuracy"]
weeks_in = len(story["actual"])
off_plan = [
    c for c, v in pacing["channels"].items() if v["status"] not in ("on plan", "not planned")
]

first, second, third, fourth = st.columns(4)
first.metric("Weeks into the quarter", f"{weeks_in} of {len(story['schedule'])}")
second.metric("Forecast error so far", f"{accuracy['mape_pct']:.1f}%")
second.caption(f"{accuracy['coverage_pct']:.0f}% of weeks inside the 94% range")
third.metric("Channels off plan", f"{len(off_plan)}")
third.caption(", ".join(label(c) for c in off_plan) or "All on plan")
fourth.metric("Model drift", "Yes" if drift["drifting"] else "No")
fourth.caption(
    f"Normal error {drift['backtest_mape_pct']:.1f}%, recent {drift['recent_mape_pct']:.1f}%"
)

st.caption(
    "Drift is judged against an as-run forecast: the model re-predicts each week with the "
    "spend and promotions that actually ran, so missed plans do not count as model error."
)
if drift["drifting"]:
    st.warning(
        "**The model may have drifted:** " + "; ".join(drift["reasons"]) + ". " + drift["action"]
    )
else:
    st.success(drift["action"])

st.subheader("Forecast for the quarter")
show(charts.forecast_vs_actual(story))
current = pd.DataFrame(story["forecasts"]["current"])["mean"].sum()
recommended = pd.DataFrame(story["forecasts"]["recommended"])["mean"].sum()
st.caption(
    f"13-week forecast: {crore(current)} on the current plan, {crore(recommended)} on the "
    "recommended plan. Known festivals and the new-year price step are included; no "
    "promotions are assumed."
)

st.subheader("Pacing against the plan")
show(charts.pacing_bars(pacing["channels"]))
rows = [
    {
        "Channel": label(c),
        "Planned so far": lakh(v["planned"]),
        "Actual so far": lakh(v["actual"]),
        "Status": v["status"].capitalize(),
    }
    for c, v in pacing["channels"].items()
]
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
if pacing["outside_weeks"]:
    st.caption(
        f"{plural(len(pacing['outside_weeks']), 'week')} outside the forecast range: "
        + ", ".join(pacing["outside_weeks"])
        + "."
    )

st.subheader("Refreshed model: what changed")
st.caption(
    f"The model was refitted on the {weeks_in} new weeks as well. Each channel's ROI before "
    "and after, and why it moved."
)
for change in story["changes"]:
    before, after = change["before"], change["after"]
    line = (
        f"**{label(change['channel'])}:** ROI {before['mean']:.2f} → {after['mean']:.2f} "
        f"({change['change_pct']:+.0f}%; likely between {after['hdi_low']:.2f} and "
        f"{after['hdi_high']:.2f} now). {change['reason']}"
    )
    st.markdown(("⚠️ " if change["big_change"] else "") + line)

st.subheader("Model versions")
versions = load_versions(brand)
if versions:
    table = pd.DataFrame(
        [
            {
                "Version": v["version"],
                "Data": f"{v['data_start']} to {v['data_end']} ({v['n_weeks']} weeks)",
                "Blended ROI": f"{v['blended_roi']['mean']:.2f}",
                **{label(c): f"{r['mean']:.2f}" for c, r in v["roi"].items()},
            }
            for v in versions
        ]
    )
    st.dataframe(table, hide_index=True, width="stretch")

with st.expander("How these weeks were made (synthetic data)"):
    simulation = story["simulation"]
    slips = ", ".join(f"{label(c)} {100 * s:+.0f}%" for c, s in simulation["slip"].items())
    st.markdown(
        f"The new weeks come from the brand's true data-generating process. Spend followed "
        f"the recommended plan except: {slips}. In the last {simulation['shock_weeks']} weeks "
        f"a competitor launch the model knows nothing about cut baseline revenue by "
        f"{-100 * simulation['shock_share']:.0f}%."
    )
