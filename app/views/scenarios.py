"""Scenario planner: move channel budgets, see predicted revenue live, save and compare."""

import charts
import streamlit as st
from common import crore, label, likely, load_runtime, setup, show

from mixlab import config
from mixlab.optimizer import compare_scenarios, last_quarter_spend, prebuilt_scenarios

brand, results = setup(
    "Scenario planner",
    "Change spend per channel and see what the model expects. Save plans to compare them.",
)
runtime = load_runtime(brand)
draws = runtime.draws
weeks = runtime.allocator.n_weeks
current = last_quarter_spend(draws, weeks)
saved_key = f"saved_scenarios_{brand}"
saved: dict[str, dict[str, float]] = st.session_state.setdefault(saved_key, {})


def slider_key(channel: str) -> str:
    """Return the session key for a channel's slider."""
    return f"scenario_{brand}_{channel}"


def apply_plan(plan: dict[str, float]) -> None:
    """Set every slider to match a plan."""
    for channel, spend in plan.items():
        base = current[channel]
        st.session_state[slider_key(channel)] = round(100 * (spend / base - 1)) if base else 0


presets = {
    ("Last quarter (no change)" if name == "current" else name.capitalize()): plan
    for name, plan in prebuilt_scenarios(current).items()
}
preset_key = f"scenario_preset_{brand}"
picker, _ = st.columns([2, 3])
picker.selectbox(
    "Start from a preset",
    list(presets),
    key=preset_key,
    on_change=lambda: apply_plan(presets[st.session_state[preset_key]]),
    help="Sets every slider below; adjust from there.",
)

st.markdown(f"**Change vs. last quarter, per channel ({weeks} weeks)**")
plan: dict[str, float] = {}
columns = st.columns(3)
for index, (channel, spend) in enumerate(current.items()):
    change = columns[index % 3].slider(
        f"{label(channel)} · {spend / config.INR_PER_LAKH:,.0f} L",
        min_value=-100,
        max_value=150,
        value=0,
        step=5,
        format="%+d%%",
        key=slider_key(channel),
    )
    plan[channel] = spend * (1 + change / 100)

row = compare_scenarios(draws, {"current": current, "scenario": plan}, weeks).loc["scenario"]
first, second, third, fourth = st.columns(4)
first.metric(
    "Total spend", crore(row["spend"], 2), f"{row['spend_change'] / config.INR_PER_CRORE:+.2f} Cr"
)
second.metric(
    "Revenue from marketing",
    crore(row["revenue_mean"], 2),
    f"{row['revenue_change_mean'] / config.INR_PER_CRORE:+.2f} Cr",
)
second.caption(likely(crore(row["revenue_hdi_low"], 2), crore(row["revenue_hdi_high"], 2)))
third.metric("Chance revenue rises", f"{row['prob_revenue_up']:.0%}")
third.caption(
    "Change "
    + likely(
        f"{row['revenue_change_hdi_low'] / config.INR_PER_CRORE:+.2f}",
        f"{row['revenue_change_hdi_high'] / config.INR_PER_CRORE:+.2f} Cr",
    )
)
fourth.metric("Net of spend", f"{row['net_change_mean'] / config.INR_PER_CRORE:+.2f} Cr")
fourth.caption("Revenue change minus spend change. Positive means the plan pays for itself.")

name_column, save_column, _ = st.columns([2, 1, 2], vertical_alignment="bottom")
name = name_column.text_input("Name this scenario", placeholder="e.g. Cut Meta, fund Search")
if save_column.button("Save scenario", type="primary", width="stretch"):
    if not name.strip():
        st.error("Give the scenario a name before saving it.")
    else:
        saved[name.strip()] = dict(plan)
        st.toast(f"Saved “{name.strip()}”")

st.subheader("Saved scenarios")
if not saved:
    st.info("Nothing saved yet. Adjust the sliders, name the plan and save it to compare here.")
else:
    table = compare_scenarios(draws, {"current": current, **saved}, weeks)
    crore_columns = {
        "spend": "Spend (₹ Cr)",
        "revenue_mean": "Revenue (₹ Cr)",
        "revenue_change_mean": "Revenue change (₹ Cr)",
        "revenue_change_hdi_low": "Change low",
        "revenue_change_hdi_high": "Change high",
        "net_change_mean": "Change minus spend change (₹ Cr)",
    }
    shown = (table[list(crore_columns)] / config.INR_PER_CRORE).rename(columns=crore_columns)
    shown["Chance revenue rises (%)"] = 100 * table["prob_revenue_up"]
    st.dataframe(shown.round(2), width="stretch")
    if len(table) > 1:
        show(charts.scenario_changes(table))
    if st.button("Clear saved scenarios"):
        st.session_state[saved_key] = {}
        st.rerun()

st.caption(
    "Assumes spend is spread evenly over the period. Revenue is revenue caused by marketing, "
    "including carryover; baseline revenue is unaffected by the plan."
)
