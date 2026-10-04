"""Scenario planner: move channel budgets, see predicted revenue live, save and compare."""

import charts
import streamlit as st
from common import chance, crore, label, load_runtime, margin, range_text, setup, show

from mixlab import config
from mixlab.optimizer import compare_scenarios, last_quarter_spend, prebuilt_scenarios

CRORE = config.INR_PER_CRORE

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


def signed_crore(value: float) -> str:
    """Format a change in INR crore with its sign and unit."""
    return f"{'-' if value < 0 else '+'}₹{abs(value) / CRORE:.2f} Cr"


presets = {
    ("Last quarter (no change)" if name == "current" else label(name).replace("_", " ")): plan
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
        f"{label(channel)} · ₹{spend / config.INR_PER_LAKH:,.0f} L",
        min_value=-100,
        max_value=150,
        value=0,
        step=5,
        format="%+d%%",
        key=slider_key(channel),
    )
    plan[channel] = spend * (1 + change / 100)

row = compare_scenarios(draws, {"current": current, "scenario": plan}, weeks).loc["scenario"]
unchanged = all(abs(plan[c] - current[c]) < 1 for c in current)
profit_change = row["revenue_change_mean"] * margin() - row["spend_change"]

first, second, third, fourth = st.columns(4)
first.metric(
    "Total spend", crore(row["spend"], 2), None if unchanged else signed_crore(row["spend_change"])
)
second.metric(
    "Marketing revenue",
    crore(row["revenue_mean"], 2),
    None if unchanged else signed_crore(row["revenue_change_mean"]),
)
second.caption(range_text(crore(row["revenue_hdi_low"], 2), crore(row["revenue_hdi_high"], 2)))
if unchanged:
    third.metric("Chance revenue rises", "—")
    third.caption("Move a slider or pick a preset to see the effect of a change.")
    fourth.metric("Profit change", "—")
    fourth.caption(f"Revenue change x {margin():.0%} margin, minus the change in spend.")
else:
    third.metric("Chance revenue rises", chance(row["prob_revenue_up"]))
    third.caption(
        "Revenue change "
        + range_text(
            signed_crore(row["revenue_change_hdi_low"]),
            signed_crore(row["revenue_change_hdi_high"]),
        )
    )
    fourth.metric("Profit change", signed_crore(profit_change))
    fourth.caption(f"Revenue change x {margin():.0%} margin, minus the change in spend.")

over_peak = [
    label(channel)
    for channel, peak in zip(draws.channels, draws.spend.max(axis=0), strict=True)
    if plan[channel] / weeks > peak
]
if over_peak:
    st.warning(
        f"This plan takes {', '.join(over_peak)} above any weekly spend seen in the data, so "
        "the prediction there is an extrapolation."
    )

name_column, save_column, _ = st.columns([2, 1, 2], vertical_alignment="bottom")
name = name_column.text_input("Name this scenario", placeholder="e.g. Cut Meta, fund Search")
if save_column.button("Save scenario", type="primary", width="stretch", disabled=unchanged):
    if not name.strip():
        st.error("Give the scenario a name before saving it.")
    else:
        saved[name.strip()] = dict(plan)
        st.toast(f"Saved “{name.strip()}”")
if unchanged:
    st.caption("Nothing to save yet: this plan is the same as last quarter.")

st.subheader("Saved scenarios")
if not saved:
    st.info("Nothing saved yet. Adjust the sliders, name the plan and save it to compare here.")
else:
    table = compare_scenarios(draws, {"current": current, **saved}, weeks)
    shown = (
        table[["spend", "revenue_mean", "revenue_change_mean"]]
        .div(CRORE)
        .rename(
            columns={
                "spend": "Spend (₹ Cr)",
                "revenue_mean": "Marketing revenue (₹ Cr)",
                "revenue_change_mean": "Revenue change (₹ Cr)",
            }
        )
    )
    shown["Revenue change, 94% range (₹ Cr)"] = [
        f"{low / CRORE:+.2f} to {high / CRORE:+.2f}"
        for low, high in zip(
            table["revenue_change_hdi_low"], table["revenue_change_hdi_high"], strict=True
        )
    ]
    shown["Profit change (₹ Cr)"] = (
        table["revenue_change_mean"] * margin() - table["spend_change"]
    ) / CRORE
    shown["Chance revenue rises"] = [chance(p) for p in table["prob_revenue_up"]]
    shown.index = ["Last quarter" if i == "current" else i for i in shown.index]
    st.dataframe(shown.round(2), width="stretch")
    show(charts.scenario_changes(table))
    if st.button("Clear saved scenarios"):
        st.session_state[saved_key] = {}
        st.rerun()

st.caption(
    "Assumes spend is spread evenly over the period. Marketing revenue is revenue caused by "
    "marketing, including carryover; baseline revenue is unaffected by the plan."
)
