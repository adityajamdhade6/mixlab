"""Budget optimizer: set a budget and limits, get a recommended allocation with its range."""

import charts
import pandas as pd
import streamlit as st
from common import chance, crore, label, load_runtime, margin, range_text, setup, show

from mixlab import config
from mixlab.optimizer import (
    OptimizationResult,
    channel_caveats,
    last_quarter_spend,
    optimize_budget,
)

LAKH = config.INR_PER_LAKH
OBJECTIVES = {
    "Expected revenue": "mean",
    f"Cautious ({config.RISK_PERCENTILE:g}th percentile)": "percentile",
}

brand, results = setup(
    "Budget optimizer",
    "Choose a budget and how far each channel may move. The model finds the best split.",
)
runtime = load_runtime(brand)
weeks = runtime.allocator.n_weeks
current = last_quarter_spend(runtime.draws, weeks)
caveats = channel_caveats(runtime.draws)
total = sum(current.values())
state_key = f"optimizer_result_{brand}"

if caveats:
    names = ", ".join(label(c) for c in caveats)
    st.info(
        f"Confidence gate: {names} cannot be measured well (see Model health), so their "
        f"default limits are ±{config.GATED_MAX_CHANGE:.0%} instead of "
        f"±{config.DEFAULT_MAX_CHANGE:.0%}. You can widen them below."
    )

with st.form("optimizer"):
    left, right = st.columns(2)
    budget = LAKH * left.number_input(
        f"Total budget for the next {weeks} weeks (₹ lakh)",
        min_value=0.0,
        value=float(round(total / LAKH)),
        step=10.0,
        help=f"Last quarter's spend was ₹{total / LAKH:,.0f} lakh.",
    )
    objective = right.radio(
        "Optimise for",
        list(OBJECTIVES),
        horizontal=True,
        help="Cautious maximises the revenue you would still get in a bad case, so it favours "
        "channels whose effect is well established.",
    )
    st.markdown("**Allowed spend per channel (₹ lakh)**")
    limits: dict[str, tuple[float, float]] = {}
    columns = st.columns(3)
    for index, (channel, spend) in enumerate(current.items()):
        value = spend / LAKH
        change = config.GATED_MAX_CHANGE if channel in caveats else config.DEFAULT_MAX_CHANGE
        # Small channels need a finer step, or a ±10% range rounds away to nothing.
        step = 0.1 if value < 20 else 1.0
        low, high = (round(value * factor / step) * step for factor in (1 - change, 1 + change))
        limits[channel] = columns[index % 3].slider(
            f"{label(channel)} (₹{value:,.0f} L)" + (" · gated" if channel in caveats else ""),
            min_value=0.0,
            max_value=float(max(round(3 * value), 10)),
            value=(float(low), float(high)),
            step=step,
            format="%.1f" if step < 1 else "%.0f",
            help=(
                f"Last quarter ₹{value:,.1f} lakh. "
                + (
                    f"Held to ±{change:.0%} by default: {caveats[channel]}."
                    if channel in caveats
                    else ""
                )
            ),
        )
    submitted = st.form_submit_button("Run optimization", type="primary")

if submitted:
    try:
        with st.spinner("Finding the best allocation…"):
            st.session_state[state_key] = optimize_budget(
                runtime.allocator,
                runtime.draws,
                total_budget=budget,
                minimum={c: low * LAKH for c, (low, _) in limits.items()},
                maximum={c: high * LAKH for c, (_, high) in limits.items()},
                objective=OBJECTIVES[objective],
            )
    except ValueError as error:
        st.session_state.pop(state_key, None)
        st.error(f"That combination cannot be optimised. {error}")

result = st.session_state.get(state_key)
if result is None:
    result = OptimizationResult.model_validate(results["optimizer"]["expected_revenue"])
    st.caption(
        "Showing the saved recommendation for last quarter's budget. Change the inputs and run "
        "the optimization to replace it."
    )
elif st.button("Back to the saved recommendation"):
    st.session_state.pop(state_key, None)
    st.rerun()

uplift = result.uplift
first, second, third = st.columns(3)
first.metric(
    "Expected uplift", crore(uplift.mean, 2), f"{result.uplift_pct.mean:+.1f}% vs. current"
)
first.caption(range_text(crore(uplift.hdi_low, 2), crore(uplift.hdi_high, 2)))
second.metric(
    "Realistic uplift",
    crore(result.realistic_uplift, 2),
    f"{result.realistic_uplift_pct:+.1f}% vs. current",
)
second.caption(
    f"About {crore(result.realistic_uplift * margin(), 2)} of gross profit at a "
    f"{margin():.0%} margin"
)
third.metric("Chance it beats current", chance(result.prob_recommended_beats_current))
third.caption(f"Optimised for {result.objective}")
st.caption(
    f"Realistic uplift is the expected uplift x {config.UPLIFT_SHRINKAGE:.0%}: optimizers favour "
    "the channels a model happens to overestimate, and on synthetic brands with known truth "
    "that share of the expected uplift was actually delivered."
)

if not result.converged:
    st.warning("The solver stopped before fully converging, so this split may be slightly off.")
if result.extrapolated_channels:
    names = ", ".join(label(c) for c in result.extrapolated_channels)
    st.warning(
        f"This plan takes {names} above any weekly spend seen in the data. The model is "
        "extrapolating there; step up gradually."
    )

show(charts.allocation(result.current.spend, result.recommended.spend, weeks))
st.dataframe(
    pd.DataFrame(
        {
            "Channel": [label(c) for c in result.current.spend],
            "Current (₹ L)": [v / LAKH for v in result.current.spend.values()],
            "Recommended (₹ L)": [v / LAKH for v in result.recommended.spend.values()],
            "Change (%)": [
                100 * (result.recommended.spend[c] / v - 1) if v else None
                for c, v in result.current.spend.items()
            ],
            "Caveat": [result.caveats.get(c, "").capitalize() for c in result.current.spend],
        }
    ),
    hide_index=True,
    width="stretch",
    column_config={
        "Channel": st.column_config.TextColumn(pinned=True),
        "Current (₹ L)": st.column_config.NumberColumn(format="%.0f"),
        "Recommended (₹ L)": st.column_config.NumberColumn(format="%.0f"),
        "Change (%)": st.column_config.NumberColumn(format="%+.0f%%"),
        "Caveat": st.column_config.TextColumn(width="large"),
    },
)

st.subheader("Is the total budget the right size?")
show(charts.budget_curve(results["optimizer"]))
st.caption(
    "Each point is the best the model can do with that total budget. Where the curve gets "
    "flatter than the dotted line, an extra ₹1 returns less than ₹1 of revenue."
)
