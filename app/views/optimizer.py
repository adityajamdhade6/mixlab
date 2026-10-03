"""Budget optimizer: set a budget and limits, get a recommended allocation with its range."""

import charts
import pandas as pd
import streamlit as st
from common import crore, label, likely, load_runtime, setup, show

from mixlab import config
from mixlab.optimizer import OptimizationResult, last_quarter_spend, optimize_budget

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
total = sum(current.values())
state_key = f"optimizer_result_{brand}"

with st.form("optimizer"):
    left, right = st.columns(2)
    budget = LAKH * left.number_input(
        f"Total budget for the next {weeks} weeks (₹ lakh)",
        min_value=0.0,
        value=float(round(total / LAKH)),
        step=10.0,
        help=f"Last quarter's spend was {total / LAKH:,.0f} lakh.",
    )
    objective = right.radio(
        "Optimise for",
        list(OBJECTIVES),
        horizontal=True,
        help="Cautious maximises the revenue you would still get in a bad case, so it favours "
        "channels whose effect is well established.",
    )
    st.markdown("**Allowed spend per channel (₹ lakh)**")
    st.caption(
        f"Defaults allow each channel to move {config.DEFAULT_MAX_CHANGE:.0%} either way "
        "from last quarter."
    )
    limits: dict[str, tuple[float, float]] = {}
    columns = st.columns(3)
    for index, (channel, spend) in enumerate(current.items()):
        value = spend / LAKH
        limits[channel] = columns[index % 3].slider(
            f"{label(channel)} · last quarter {value:,.0f}",
            min_value=0.0,
            max_value=float(max(round(3 * value), 10)),
            value=(
                float(round(value * (1 - config.DEFAULT_MAX_CHANGE))),
                float(round(value * (1 + config.DEFAULT_MAX_CHANGE))),
            ),
            step=1.0,
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
    st.info(
        "Showing the saved recommendation for last quarter's budget. Change the inputs and "
        "run the optimization to replace it."
    )

uplift, uplift_pct = result.uplift, result.uplift_pct
revenue = result.recommended.incremental_revenue
first, second, third = st.columns(3)
first.metric("Expected uplift", f"{crore(uplift.mean, 2)}", f"{uplift_pct.mean:+.1f}% vs. current")
first.caption(likely(crore(uplift.hdi_low, 2), crore(uplift.hdi_high, 2)))
second.metric("Chance it beats the current plan", f"{result.prob_recommended_beats_current:.0%}")
second.caption(f"Optimised for {result.objective}")
third.metric("Revenue from marketing", crore(revenue.mean, 2))
third.caption(likely(crore(revenue.hdi_low, 2), crore(revenue.hdi_high, 2)))

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
        }
    ),
    hide_index=True,
    width="stretch",
    column_config={
        "Current (₹ L)": st.column_config.NumberColumn(format="%.0f"),
        "Recommended (₹ L)": st.column_config.NumberColumn(format="%.0f"),
        "Change (%)": st.column_config.NumberColumn(format="%+.0f%%"),
    },
)

st.subheader("Is the total budget the right size?")
show(charts.budget_curve(results["optimizer"]))
st.caption(
    "Each point is the best the model can do with that total budget. Where the curve gets "
    "flatter than the dotted line, an extra rupee returns less than a rupee of revenue."
)
