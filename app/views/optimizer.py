"""Budget optimizer: budget, objective and limits in; a plan with its range and rollout out."""

import charts
import pandas as pd
import streamlit as st
from common import (
    chance,
    crore,
    label,
    load_benchmark,
    load_runtime,
    margin,
    range_text,
    setup,
    show,
)

from mixlab import config
from mixlab.optimizer import (
    OptimizationResult,
    channel_limits,
    last_quarter_spend,
    optimize_budget,
    optimize_goal,
    rollout_plan,
    weekly_plan,
)

LAKH = config.INR_PER_LAKH
CRORE = config.INR_PER_CRORE
OBJECTIVES = {
    "Expected revenue": "mean",
    "Expected profit (uses the margin)": "profit",
    "Risk-adjusted (penalise uncertainty)": "risk_adjusted",
    f"Cautious ({config.RISK_PERCENTILE:g}th percentile)": "percentile",
}

brand, results = setup(
    "Budget optimizer",
    "Choose a budget and how far each channel may move. The model finds the best split.",
)
runtime = load_runtime(brand)
draws = runtime.draws
weeks = runtime.allocator.n_weeks
current = last_quarter_spend(draws, weeks)
limits = channel_limits(draws, weeks)
saved = results["optimizer"]
haircut = (saved.get("optimism") or {}).get("shrinkage", config.UPLIFT_SHRINKAGE)
total = sum(current.values())
state_key = f"optimizer_result_{brand}"

with st.expander("Why each channel has the default limit it does", expanded=False):
    st.dataframe(
        pd.DataFrame(
            {
                "Channel": [label(c) for c in limits],
                "Default limit": [f"±{info['max_change']:.0%}" for info in limits.values()],
                "Why": [info["reason"].capitalize() + "." for info in limits.values()],
            }
        ),
        hide_index=True,
        width="stretch",
        column_config={"Why": st.column_config.TextColumn(width="large")},
    )
    st.caption(
        "Limits are set from the evidence: tighter where a channel is hard to measure or its "
        "ROI range is wide. No channel is pushed above its highest week on record by default."
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
    objective = right.selectbox(
        "Optimise for",
        list(OBJECTIVES),
        help="Every option scores a plan across the model's uncertainty, not a single best "
        "guess. Risk-adjusted and Cautious give up some expected revenue for a plan the model "
        "is surer about.",
    )
    st.markdown("**Allowed spend per channel (₹ lakh)**")
    chosen: dict[str, tuple[float, float]] = {}
    columns = st.columns(3)
    for index, (channel, spend) in enumerate(current.items()):
        value = spend / LAKH
        change = limits[channel]["max_change"]
        # Small channels need a finer step, or a ±10% range rounds away to nothing.
        step = 0.1 if value < 20 else 1.0
        ceiling = min(value * (1 + change), limits[channel]["ceiling"] / LAKH)
        low, high = (round(v / step) * step for v in (value * (1 - change), ceiling))
        chosen[channel] = columns[index % 3].slider(
            f"{label(channel)} (₹{value:,.0f} L) · ±{change:.0%}",
            min_value=0.0,
            max_value=float(max(round(3 * value), 10)),
            value=(float(low), float(max(high, low))),
            step=step,
            format="%.1f" if step < 1 else "%.0f",
            help=f"Last quarter ₹{value:,.1f} lakh. Default limit ±{change:.0%}: "
            f"{limits[channel]['reason']}.",
        )
    submitted = st.form_submit_button("Run optimization", type="primary")

if submitted:
    try:
        with st.spinner("Finding the best allocation…"):
            st.session_state[state_key] = optimize_budget(
                weeks,
                draws,
                total_budget=budget,
                minimum={c: low * LAKH for c, (low, _) in chosen.items()},
                maximum={c: high * LAKH for c, (_, high) in chosen.items()},
                objective=OBJECTIVES[objective],
                margin=margin(),
                shrinkage=haircut,
            )
    except ValueError as error:
        st.session_state.pop(state_key, None)
        st.error(f"That combination cannot be optimised. {error}")

result = st.session_state.get(state_key)
is_saved = result is None
if is_saved:
    result = OptimizationResult.model_validate(saved["expected_revenue"])
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
optimism = saved.get("optimism") or {}
haircut_range = (
    f" (it could plausibly be {optimism['shrinkage_low']:.0%} to {optimism['shrinkage_high']:.0%}; "
    "it is estimated from six refits)"
    if "shrinkage_low" in optimism
    else ""
)
st.caption(
    f"Realistic uplift is the expected uplift x {result.shrinkage:.0%}{haircut_range}. "
    "Optimizers favour the channels a model happens to overestimate; refitting the model on "
    "simulated histories and re-optimizing shows that share of the expected uplift is actually "
    "delivered."
)

if result.corner_solution:
    st.info(result.corner_note)
widest = max(
    (abs(result.recommended.spend[c] / v - 1) for c, v in result.current.spend.items() if v),
    default=0.0,
)
if widest > config.DEFAULT_MAX_CHANGE + 0.005:
    st.warning(
        f"This plan moves a channel by {widest:.0%}. The realistic-uplift haircut was estimated "
        f"with moves capped at {config.DEFAULT_MAX_CHANGE:.0%}; larger moves lean harder on the "
        "model's curves, so treat the figures above as optimistic."
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

st.subheader("How to roll it out")
steps = saved["rollout"] if is_saved else rollout_plan(draws, result)
st.dataframe(
    pd.DataFrame(
        {
            "Step": [f"{s['step']} of {len(steps)}" for s in steps],
            "Weeks": [s["weeks"] for s in steps],
            "Share of the change": [f"{s['share_of_change']:.0%}" for s in steps],
            "Expected revenue change (₹ L / week)": [
                f"{s['expected_weekly_revenue_change']['mean'] / LAKH:+.1f} "
                f"({s['expected_weekly_revenue_change']['hdi_low'] / LAKH:+.1f} to "
                f"{s['expected_weekly_revenue_change']['hdi_high'] / LAKH:+.1f})"
                for s in steps
            ],
            "Confirm if": [s["confirm"] for s in steps],
            "Stop if": [
                "Average weekly revenue over the step runs more than "
                f"₹{2 * s['weekly_noise'] / config.ROLLOUT_WEEKS_PER_STEP**0.5 / LAKH:,.1f} L "
                "below forecast."
                for s in steps
            ],
        }
    ),
    hide_index=True,
    width="stretch",
    column_config={
        "Confirm if": st.column_config.TextColumn(width="large"),
        "Stop if": st.column_config.TextColumn(width="large"),
    },
)
st.caption(
    f"Weekly revenue normally moves by about ₹{steps[0]['weekly_noise'] / LAKH:,.1f} lakh for "
    "reasons unrelated to marketing, so small steps cannot be confirmed from revenue alone."
)

bursts = saved.get("burst_minimum", {})
schedule = (
    pd.DataFrame(saved["weekly_plan"])
    if is_saved
    else weekly_plan(draws, result.recommended.spend, weeks, bursts)
)
show(charts.weekly_schedule(schedule))
st.caption(
    "Each channel's total is either spread evenly or concentrated into one burst, whichever "
    "the model expects to earn more once carryover is counted. Burst channels keep at least "
    "their smallest historical on-air week. The model does not know which calendar weeks are "
    "better for media, so place bursts against festivals yourself."
)

with st.expander("Other goals: hit a revenue number or a target ROI"):
    goal = st.radio(
        "Goal",
        ["Lowest spend for a revenue target", "Largest plan at a target ROI"],
        horizontal=True,
    )
    now = result.current.incremental_revenue.mean
    if goal.startswith("Lowest"):
        target = CRORE * st.number_input(
            f"Marketing revenue to reach over {weeks} weeks (₹ crore)",
            min_value=0.0,
            value=round(now / CRORE, 2),
            step=0.1,
        )
        arguments = {"revenue_target": target}
    else:
        target = st.number_input(
            "Minimum ROI (₹ revenue per ₹1 spent)",
            min_value=0.0,
            value=round(result.current.roi.mean, 2),
            step=0.05,
        )
        arguments = {"roi_target": target}
    if st.button("Find the plan"):
        try:
            with st.spinner("Searching…"):
                plan = optimize_goal(draws, weeks, max_change=0.5, **arguments)
        except ValueError as error:
            st.error(f"No plan can meet that goal. {error}")
        else:
            one, two, three = st.columns(3)
            one.metric(
                "Spend",
                crore(plan.recommended.total_spend, 2),
                f"{(plan.recommended.total_spend - plan.current.total_spend) / CRORE:+.2f} Cr",
            )
            revenue = plan.recommended.incremental_revenue
            two.metric("Marketing revenue", crore(revenue.mean, 2))
            two.caption(range_text(crore(revenue.hdi_low, 2), crore(revenue.hdi_high, 2)))
            three.metric("ROI", f"₹{plan.recommended.roi.mean:.2f} per ₹1")
            three.caption(f"Chance the goal is met: {chance(plan.goal_probability)}")
            show(charts.allocation(plan.current.spend, plan.recommended.spend, weeks))
            st.caption("Channels may move up to 50% either way for this search.")

benchmark = load_benchmark()
if benchmark:
    st.subheader("How this optimizer did when the truth was known")
    naive, robust = benchmark["naive"], benchmark["robust"]
    st.dataframe(
        pd.DataFrame(
            {
                "Optimizer": ["Naive (no limits, no haircut)", "This one (limits + haircut)"],
                "Promised uplift (%)": [
                    naive["mean_promised_uplift_pct"],
                    robust["mean_promised_uplift_pct"],
                ],
                "True uplift (%)": [naive["mean_true_uplift_pct"], robust["mean_true_uplift_pct"]],
                "Brands made worse (%)": [
                    100 * naive["share_worse_than_current"],
                    100 * robust["share_worse_than_current"],
                ],
                "Worst case (%)": [naive["worst_true_uplift_pct"], robust["worst_true_uplift_pct"]],
            }
        ),
        hide_index=True,
        width="stretch",
        column_config={
            "Promised uplift (%)": st.column_config.NumberColumn(format="%+.1f"),
            "True uplift (%)": st.column_config.NumberColumn(format="%+.1f"),
            "Brands made worse (%)": st.column_config.NumberColumn(format="%.0f"),
            "Worst case (%)": st.column_config.NumberColumn(format="%+.1f"),
        },
    )
    st.caption(
        f"Average over {benchmark['brands']} randomly generated brands, each scored against its "
        "true response curves. The cautious optimizer gives up some average uplift in exchange "
        "for promises it keeps and far smaller losses when the model is wrong."
    )
