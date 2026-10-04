"""Overview: is marketing paying for itself, what to change, and where revenue comes from."""

import re

import charts
import streamlit as st
from common import ai_context, chance, crore, label, margin, per_rupee, range_text, setup, show

from mixlab import config
from mixlab.ai_explainer import (
    TONE_PRESETS,
    ExplainerError,
    brief_to_pdf,
    cached_brief,
    generate_brief,
    template_brief,
)
from mixlab.optimizer import OptimizationResult

TONE_LABELS = {"cmo": "CMO", "analyst": "Analyst", "founder": "Founder"}

brand, results = setup(
    "Overview", "What marketing delivered, what to change next, and how sure the model is."
)
insights = results["insights"]
totals = insights["totals"]
baseline = insights["decomposition"]["baseline"]["pct_of_revenue"]
media_revenue, roi, share = (
    totals["media_revenue"],
    totals["blended_media_roi"],
    totals["media_pct_of_revenue"],
)
share_of_margin = margin()
profit = {key: value * share_of_margin for key, value in roi.items()}

verdict = (
    f"each ₹1 of spend returns {per_rupee(profit['mean'])} of gross profit "
    f"({range_text(per_rupee(profit['hdi_low']), per_rupee(profit['hdi_high']))})"
)
if profit["mean"] >= config.PROFIT_BREAKEVEN:
    st.success(f"At a {share_of_margin:.0%} margin, marketing pays for itself: {verdict}.")
else:
    st.warning(
        f"At a {share_of_margin:.0%} margin, marketing does not pay for itself on incremental "
        f"revenue alone: {verdict}. Change the margin in the sidebar to match your product."
    )

first, second, third = st.columns(3)
first.metric("Media spend", crore(totals["media_spend"]))
first.caption(f"Measured over {insights['period']['n_weeks']} weeks")
second.metric("Marketing revenue", crore(media_revenue["mean"]))
second.caption(range_text(crore(media_revenue["hdi_low"]), crore(media_revenue["hdi_high"])))
third.metric("Marketing share of revenue", f"{share['mean']:.0f}%")
third.caption(
    range_text(str(round(share["hdi_low"])) + "%", str(round(share["hdi_high"])) + "%")
    + f" · baseline {baseline['mean']:.0f}%"
)

st.subheader("Return on each ₹1 of spend")
left, right, _ = st.columns(3)
left.metric("ROI: revenue per ₹1", per_rupee(roi["mean"]))
left.caption(range_text(per_rupee(roi["hdi_low"]), per_rupee(roi["hdi_high"])))
right.metric("Profit ROI: profit per ₹1", per_rupee(profit["mean"]))
right.caption(
    range_text(per_rupee(profit["hdi_low"]), per_rupee(profit["hdi_high"]))
    + f" · at a {share_of_margin:.0%} margin · pays for itself above ₹1.00"
)

st.subheader("Recommended reallocation")
recommendation = OptimizationResult.model_validate(results["optimizer"]["expected_revenue"])
uplift = recommendation.uplift
expected, realistic, odds = st.columns(3)
expected.metric("Expected uplift", crore(uplift.mean, 2), f"{recommendation.uplift_pct.mean:+.1f}%")
expected.caption(range_text(crore(uplift.hdi_low, 2), crore(uplift.hdi_high, 2)))
realistic.metric(
    "Realistic uplift",
    crore(recommendation.realistic_uplift, 2),
    f"{recommendation.realistic_uplift_pct:+.1f}%",
)
realistic.caption(f"Expected uplift x {recommendation.shrinkage:.0%}")
odds.metric("Chance it beats current", chance(recommendation.prob_recommended_beats_current))
odds.caption(f"Same budget, next {recommendation.current.n_weeks} weeks")
st.caption(
    "Why two numbers: an optimizer moves money to the channels the model rates highest, and "
    "the highest ratings are disproportionately overestimates. Refitting the model on "
    f"simulated histories shows about {recommendation.shrinkage:.0%} of the expected uplift is "
    "actually delivered, so plan on the realistic figure. Details are on the Budget optimizer "
    "page."
)
if recommendation.corner_solution:
    st.caption(recommendation.corner_note)

st.subheader("Executive summary")
context = ai_context(brand)
picker, action, _ = st.columns([2, 2, 3], vertical_alignment="bottom")
tone = picker.selectbox("Written for", list(TONE_PRESETS), format_func=TONE_LABELS.get)
if action.button("Generate AI brief", help="Calls the Claude API once; the result is cached."):
    try:
        with st.spinner("Writing the brief…"):
            generate_brief(context, tone)
    except ExplainerError as error:
        st.error(str(error))

brief = cached_brief(context, tone)
if brief is not None:
    text = brief.text
    source = f"Written by {brief.model} for the {TONE_LABELS[tone]}. {brief.number_check.summary()}"
else:
    text = template_brief(context.facts, label)
    source = (
        "Standard summary filled in from the model's numbers (no AI). Click Generate AI brief "
        "for a version written for the selected reader."
    )
with st.container(border=True):
    # The brief's own headings are demoted so they sit below the page's section titles.
    st.markdown(re.sub(r"^#{1,6}\s*", "##### ", text, flags=re.MULTILINE))
st.caption(source)


@st.cache_data(show_spinner=False)
def pdf_bytes(markdown: str, title: str) -> bytes:
    """Cache the rendered PDF so reruns do not rebuild it."""
    return brief_to_pdf(markdown, title)


st.download_button(
    "Download brief as PDF",
    pdf_bytes(text, f"MixLab brief · {brand.replace('_', ' ')}"),
    file_name=f"mixlab_brief_{brand}_{tone}.pdf",
    mime="application/pdf",
    icon=":material/download:",
)

st.subheader("Revenue decomposition")
show(charts.decomposition(results["weekly"], list(insights["channels"])))
st.caption(
    "Each band is the model's best estimate of weekly revenue (₹ lakh) from that driver. Bands "
    "below zero (price, seasonal troughs) reduce revenue."
)
