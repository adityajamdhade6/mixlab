"""Overview: headline numbers, the executive summary and where revenue comes from."""

import re

import charts
import streamlit as st
from common import ai_context, crore, label, likely, setup, show

from mixlab.ai_explainer import (
    TONE_PRESETS,
    ExplainerError,
    brief_to_pdf,
    cached_brief,
    generate_brief,
    template_brief,
)

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

first, second, third, fourth = st.columns(4)
first.metric("Media spend", crore(totals["media_spend"]))
first.caption(f"Measured over {insights['period']['n_weeks']} weeks")
second.metric("Marketing revenue", crore(media_revenue["mean"]))
second.caption(likely(crore(media_revenue["hdi_low"]), crore(media_revenue["hdi_high"])))
third.metric("Blended ROI", f"{roi['mean']:.2f}")
third.caption(likely(f"{roi['hdi_low']:.2f}", f"{roi['hdi_high']:.2f}"))
fourth.metric("Marketing share", f"{share['mean']:.0f}%")
share_range = likely(f"{share['hdi_low']:.0f}%", f"{share['hdi_high']:.0f}%")
fourth.caption(f"{share_range} · baseline {baseline['mean']:.0f}%")

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
    "Each band is the model's best estimate of weekly revenue from that driver. Bands below "
    "zero (price, seasonal troughs) reduce revenue."
)
