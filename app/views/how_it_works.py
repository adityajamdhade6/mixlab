"""How it works: the pipeline in one diagram and marketing mix modelling in five lines."""

import streamlit as st
from common import header

from mixlab import config

PIPELINE = [
    ("Weekly data", "Spend per channel, revenue, promotions, holidays and price."),
    ("Validation", "Checks the data and scores how ready it is for modelling."),
    ("Bayesian MMM", "Estimates each channel's effect with carryover and diminishing returns."),
    ("Insights", "ROI, what the next ₹1 earns, and the range around each."),
    ("Optimizer", "Finds the best split of a budget and simulates what-if plans."),
    ("AI brief", "Explains the result in plain English; every number is checked."),
]

header()
st.title("How it works")
st.caption("What MixLab does with the data, in the order it does it.")
for start in range(0, len(PIPELINE), 3):
    for offset, (column, (title, detail)) in enumerate(
        zip(st.columns(3), PIPELINE[start : start + 3], strict=True)
    ):
        with column.container(border=True):
            step = start + offset + 1
            arrow = "" if step == len(PIPELINE) else " →"
            st.markdown(f"**{step}. {title}**{arrow}")
            st.caption(detail)

st.subheader("Marketing mix modelling in five lines")
st.markdown(
    """
1. Every week, revenue is the sum of a **baseline** (what you would have sold anyway) and
   what each **marketing channel** added.
2. The model looks at three years of weeks where spend went up and down, and works out how
   much revenue moved with each channel.
3. It allows for **carryover** (an ad keeps working for weeks) and **diminishing returns**
   (the tenth lakh buys less than the first).
4. Because many explanations fit the same data, every answer comes as a **94% range**, not a
   single number.
5. The optimizer then asks: given those curves, which split of the same budget earns the
   most, and how sure are we?
"""
)

st.subheader("Why not just use platform reports?")
st.markdown(
    "Platform and last-click reports credit a sale to an ad that was touched. They cannot see "
    "sales that would have happened anyway, so every channel looks better than it is. The "
    "Channel performance page shows the size of that gap for each channel."
)

st.subheader("How we know whether to believe it")
st.markdown(
    "These demo brands are synthetic: the true effect of every channel is known. The Model "
    "health page scores the model against that truth, against weeks it never saw, and lists "
    "the situations where it should not be trusted."
)
st.markdown(
    f"[Read the case study]({config.CASE_STUDY_URL}) · [Browse the code]({config.REPO_URL})"
)
