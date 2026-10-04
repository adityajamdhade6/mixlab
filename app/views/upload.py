"""Upload data: check a CSV against the data contract and score its readiness for an MMM."""

import pandas as pd
import streamlit as st
from common import label, plural, setup

from mixlab import config
from mixlab.validate import Severity, validate

setup(
    "Upload data",
    "Check your own weekly data before modelling. Nothing is stored and nothing is fitted here.",
)
with st.expander("What the file needs to look like"):
    st.markdown(
        f"- One row per week, with a `{config.DATE_COL}` column (week start).\n"
        f"- A `{config.TARGET_COL}` column.\n"
        f"- One `{config.SPEND_PREFIX}<channel>` column per channel.\n"
        "- Optional controls: promo flags, holidays, price index.\n"
        "- All money in INR."
    )

uploaded = st.file_uploader("Weekly data (CSV)", type="csv")
if uploaded is None:
    st.info("Upload a CSV to see its readiness score and anything that needs fixing.")
    example = (
        config.ARTIFACTS_DIR / config.PERFORMANCE_HEAVY_BRAND.name / config.WEEKLY_DATA_FILENAME
    )
    if example.exists():
        st.download_button(
            "Download an example file to try",
            example.read_bytes(),
            file_name="mixlab_example_weekly.csv",
            mime="text/csv",
            icon=":material/download:",
        )
    st.stop()

try:
    frame = pd.read_csv(uploaded)
except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError):
    st.error(
        "This file could not be read as a CSV. Check that it is comma-separated, saved as "
        "UTF-8, and has a header row."
    )
    st.stop()

with st.spinner("Checking the data…"):
    report = validate(frame)

critical, warnings = report.count(Severity.CRITICAL), report.count(Severity.WARNING)
first, second, third = st.columns(3)
first.metric("Readiness score", f"{report.readiness_score}/{config.MAX_READINESS_SCORE}")
second.metric("Weeks of data", report.n_weeks)
third.metric("Channels found", len(report.channels))
second.caption("At least 104 weeks recommended")
if frame.empty:
    st.error("The file has a header but no rows of data.")
    st.stop()
if critical:
    st.error(
        f"{plural(critical, 'critical issue')} must be fixed before this data can be modelled."
    )
elif warnings:
    st.warning(
        f"Usable, with {plural(warnings, 'warning')} that will make some estimates less certain."
    )
else:
    st.success("This data is ready to model.")

if report.issues:
    st.dataframe(
        pd.DataFrame(
            {
                "Severity": [issue.severity.value.title() for issue in report.issues],
                "Finding": [issue.message for issue in report.issues],
            }
        ),
        hide_index=True,
        width="stretch",
    )
if report.spend_share:
    st.subheader("Per channel")
    st.dataframe(
        pd.DataFrame(
            {
                "Channel": [label(c) for c in report.channels],
                "Share of spend (%)": [100 * report.spend_share.get(c, 0) for c in report.channels],
                "Spend variation": [report.spend_cv.get(c) for c in report.channels],
                "Overlap with other channels (VIF)": [report.vif.get(c) for c in report.channels],
            }
        ).round(2),
        hide_index=True,
        width="stretch",
    )

st.subheader("Next step")
st.markdown("Model training runs offline for now. Save the file and run:")
st.code(
    "uv run python scripts/train.py --data path/to/your.csv --config configs/default.yaml",
    language="bash",
)
