"""Model health: data checks, sampler diagnostics, ground-truth recovery and trust notes."""

import charts
import pandas as pd
import streamlit as st
from common import label, setup, show

from mixlab import config
from mixlab.evaluate import explain_convergence, is_converged, roi_recovery, trust_notes
from mixlab.validate import Severity

brand, results = setup(
    "Model health", "How far these results can be trusted, and where they should not be."
)
report = results["report"]
diagnostics = results["meta"].get("diagnostics")
truth = results["truth"]
recovery = roi_recovery(results["insights"], truth) if truth else None

first, second, third = st.columns(3)
first.metric("Data readiness", f"{report.readiness_score}/{config.MAX_READINESS_SCORE}")
first.caption(
    f"{report.count(Severity.CRITICAL)} critical, {report.count(Severity.WARNING)} warnings"
)
if diagnostics:
    second.metric("Sampler checks", "All passed" if is_converged(diagnostics) else "Problems found")
    second.caption(f"{diagnostics['chains']} chains × {diagnostics['draws_per_chain']} draws")
if recovery is not None:
    hits = int(recovery["truth_inside_range"].sum())
    third.metric("True ROI recovered", f"{hits} of {len(recovery)}")
    third.caption("Channels with the truth inside the model's 94% range")

st.subheader("When not to trust this model")
notes = trust_notes(results["insights"], results["optimizer"], diagnostics, recovery, label)
for start in range(0, len(notes), 2):
    for column, note in zip(st.columns(2), notes[start : start + 2], strict=False):
        with column.container(border=True):
            st.markdown(f"**{note['title']}**")
            st.markdown(note["detail"])

if recovery is not None:
    st.subheader("Ground-truth recovery")
    show(charts.recovery(recovery))
    st.caption(
        "This brand is synthetic, so the true ROI is known. A black tick inside the blue range "
        "means the model's range contains the truth. With real data this check is impossible, "
        "which is why it is run here first."
    )

st.subheader("Sampler diagnostics")
if diagnostics:
    for line in explain_convergence(diagnostics):
        verdict, _, detail = line.partition(" ")
        (st.success if verdict == "PASS" else st.error)(detail)
else:
    st.info("No sampler diagnostics were saved with this model.")

st.subheader("Data checks")
issues = [issue for issue in report.issues if issue.severity != Severity.INFO]
if issues:
    st.dataframe(
        pd.DataFrame(
            {
                "Severity": [i.severity.value.title() for i in issues],
                "Finding": [i.message for i in issues],
            }
        ),
        hide_index=True,
        width="stretch",
    )
else:
    st.success("No critical issues or warnings in the data.")
with st.expander(f"Minor notes ({report.count(Severity.INFO)})"):
    for issue in report.issues:
        if issue.severity == Severity.INFO:
            st.markdown(f"- {issue.message}")
