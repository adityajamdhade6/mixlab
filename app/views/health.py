"""Model health: out-of-sample accuracy, ground-truth recovery, diagnostics and trust notes."""

import charts
import pandas as pd
import streamlit as st
from common import label, plural, setup, show

from mixlab import config
from mixlab.evaluate import explain_convergence, is_converged, roi_recovery, trust_notes
from mixlab.validate import Severity

brand, results = setup(
    "Model health", "How far these results can be trusted, and where they should not be."
)
report = results["report"]
diagnostics = results["meta"].get("diagnostics")
truth = results["truth"]
backtest = results["backtest"]
recovery = roi_recovery(results["insights"], truth) if truth else None

first, second, third, fourth = st.columns(4)
first.metric("Data readiness", f"{report.readiness_score} / {config.MAX_READINESS_SCORE}")
first.caption(
    f"{plural(report.count(Severity.CRITICAL), 'critical issue')}, "
    f"{plural(report.count(Severity.WARNING), 'warning')}"
)
if backtest:
    holdout = backtest["holdout"]
    second.metric("Holdout error (MAPE)", f"{holdout['mape_pct']:.1f}%")
    second.caption(f"Last {backtest['horizon_weeks']} weeks, never shown to the model")
if diagnostics:
    third.metric("Sampler checks", "Passed" if is_converged(diagnostics) else "Problems")
    third.caption(f"{diagnostics['chains']} chains x {diagnostics['draws_per_chain']:,} draws")
if recovery is not None:
    hits = int(recovery["truth_inside_range"].sum())
    fourth.metric("True ROI recovered", f"{hits} of {len(recovery)}")
    fourth.caption("Channels with the truth inside the model's 94% range")

st.subheader("When not to trust this model")
notes = trust_notes(
    results["insights"], results["optimizer"], diagnostics, recovery, label, backtest
)
for start in range(0, len(notes), 2):
    for column, note in zip(st.columns(2), notes[start : start + 2], strict=False):
        with column.container(border=True):
            st.markdown(f"**{note['title']}**")
            st.markdown(note["detail"])

st.subheader("Out-of-sample accuracy")
if backtest:
    holdout = backtest["holdout"]
    left, middle, right = st.columns(3)
    left.metric("Holdout MAPE", f"{holdout['mape_pct']:.1f}%")
    left.caption("Average miss as a share of actual weekly revenue")
    middle.metric("Holdout R²", f"{holdout['r2']:.2f}")
    middle.caption("Share of week-to-week variation explained (1.00 is perfect)")
    right.metric("Weeks inside the 94% range", f"{holdout['coverage_pct']:.0f}%")
    right.caption(f"{holdout['test_start']} to {holdout['test_end']}")
    show(charts.actual_vs_predicted(holdout["weeks"]))
    st.markdown("**Rolling backtest**")
    st.dataframe(
        pd.DataFrame(
            {
                "Trained on": [f"{fold['train_weeks']} weeks" for fold in backtest["folds"]],
                "Predicted": [
                    f"{fold['test_start']} to {fold['test_end']}" for fold in backtest["folds"]
                ],
                "MAPE (%)": [fold["mape_pct"] for fold in backtest["folds"]],
                "R²": [fold["r2"] for fold in backtest["folds"]],
                "Weeks inside 94% range (%)": [fold["coverage_pct"] for fold in backtest["folds"]],
            }
        ),
        hide_index=True,
        width="stretch",
        column_config={
            "MAPE (%)": st.column_config.NumberColumn(format="%.1f"),
            "R²": st.column_config.NumberColumn(format="%.2f"),
            "Weeks inside 94% range (%)": st.column_config.NumberColumn(format="%.0f"),
        },
    )
    st.caption(
        "Each row refits the model on an earlier cut of the data and predicts the following "
        f"{backtest['horizon_weeks']} weeks. Good prediction shows the model tracks revenue; it "
        "does not by itself prove the split between channels is right."
    )
else:
    st.info("No backtest was saved with this model. Run `make train` to produce one.")

if recovery is not None:
    st.subheader("Ground-truth recovery")
    show(charts.recovery(recovery))
    st.caption(
        "This brand is synthetic, so the true ROI is known. A black tick inside the blue line "
        "means the model's 94% range contains the truth. With real data this check is "
        "impossible, which is why it is run here first."
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
with st.expander(plural(report.count(Severity.INFO), "minor note")):
    for issue in report.issues:
        if issue.severity == Severity.INFO:
            st.markdown(f"- {issue.message}")
