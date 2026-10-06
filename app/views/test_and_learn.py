"""Test and learn: what to test, how to test it, what the test found, and what changed."""

import charts
import pandas as pd
import streamlit as st
from common import (
    crore,
    header,
    label,
    lakh,
    load_experiment_loop,
    load_geo,
    range_text,
    show,
    use_hosted_secret,
)

from mixlab import config

LOOP_COMMAND = "uv run python scripts/experiment_loop.py"

use_hosted_secret()
header(config.GEO_DEMO_BRAND)
st.title("Test and learn")
st.caption(
    "The model is unsure about some channels. This page follows one full loop on the regional "
    "demo brand: choose what to test, design the test, run it (simulated from the true "
    "process), measure it, and feed the result back into the model."
)

story = load_experiment_loop()
if story is None:
    st.error(
        "The test-and-learn loop has not been run yet, so there is nothing to show. Run it "
        "once with the command below (about five minutes), then reload this page."
    )
    st.code(LOOP_COMMAND, language="bash")
    st.stop()

regions = (load_geo().get("truth") or {}).get("regions", {})


def region_names(names: list[str]) -> str:
    """Return region ids as a readable list of names."""
    return ", ".join(regions.get(r, {}).get("label", r) for r in names)


st.subheader("1. What to test first")
st.markdown(story["method_note"])
show(charts.value_of_tests(story["value_of_information"]))
top = story["value_of_information"][0]
st.caption(
    f"Testing {label(top['channel'])} is worth the most: it cuts the expected cost of a wrong "
    f"call from {lakh(top['expected_loss_now'], 2)} to "
    f"{lakh(top['expected_loss_after_test'], 2)} over 13 weeks."
)

st.subheader("2. How to test it")
for test in story["tests"]:
    channel = label(test["channel"])
    design = test["geo_design"]
    if test["kind"] == "holdout":
        largest = design["options"][-1]
        weeks = largest["weeks_needed"]
        st.markdown(
            f"**{channel}: a geo test would not work.** Even at "
            f"{largest['multiplier']:g}x spend in the best-matched regions, the extra revenue "
            f"is too small to see in regional sales within {config.MAX_TEST_WEEKS} weeks "
            f"(it would need {weeks if weeks else 'far more'} weeks). Small channels are "
            f"measured with a user-level holdout instead: the platform keeps ads from a random "
            f"share of customers for {test['weeks']} weeks and compares the two groups."
        )
    else:
        st.markdown(f"**{channel}: a geo-lift test.**")
        with st.container(border=True):
            st.markdown(test["plan_text"])
        options = pd.DataFrame(
            [
                {
                    "Spend in test regions": f"{o['multiplier']:g}x",
                    "Extra a week": lakh(o["extra_weekly_spend"], 1),
                    "Weeks for 80% power": o["weeks_needed"] or "never",
                    "Fits in a test?": "Yes" if o["feasible"] else "No",
                }
                for o in design["options"]
            ]
        )
        st.dataframe(options, hide_index=True, width="stretch")

st.subheader("3. What the tests found")
for test in story["tests"]:
    channel = label(test["channel"])
    low = test["measured"] - config.Z_95 * test["standard_error"]
    high = test["measured"] + config.Z_95 * test["standard_error"]
    left, right = st.columns(2)
    left.metric(f"{channel}: measured extra revenue", crore(test["measured"], 2))
    left.caption(range_text(crore(low, 2), crore(high, 2)).replace("94%", "95%"))
    if test["kind"] == "geo":
        analysis = test["analysis"]
        right.metric("True extra revenue (synthetic data)", crore(test["true_lift"], 2))
        verdict = "passes" if analysis["passes_placebo"] else "does not pass"
        if abs(analysis["z_score"]) < config.Z_95:
            st.warning(
                f"The {channel} test was inconclusive: its range includes zero. It was sized "
                "on the model's own estimate of the channel's return, which was too low, so "
                "it ran too short. It still narrows the model's range, but on its own it "
                "would not justify a budget move. Extend it or repeat it before acting."
            )
        fakes = len(analysis["placebo_z_scores"])
        right.caption(
            f"Placebo check {verdict}: p = {analysis['placebo_p_value']:.2f} across {fakes} "
            f"fake tests on {region_names(analysis['control_regions'])}."
        )
        show(charts.test_gap(analysis))
        st.caption(
            f"Difference-in-differences cross-check: {crore(analysis['did_effect'], 2)}. "
            f"Before the test the synthetic control tracked the test regions within "
            f"{analysis['pre_period_mape_pct']:.1f}% a week."
        )
    else:
        right.metric("Measured by", "Holdout study")
        right.caption("Simulated from the true process with a 10% measurement error.")

st.subheader("4. What changed in the model")
rows = story["roi_before_after"]
tested = [t["channel"] for t in story["tests"]]
show(charts.calibration_ranges(rows, tested))
for row in rows:
    if row["channel"] in tested:
        st.markdown(
            f"- **{label(row['channel'])}:** ROI likely between {row['before_low']:.2f} and "
            f"{row['before_high']:.2f} before, {row['after_low']:.2f} and "
            f"{row['after_high']:.2f} after ({row['range_shrink_pct']:.0f}% narrower; "
            f"true {row['true_roi']:.2f})."
        )

st.subheader("5. What changed in the recommendation")
before, after = story["recommendation_before"], story["recommendation_after"]
table = pd.DataFrame(
    [
        {
            "Channel": label(channel),
            "Today": lakh(spend),
            "Recommended before tests": lakh(before["recommended"][channel]),
            "Recommended after tests": lakh(after["recommended"][channel]),
            "Limit after tests": f"±{100 * after['limits'][channel]:.0f}%",
        }
        for channel, spend in before["current"].items()
    ]
)
st.dataframe(table, hide_index=True, width="stretch")
st.caption(
    "Same budget, 13 weeks. A channel measured by an experiment no longer carries the "
    "'too small to measure' or 'ran in bursts' caveat, so its limit is set by its calibrated "
    "range instead; it still never goes above its highest week on record. "
    f"Expected uplift: {crore(before['uplift']['mean'], 2)} before, "
    f"{crore(after['uplift']['mean'], 2)} after."
)
