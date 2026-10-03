"""Run the full pipeline on a public dataset and write the results to a report.

Dataset: ``mmm_example.csv`` from the pymc-marketing repository (179 simulated weeks, two
channels, two event flags), stored at ``data/public/pymc_marketing_mmm_example.csv``.

Example:
    uv run python scripts/public_dataset_test.py

"""

import numpy as np
import pandas as pd

import mixlab  # noqa: F401  (sets the PyTensor backend before PyMC is imported)
from mixlab import config
from mixlab.config import ModelSettings, SamplerSettings
from mixlab.evaluate import convergence_diagnostics, explain_convergence
from mixlab.insights import build_summary, extract_draws
from mixlab.model import MixLabModel
from mixlab.onboarding import map_generic
from mixlab.validate import validate

SOURCE = "https://github.com/pymc-labs/pymc-marketing/blob/main/data/mmm_example.csv"
DATA_FILE = config.PUBLIC_DATA_DIR / "pymc_marketing_mmm_example.csv"
REPORT_FILE = config.REPORTS_DIR / "public_dataset_test.md"
# Values used to simulate the dataset, as stated in the pymc-marketing MMM example notebook.
DOCUMENTED_DECAY = {"x1": 0.4, "x2": 0.2}
TARGET_ACCEPT = 0.95


def main() -> None:
    """Map, validate, fit and report."""
    frame = map_generic(
        pd.read_csv(DATA_FILE),
        date_column="date_week",
        revenue_column="y",
        spend_columns={"x1": "x1", "x2": "x2"},
        control_columns={"event_1": "holiday_event_1", "event_2": "holiday_event_2"},
    )
    report = validate(frame)
    settings = ModelSettings(sampler=SamplerSettings(target_accept=TARGET_ACCEPT))
    model = MixLabModel().build(frame, settings)
    model.fit(progressbar=False)
    diagnostics = convergence_diagnostics(model.idata, model.parameter_names)
    draws = extract_draws(model, frame)
    summary = build_summary(draws)
    prediction = model.predict(frame)
    actual = frame[config.TARGET_COL]
    mape = float((prediction["mean"] - actual).abs().div(actual).mean() * 100)
    covered = float(
        ((actual >= prediction["lower"]) & (actual <= prediction["upper"])).mean() * 100
    )

    lines = [
        "# Public dataset test",
        "",
        f"Source: [{SOURCE.rsplit('/', 1)[-1]}]({SOURCE}) from the pymc-marketing repository "
        "(simulated data; values are unitless, not INR).",
        "",
        "## Mapping",
        "`date_week` to `date`, `y` to `revenue`, `x1`/`x2` to `spend_x1`/`spend_x2`, and "
        "`event_1`/`event_2` to `holiday_event_1`/`holiday_event_2`. `t` and `dayofyear` were "
        "dropped because the model builds its own trend and seasonality.",
        "",
        "## Validation",
        f"Readiness score **{report.readiness_score}/100** over {report.n_weeks} weeks and "
        f"{len(report.channels)} channels.",
        "",
        *[f"- {issue.severity.value}: {issue.message}" for issue in report.issues],
        "",
        "## Fit",
        f"Default priors, no channel-specific beliefs, target_accept {TARGET_ACCEPT}.",
        "",
        *[f"- {line}" for line in explain_convergence(diagnostics)],
        f"- In-sample error (MAPE): **{mape:.1f}%**. {covered:.0f}% of weeks fall inside the "
        "model's 94% predictive range.",
        "",
        "## Results",
        "",
        "| Channel | Adstock decay (94% range) | Documented decay | Share of revenue (94% range) |",
        "|---|---|---|---|",
    ]
    for channel, metrics in summary["channels"].items():
        decay, share = metrics["adstock_decay"], metrics["contribution_pct"]
        truth = DOCUMENTED_DECAY[channel]
        inside = "inside" if decay["hdi_low"] <= truth <= decay["hdi_high"] else "outside"
        lines.append(
            f"| {channel} | {decay['mean']:.2f} ({decay['hdi_low']:.2f} to "
            f"{decay['hdi_high']:.2f}) | {truth} ({inside} range) | {share['mean']:.1f}% "
            f"({share['hdi_low']:.1f}% to {share['hdi_high']:.1f}%) |"
        )
    media = summary["totals"]["media_pct_of_revenue"]
    lines += [
        "",
        f"Media drives {media['mean']:.0f}% of the target "
        f"({media['hdi_low']:.0f}% to {media['hdi_high']:.0f}%).",
        "",
        "The documented decay values are the ones the pymc-marketing example notebook says "
        "were used to simulate the data. ROI is not reported because `x1` and `x2` are scaled "
        "activity levels, not spend in currency.",
    ]
    REPORT_FILE.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    assert np.isfinite(mape)


if __name__ == "__main__":
    main()
