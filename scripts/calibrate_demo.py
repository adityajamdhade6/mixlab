"""Show what a lift test buys: ROI before vs. after calibration on a synthetic brand.

Uses the uncalibrated demo fit in ``artifacts/<brand>/`` as "before", asks the test planner
which channel to test, simulates that test from ground truth, refits with it, and compares.

Example:
    uv run python scripts/calibrate_demo.py --brand performance_heavy

"""

import argparse
import json
from pathlib import Path

import pandas as pd

import mixlab  # noqa: F401  (sets the PyTensor backend before PyMC is imported)
from mixlab import config
from mixlab.calibration import (
    compare_roi,
    fit_calibrated,
    plan_tests,
    plot_before_after,
    simulate_lift_test,
)
from mixlab.config import ModelSettings
from mixlab.evaluate import convergence_diagnostics, is_converged
from mixlab.insights import build_summary, extract_draws


def main() -> None:
    """Run the before/after comparison and save the table, figure and experiment file."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--brand", default=config.PERFORMANCE_HEAVY_BRAND.name)
    parser.add_argument("--start", default="2025-03-03")
    parser.add_argument("--end", default="2025-04-21", help="Monday of the last test week.")
    parser.add_argument("--config", type=Path, default=config.PROJECT_ROOT / "configs/demo.yaml")
    args = parser.parse_args()

    folder = config.ARTIFACTS_DIR / args.brand
    df = pd.read_csv(folder / config.WEEKLY_DATA_FILENAME, parse_dates=[config.DATE_COL])
    before = json.loads((folder / config.INSIGHTS_SUMMARY_FILENAME).read_text())
    truth = json.loads((folder / config.GROUND_TRUTH_FILENAME).read_text())

    plan = plan_tests(before, df)
    print("== Test planner ==")
    print(plan.round(2).to_string(index=False))
    channel = plan.loc[0, "channel"]

    experiment = simulate_lift_test(
        config.BRAND_PRESETS[args.brand], df, channel, args.start, args.end
    )
    pd.DataFrame([experiment.model_dump()]).to_csv(folder / "experiments.csv", index=False)
    crore = config.INR_PER_CRORE
    print(
        f"\n== Simulated lift test on {channel} ({experiment.weeks:.0f} weeks) ==\n"
        f"Measured incremental revenue {experiment.incremental_revenue / crore:.2f} Cr "
        f"(standard error {experiment.standard_error / crore:.2f} Cr)"
    )

    model = fit_calibrated(df, ModelSettings.from_yaml(args.config), [experiment])
    diagnostics = convergence_diagnostics(model.idata, model.parameter_names)
    after = build_summary(extract_draws(model, df))
    table = compare_roi(before, after, truth)
    print(f"\n== ROI before vs. after (sampler healthy: {is_converged(diagnostics)}) ==")
    print(table.round(2).to_string(index=False))

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_before_after(table, [channel]).savefig(
        config.FIGURES_DIR / config.CALIBRATION_FIGURE, dpi=config.FIGURE_DPI
    )
    summary = {
        "brand": args.brand,
        "tested_channel": channel,
        "experiment": json.loads(experiment.model_dump_json()),
        "test_plan": plan.to_dict(orient="records"),
        "roi_before_after": table.to_dict(orient="records"),
        "diagnostics": diagnostics,
    }
    path = config.REPORTS_DIR / config.CALIBRATION_SUMMARY_FILENAME
    path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
