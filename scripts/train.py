"""Train the MixLab MMM: validate, check priors, fit, diagnose and save.

Example:
    uv run python scripts/train.py --data data/synthetic/mmm_weekly.csv \
        --config configs/default.yaml

"""

import argparse
import sys
from pathlib import Path

import pandas as pd

import mixlab  # noqa: F401  (sets the PyTensor backend before PyMC is imported)
from mixlab import config
from mixlab.config import ModelSettings
from mixlab.evaluate import convergence_diagnostics, explain_convergence
from mixlab.model import (
    MixLabModel,
    interval_summary,
    plot_prior_predictive,
    prior_predictive_check,
)
from mixlab.validate import Severity, validate


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="Weekly CSV (data contract).")
    parser.add_argument("--config", type=Path, required=True, help="Model settings YAML.")
    parser.add_argument("--out", type=Path, default=config.MODELS_DIR, help="Output folder.")
    return parser.parse_args()


def main() -> None:
    """Run the full training pipeline."""
    args = parse_args()
    df = pd.read_csv(args.data, parse_dates=[config.DATE_COL])
    settings = ModelSettings.from_yaml(args.config)

    print("== 1. Validate data ==")
    report = validate(df)
    print(f"Readiness score {report.readiness_score}/{config.MAX_READINESS_SCORE}")
    if report.count(Severity.CRITICAL):
        print(report.summary())
        sys.exit("Critical data issues found; fix them before training.")

    print("\n== 2. Build model ==")
    model = MixLabModel().build(df, settings)
    print(f"Channels: {', '.join(model.channels)}")
    print(f"Controls: {', '.join(model.controls)}")

    print("\n== 3. Prior predictive check (before fitting) ==")
    prior = model.sample_prior_predictive()
    check = prior_predictive_check(prior, model.y)
    lakh = config.INR_PER_LAKH
    print(
        f"Observed average week: {check['observed_mean_weekly_revenue'] / lakh:,.0f} lakh | "
        f"priors allow {check['prior_mean_weekly_revenue_low'] / lakh:,.0f} to "
        f"{check['prior_mean_weekly_revenue_high'] / lakh:,.0f} lakh "
        f"(median {check['prior_mean_weekly_revenue_median'] / lakh:,.0f}); "
        f"{check['share_of_negative_weeks']:.1%} of simulated weeks are negative"
    )
    print("Verdict:", "believable" if check["believable"] else "NOT believable; revisit priors")
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = config.FIGURES_DIR / config.PRIOR_PREDICTIVE_FIGURE
    plot_prior_predictive(interval_summary(prior), model.y).savefig(
        figure_path, dpi=config.FIGURE_DPI
    )
    print(f"Saved {figure_path}")

    print("\n== 4. Fit ==")
    model.fit()

    print("\n== 5. Convergence diagnostics ==")
    diagnostics = convergence_diagnostics(model.idata, model.parameter_names)
    for line in explain_convergence(diagnostics):
        print(line)

    print("\n== 6. Save ==")
    path = model.save(args.out, extra={"diagnostics": diagnostics, "prior_check": check})
    print(f"Saved {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
