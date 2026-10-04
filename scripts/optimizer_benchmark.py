"""Compare a naive and a robust optimizer on many synthetic brands, scored against the truth.

Naive: maximise expected revenue with no limits on channel moves and no haircut (v1 style).
Robust: uncertainty-aware limits, a risk-adjusted objective, and the bootstrap haircut on the
promised uplift. For every brand both recommendations are scored with the true
data-generating process.

Example:
    uv run python scripts/optimizer_benchmark.py --brands 20

"""

import argparse
import json
from pathlib import Path

import numpy as np

import mixlab  # noqa: F401  (sets the PyTensor backend before PyMC is imported)
from mixlab import config
from mixlab.benchmark import random_brand
from mixlab.config import ModelSettings
from mixlab.data_gen import generate
from mixlab.insights import extract_draws
from mixlab.model import MixLabModel
from mixlab.optimizer import optimize_budget, validate_against_truth

OUTPUT = config.REPORTS_DIR / config.OPTIMIZER_BENCHMARK_FILENAME


def saved_haircut() -> float:
    """Return the bootstrap haircut estimated on the main demo brand, or the default."""
    path = (
        config.ARTIFACTS_DIR
        / config.PERFORMANCE_HEAVY_BRAND.name
        / config.OPTIMIZER_SUMMARY_FILENAME
    )
    if path.exists():
        optimism = json.loads(path.read_text()).get("optimism")
        if optimism:
            return float(optimism["shrinkage"])
    return config.UPLIFT_SHRINKAGE


def main() -> None:
    """Fit each random brand, run both optimizers, score them against the truth."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--brands", type=int, default=20)
    parser.add_argument("--config", type=Path, default=config.PROJECT_ROOT / "configs/demo.yaml")
    args = parser.parse_args()
    base = ModelSettings.from_yaml(args.config)
    draws_per_chain = config.BACKTEST_DRAWS
    settings = base.model_copy(
        update={
            "sampler": base.sampler.model_copy(
                update={"draws": draws_per_chain, "tune": draws_per_chain}
            )
        }
    )
    haircut = saved_haircut()
    weeks = config.OPTIMIZER_WEEKS
    rows = []
    for index in range(args.brands):
        brand = random_brand(index)
        data = generate(brand).data
        model = MixLabModel().build(data, settings)
        model.fit(progressbar=False)
        draws = extract_draws(model, data)
        row: dict = {"brand": brand.name}
        for label, result in {
            "naive": optimize_budget(weeks, draws),
            "robust": optimize_budget(
                weeks,
                draws,
                max_change=config.DEFAULT_MAX_CHANGE,
                gate=True,
                objective="risk_adjusted",
                shrinkage=haircut,
            ),
        }.items():
            check = validate_against_truth(brand, result)
            promised = result.uplift_pct.mean if label == "naive" else result.realistic_uplift_pct
            row[label] = {
                "promised_uplift_pct": promised,
                "expected_uplift_pct": result.uplift_pct.mean,
                "true_uplift_pct": check["true_uplift_pct"],
                "corner_solution": result.corner_solution,
            }
        rows.append(row)
        print(
            f"{brand.name}: naive promised {row['naive']['promised_uplift_pct']:+.1f}% true "
            f"{row['naive']['true_uplift_pct']:+.1f}% | robust promised "
            f"{row['robust']['promised_uplift_pct']:+.1f}% true "
            f"{row['robust']['true_uplift_pct']:+.1f}%",
            flush=True,
        )

    summary: dict = {"brands": args.brands, "haircut": haircut, "rows": rows}
    for label in ("naive", "robust"):
        promised = np.array([row[label]["promised_uplift_pct"] for row in rows])
        true = np.array([row[label]["true_uplift_pct"] for row in rows])
        summary[label] = {
            "mean_promised_uplift_pct": float(promised.mean()),
            "mean_true_uplift_pct": float(true.mean()),
            "promised_over_true": float(promised.mean() / true.mean()) if true.mean() else None,
            "share_worse_than_current": float((true < 0).mean()),
            "worst_true_uplift_pct": float(true.min()),
            "mean_absolute_gap_pct_points": float(np.abs(promised - true).mean()),
        }
    OUTPUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("haircut", "naive", "robust")}, indent=1))
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
