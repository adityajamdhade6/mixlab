"""Build the demo brands for the app: generate data, fit, and save every result per brand.

Each brand gets its own folder under ``artifacts/`` holding the data, ground truth, fitted
model, insights summary, optimizer summary and weekly decomposition.

Example:
    uv run python scripts/build_demo.py --brands performance_heavy tv_heavy influencer_led

"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

import mixlab  # noqa: F401  (sets the PyTensor backend before PyMC is imported)
from mixlab import config
from mixlab.config import ModelSettings
from mixlab.data_gen import generate, save_dataset
from mixlab.evaluate import convergence_diagnostics, explain_convergence
from mixlab.insights import build_summary, decomposition_weekly, export_summary, extract_draws
from mixlab.model import MixLabModel
from mixlab.optimizer import BudgetAllocator, build_optimizer_summary


def generate_brand(name: str, root: Path = config.ARTIFACTS_DIR) -> Path:
    """Generate one brand's synthetic data and ground truth; return its folder."""
    folder = root / name
    save_dataset(generate(config.BRAND_PRESETS[name]), folder)
    return folder


def train_brand(name: str, settings: ModelSettings, root: Path = config.ARTIFACTS_DIR) -> Path:
    """Fit one brand from its saved data and save the model, insights and optimizer results."""
    folder = root / name
    data = pd.read_csv(folder / config.WEEKLY_DATA_FILENAME, parse_dates=[config.DATE_COL])

    model = MixLabModel().build(data, settings)
    model.sample_prior_predictive()
    model.fit(progressbar=False)
    diagnostics = convergence_diagnostics(model.idata, model.parameter_names)
    for line in explain_convergence(diagnostics):
        print("  " + line)
    model.save(folder, extra={"diagnostics": diagnostics, "brand": name})

    draws = extract_draws(model, data)
    export_summary(build_summary(draws), folder)
    decomposition_weekly(draws).to_csv(folder / config.DECOMPOSITION_FILENAME)
    optimizer = build_optimizer_summary(BudgetAllocator(model), draws, config.BRAND_PRESETS[name])
    (folder / config.OPTIMIZER_SUMMARY_FILENAME).write_text(json.dumps(optimizer, indent=2) + "\n")
    return folder


def build_brand(name: str, settings: ModelSettings, root: Path = config.ARTIFACTS_DIR) -> Path:
    """Generate, fit and save one brand; return its folder."""
    generate_brand(name, root)
    return train_brand(name, settings, root)


def main() -> None:
    """Build every requested brand, or only one stage of the build."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--brands", nargs="+", default=list(config.BRAND_PRESETS))
    parser.add_argument("--config", type=Path, default=config.PROJECT_ROOT / "configs/demo.yaml")
    parser.add_argument("--stage", choices=["data", "train", "all"], default="all")
    args = parser.parse_args()
    settings = ModelSettings.from_yaml(args.config)
    for name in args.brands:
        start = time.perf_counter()
        print(f"== {name} ({args.stage}) ==")
        if args.stage in ("data", "all"):
            folder = generate_brand(name)
        if args.stage in ("train", "all"):
            folder = train_brand(name, settings)
        print(f"  saved to {folder} in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
