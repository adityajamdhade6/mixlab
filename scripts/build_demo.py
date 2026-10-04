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
from mixlab.evaluate import convergence_diagnostics, explain_convergence, rolling_backtest
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
    write_optimizer_summary(name, model, data, folder)
    backtest_brand(name, settings, root)
    return folder


def write_optimizer_summary(
    name: str, model: MixLabModel, data: pd.DataFrame, folder: Path
) -> None:
    """Run the optimizer analyses for a fitted model and save them."""
    draws = extract_draws(model, data)
    optimizer = build_optimizer_summary(BudgetAllocator(model), draws, config.BRAND_PRESETS[name])
    (folder / config.OPTIMIZER_SUMMARY_FILENAME).write_text(json.dumps(optimizer, indent=2) + "\n")


def optimize_brand(name: str, root: Path = config.ARTIFACTS_DIR) -> Path:
    """Recompute the optimizer summary from the saved model, without refitting."""
    folder = root / name
    data = pd.read_csv(folder / config.WEEKLY_DATA_FILENAME, parse_dates=[config.DATE_COL])
    write_optimizer_summary(name, MixLabModel.load(folder), data, folder)
    return folder


def backtest_brand(name: str, settings: ModelSettings, root: Path = config.ARTIFACTS_DIR) -> Path:
    """Run the rolling backtest (refits on shorter histories with fewer draws) and save it."""
    folder = root / name
    data = pd.read_csv(folder / config.WEEKLY_DATA_FILENAME, parse_dates=[config.DATE_COL])
    draws = min(config.BACKTEST_DRAWS, settings.sampler.draws)
    quick = settings.model_copy(
        update={"sampler": settings.sampler.model_copy(update={"draws": draws, "tune": draws})}
    )

    def fit_and_predict(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
        model = MixLabModel().build(train, quick)
        model.fit(progressbar=False)
        return model.predict(test)

    result = rolling_backtest(data, fit_and_predict)
    (folder / config.BACKTEST_FILENAME).write_text(json.dumps(result, indent=2) + "\n")
    holdout = result["holdout"]
    print(f"  backtest holdout: MAPE {holdout['mape_pct']:.1f}%, R2 {holdout['r2']:.2f}")
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
    parser.add_argument(
        "--stage", choices=["data", "train", "optimize", "backtest", "all"], default="all"
    )
    args = parser.parse_args()
    settings = ModelSettings.from_yaml(args.config)
    for name in args.brands:
        start = time.perf_counter()
        print(f"== {name} ({args.stage}) ==")
        if args.stage in ("data", "all"):
            folder = generate_brand(name)
        if args.stage in ("train", "all"):
            folder = train_brand(name, settings)
        if args.stage == "optimize":
            folder = optimize_brand(name)
        if args.stage == "backtest":
            folder = backtest_brand(name, settings)
        print(f"  saved to {folder} in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
