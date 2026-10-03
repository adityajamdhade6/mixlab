"""Out-of-sample and robustness checks for the demo brands.

1. Holdout: fit on all but the last 13 weeks, predict those weeks, report MAPE and coverage.
2. Prior sensitivity: refit one brand WITHOUT the channel-specific priors that happen to agree
   with the ground truth, and see whether ROI recovery survives.
3. Seeds: regenerate one brand with several seeds and count how often the true ROI falls
   inside the model's 94% range, so recovery is not judged on one lucky draw.

Example:
    uv run python scripts/evaluate_models.py

"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

import mixlab  # noqa: F401  (sets the PyTensor backend before PyMC is imported)
from mixlab import config
from mixlab.config import ModelSettings
from mixlab.data_gen import generate
from mixlab.evaluate import prediction_error, roi_recovery
from mixlab.insights import build_summary, extract_draws
from mixlab.model import MixLabModel

OUTPUT = config.REPORTS_DIR / "evaluation_summary.json"


def fit(df: pd.DataFrame, settings: ModelSettings) -> MixLabModel:
    """Build and fit quietly."""
    model = MixLabModel().build(df, settings)
    model.fit(progressbar=False)
    return model


def holdout(df: pd.DataFrame, settings: ModelSettings, weeks: int) -> dict[str, float]:
    """Fit on the early weeks and score predictions for the last ``weeks``."""
    train, test = df.iloc[:-weeks], df.iloc[-weeks:]
    prediction = fit(train, settings).predict(test)
    return prediction_error(prediction, test[config.TARGET_COL])


def recovery(df: pd.DataFrame, truth: dict, settings: ModelSettings) -> pd.DataFrame:
    """Fit and compare estimated ROI with the truth."""
    model = fit(df, settings)
    return roi_recovery(build_summary(extract_draws(model, df)), truth)


def main() -> None:
    """Run all three checks and save one JSON summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=config.PROJECT_ROOT / "configs/demo.yaml")
    parser.add_argument("--holdout-weeks", type=int, default=config.OPTIMIZER_WEEKS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--brand", default=config.PERFORMANCE_HEAVY_BRAND.name)
    args = parser.parse_args()
    settings = ModelSettings.from_yaml(args.config)
    summary: dict = {"holdout_weeks": args.holdout_weeks, "holdout": {}}

    for name in config.BRAND_PRESETS:
        df = generate(config.BRAND_PRESETS[name]).data
        summary["holdout"][name] = holdout(df, settings, args.holdout_weeks)
        print(f"holdout {name}: {summary['holdout'][name]}", flush=True)

    brand = config.BRAND_PRESETS[args.brand]
    dataset = generate(brand)
    plain = settings.model_copy(update={"channel_priors": {}})
    table = recovery(dataset.data, dataset.ground_truth, plain)
    summary["no_channel_priors"] = {
        "brand": args.brand,
        "channels_recovered": int(table["truth_inside_range"].sum()),
        "channels": len(table),
        "table": table.to_dict(orient="records"),
    }
    print(f"no channel priors: {summary['no_channel_priors']['channels_recovered']}/6", flush=True)

    hits: dict[str, int] = dict.fromkeys((c.name for c in brand.channels), 0)
    for seed in args.seeds:
        seeded = generate(replace(brand, seed=seed))
        table = recovery(seeded.data, seeded.ground_truth, settings)
        for row in table.itertuples():
            hits[row.channel] += int(row.truth_inside_range)
        print(f"seed {seed}: {int(table['truth_inside_range'].sum())}/6", flush=True)
    total = sum(hits.values())
    summary["seed_study"] = {
        "brand": args.brand,
        "seeds": args.seeds,
        "recovered_by_channel": hits,
        "recovered_pct": 100 * total / (len(args.seeds) * len(hits)),
    }
    OUTPUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
