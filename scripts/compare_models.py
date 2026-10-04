"""Compare model variants on the synthetic brand, where the truth is known.

For each variant: leave-one-out cross-validation, a 12-week holdout, ROI recovery against
ground truth on the demo seed and on other seeds, and the width of the ROI ranges. Also runs a
prior sensitivity sweep for the pooled ROI prior.

Example:
    uv run python scripts/compare_models.py

"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

import mixlab  # noqa: F401  (sets the PyTensor backend before PyMC is imported)
from mixlab import config
from mixlab.config import ModelSettings, RoiPriorSettings
from mixlab.data_gen import generate
from mixlab.evaluate import (
    convergence_diagnostics,
    is_converged,
    loo,
    prediction_error,
    roi_from_posterior,
    roi_recovery,
)
from mixlab.model import MixLabModel

OUTPUT = config.REPORTS_DIR / config.MODEL_COMPARISON_FILENAME


def variants(base: ModelSettings) -> dict[str, ModelSettings]:
    """Return the model variants under comparison.

    Every variant is defined relative to the v1 model (logistic saturation, no ROI prior), not
    to whatever the settings file currently selects, so the comparison stays stable when the
    default changes.
    """
    v1 = base.model_copy(update={"saturation": "logistic", "roi_prior": None})
    independent = RoiPriorSettings(mode="independent", spread=1.0)
    return {
        "v1_baseline": v1,
        "pooled_roi_prior": v1.model_copy(update={"roi_prior": RoiPriorSettings(spread=1.0)}),
        "independent_roi_prior": v1.model_copy(update={"roi_prior": independent}),
        "delayed_adstock": v1.model_copy(update={"adstock": "delayed"}),
        "time_varying_baseline": v1.model_copy(update={"time_varying_intercept": True}),
        "hill_saturation": v1.model_copy(update={"saturation": "hill"}),
        "hill_independent_roi_prior": v1.model_copy(
            update={"saturation": "hill", "roi_prior": independent}
        ),
    }


def quick(settings: ModelSettings) -> ModelSettings:
    """Return the same model with fewer draws, for repeated fits."""
    draws = config.BACKTEST_DRAWS
    sampler = settings.sampler.model_copy(update={"draws": draws, "tune": draws})
    return settings.model_copy(update={"sampler": sampler})


def fit(df: pd.DataFrame, settings: ModelSettings) -> MixLabModel:
    """Build and fit quietly."""
    model = MixLabModel().build(df, settings)
    model.fit(progressbar=False)
    return model


def recovery_record(model: MixLabModel, df: pd.DataFrame, truth: dict) -> dict:
    """Return recovery and range-width results for one fitted model."""
    table = roi_recovery(roi_from_posterior(model.idata, df), truth)
    table["width"] = table["high"] - table["low"]
    return {
        "recovered": int(table["truth_inside_range"].sum()),
        "channels": len(table),
        "missed": table.loc[~table["truth_inside_range"], "channel"].tolist(),
        "mean_range_width": float(table["width"].mean()),
        "table": table.to_dict(orient="records"),
    }


def evaluate_variant(settings: ModelSettings, brand: config.BrandConfig, seeds: list[int]) -> dict:
    """Run every check for one variant."""
    dataset = generate(brand)
    df, truth = dataset.data, dataset.ground_truth
    model = fit(df, settings)
    diagnostics = convergence_diagnostics(model.idata, model.parameter_names)
    result = {
        "converged": is_converged(diagnostics),
        "divergences": diagnostics["divergences"],
        "fit_seconds": model.fit_seconds,
        "loo": loo(model.idata),
        **recovery_record(model, df, truth),
    }
    weeks = config.BACKTEST_HORIZON_WEEKS
    train, test = df.iloc[:-weeks], df.iloc[-weeks:]
    result["holdout"] = prediction_error(
        fit(train, quick(settings)).predict(test), test[config.TARGET_COL]
    )
    hits = total = 0
    for seed in seeds:
        seeded = generate(replace(brand, seed=seed))
        record = recovery_record(
            fit(seeded.data, quick(settings)), seeded.data, seeded.ground_truth
        )
        hits, total = hits + record["recovered"], total + record["channels"]
    result["other_seeds"] = {"seeds": seeds, "recovered": hits, "of": total}
    return result


def prior_sensitivity(base: ModelSettings, brand: config.BrandConfig, spreads: list[float]) -> list:
    """Refit the pooled ROI prior at several strengths and record each channel's ROI."""
    dataset = generate(brand)
    rows = []
    v1 = base.model_copy(update={"saturation": "logistic", "roi_prior": None})
    for spread in spreads:
        settings = quick(v1.model_copy(update={"roi_prior": RoiPriorSettings(spread=spread)}))
        record = recovery_record(fit(dataset.data, settings), dataset.data, dataset.ground_truth)
        rows.append({"spread": spread, **{k: record[k] for k in ("recovered", "table")}})
        print(f"sensitivity spread {spread}: {record['recovered']}/6", flush=True)
    return rows


def main() -> None:
    """Run the comparison and save one JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=config.PROJECT_ROOT / "configs/demo.yaml")
    parser.add_argument("--brand", default=config.PERFORMANCE_HEAVY_BRAND.name)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--only", nargs="+", help="Variant names to run (default: all).")
    args = parser.parse_args()
    base = ModelSettings.from_yaml(args.config)
    brand = config.BRAND_PRESETS[args.brand]
    summary = json.loads(OUTPUT.read_text()) if OUTPUT.exists() else {"variants": {}}
    for name, settings in variants(base).items():
        if args.only and name not in args.only:
            continue
        result = evaluate_variant(settings, brand, args.seeds)
        summary["variants"][name] = result
        print(
            f"{name}: elpd {result['loo']['elpd_loo']:.1f} | holdout MAPE "
            f"{result['holdout']['mape_pct']:.2f}% | recovered {result['recovered']}/6 "
            f"(missed {result['missed']}) | other seeds {result['other_seeds']['recovered']}/"
            f"{result['other_seeds']['of']} | width {result['mean_range_width']:.2f}",
            flush=True,
        )
        OUTPUT.write_text(json.dumps(summary, indent=2) + "\n")
    if not args.only:
        summary["prior_sensitivity"] = prior_sensitivity(base, brand, [0.25, 0.5, 1.0, 2.0])
        OUTPUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
