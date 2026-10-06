"""Build the geo demo: generate the regional panel, fit national and geo models, compare them.

Everything lands in ``artifacts/india_regions/``:

- ``geo_weekly.csv`` and ``geo_ground_truth.json``: the regional panel and its truth;
- ``mmm_weekly.csv``: the same data summed to national weeks;
- ``national_model/`` and ``geo_model/``: the two fitted models (large, gitignored);
- ``geo_draws.npz``: the geo model's response parameters per region (small, for the app);
- ``geo_summary.json`` and ``geo_comparison.json``: regional insights, the regional optimizer
  and the national-versus-geo comparison.

Example:
    uv run python scripts/build_geo_demo.py               # everything (about 30 minutes)
    uv run python scripts/build_geo_demo.py --stage analyze   # recompute from saved models

"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

import mixlab  # noqa: F401  (sets the PyTensor backend before PyMC is imported)
from mixlab import config
from mixlab.config import ModelSettings
from mixlab.evaluate import convergence_diagnostics, explain_convergence
from mixlab.geo_data import aggregate_national, generate_geo, save_geo_dataset
from mixlab.geo_insights import build_geo_outputs, save_geo_draws
from mixlab.geo_model import GeoMixLabModel, extract_geo_draws
from mixlab.insights import extract_draws
from mixlab.model import MixLabModel

FOLDER: Path = config.ARTIFACTS_DIR / config.GEO_DEMO_BRAND


def generate_data(folder: Path = FOLDER) -> None:
    """Write the regional panel, its ground truth and the national aggregate."""
    dataset = generate_geo(config.PERFORMANCE_HEAVY_BRAND)
    save_geo_dataset(dataset, folder)
    aggregate_national(dataset.data).to_csv(folder / config.WEEKLY_DATA_FILENAME, index=False)


def fit_and_save(
    model: MixLabModel, data: pd.DataFrame, settings: ModelSettings, out: Path
) -> None:
    """Build, fit, diagnose and save one model."""
    model.build(data, settings)
    model.fit(progressbar=False)
    diagnostics = convergence_diagnostics(model.idata, model.parameter_names)
    for line in explain_convergence(diagnostics):
        print("  " + line)
    model.save(out, extra={"diagnostics": diagnostics, "brand": config.GEO_DEMO_BRAND})


def train_national(settings: ModelSettings, folder: Path = FOLDER) -> None:
    """Fit the national model on the summed panel (the comparison baseline)."""
    data = pd.read_csv(folder / config.WEEKLY_DATA_FILENAME, parse_dates=[config.DATE_COL])
    out = folder / config.NATIONAL_MODEL_SUBDIR
    fit_and_save(MixLabModel(), data, settings, out)


def train_geo(settings: ModelSettings, folder: Path = FOLDER) -> None:
    """Fit the hierarchical geo model on the regional panel."""
    data = pd.read_csv(folder / config.GEO_WEEKLY_FILENAME, parse_dates=[config.DATE_COL])
    out = folder / config.GEO_MODEL_SUBDIR
    fit_and_save(GeoMixLabModel(), data, settings, out)


def analyze(folder: Path = FOLDER) -> None:
    """Compute regional insights, the regional optimizer and the comparison from saved models."""
    panel = pd.read_csv(folder / config.GEO_WEEKLY_FILENAME, parse_dates=[config.DATE_COL])
    national_data = pd.read_csv(folder / config.WEEKLY_DATA_FILENAME, parse_dates=[config.DATE_COL])
    truth = json.loads((folder / config.GEO_GROUND_TRUTH_FILENAME).read_text())
    geo_model = GeoMixLabModel.load(folder / config.GEO_MODEL_SUBDIR)
    national_model = MixLabModel.load(folder / config.NATIONAL_MODEL_SUBDIR)
    by_geo = extract_geo_draws(geo_model, panel)
    national = extract_draws(national_model, national_data)
    meta = json.loads((folder / config.GEO_MODEL_SUBDIR / config.MODEL_META_FILENAME).read_text())
    national_meta = json.loads(
        (folder / config.NATIONAL_MODEL_SUBDIR / config.MODEL_META_FILENAME).read_text()
    )
    summary, comparison = build_geo_outputs(
        by_geo, national, truth, meta.get("diagnostics"), national_meta.get("diagnostics")
    )
    (folder / config.GEO_SUMMARY_FILENAME).write_text(json.dumps(summary, indent=2) + "\n")
    (folder / config.GEO_COMPARISON_FILENAME).write_text(json.dumps(comparison, indent=2) + "\n")
    save_geo_draws(by_geo, folder / config.GEO_DRAWS_FILENAME)
    for line in comparison["headline"]:
        print("  " + line)


def main() -> None:
    """Run every stage, or one of them."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=config.PROJECT_ROOT / "configs/geo.yaml")
    parser.add_argument(
        "--national-config", type=Path, default=config.PROJECT_ROOT / "configs/demo.yaml"
    )
    parser.add_argument(
        "--stage", choices=["data", "national", "geo", "analyze", "all"], default="all"
    )
    args = parser.parse_args()
    FOLDER.mkdir(parents=True, exist_ok=True)
    stages = {
        "data": generate_data,
        "national": lambda: train_national(ModelSettings.from_yaml(args.national_config)),
        "geo": lambda: train_geo(ModelSettings.from_yaml(args.config)),
        "analyze": analyze,
    }
    for name, run in stages.items():
        if args.stage in (name, "all"):
            start = time.perf_counter()
            print(f"== {name} ==")
            run()
            print(f"  done in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
