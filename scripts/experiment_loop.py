"""Run the test-and-learn loop on the synthetic India brand and save the story for the app.

1. Rank channels by the value of a test (``experiments.value_of_information``).
2. Design a geo test for each; channels too small for a geo test get a holdout study instead.
3. Simulate the tests from the true data-generating process and analyse them.
4. Recalibrate the national model with the measured lifts and compare ROI ranges and the
   optimizer's recommendation before and after.

Needs ``scripts/build_geo_demo.py`` to have run. Writes
``artifacts/india_regions/experiment_loop.json`` (the calibrated model itself is gitignored).

Example:
    uv run python scripts/experiment_loop.py      # about five minutes (one refit)

"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import mixlab  # noqa: F401  (sets the PyTensor backend before PyMC is imported)
from mixlab import config
from mixlab.calibration import Experiment, compare_roi, fit_calibrated
from mixlab.config import ModelSettings
from mixlab.experiments.analyze import analyze_test
from mixlab.experiments.design import design_test, plan_text
from mixlab.experiments.loop import (
    simulate_geo_test,
    simulate_holdout_study,
    to_experiment,
    trial_window,
    truth_brands,
)
from mixlab.experiments.value_of_information import METHOD_NOTE, value_of_information
from mixlab.insights import PosteriorDraws, build_summary, extract_draws, marginal_roi_draws
from mixlab.model import MixLabModel
from mixlab.optimizer import optimize_budget

FOLDER: Path = config.ARTIFACTS_DIR / config.GEO_DEMO_BRAND


def recommendation(draws: PosteriorDraws, measured: frozenset[str] = frozenset()) -> dict[str, Any]:
    """Return the gated optimizer's plan and uplift; ``measured`` channels were tested."""
    result = optimize_budget(
        config.OPTIMIZER_WEEKS,
        draws,
        max_change=config.DEFAULT_MAX_CHANGE,
        gate=True,
        measured=measured,
    )
    return {
        "current": result.current.spend,
        "recommended": result.recommended.spend,
        "uplift": result.uplift.model_dump(),
        "limits": {c: v["max_change"] for c, v in result.limits.items()},
    }


def holdout_story(
    panel: pd.DataFrame, brands: dict[str, Any], channel: str, plan: dict[str, Any]
) -> tuple[Experiment, dict[str, Any]]:
    """Run a holdout study for a channel too small for a geo test."""
    weeks = config.OPTIMIZER_WEEKS
    experiment = simulate_holdout_study(panel, brands, channel, weeks)
    story = {
        "channel": channel,
        "kind": "holdout",
        "geo_design": plan,
        "weeks": weeks,
        "measured": experiment.incremental_revenue,
        "standard_error": experiment.standard_error,
    }
    return experiment, story


def geo_story(
    panel: pd.DataFrame, national: pd.DataFrame, brands: dict[str, Any], plan: dict[str, Any]
) -> tuple[Experiment, dict[str, Any]]:
    """Simulate and analyse the designed geo test; return the calibration input and story."""
    chosen = plan["chosen"]
    weeks, channel = chosen["weeks_needed"], plan["channel"]
    tested, true_lift = simulate_geo_test(
        panel, brands, channel, plan["test_regions"], weeks, chosen["multiplier"]
    )
    start, end = trial_window(panel, weeks)
    analysis = analyze_test(tested, plan["test_regions"], plan["control_regions"], start)
    experiment = to_experiment(analysis, national, channel, chosen["extra_weekly_spend"])
    labels = {
        g: r["label"]
        for g, r in json.loads((FOLDER / config.GEO_GROUND_TRUTH_FILENAME).read_text())[
            "regions"
        ].items()
    }
    story = {
        "channel": channel,
        "kind": "geo",
        "geo_design": plan,
        "plan_text": plan_text(plan, labels, str(start.date()), str(end.date())),
        "analysis": analysis,
        "true_lift": true_lift,
        "measured": analysis["effect"],
        "standard_error": analysis["standard_error"],
    }
    return experiment, story


def run(settings: ModelSettings) -> dict[str, Any]:
    """Run every step of the loop and return the story."""
    panel = pd.read_csv(FOLDER / config.GEO_WEEKLY_FILENAME, parse_dates=[config.DATE_COL])
    national = pd.read_csv(FOLDER / config.WEEKLY_DATA_FILENAME, parse_dates=[config.DATE_COL])
    truth = json.loads((FOLDER / config.GEO_GROUND_TRUTH_FILENAME).read_text())
    brands = truth_brands(truth)
    before_model = MixLabModel.load(FOLDER / config.NATIONAL_MODEL_SUBDIR)
    before = extract_draws(before_model, national)

    ranking = value_of_information(before)
    marginal = np.median(marginal_roi_draws(before), axis=0)
    designs = {c: design_test(panel, c, float(marginal[i])) for i, c in enumerate(before.channels)}
    target = config.EXPERIMENT_CHANNEL
    experiments, stories = [], []
    if designs[target]["feasible"]:
        experiment, story = geo_story(panel, national, brands, designs[target])
    else:
        experiment, story = holdout_story(panel, brands, target, designs[target])
    experiments.append(experiment)
    stories.append(story)
    geo_channel = next(
        (
            row["channel"]
            for row in ranking
            if row["channel"] != target and designs[row["channel"]]["feasible"]
        ),
        None,
    )
    if geo_channel is not None:
        experiment, story = geo_story(panel, national, brands, designs[geo_channel])
        experiments.append(experiment)
        stories.append(story)

    print(f"  refitting with {len(experiments)} lift measurements")
    after_model = fit_calibrated(national, settings, experiments)
    after_model.save(
        FOLDER / config.CALIBRATED_MODEL_SUBDIR, extra={"brand": config.GEO_DEMO_BRAND}
    )
    after = extract_draws(after_model, national)
    table = compare_roi(build_summary(before), build_summary(after), truth)
    return {
        "method_note": METHOD_NOTE,
        "value_of_information": ranking,
        "designs": {
            c: {k: v for k, v in d.items() if k != "region_ranking"} for c, d in designs.items()
        },
        "region_ranking": designs[target]["region_ranking"],
        "tests": stories,
        "experiments": [e.model_dump(mode="json") for e in experiments],
        "roi_before_after": table.to_dict(orient="records"),
        "recommendation_before": recommendation(before),
        "recommendation_after": recommendation(after, frozenset(e.channel for e in experiments)),
    }


def refresh(settings: ModelSettings) -> dict[str, Any]:
    """Recompute the recommendations from the saved models without refitting."""
    path = FOLDER / config.EXPERIMENT_LOOP_FILENAME
    story = json.loads(path.read_text())
    national = pd.read_csv(FOLDER / config.WEEKLY_DATA_FILENAME, parse_dates=[config.DATE_COL])
    before = extract_draws(MixLabModel.load(FOLDER / config.NATIONAL_MODEL_SUBDIR), national)
    after = extract_draws(MixLabModel.load(FOLDER / config.CALIBRATED_MODEL_SUBDIR), national)
    tested = frozenset(e["channel"] for e in story["experiments"])
    story["recommendation_before"] = recommendation(before)
    story["recommendation_after"] = recommendation(after, tested)
    return story


def main() -> None:
    """Run the loop and save the story JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=config.PROJECT_ROOT / "configs/demo.yaml")
    parser.add_argument(
        "--refresh", action="store_true", help="Recompute recommendations without refitting."
    )
    args = parser.parse_args()
    settings = ModelSettings.from_yaml(args.config)
    story = refresh(settings) if args.refresh else run(settings)
    path = FOLDER / config.EXPERIMENT_LOOP_FILENAME
    path.write_text(json.dumps(story, indent=2, default=float) + "\n")
    for row in story["roi_before_after"]:
        print(
            f"  {row['channel']:<14} true {row['true_roi']:.2f} | before {row['before_low']:.2f}"
            f"-{row['before_high']:.2f} | after {row['after_low']:.2f}-{row['after_high']:.2f} "
            f"| {row['range_shrink_pct']:+.0f}% narrower"
        )
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
