"""Simulate a quarter in progress: forecast, 8 new weeks, pacing, drift, refresh, versions.

For the performance-heavy demo brand:

1. Forecast the next 13 weeks under the current and the recommended plan.
2. Generate 8 new weeks from the true process: spend follows the recommended plan with
   realistic slippage (``config.SIM_PACING_SLIP``) and an unmodelled competitor launch hits
   the last weeks (``config.SIM_SHOCK_SHARE``).
3. Compare them with the plan and the forecast (pacing) and check for drift.
4. Refit on the longer history and explain how each channel's ROI moved.
5. Record both model versions.

Writes ``artifacts/performance_heavy/monitoring.json`` and ``versions.json``.

Example:
    uv run python scripts/simulate_weeks.py      # about five minutes (one refit)

"""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

import mixlab  # noqa: F401  (sets the PyTensor backend before PyMC is imported)
from mixlab import config
from mixlab.config import ModelSettings
from mixlab.insights import build_summary, extract_draws
from mixlab.model import MixLabModel
from mixlab.monitoring import (
    as_run_frame,
    drift_check,
    even_schedule,
    explain_changes,
    forecast,
    future_dates,
    pacing_report,
    record_version,
    simulate_new_weeks,
    version_entry,
)

BRAND = "performance_heavy"
FOLDER: Path = config.ARTIFACTS_DIR / BRAND


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Return a frame as JSON-ready records with ISO dates."""
    out = frame.copy()
    out[config.DATE_COL] = pd.to_datetime(out[config.DATE_COL]).dt.strftime("%Y-%m-%d")
    return out.to_dict(orient="records")


def main() -> None:
    """Run the simulated quarter and save everything the Pacing page shows."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=config.PROJECT_ROOT / "configs/demo.yaml")
    args = parser.parse_args()
    brand = config.BRAND_PRESETS[BRAND]
    history = pd.read_csv(FOLDER / config.WEEKLY_DATA_FILENAME, parse_dates=[config.DATE_COL])
    optimizer = json.loads((FOLDER / config.OPTIMIZER_SUMMARY_FILENAME).read_text())
    plans = {
        "current": optimizer["expected_revenue"]["current"]["spend"],
        "recommended": optimizer["expected_revenue"]["recommended"]["spend"],
    }
    model = MixLabModel.load(FOLDER)
    dates = future_dates(history, config.FORECAST_WEEKS)
    schedules = {name: even_schedule(plan, dates) for name, plan in plans.items()}
    forecasts = {name: forecast(model, history, s, brand) for name, s in schedules.items()}

    print("== new weeks ==")
    schedule = schedules["recommended"].iloc[: config.SIM_NEW_WEEKS]
    actual, truth = simulate_new_weeks(
        brand,
        history,
        schedule,
        config.SIM_PACING_SLIP,
        config.SIM_SHOCK_SHARE,
        config.SIM_SHOCK_WEEKS,
    )
    report = pacing_report(schedules["recommended"], actual, forecasts["recommended"])
    # Drift is judged against what actually ran (spend slippage and promotions included),
    # so that only revenue the model cannot explain counts as error.
    as_run = model.predict(as_run_frame(actual))
    as_run_report = pacing_report(schedules["recommended"], actual, as_run)
    backtest = json.loads((FOLDER / config.BACKTEST_FILENAME).read_text())
    drift = drift_check(as_run_report, backtest["holdout"]["mape_pct"])
    print(f"  accuracy {report['accuracy']}; drifting: {drift['drifting']}")

    print("== refresh ==")
    longer = pd.concat([history, actual], ignore_index=True)
    settings = ModelSettings.from_yaml(args.config)
    refreshed = MixLabModel().build(longer, settings)
    refreshed.fit(progressbar=False)
    refreshed.save(FOLDER / config.REFRESH_MODEL_SUBDIR, extra={"brand": BRAND})
    old_insights = json.loads((FOLDER / config.INSIGHTS_SUMMARY_FILENAME).read_text())
    new_insights = build_summary(extract_draws(refreshed, longer))
    changes = explain_changes(old_insights, new_insights, history, longer)

    versions_path = FOLDER / config.VERSIONS_FILENAME
    record_version(
        versions_path,
        version_entry(
            "v1", history, old_insights, plans["recommended"], {"holdout": backtest["holdout"]}
        ),
    )
    record_version(
        versions_path,
        version_entry(
            "v2 (refresh)",
            longer,
            new_insights,
            plans["recommended"],
            {"pacing_accuracy": report["accuracy"], "drift": drift["drifting"]},
        ),
    )
    story = {
        "plans": plans,
        "forecasts": {name: records(f) for name, f in forecasts.items()},
        "schedule": records(schedules["recommended"].reset_index()),
        "actual": records(actual),
        "truth": records(truth),
        "pacing": report,
        "as_run": as_run_report,
        "drift": drift,
        "changes": changes,
        "simulation": {
            "slip": config.SIM_PACING_SLIP,
            "shock_share": config.SIM_SHOCK_SHARE,
            "shock_weeks": config.SIM_SHOCK_WEEKS,
        },
    }
    path = FOLDER / config.MONITORING_FILENAME
    path.write_text(json.dumps(story, indent=2, default=float) + "\n")
    for row in changes:
        print(f"  {row['channel']:<14} {row['change_pct']:+.0f}%  {row['reason']}")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
