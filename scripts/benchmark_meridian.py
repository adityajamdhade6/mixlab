"""Fit Google Meridian on a demo brand and record its channel ROI estimates.

Meridian needs TensorFlow/JAX builds that conflict with this project's environment, so this
script deliberately imports nothing from ``mixlab`` and runs in its own environment:

    uv run --isolated --python 3.11 --with google-meridian --with pandas \
        python scripts/benchmark_meridian.py

It writes ``reports/meridian_benchmark.json``; ``docs/model_log.md`` compares it with MixLab.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from meridian.analysis import analyzer
from meridian.data import load
from meridian.model import model, spec

ROOT = Path(__file__).resolve().parents[1]
INTERVAL = (3.0, 97.0)


def main() -> None:
    """Load a brand, fit Meridian's national model with default priors, save ROI estimates."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--brand", default="performance_heavy")
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--adapt", type=int, default=500)
    parser.add_argument("--burnin", type=int, default=500)
    parser.add_argument("--keep", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    folder = ROOT / "artifacts" / args.brand
    df = pd.read_csv(folder / "mmm_weekly.csv")
    truth = json.loads((folder / "ground_truth.json").read_text())["channels"]
    spend = [c for c in df.columns if c.startswith("spend_")]
    controls = [c for c in df.columns if c.startswith("holiday_")] + ["promo_flag", "price_index"]
    names = {column: column.removeprefix("spend_") for column in spend}

    data = load.DataFrameDataLoader(
        df=df[["date", "revenue", *spend, *controls]],
        kpi_type="revenue",
        coord_to_columns=load.CoordToColumns(
            time="date", kpi="revenue", controls=controls, media=spend, media_spend=spend
        ),
        media_to_channel=names,
        media_spend_to_channel=names,
    ).load()

    mmm = model.Meridian(input_data=data, model_spec=spec.ModelSpec())
    start = time.perf_counter()
    mmm.sample_prior(500, seed=args.seed)
    mmm.sample_posterior(
        n_chains=args.chains,
        n_adapt=args.adapt,
        n_burnin=args.burnin,
        n_keep=args.keep,
        seed=args.seed,
    )
    seconds = time.perf_counter() - start

    roi = np.asarray(analyzer.Analyzer(mmm).roi())
    roi = roi.reshape(-1, roi.shape[-1])
    channels = [str(c) for c in data.media_channel.values]
    results = {}
    for index, channel in enumerate(channels):
        low, high = np.percentile(roi[:, index], INTERVAL)
        results[channel] = {
            "true_roi": truth[channel]["true_roi"],
            "mean": float(roi[:, index].mean()),
            "low": float(low),
            "high": float(high),
            "truth_inside_range": bool(low <= truth[channel]["true_roi"] <= high),
        }
        print(
            f"{channel:14s} true {truth[channel]['true_roi']:.2f} | Meridian "
            f"{results[channel]['mean']:.2f} ({low:.2f} to {high:.2f})"
        )
    output = ROOT / "reports" / "meridian_benchmark.json"
    output.write_text(
        json.dumps(
            {
                "brand": args.brand,
                "interval": "3rd to 97th percentile",
                "sampling": vars(args),
                "fit_seconds": seconds,
                "recovered": sum(r["truth_inside_range"] for r in results.values()),
                "channels": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved {output} ({seconds:.0f}s)")


if __name__ == "__main__":
    main()
