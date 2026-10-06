"""Tests for forecasting, pacing, drift checks, refresh explanations and version history."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mixlab import config
from mixlab.model import MixLabModel
from mixlab.monitoring import (
    drift_check,
    even_schedule,
    explain_changes,
    forecast,
    future_dates,
    future_frame,
    load_versions,
    pacing_report,
    pacing_status,
    record_version,
    simulate_new_weeks,
    version_entry,
)

BRAND = config.PERFORMANCE_HEAVY_BRAND
PLAN = {"meta_ads": 13e6, "tv": 26e6, "email": 1.3e6}


def schedule_for(df: pd.DataFrame, weeks: int = 13) -> pd.DataFrame:
    return even_schedule(PLAN, future_dates(df, weeks))


def test_schedule_spreads_the_plan_over_the_next_weeks(df: pd.DataFrame) -> None:
    schedule = schedule_for(df)
    assert schedule.index[0] == df[config.DATE_COL].max() + pd.Timedelta(days=7)
    assert schedule["spend_tv"].sum() == pytest.approx(PLAN["tv"])
    assert len(schedule) == 13


def test_future_frame_knows_the_calendar(df: pd.DataFrame) -> None:
    frame = future_frame(df, schedule_for(df), BRAND)
    new_year = frame[frame[config.DATE_COL] == pd.Timestamp("2025-12-29")]
    assert int(new_year["holiday_new_year"].iloc[0]) == 1  # 1 Jan 2026 falls in that week
    assert (frame[config.PROMO_COL] == 0).all()
    last_price = df[config.PRICE_COL].iloc[-1]
    january = frame[frame[config.DATE_COL].dt.year == 2026][config.PRICE_COL]
    assert january.iloc[0] == pytest.approx(
        last_price * (1 + BRAND.price_annual_increase), abs=1e-4
    )
    assert set(c for c in df.columns if c.startswith(config.HOLIDAY_PREFIX)) <= set(frame.columns)


def test_forecast_has_a_range_for_every_week(df: pd.DataFrame, fitted: MixLabModel) -> None:
    full = {c.removeprefix("spend_"): df[c].tail(13).sum() for c in fitted.channels}
    schedule = even_schedule(full, future_dates(df, 6))
    prediction = forecast(fitted, df, schedule, BRAND)
    assert len(prediction) == 6
    assert (prediction["lower"] < prediction["mean"]).all()
    assert (prediction["mean"] < prediction["upper"]).all()


def test_pacing_status_thresholds() -> None:
    assert pacing_status(1.2) == "over-pacing"
    assert pacing_status(0.8) == "under-pacing"
    assert pacing_status(1.05) == "on plan"


def fake_actual(schedule: pd.DataFrame, weeks: int, revenue: list[float]) -> pd.DataFrame:
    actual = schedule.iloc[:weeks].reset_index()
    actual["spend_meta_ads"] *= 1.3
    actual[config.TARGET_COL] = revenue
    return actual


def fake_forecast(schedule: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            config.DATE_COL: schedule.index,
            "mean": 100.0,
            "lower": 90.0,
            "upper": 110.0,
        }
    )


def test_pacing_report_flags_channels_and_weeks(df: pd.DataFrame) -> None:
    schedule = schedule_for(df)
    actual = fake_actual(schedule, 4, [100.0, 105.0, 80.0, 70.0])
    report = pacing_report(schedule, actual, fake_forecast(schedule))
    assert report["channels"]["meta_ads"]["status"] == "over-pacing"
    assert report["channels"]["tv"]["status"] == "on plan"
    assert report["outside_weeks"] == [str(d.date()) for d in schedule.index[2:4]]
    assert report["below_range"] == 2 and report["above_range"] == 0
    drift = drift_check(report, backtest_mape=3.0)
    assert drift["drifting"] and len(drift["reasons"]) == 3
    assert "below" in drift["action"]


def test_no_drift_when_the_forecast_holds(df: pd.DataFrame) -> None:
    schedule = schedule_for(df)
    report = pacing_report(
        schedule, fake_actual(schedule, 4, [100.0, 101.0, 99.0, 100.0]), fake_forecast(schedule)
    )
    drift = drift_check(report, backtest_mape=3.0)
    assert not drift["drifting"] and drift["reasons"] == []


def test_a_sustained_shortfall_is_caught_inside_the_range(df: pd.DataFrame) -> None:
    from mixlab.monitoring import bias_z

    schedule = schedule_for(df)
    # Every week 6% short: no single week leaves the 90 to 110 range, but the run does.
    report = pacing_report(
        schedule, fake_actual(schedule, 4, [94.0, 94.0, 94.0, 94.0]), fake_forecast(schedule)
    )
    assert report["outside_weeks"] == []
    assert bias_z(report["weeks"]) < -config.Z_95
    drift = drift_check(report, backtest_mape=3.0)
    assert drift["drifting"] and "below" in drift["reasons"][-1] and "below" in drift["action"]


def summary(rois: dict[str, float]) -> dict:
    return {
        "channels": {
            c: {"roi": {"mean": v, "hdi_low": v / 2, "hdi_high": v * 2}} for c, v in rois.items()
        },
        "totals": {"blended_media_roi": {"mean": 1.0}},
    }


def test_explain_changes_gives_reasons_for_big_moves(df: pd.DataFrame) -> None:
    longer = df.copy()
    extra = df.tail(4).copy()
    extra[config.DATE_COL] = extra[config.DATE_COL] + pd.Timedelta(weeks=4)
    extra["spend_tv"] = df["spend_tv"].max() * 2
    longer = pd.concat([df, extra], ignore_index=True)
    old = summary({"tv": 1.0, "email": 3.0, "meta_ads": 1.0})
    new = summary({"tv": 0.6, "email": 3.1, "meta_ads": 0.85})
    rows = explain_changes(old, new, df, longer)
    assert rows[0]["channel"] == "tv" and rows[0]["big_change"]
    assert "previous peak" in rows[0]["reason"]
    email = next(r for r in rows if r["channel"] == "email")
    assert not email["big_change"] and "Little change" in email["reason"]
    meta = next(r for r in rows if r["channel"] == "meta_ads")
    assert not meta["big_change"] and "inside the old 94% range" in meta["reason"]


def test_versions_are_appended_and_replaced(df: pd.DataFrame, tmp_path: Path) -> None:
    path = tmp_path / config.VERSIONS_FILENAME
    insights = summary({"tv": 1.0})
    record_version(path, version_entry("v1", df, insights, PLAN))
    record_version(path, version_entry("v2", df, insights, PLAN))
    versions = record_version(path, version_entry("v1", df.tail(10), insights, PLAN))
    assert [v["version"] for v in versions] == ["v2", "v1"]
    assert load_versions(path)[1]["n_weeks"] == 10
    assert load_versions(tmp_path / "missing.json") == []


def test_simulated_weeks_follow_the_plan_with_slip_and_shock(df: pd.DataFrame) -> None:
    full = {c.name: float(df[f"spend_{c.name}"].tail(13).sum()) for c in BRAND.channels}
    schedule = even_schedule(full, future_dates(df, 6))
    calm, _ = simulate_new_weeks(BRAND, df, schedule)
    shocked, truth = simulate_new_weeks(BRAND, df, schedule, {"tv": -0.5}, -0.1, 2)
    assert list(calm.columns) == list(df.columns) and len(calm) == 6
    assert np.allclose(calm["spend_meta_ads"], schedule["spend_meta_ads"].round())
    assert np.allclose(shocked["spend_tv"], (schedule["spend_tv"] * 0.5).round())
    assert (shocked[config.TARGET_COL].iloc[-2:] < calm[config.TARGET_COL].iloc[-2:]).all()
    assert truth["shock"].iloc[-1] == pytest.approx(-0.1 * BRAND.baseline_revenue)
