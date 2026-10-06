"""Week-to-week use: forecast, pacing against plan, drift, model refresh and version history.

- ``forecast``: weekly revenue for the next weeks under any plan, with a 94% range, including
  known festivals, the yearly price step and carryover from the last weeks of history.
- ``pacing_report``: actual spend and revenue against the plan and the forecast.
- ``drift_check``: are recent forecast errors larger than the model's normal error?
- ``explain_changes``: after a refit, which ROI estimates moved and why.
- ``record_version`` / ``load_versions``: every model version with its data and results.
- ``simulate_new_weeks``: new weeks for a synthetic brand, from the true process.
"""

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mixlab import config
from mixlab.config import BrandConfig
from mixlab.data_gen import channel_contribution, event_flags, generate
from mixlab.evaluate import prediction_error
from mixlab.model import MixLabModel

# --- Forecast -------------------------------------------------------------------------------


def future_dates(history: pd.DataFrame, n_weeks: int) -> pd.DatetimeIndex:
    """Return the next ``n_weeks`` week-start dates after the history."""
    last = pd.Timestamp(history[config.DATE_COL].max())
    return pd.date_range(last + pd.Timedelta(days=config.DAYS_PER_WEEK), periods=n_weeks, freq="7D")


def even_schedule(plan: dict[str, float], dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Return a weekly spend schedule spreading each channel's total evenly over ``dates``."""
    weekly = {f"{config.SPEND_PREFIX}{c}": total / len(dates) for c, total in plan.items()}
    return pd.DataFrame(weekly, index=dates).rename_axis(config.DATE_COL)


def future_frame(
    history: pd.DataFrame, schedule: pd.DataFrame, brand: BrandConfig | None = None
) -> pd.DataFrame:
    """Return model inputs for future weeks: planned spend plus the known calendar.

    Holidays come from the brand's event calendar; no promotion is assumed; the price index
    stays at its last level and steps up by the brand's yearly increase at a new year.
    """
    dates = pd.DatetimeIndex(schedule.index)
    frame = schedule.reset_index()
    holidays = [c for c in history.columns if c.startswith(config.HOLIDAY_PREFIX)]
    if brand is not None:
        flags = event_flags(dates, brand.events)
        for column in holidays:
            frame[column] = flags[column].to_numpy() if column in flags else 0
    else:
        for column in holidays:
            frame[column] = 0
    if config.PROMO_COL in history:
        frame[config.PROMO_COL] = 0
    if config.PRICE_COL in history:
        last = history.sort_values(config.DATE_COL).iloc[-1]
        step = 1 + (brand.price_annual_increase if brand else 0.0)
        years = dates.year - pd.Timestamp(last[config.DATE_COL]).year
        frame[config.PRICE_COL] = np.round(last[config.PRICE_COL] * step ** np.asarray(years), 4)
    return frame


def forecast(
    model: MixLabModel,
    history: pd.DataFrame,
    schedule: pd.DataFrame,
    brand: BrandConfig | None = None,
) -> pd.DataFrame:
    """Return the weekly revenue forecast (date, mean, lower, upper) for a spend schedule."""
    return model.predict(future_frame(history, schedule, brand))


# --- Pacing and drift -----------------------------------------------------------------------


def pacing_status(ratio: float, tolerance: float = config.PACING_TOLERANCE) -> str:
    """Return "over-pacing", "under-pacing" or "on plan" for actual / planned spend."""
    if ratio > 1 + tolerance:
        return "over-pacing"
    if ratio < 1 - tolerance:
        return "under-pacing"
    return "on plan"


def pacing_report(
    schedule: pd.DataFrame, actual: pd.DataFrame, prediction: pd.DataFrame
) -> dict[str, Any]:
    """Compare actual weeks with the plan and the forecast made for them.

    Args:
        schedule: Planned weekly spend (dates x ``spend_*`` columns).
        actual: Actual weekly rows (date, revenue, ``spend_*``), a prefix of the plan's weeks.
        prediction: Forecast for the plan's weeks (date, mean, lower, upper).

    """
    weeks = pd.DatetimeIndex(actual[config.DATE_COL])
    planned = schedule.loc[weeks]
    channels = {}
    for column in planned.columns:
        plan_total = float(planned[column].sum())
        actual_total = float(actual[column].sum())
        ratio = actual_total / plan_total if plan_total > 0 else float("nan")
        channels[column.removeprefix(config.SPEND_PREFIX)] = {
            "planned": plan_total,
            "actual": actual_total,
            "pace": ratio,
            "status": pacing_status(ratio) if plan_total > 0 else "not planned",
        }
    forecast_rows = prediction.set_index(config.DATE_COL).loc[weeks]
    revenue = actual[config.TARGET_COL].to_numpy(dtype=float)
    lower, upper = forecast_rows["lower"].to_numpy(), forecast_rows["upper"].to_numpy()
    week_rows = [
        {
            "date": str(date.date()),
            "actual": float(value),
            "forecast": float(row.mean),
            "lower": float(row.lower),
            "upper": float(row.upper),
            "outside": bool(value < row.lower or value > row.upper),
        }
        for date, value, row in zip(weeks, revenue, forecast_rows.itertuples(), strict=True)
    ]
    return {
        "channels": channels,
        "weeks": week_rows,
        "accuracy": prediction_error(forecast_rows.reset_index(), actual[config.TARGET_COL]),
        "outside_weeks": [w["date"] for w in week_rows if w["outside"]],
        "below_range": int(np.sum(revenue < lower)),
        "above_range": int(np.sum(revenue > upper)),
    }


def bias_z(weeks: list[dict[str, Any]]) -> float:
    """Return how many standard errors the average weekly miss sits from zero.

    Each week's spread is read off its 94% range; a run of misses in one direction adds up
    even when no single week falls outside its range.
    """
    if not weeks:
        return 0.0
    errors = np.array([w["actual"] - w["forecast"] for w in weeks])
    sd = np.array([(w["upper"] - w["lower"]) / (2 * config.Z_94) for w in weeks])
    return float(errors.mean() / (np.sqrt((sd**2).mean()) / np.sqrt(len(weeks))))


def drift_check(report: dict[str, Any], backtest_mape: float) -> dict[str, Any]:
    """Decide whether the model has drifted, from forecast errors on recent weeks.

    ``report`` should compare actual revenue with an as-run forecast (actual spend and
    promotions), so that what is left is revenue the model cannot explain. Three signals:
    recent error well beyond the backtest error, several weeks outside the range, or a
    sustained bias in one direction (``bias_z`` beyond ``config.Z_95``).
    """
    recent = report["weeks"][-config.DRIFT_WINDOW :]
    errors = [abs(w["actual"] - w["forecast"]) / w["actual"] * 100 for w in recent]
    recent_mape = float(np.mean(errors)) if errors else 0.0
    outside = sum(w["outside"] for w in recent)
    z = bias_z(recent)
    signed = (
        float(np.mean([(w["actual"] - w["forecast"]) / w["forecast"] for w in recent]))
        if recent
        else 0.0
    )
    reasons = []
    if recent_mape > config.DRIFT_ERROR_RATIO * backtest_mape:
        reasons.append(
            f"the last {len(recent)} weeks missed by {recent_mape:.1f}% on average, over "
            f"{config.DRIFT_ERROR_RATIO:g}x the model's normal {backtest_mape:.1f}%"
        )
    if outside >= config.DRIFT_OUTSIDE_WEEKS:
        reasons.append(f"{outside} of the last {len(recent)} weeks fell outside the 94% range")
    if abs(z) > config.Z_95:
        reasons.append(
            f"revenue has run {abs(100 * signed):.1f}% {'below' if signed < 0 else 'above'} "
            f"the forecast on average over {len(recent)} weeks ({abs(z):.1f} standard errors), "
            "more than chance explains"
        )
    direction = "below" if signed < 0 else "above"
    action = (
        f"Revenue is running {direction} what actual spend explains. Check for something the "
        "model does not know about (a competitor, stock-outs, a site problem) before "
        "refitting; if it persists, refit so the baseline catches up."
        if reasons
        else "No action needed: forecast errors are within the model's normal range."
    )
    return {
        "drifting": bool(reasons),
        "recent_mape_pct": recent_mape,
        "backtest_mape_pct": backtest_mape,
        "weeks_outside": outside,
        "bias_pct": 100 * signed,
        "bias_z": z,
        "reasons": reasons,
        "action": action,
    }


def as_run_frame(actual: pd.DataFrame) -> pd.DataFrame:
    """Return the actual weeks as model inputs, so the forecast uses what really ran."""
    return actual.drop(columns=[config.TARGET_COL])


# --- Refresh --------------------------------------------------------------------------------


def explain_changes(
    old: dict[str, Any], new: dict[str, Any], old_data: pd.DataFrame, new_data: pd.DataFrame
) -> list[dict[str, Any]]:
    """Return each channel's ROI before and after a refit, with a reason for big moves.

    Reasons are read from the data the refit added: whether the new weeks pushed spend above
    anything seen before (the curve is now measured further out), changed the channel's share
    of spend, or simply added evidence that moved the estimate.
    """
    added = new_data[~new_data[config.DATE_COL].isin(old_data[config.DATE_COL])]
    rows = []
    for channel, metrics in new["channels"].items():
        before, after = old["channels"][channel]["roi"], metrics["roi"]
        change = after["mean"] / before["mean"] - 1 if before["mean"] > 0 else 0.0
        column = f"{config.SPEND_PREFIX}{channel}"
        old_peak = float(old_data[column].max())
        new_peak = float(added[column].max()) if len(added) else 0.0
        old_share = float(
            old_data[column].sum() / old_data.filter(like=config.SPEND_PREFIX).sum().sum()
        )
        new_share = float(
            added[column].sum() / max(added.filter(like=config.SPEND_PREFIX).sum().sum(), 1.0)
        )
        big = abs(change) >= config.ROI_CHANGE_FLAG
        if abs(change) < config.ROI_CHANGE_FLAG / 2:
            reason = "Little change: the new weeks agree with what the model already knew."
        elif not big:
            inside = before["hdi_low"] <= after["mean"] <= before["hdi_high"]
            reason = (
                f"A {'fall' if change < 0 else 'rise'} that stays inside the old 94% range: the "
                "new weeks refine the estimate rather than overturn it."
                if inside
                else f"A {'fall' if change < 0 else 'rise'} to just outside the old range; "
                "worth watching at the next refresh."
            )
        elif new_peak > old_peak:
            peak = f"₹{new_peak / config.INR_PER_LAKH:,.1f} L a week"
            reason = (
                f"New weeks pushed spend above its previous peak ({peak}), so the curve is "
                "now measured where returns are lower."
                if change < 0
                else f"New weeks at record spend ({peak}) earned more than the old curve expected."
            )
        elif abs(new_share - old_share) > 0.5 * old_share:
            reason = (
                f"Its share of spend moved from {100 * old_share:.0f}% to {100 * new_share:.0f}% "
                "in the new weeks, which gave the model a cleaner read on it."
            )
        else:
            reason = (
                "The new weeks' revenue fitted a "
                + ("lower" if change < 0 else "higher")
                + " return better; nothing unusual happened to its spend."
            )
        rows.append(
            {
                "channel": channel,
                "before": before,
                "after": after,
                "change_pct": 100 * change,
                "big_change": big,
                "reason": reason,
            }
        )
    return sorted(rows, key=lambda row: abs(row["change_pct"]), reverse=True)


# --- Versions -------------------------------------------------------------------------------


def load_versions(path: Path) -> list[dict[str, Any]]:
    """Return the saved model versions, oldest first (empty if none)."""
    return json.loads(path.read_text()) if path.exists() else []


def record_version(path: Path, entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Append (or replace, by ``version``) one model version and save the history."""
    versions = [v for v in load_versions(path) if v["version"] != entry["version"]]
    versions.append(entry)
    path.write_text(json.dumps(versions, indent=2) + "\n")
    return versions


def version_entry(
    version: str,
    data: pd.DataFrame,
    insights: dict[str, Any],
    recommendation: dict[str, float],
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one version record: data range, headline ROI and the recommendation."""
    return {
        "version": version,
        "saved_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "data_start": str(pd.Timestamp(data[config.DATE_COL].min()).date()),
        "data_end": str(pd.Timestamp(data[config.DATE_COL].max()).date()),
        "n_weeks": len(data),
        "roi": {c: m["roi"] for c, m in insights["channels"].items()},
        "blended_roi": insights["totals"]["blended_media_roi"],
        "recommendation": recommendation,
        "metrics": metrics or {},
    }


# --- Simulation (synthetic brands only) -----------------------------------------------------


def simulate_new_weeks(
    brand: BrandConfig,
    history: pd.DataFrame,
    schedule: pd.DataFrame,
    slip: dict[str, float] | None = None,
    shock_share: float = 0.0,
    shock_weeks: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return new weekly rows for a synthetic brand, generated from the true process.

    Actual spend follows ``schedule`` with each channel off by its ``slip`` share; organic
    revenue, controls and noise come from the generator run over the longer period; media
    revenue is recomputed with the true curves, carrying over from the history's spend. The
    last ``shock_weeks`` lose ``shock_share`` of baseline revenue to something the model does
    not know about (a competitor launch). Also returns the true contributions of those weeks.
    """
    n_new = len(schedule)
    longer = generate(dataclasses.replace(brand, n_weeks=len(history) + n_new))
    tail = longer.data.iloc[len(history) :].reset_index(drop=True)
    parts = longer.contributions.iloc[len(history) :].reset_index(drop=True)
    new = tail.copy()
    new[config.DATE_COL] = schedule.index
    media = {}
    for spec in brand.channels:
        column = f"{config.SPEND_PREFIX}{spec.name}"
        planned = schedule[column].to_numpy(dtype=float)
        actual = np.round(planned * (1 + (slip or {}).get(spec.name, 0.0)))
        series = np.concatenate([history[column].to_numpy(dtype=float), actual])
        effect = channel_contribution(series, spec, brand.adstock_l_max, brand.true_saturation)
        media[spec.name] = effect[len(history) :]
        new[column] = actual
    organic = parts[["baseline", "trend", "seasonality", "holiday", "promo", "price"]]
    shock = np.zeros(n_new)
    if shock_weeks:
        shock[-shock_weeks:] = shock_share * brand.baseline_revenue
    revenue = organic.sum(axis=1).to_numpy() + sum(media.values()) + parts["noise"] + shock
    new[config.TARGET_COL] = np.round(revenue)
    truth = organic.assign(**media, shock=shock, noise=parts["noise"])
    truth.insert(0, config.DATE_COL, schedule.index)
    return new[history.columns], truth
