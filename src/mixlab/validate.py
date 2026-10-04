"""Data-contract validation for weekly MMM input frames.

``validate(df)`` runs schema, data-quality and MMM-readiness checks and returns a
``ValidationReport``: a readiness score out of 100 plus issues ranked by severity. Nothing is
ever fixed or deleted; every problem is reported for a human to decide on.
"""

import argparse
import sys
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from mixlab import config


class Severity(StrEnum):
    """How much an issue threatens the model: critical blocks it, warning weakens it."""

    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


SEVERITY_RANK: dict[Severity, int] = {Severity.CRITICAL: 0, Severity.WARNING: 1, Severity.INFO: 2}


class Issue(BaseModel):
    """One finding from one check, with a plain-English message."""

    check: str
    severity: Severity
    message: str
    column: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ValidationReport(BaseModel):
    """Structured result of validating one weekly frame."""

    readiness_score: int
    n_weeks: int
    channels: list[str]
    issues: list[Issue]
    spend_share: dict[str, float] = Field(default_factory=dict)
    spend_cv: dict[str, float] = Field(default_factory=dict)
    correlation: dict[str, dict[str, float]] = Field(default_factory=dict)
    vif: dict[str, float] = Field(default_factory=dict)

    def count(self, severity: Severity) -> int:
        """Return how many issues have the given severity."""
        return sum(issue.severity == severity for issue in self.issues)

    def summary(self) -> str:
        """Return a short plain-text summary: score, counts, then issues in ranked order."""
        lines = [
            f"Readiness score: {self.readiness_score}/{config.MAX_READINESS_SCORE} "
            f"({self.n_weeks} weeks, {len(self.channels)} channels)",
            f"{self.count(Severity.CRITICAL)} critical, {self.count(Severity.WARNING)} warning, "
            f"{self.count(Severity.INFO)} info",
        ]
        lines += [f"[{issue.severity.upper()}] {issue.message}" for issue in self.issues]
        return "\n".join(lines)


# --- Helpers --------------------------------------------------------------------------------


def spend_columns(df: pd.DataFrame) -> list[str]:
    """Return the spend columns, in frame order."""
    return [c for c in df.columns if c.startswith(config.SPEND_PREFIX)]


def channel_name(column: str) -> str:
    """Strip the spend prefix from a column name."""
    return column.removeprefix(config.SPEND_PREFIX)


def count_noun(count: int, noun: str) -> str:
    """Return a count with its noun correctly pluralised, e.g. "1 week" or "3 weeks"."""
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def verb(count: int, singular: str, plural: str) -> str:
    """Return the verb form that agrees with ``count``."""
    return singular if count == 1 else plural


def _format_dates(dates: pd.Series | pd.DatetimeIndex) -> str:
    """Return the first few dates as text, noting how many more there are."""
    shown = [d.strftime("%Y-%m-%d") for d in list(dates)[: config.MAX_DATES_IN_MESSAGE]]
    extra = len(dates) - len(shown)
    return ", ".join(shown) + (f" and {extra} more" if extra > 0 else "")


def _numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return the given columns coerced to numbers (unparseable values become NaN)."""
    return df[columns].apply(pd.to_numeric, errors="coerce")


def _varying(spend: pd.DataFrame) -> pd.DataFrame:
    """Return only complete rows and the columns whose spend actually varies."""
    clean = spend.dropna()
    return clean.loc[:, clean.std() > 0]


# --- Schema checks --------------------------------------------------------------------------


def check_required_columns(df: pd.DataFrame) -> list[Issue]:
    """Require the date column, the target column and at least one spend column."""
    issues = [
        Issue(
            check="required_columns",
            severity=Severity.CRITICAL,
            message=f"Required column '{column}' is missing.",
            column=column,
        )
        for column in (config.DATE_COL, config.TARGET_COL)
        if column not in df.columns
    ]
    if not spend_columns(df):
        issues.append(
            Issue(
                check="required_columns",
                severity=Severity.CRITICAL,
                message=f"No spend columns found (expected names like "
                f"'{config.SPEND_PREFIX}<channel>').",
            )
        )
    return issues


def check_dtypes(df: pd.DataFrame) -> list[Issue]:
    """Require a parseable date column and numeric revenue and spend columns."""
    issues: list[Issue] = []
    parsed = pd.to_datetime(df[config.DATE_COL], errors="coerce")
    unparseable = int((parsed.isna() & df[config.DATE_COL].notna()).sum())
    if unparseable:
        issues.append(
            Issue(
                check="dtypes",
                severity=Severity.CRITICAL,
                message=f"{count_noun(unparseable, 'value')} in '{config.DATE_COL}' "
                f"{verb(unparseable, 'is', 'are')} not a valid date.",
                column=config.DATE_COL,
            )
        )
    for column in [config.TARGET_COL, *spend_columns(df)]:
        if not pd.api.types.is_numeric_dtype(df[column]):
            issues.append(
                Issue(
                    check="dtypes",
                    severity=Severity.CRITICAL,
                    message=f"'{column}' should be numeric but is stored as {df[column].dtype}.",
                    column=column,
                )
            )
    return issues


def check_weekly_frequency(dates: pd.Series) -> list[Issue]:
    """Require one row per week: no duplicate weeks and no gaps."""
    issues: list[Issue] = []
    valid = dates.dropna()
    duplicated = valid[valid.duplicated()].drop_duplicates()
    if len(duplicated):
        issues.append(
            Issue(
                check="weekly_frequency",
                severity=Severity.CRITICAL,
                message=f"{count_noun(len(duplicated), 'week')} "
                f"{verb(len(duplicated), 'appears', 'appear')} more than once: "
                f"{_format_dates(duplicated)}. Each week must be a single row.",
                column=config.DATE_COL,
                details={"duplicate_weeks": [str(d.date()) for d in duplicated]},
            )
        )
    unique = valid.drop_duplicates().sort_values()
    if len(unique) < 2:
        return issues
    expected = pd.date_range(unique.iloc[0], unique.iloc[-1], freq=f"{config.DAYS_PER_WEEK}D")
    missing = expected.difference(pd.DatetimeIndex(unique))
    off_grid = pd.DatetimeIndex(unique).difference(expected)
    if len(off_grid):
        issues.append(
            Issue(
                check="weekly_frequency",
                severity=Severity.CRITICAL,
                message=f"{count_noun(len(off_grid), 'date')} {verb(len(off_grid), 'is', 'are')} "
                f"not exactly {config.DAYS_PER_WEEK} days "
                f"apart from the first week: {_format_dates(off_grid)}.",
                column=config.DATE_COL,
            )
        )
    elif len(missing):
        issues.append(
            Issue(
                check="weekly_frequency",
                severity=Severity.CRITICAL,
                message=f"{count_noun(len(missing), 'week')} {verb(len(missing), 'is', 'are')} "
                f"missing: {_format_dates(missing)}. "
                "Carryover effects cannot be estimated across gaps.",
                column=config.DATE_COL,
                details={"missing_weeks": [str(d.date()) for d in missing]},
            )
        )
    return issues


# --- Data quality checks --------------------------------------------------------------------


def check_missing_values(df: pd.DataFrame) -> list[Issue]:
    """Flag missing values: critical in date, revenue or spend; warning in controls."""
    core = {config.DATE_COL, config.TARGET_COL, *spend_columns(df)}
    issues: list[Issue] = []
    for column, count in df.isna().sum().items():
        if count:
            issues.append(
                Issue(
                    check="missing_values",
                    severity=Severity.CRITICAL if column in core else Severity.WARNING,
                    message=f"'{column}' has {count_noun(int(count), 'missing value')}.",
                    column=str(column),
                    details={"count": int(count)},
                )
            )
    return issues


def check_negative_values(df: pd.DataFrame) -> list[Issue]:
    """Flag negative spend or revenue, which usually means refunds or a data error."""
    columns = [config.TARGET_COL, *spend_columns(df)]
    counts = (_numeric(df, columns) < 0).sum()
    return [
        Issue(
            check="negative_values",
            severity=Severity.CRITICAL,
            message=f"'{column}' has {count_noun(int(count), 'negative value')}. Spend and "
            "revenue cannot be "
            "below zero; check for refunds, credits or sign errors.",
            column=column,
            details={"count": int(count)},
        )
        for column, count in counts.items()
        if count
    ]


def check_zero_spend(df: pd.DataFrame) -> list[Issue]:
    """Flag channels that never spend (critical) or are off most weeks (info)."""
    issues: list[Issue] = []
    spend = _numeric(df, spend_columns(df))
    for column in spend.columns:
        zero_share = float((spend[column] == 0).mean())
        if zero_share == 1.0:
            issues.append(
                Issue(
                    check="zero_spend",
                    severity=Severity.CRITICAL,
                    message=f"'{column}' is zero in every week. Remove the channel; there is "
                    "nothing to measure.",
                    column=column,
                )
            )
        elif zero_share >= config.FLIGHTED_ZERO_SHARE:
            issues.append(
                Issue(
                    check="zero_spend",
                    severity=Severity.INFO,
                    message=f"'{column}' is zero in {zero_share:.0%} of weeks. That is fine for "
                    "a channel run in bursts, but its estimate rests on few active weeks and "
                    "will be less certain.",
                    column=column,
                    details={"zero_share": zero_share},
                )
            )
    return issues


def iqr_outlier_mask(values: pd.Series, multiplier: float) -> pd.Series:
    """Return a boolean mask of values outside ``[Q1 - k*IQR, Q3 + k*IQR]``."""
    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    spread = multiplier * (q3 - q1)
    return (values < q1 - spread) | (values > q3 + spread)


def check_outliers(df: pd.DataFrame, dates: pd.Series) -> list[Issue]:
    """Flag unusual weeks with the IQR rule. Outliers are reported, never removed.

    Spend columns are judged on active (non-zero) weeks only, so a burst channel is not
    flagged simply for being switched on.
    """
    issues: list[Issue] = []
    columns = [config.TARGET_COL, *spend_columns(df)]
    numeric = _numeric(df, columns)
    for column in columns:
        values = numeric[column].dropna()
        if column != config.TARGET_COL:
            values = values[values != 0]
        if len(values) < config.MIN_POINTS_FOR_OUTLIERS:
            continue
        mask = iqr_outlier_mask(values, config.OUTLIER_IQR_MULTIPLIER)
        if mask.any():
            flagged = dates.loc[mask[mask].index].dropna()
            issues.append(
                Issue(
                    check="outliers",
                    severity=Severity.INFO,
                    message=f"'{column}' has {count_noun(int(mask.sum()), 'unusual week')}: "
                    f"{_format_dates(flagged)}. Check whether they are real (festive peaks, "
                    "launches) or errors. Nothing has been removed.",
                    column=column,
                    details={"weeks": [str(d.date()) for d in flagged]},
                )
            )
    return issues


# --- MMM readiness checks -------------------------------------------------------------------


def check_history_length(n_weeks: int) -> list[Issue]:
    """Warn when there are fewer than two years of weekly data."""
    if n_weeks >= config.MIN_WEEKS_HISTORY:
        return []
    return [
        Issue(
            check="history_length",
            severity=Severity.WARNING,
            message=f"Only {n_weeks} weeks of history; at least {config.MIN_WEEKS_HISTORY} are "
            "recommended. With under two years the model sees each season once and cannot "
            "tell seasonality apart from marketing.",
            details={"n_weeks": n_weeks},
        )
    ]


def spend_cv(df: pd.DataFrame) -> dict[str, float]:
    """Return the coefficient of variation (std / mean) of each channel's weekly spend."""
    spend = _numeric(df, spend_columns(df))
    means = spend.mean()
    return {
        channel_name(c): float(spend[c].std() / means[c]) if means[c] > 0 else 0.0
        for c in spend.columns
    }


def check_spend_variation(cv: dict[str, float]) -> list[Issue]:
    """Warn about channels whose spend barely moves from week to week."""
    return [
        Issue(
            check="spend_variation",
            severity=Severity.WARNING,
            message=f"'{channel}' spend barely changes (coefficient of variation {value:.2f}). "
            "The model can't learn from a channel whose spend never changes.",
            column=f"{config.SPEND_PREFIX}{channel}",
            details={"cv": value},
        )
        for channel, value in cv.items()
        if value < config.MIN_SPEND_CV
    ]


def spend_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Return the correlation matrix between channels whose spend varies."""
    spend = _varying(_numeric(df, spend_columns(df)))
    return spend.rename(columns=channel_name).corr()


def check_correlation(corr: pd.DataFrame) -> list[Issue]:
    """Warn about channel pairs that move together too closely to tell apart."""
    issues: list[Issue] = []
    names = list(corr.columns)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            r = float(corr.loc[first, second])
            if abs(r) < config.MAX_CHANNEL_CORRELATION:
                continue
            duplicate = abs(r) >= config.DUPLICATE_CHANNEL_CORRELATION
            consequence = (
                "They are effectively one channel to the model; merge them or the split of "
                "credit between them will be arbitrary."
                if duplicate
                else "When two channels always rise and fall together the model cannot tell "
                "which one drove sales, so their individual ROI will be unreliable."
            )
            issues.append(
                Issue(
                    check="multicollinearity",
                    severity=Severity.CRITICAL if duplicate else Severity.WARNING,
                    message=f"'{first}' and '{second}' spend are {r:.2f} correlated. "
                    + consequence,
                    details={"channels": [first, second], "correlation": r},
                )
            )
    return issues


def variance_inflation_factors(df: pd.DataFrame) -> dict[str, float]:
    """Return each channel's VIF: how well the other channels' spend predicts it.

    A VIF of 1 means the channel moves independently; infinity means it is fully explained
    by the others.
    """
    spend = _varying(_numeric(df, spend_columns(df)))
    if spend.shape[1] < 2:
        return {channel_name(c): 1.0 for c in spend.columns}
    standardized = ((spend - spend.mean()) / spend.std()).to_numpy()
    vif: dict[str, float] = {}
    for j, column in enumerate(spend.columns):
        others = np.delete(standardized, j, axis=1)
        coefs, *_ = np.linalg.lstsq(others, standardized[:, j], rcond=None)
        residual = standardized[:, j] - others @ coefs
        unexplained = float(residual @ residual / (standardized[:, j] @ standardized[:, j]))
        vif[channel_name(column)] = float("inf") if np.isclose(unexplained, 0) else 1 / unexplained
    return vif


def check_vif(vif: dict[str, float]) -> list[Issue]:
    """Warn about channels largely predictable from a combination of the others."""
    return [
        Issue(
            check="multicollinearity",
            severity=Severity.WARNING,
            message=f"'{channel}' spend can be predicted from the other channels "
            f"(VIF {value:.1f}, limit {config.MAX_VIF:.0f}). Its effect will be hard to "
            "separate from theirs.",
            column=f"{config.SPEND_PREFIX}{channel}",
            details={"vif": value},
        )
        for channel, value in vif.items()
        if value > config.MAX_VIF
    ]


def spend_share(df: pd.DataFrame) -> dict[str, float]:
    """Return each channel's share of total spend (0-1)."""
    totals = _numeric(df, spend_columns(df)).clip(lower=0).sum()
    grand_total = float(totals.sum())
    return {
        channel_name(c): float(t / grand_total) if grand_total else 0.0 for c, t in totals.items()
    }


def check_spend_share(share: dict[str, float]) -> list[Issue]:
    """Warn about channels too small for their effect to stand out from noise."""
    return [
        Issue(
            check="spend_share",
            severity=Severity.WARNING,
            message=f"'{channel}' is only {value:.1%} of total spend. A channel this small "
            "moves revenue by less than the week-to-week noise, so expect a wide range on its "
            "ROI rather than a precise number.",
            column=f"{config.SPEND_PREFIX}{channel}",
            details={"share": value},
        )
        for channel, value in share.items()
        if 0 < value < config.MIN_SPEND_SHARE
    ]


# --- Report ---------------------------------------------------------------------------------


def readiness_score(issues: list[Issue]) -> int:
    """Return 100 minus a fixed penalty per issue by severity, floored at 0."""
    penalty = sum(config.SEVERITY_PENALTY[issue.severity.value] for issue in issues)
    return max(config.MAX_READINESS_SCORE - penalty, 0)


def rank_issues(issues: list[Issue]) -> list[Issue]:
    """Return issues ordered critical first, then warning, then info."""
    return sorted(issues, key=lambda issue: SEVERITY_RANK[issue.severity])


def validate(df: pd.DataFrame) -> ValidationReport:
    """Run every check on a weekly frame and return the structured report.

    If required columns are missing, only that failure is reported, since no other check can
    run meaningfully.
    """
    issues = check_required_columns(df)
    channels = [channel_name(c) for c in spend_columns(df)]
    if issues:
        return ValidationReport(
            readiness_score=readiness_score(issues),
            n_weeks=len(df),
            channels=channels,
            issues=rank_issues(issues),
        )

    dates = pd.to_datetime(df[config.DATE_COL], errors="coerce")
    cv = spend_cv(df)
    share = spend_share(df)
    corr = spend_correlation(df)
    vif = variance_inflation_factors(df)
    issues = [
        *check_dtypes(df),
        *check_weekly_frequency(dates),
        *check_missing_values(df),
        *check_negative_values(df),
        *check_zero_spend(df),
        *check_outliers(df, dates),
        *check_history_length(int(dates.nunique())),
        *check_spend_variation(cv),
        *check_correlation(corr),
        *check_vif(vif),
        *check_spend_share(share),
    ]
    return ValidationReport(
        readiness_score=readiness_score(issues),
        n_weeks=int(dates.nunique()),
        channels=channels,
        issues=rank_issues(issues),
        spend_share=share,
        spend_cv=cv,
        correlation={k: {i: float(x) for i, x in v.items()} for k, v in corr.to_dict().items()},
        vif=vif,
    )


def validate_file(path: Path) -> ValidationReport:
    """Read a weekly CSV and validate it."""
    return validate(pd.read_csv(path))


def main() -> None:
    """Validate one or more weekly CSV files from the command line.

    Exits with status 1 if any file has a critical issue.
    """
    parser = argparse.ArgumentParser(description="Validate weekly MMM data files.")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    failed = False
    for path in args.paths:
        report = validate_file(path)
        print(f"== {path} ==\n{report.summary()}\n")
        failed = failed or report.count(Severity.CRITICAL) > 0
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
