"""Tests for the validator, mostly against deliberately broken data."""

import pandas as pd
import pytest

from mixlab import config
from mixlab.data_gen import generate
from mixlab.validate import Issue, Severity, ValidationReport, validate, validate_file


@pytest.fixture(scope="module")
def clean() -> pd.DataFrame:
    return generate(config.PERFORMANCE_HEAVY_BRAND).data


def found(report: ValidationReport, check: str, severity: Severity) -> list[Issue]:
    return [i for i in report.issues if i.check == check and i.severity == severity]


def test_clean_data_has_no_critical_issues(clean: pd.DataFrame) -> None:
    report = validate(clean)
    assert report.count(Severity.CRITICAL) == 0
    assert report.n_weeks == 156
    assert len(report.channels) == 6
    assert sum(report.spend_share.values()) == pytest.approx(1.0)
    assert set(report.vif) == set(report.channels)


def test_score_drops_when_data_is_broken(clean: pd.DataFrame) -> None:
    broken = clean.drop(index=[10, 11])
    assert validate(broken).readiness_score < validate(clean).readiness_score


def test_missing_required_column_stops_early(clean: pd.DataFrame) -> None:
    report = validate(clean.drop(columns=[config.TARGET_COL]))
    assert [i.check for i in report.issues] == ["required_columns"]
    assert report.issues[0].column == config.TARGET_COL


def test_no_spend_columns_is_critical(clean: pd.DataFrame) -> None:
    report = validate(clean[[config.DATE_COL, config.TARGET_COL]])
    assert found(report, "required_columns", Severity.CRITICAL)


def test_non_numeric_spend_is_critical(clean: pd.DataFrame) -> None:
    broken = clean.assign(spend_email=clean["spend_email"].astype(str) + " INR")
    assert found(validate(broken), "dtypes", Severity.CRITICAL)


def test_missing_weeks_are_reported(clean: pd.DataFrame) -> None:
    report = validate(clean.drop(index=[20, 21, 60]))
    (issue,) = found(report, "weekly_frequency", Severity.CRITICAL)
    assert len(issue.details["missing_weeks"]) == 3
    assert str(clean[config.DATE_COL].iloc[20].date()) in issue.details["missing_weeks"]


def test_duplicate_weeks_are_reported(clean: pd.DataFrame) -> None:
    broken = pd.concat([clean, clean.iloc[[5]]], ignore_index=True)
    (issue,) = found(validate(broken), "weekly_frequency", Severity.CRITICAL)
    assert "more than once" in issue.message


def test_missing_values_are_reported(clean: pd.DataFrame) -> None:
    broken = clean.copy()
    broken.loc[3, "spend_meta_ads"] = None
    broken.loc[4, config.PRICE_COL] = None
    report = validate(broken)
    assert found(report, "missing_values", Severity.CRITICAL)[0].column == "spend_meta_ads"
    assert found(report, "missing_values", Severity.WARNING)[0].column == config.PRICE_COL


def test_negative_spend_is_critical(clean: pd.DataFrame) -> None:
    broken = clean.copy()
    broken.loc[[7, 8], "spend_youtube"] = -5000.0
    (issue,) = found(validate(broken), "negative_values", Severity.CRITICAL)
    assert issue.column == "spend_youtube"
    assert issue.details["count"] == 2


def test_perfectly_correlated_channels_are_critical(clean: pd.DataFrame) -> None:
    broken = clean.assign(spend_meta_copy=clean["spend_meta_ads"] * 0.5)
    report = validate(broken)
    (issue,) = found(report, "multicollinearity", Severity.CRITICAL)
    assert set(issue.details["channels"]) == {"meta_ads", "meta_copy"}
    assert issue.details["correlation"] == pytest.approx(1.0)
    assert report.vif["meta_copy"] == float("inf")


def test_all_zero_channel_is_critical(clean: pd.DataFrame) -> None:
    report = validate(clean.assign(spend_radio=0.0))
    assert found(report, "zero_spend", Severity.CRITICAL)[0].column == "spend_radio"


def test_flighted_channel_is_info_not_error(clean: pd.DataFrame) -> None:
    assert found(validate(clean), "zero_spend", Severity.INFO)[0].column == "spend_tv"


def test_flat_spend_warns_model_cannot_learn(clean: pd.DataFrame) -> None:
    report = validate(clean.assign(spend_meta_ads=900_000.0))
    (issue,) = found(report, "spend_variation", Severity.WARNING)
    assert "can't learn from a channel whose spend never changes" in issue.message


def test_short_history_warns(clean: pd.DataFrame) -> None:
    report = validate(clean.head(60))
    assert found(report, "history_length", Severity.WARNING)
    assert not found(validate(clean), "history_length", Severity.WARNING)


def test_tiny_channel_warns(clean: pd.DataFrame) -> None:
    report = validate(clean.assign(spend_email=clean["spend_email"] * 0.01))
    assert "spend_email" in [i.column for i in found(report, "spend_share", Severity.WARNING)]


def test_outliers_are_flagged_not_deleted(clean: pd.DataFrame) -> None:
    broken = clean.copy()
    broken.loc[30, "spend_google_search"] *= 20
    report = validate(broken)
    flagged = [
        i for i in found(report, "outliers", Severity.INFO) if i.column == "spend_google_search"
    ]
    assert str(clean[config.DATE_COL].iloc[30].date()) in flagged[0].details["weeks"]
    assert len(broken) == len(clean)


def test_issues_are_ranked_by_severity(clean: pd.DataFrame) -> None:
    broken = clean.drop(index=[20]).assign(spend_meta_ads=900_000.0)
    order = [i.severity for i in validate(broken).issues]
    rank = {Severity.CRITICAL: 0, Severity.WARNING: 1, Severity.INFO: 2}
    assert order == sorted(order, key=rank.__getitem__)


def test_score_is_bounded(clean: pd.DataFrame) -> None:
    wrecked = clean.drop(index=range(5, 40)).assign(spend_a=0.0, spend_b=0.0, spend_c=-1.0)
    assert validate(wrecked).readiness_score == 0
    assert validate(clean).readiness_score <= config.MAX_READINESS_SCORE


def test_validate_file_reads_csv(clean: pd.DataFrame, tmp_path) -> None:
    path = tmp_path / "weekly.csv"
    clean.to_csv(path, index=False)
    assert validate_file(path).readiness_score == validate(clean).readiness_score
    assert "Readiness score" in validate_file(path).summary()


def test_command_line_reports_and_fails_on_critical_issues(
    clean: pd.DataFrame, tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    from mixlab import validate as module

    good, bad = tmp_path / "good.csv", tmp_path / "bad.csv"
    clean.to_csv(good, index=False)
    clean.drop(index=[5, 6]).to_csv(bad, index=False)
    monkeypatch.setattr("sys.argv", ["validate", str(good)])
    module.main()
    assert "Readiness score" in capsys.readouterr().out
    monkeypatch.setattr("sys.argv", ["validate", str(good), str(bad)])
    with pytest.raises(SystemExit) as exit_info:
        module.main()
    assert exit_info.value.code == 1


def test_messages_use_correct_singular_and_plural(clean: pd.DataFrame) -> None:
    one = validate(clean.drop(index=[20]))
    many = validate(clean.drop(index=[20, 21]))
    assert "1 week is missing" in found(one, "weekly_frequency", Severity.CRITICAL)[0].message
    assert "2 weeks are missing" in found(many, "weekly_frequency", Severity.CRITICAL)[0].message
    broken = clean.copy()
    broken.loc[3, "spend_meta_ads"] = None
    assert (
        "1 missing value."
        in found(validate(broken), "missing_values", Severity.CRITICAL)[0].message
    )
    assert not any("(s)" in issue.message for issue in many.issues)
