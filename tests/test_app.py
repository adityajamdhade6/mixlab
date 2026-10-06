"""Headless tests of the Streamlit app: every page renders and the key interactions work."""

import json
import sys
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from conftest import TINY
from streamlit.testing.v1 import AppTest

from mixlab import ai_explainer, config
from mixlab.ai_explainer import Explanation, NumberCheck
from mixlab.geo_data import GeoDataset
from mixlab.geo_insights import build_geo_outputs, save_geo_draws
from mixlab.geo_model import GeoMixLabModel, extract_geo_draws
from mixlab.insights import extract_draws
from mixlab.model import MixLabModel

APP_DIR = config.PROJECT_ROOT / "app"
VIEWS = [
    "overview",
    "channels",
    "optimizer",
    "scenarios",
    "ask",
    "regions",
    "test_and_learn",
    "health",
    "upload",
    "how_it_works",
]
TIMEOUT = 180


def write_geo_demo(
    root: Path,
    geo_dataset: GeoDataset,
    geo_fitted: GeoMixLabModel,
    df: pd.DataFrame,
    fitted: MixLabModel,
) -> None:
    """Save the tiny geo fit's regional results where the Regions page looks for them."""
    folder = root / config.GEO_DEMO_BRAND
    folder.mkdir(parents=True, exist_ok=True)
    by_geo = extract_geo_draws(geo_fitted, geo_dataset.data)
    truth = geo_dataset.ground_truth
    summary, comparison = build_geo_outputs(by_geo, extract_draws(fitted, df), truth)
    for name, payload in (
        (config.GEO_SUMMARY_FILENAME, summary),
        (config.GEO_COMPARISON_FILENAME, comparison),
        (config.GEO_GROUND_TRUTH_FILENAME, truth),
    ):
        (folder / name).write_text(json.dumps(payload))
    save_geo_draws(by_geo, folder / config.GEO_DRAWS_FILENAME)
    loop = config.ARTIFACTS_DIR / config.GEO_DEMO_BRAND / config.EXPERIMENT_LOOP_FILENAME
    if loop.exists():  # the real saved story is small; reuse it for the page test
        (folder / config.EXPERIMENT_LOOP_FILENAME).write_text(loop.read_text())


@pytest.fixture(scope="module")
def demo_root(
    tmp_path_factory: pytest.TempPathFactory,
    geo_dataset: GeoDataset,
    geo_fitted: GeoMixLabModel,
    df: pd.DataFrame,
    fitted: MixLabModel,
) -> Iterator[Path]:
    """Build one tiny demo brand and the regional demo in a temp folder; point the app at it."""
    sys.path.insert(0, str(config.PROJECT_ROOT / "scripts"))
    sys.path.insert(0, str(APP_DIR))
    from build_demo import build_brand

    root = tmp_path_factory.mktemp("artifacts")
    build_brand("performance_heavy", TINY, root, estimate_curse=False)
    write_geo_demo(root, geo_dataset, geo_fitted, df, fitted)
    patch = pytest.MonkeyPatch()
    patch.setattr(config, "ARTIFACTS_DIR", root)
    patch.setattr(config, "AI_CACHE_DIR", root / "ai_cache")
    st.cache_data.clear()
    st.cache_resource.clear()
    yield root
    patch.undo()
    st.cache_data.clear()
    st.cache_resource.clear()


def run_view(name: str) -> AppTest:
    app = AppTest.from_file(str(APP_DIR / "views" / f"{name}.py"), default_timeout=TIMEOUT).run()
    assert not app.exception, [e.value for e in app.exception]
    return app


@pytest.mark.parametrize("view", VIEWS)
def test_every_page_renders(demo_root: Path, view: str) -> None:
    app = run_view(view)
    assert app.title[0].value
    if view != "health":  # the health page reports failed sampler checks of the tiny test fit
        assert not app.error


def test_overview_shows_profit_verdict_kpis_and_both_uplifts(demo_root: Path) -> None:
    app = run_view("overview")
    labels = [m.label for m in app.metric]
    assert labels[:3] == ["Media spend", "Marketing revenue", "Marketing share of revenue"]
    assert "ROI: revenue per ₹1" in labels and "Profit ROI: profit per ₹1" in labels
    assert not any(value.endswith("…") for value in (m.value for m in app.metric))
    assert any("Performance-heavy demo brand" in caption.value for caption in app.caption)
    assert labels.index("Realistic uplift") == labels.index("Expected uplift") + 1
    verdict = [*app.success, *app.warning][0].value
    assert "margin" in verdict and "gross profit" in verdict
    assert any("94% range:" in caption.value for caption in app.caption)
    assert any("disproportionately overestimates" in caption.value for caption in app.caption)
    assert any("##### Headline finding" in block.value for block in app.markdown)
    assert any(config.AUTHOR_NAME in caption.value for caption in app.caption)


def test_margin_slider_changes_the_profit_verdict(demo_root: Path) -> None:
    app = run_view("overview")
    app.slider[0].set_value(95).run()
    high = [*app.success, *app.warning][0].value
    app.slider[0].set_value(5).run()
    low = [*app.success, *app.warning][0].value
    assert "95% margin" in high and "5% margin" in low
    assert "does not pay for itself" in low


def test_overview_explains_a_missing_api_key(
    demo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_key(*args: object, **kwargs: object) -> None:
        raise ai_explainer.ExplainerError(ai_explainer.NO_CREDENTIALS)

    monkeypatch.setattr(ai_explainer, "generate_brief", no_key)
    app = run_view("overview")
    app.button[0].click().run()
    assert not app.exception
    assert "ANTHROPIC_API_KEY" in app.error[0].value


def test_optimizer_runs_and_respects_the_budget(demo_root: Path) -> None:
    app = run_view("optimizer")
    assert any("saved recommendation" in caption.value for caption in app.caption)
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert not any("Showing the saved recommendation" in caption.value for caption in app.caption)
    assert app.metric[0].label == "Expected uplift"


def test_optimizer_explains_impossible_limits(demo_root: Path) -> None:
    app = run_view("optimizer")
    app.number_input[0].set_value(1.0)
    app.button[0].click().run()
    assert not app.exception
    assert "cannot be met" in app.error[0].value


def scenario_sliders(app: AppTest) -> list:
    return [slider for slider in app.slider if slider.label != "Product margin"]


def test_scenario_page_starts_in_a_neutral_state(demo_root: Path) -> None:
    app = run_view("scenarios")
    values = {m.label: m.value for m in app.metric}
    assert values["Chance revenue rises"] == "—" and values["Profit change"] == "—"
    assert next(b for b in app.button if b.label == "Save scenario").disabled


def test_scenario_sliders_update_revenue_and_scenarios_can_be_saved(demo_root: Path) -> None:
    app = run_view("scenarios")
    before = app.metric[1].value
    for slider in scenario_sliders(app):
        slider.set_value(-100)
    app.run()
    assert app.metric[0].value == "₹0.00 Cr" and app.metric[1].value != before
    assert {m.label: m.value for m in app.metric}["Chance revenue rises"] == "<1%"
    app.text_input[0].set_value("All off")
    next(b for b in app.button if b.label == "Save scenario").click().run()
    assert not app.exception
    assert len(app.dataframe) == 1 and "All off" in app.dataframe[0].value.index


def test_saving_an_unnamed_scenario_is_refused(demo_root: Path) -> None:
    app = run_view("scenarios")
    scenario_sliders(app)[0].set_value(20)
    app.run()
    next(b for b in app.button if b.label == "Save scenario").click().run()
    assert "name" in app.error[0].value


def test_ask_page_shows_answer_and_grounding(
    demo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_ask(context: object, question: str, tone: str) -> Explanation:
        return Explanation(
            kind="answer",
            tone=tone,
            model="test-model",
            text=f"Answer to: {question}",
            number_check=NumberCheck(n_numbers=0, n_verified=0),
        )

    monkeypatch.setattr(ai_explainer, "ask", fake_ask)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-key-for-tests")
    app = run_view("ask")
    assert not app.warning
    app.button[0].click().run()
    assert not app.exception
    assert [m.name for m in app.chat_message] == ["user", "assistant"]
    assert "Answer to: Why should I cut Meta?" in app.chat_message[1].markdown[0].value


def test_ask_page_says_upfront_when_no_key_is_set(
    demo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(config, "ENV_FILE", demo_root / "no.env")
    app = run_view("ask")
    assert "needs a Claude API key" in app.warning[0].value
    assert all(button.disabled for button in app.button)


def test_health_page_reports_out_of_sample_accuracy_recovery_and_notes(demo_root: Path) -> None:
    app = run_view("health")
    labels = [m.label for m in app.metric]
    assert {"True ROI recovered", "Holdout MAPE", "Holdout R²"} <= set(labels)
    assert any("revenue, not profit" in block.value for block in app.markdown)
    assert any("Rolling backtest" in block.value for block in app.markdown)
    backtest = app.dataframe[0].value
    assert len(backtest) == config.BACKTEST_FOLDS and "MAPE (%)" in backtest.columns
    captions = " ".join(caption.value for caption in app.caption)
    assert "(s)" not in captions and "1 warnings" not in captions


def test_optimizer_page_explains_limits_rollout_and_schedule(demo_root: Path) -> None:
    app = run_view("optimizer")
    limits = app.dataframe[0].value.set_index("Channel")
    assert limits.loc["TV", "Default limit"] == "±10%" and "bursts" in limits.loc["TV", "Why"]
    assert limits.loc["Email", "Default limit"] == "±10%"
    labels = [slider.label for slider in app.slider]
    assert any(label.startswith("TV") and "±10%" in label for label in labels)
    allocation = app.dataframe[1].value.set_index("Channel")
    assert allocation.loc["TV", "Caveat"] and not allocation.loc["Meta Ads", "Caveat"]
    assert "Realistic uplift" in [m.label for m in app.metric]
    rollout = app.dataframe[2].value
    assert len(rollout) == config.ROLLOUT_STEPS and "Stop if" in rollout.columns
    assert any("How to roll it out" in h.value for h in app.subheader)


def test_optimizer_goal_search_reports_impossible_targets(demo_root: Path) -> None:
    app = run_view("optimizer")
    target = next(n for n in app.number_input if "revenue to reach" in n.label)
    target.set_value(9999.0)
    next(b for b in app.button if b.label == "Find the plan").click().run()
    assert not app.exception
    assert "No plan can meet that goal" in app.error[0].value


def test_app_explains_when_nothing_is_built(
    demo_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path)
    app = AppTest.from_file(str(APP_DIR / "views" / "overview.py"), default_timeout=TIMEOUT).run()
    assert not app.exception
    assert "No fitted demo brands" in app.error[0].value


def test_money_is_formatted_with_the_sign_before_the_rupee(demo_root: Path) -> None:
    from common import crore, lakh

    assert crore(-200_000, 2) == "-₹0.02 Cr" and crore(54_600_000) == "₹5.5 Cr"
    assert lakh(-150_000, 1) == "-₹1.5 L"


def test_small_channel_keeps_a_usable_gated_range(demo_root: Path) -> None:
    app = run_view("optimizer")
    email = next(slider for slider in app.slider if slider.label.startswith("Email"))
    low, high = email.value
    assert high > low and email.step == 0.1


def test_regions_page_shows_map_comparison_and_optimizer(demo_root: Path) -> None:
    app = run_view("regions")
    labels = [m.label for m in app.metric]
    assert labels[:3] == ["Regions", "ROI ranges vs. national model", "True ROI recovered"]
    assert "True extra revenue" in labels
    assert any("narrower" in block.value for block in app.markdown)
    assert any("94% range:" in caption.value for caption in app.caption)
    assert len(app.get("plotly_chart")) == 4
    app.radio[0].set_value("TV").run()
    assert not app.exception
    app.selectbox(key="region").set_value("karnataka").run()
    assert any("**Karnataka**" in block.value for block in app.markdown)


def test_regions_page_reruns_the_optimizer_with_a_new_limit(demo_root: Path) -> None:
    app = run_view("regions")
    app.slider[0].set_value(15).run()
    assert any("saved plan" in info.value for info in app.info)
    app.button[0].click().run()
    assert not app.exception
    assert not app.info  # the re-run result for 15% is now shown


def test_regions_page_explains_when_the_geo_demo_is_missing(
    demo_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path)
    st.cache_data.clear()
    app = run_view("regions")
    assert "has not been built" in app.error[0].value
    assert "build_geo_demo.py" in app.code[0].value


def test_test_and_learn_page_tells_the_whole_loop(demo_root: Path) -> None:
    if not (demo_root / config.GEO_DEMO_BRAND / config.EXPERIMENT_LOOP_FILENAME).exists():
        pytest.skip("experiment_loop.json has not been built")
    app = run_view("test_and_learn")
    headers = [h.value for h in app.subheader]
    assert headers[0].startswith("1.") and headers[-1].startswith("5.")
    assert len(app.get("plotly_chart")) >= 2
    assert any("narrower" in block.value for block in app.markdown)


def test_test_and_learn_page_explains_when_not_run(
    demo_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path)
    st.cache_data.clear()
    app = run_view("test_and_learn")
    assert "has not been run" in app.error[0].value
