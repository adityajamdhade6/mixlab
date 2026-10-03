"""Headless tests of the Streamlit app: every page renders and the key interactions work."""

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
import streamlit as st
from conftest import TINY
from streamlit.testing.v1 import AppTest

from mixlab import ai_explainer, config
from mixlab.ai_explainer import Explanation, NumberCheck

APP_DIR = config.PROJECT_ROOT / "app"
VIEWS = ["overview", "channels", "optimizer", "scenarios", "ask", "health", "upload"]
TIMEOUT = 180


@pytest.fixture(scope="module")
def demo_root(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Build one tiny demo brand in a temp folder and point the app at it."""
    sys.path.insert(0, str(config.PROJECT_ROOT / "scripts"))
    sys.path.insert(0, str(APP_DIR))
    from build_demo import build_brand

    root = tmp_path_factory.mktemp("artifacts")
    build_brand("performance_heavy", TINY, root)
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


def test_overview_shows_kpis_and_a_grounded_summary(demo_root: Path) -> None:
    app = run_view("overview")
    assert [m.label for m in app.metric] == [
        "Media spend",
        "Marketing revenue",
        "Blended ROI",
        "Marketing share",
    ]
    assert any("likely between" in caption.value for caption in app.caption)
    assert any("##### Headline finding" in block.value for block in app.markdown)


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
    assert app.info  # saved recommendation shown before the first run
    app.button[0].click().run()
    assert not app.exception and not app.error and not app.info
    assert app.metric[0].label == "Expected uplift"


def test_optimizer_explains_impossible_limits(demo_root: Path) -> None:
    app = run_view("optimizer")
    app.number_input[0].set_value(1.0)
    app.button[0].click().run()
    assert not app.exception
    assert "cannot be met" in app.error[0].value


def test_scenario_sliders_update_revenue_and_scenarios_can_be_saved(demo_root: Path) -> None:
    app = run_view("scenarios")
    before = app.metric[1].value
    for slider in app.slider:
        slider.set_value(-100)
    app.run()
    assert app.metric[0].value == "₹0.00 Cr" and app.metric[1].value != before
    app.text_input[0].set_value("All off")
    next(b for b in app.button if b.label == "Save scenario").click().run()
    assert not app.exception
    assert len(app.dataframe) == 1 and "All off" in app.dataframe[0].value.index


def test_saving_an_unnamed_scenario_is_refused(demo_root: Path) -> None:
    app = run_view("scenarios")
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
    app = run_view("ask")
    app.button[0].click().run()
    assert not app.exception
    assert [m.name for m in app.chat_message] == ["user", "assistant"]
    assert "Answer to: Why should I cut Meta?" in app.chat_message[1].markdown[0].value


def test_health_page_reports_recovery_and_trust_notes(demo_root: Path) -> None:
    app = run_view("health")
    assert "True ROI recovered" in [m.label for m in app.metric]
    assert any("revenue, not profit" in block.value for block in app.markdown)


def test_app_explains_when_nothing_is_built(
    demo_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path)
    app = AppTest.from_file(str(APP_DIR / "views" / "overview.py"), default_timeout=TIMEOUT).run()
    assert not app.exception
    assert "No fitted demo brands" in app.error[0].value
