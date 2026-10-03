"""Tests for the AI explainer. The Claude API is always mocked; no network calls are made."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from mixlab import ai_explainer, config
from mixlab.ai_explainer import (
    BRIEF_TASK,
    TONE_PRESETS,
    TOOL_FUNCTIONS,
    DataContext,
    ExplainerError,
    FoundNumber,
    ask,
    build_facts,
    check_numbers,
    collect_numbers,
    execute_tool,
    extract_numbers,
    generate_brief,
    get_response_curve,
    is_supported,
    make_client,
    render,
    render_facts,
    run_scenario,
    system_blocks,
    tool_definitions,
)
from mixlab.evaluate import convergence_diagnostics
from mixlab.insights import build_summary, extract_draws
from mixlab.model import MixLabModel
from mixlab.optimizer import BudgetAllocator, build_optimizer_summary
from mixlab.validate import validate

BRAND = config.PERFORMANCE_HEAVY_BRAND


# --- Mocked Claude client -------------------------------------------------------------------


class FakeMessages:
    """Stands in for ``client.messages``: returns scripted responses and records requests."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeClient:
    def __init__(self, responses: list[Any]) -> None:
        self.messages = FakeMessages(responses)
        self.beta = SimpleNamespace(messages=self.messages)

    @property
    def requests(self) -> list[dict[str, Any]]:
        return self.messages.requests


def reply(*blocks: Any, stop_reason: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(
        content=list(blocks),
        stop_reason=stop_reason,
        model=config.AI_MODEL,
        usage=SimpleNamespace(input_tokens=100, output_tokens=50, cache_read_input_tokens=0),
    )


def text(value: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=value)


def thinking() -> SimpleNamespace:
    return SimpleNamespace(type="thinking", thinking="", signature="sig")


def tool_use(call_id: str, name: str, tool_input: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=call_id, name=name, input=tool_input)


# --- Fixtures -------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def context(df: pd.DataFrame, fitted: MixLabModel) -> DataContext:
    draws = extract_draws(fitted, df)
    optimizer = build_optimizer_summary(BudgetAllocator(fitted), draws, BRAND, curve_points=4)
    meta = {"diagnostics": convergence_diagnostics(fitted.idata, fitted.parameter_names)}
    return DataContext(build_facts(build_summary(draws), optimizer, meta, validate(df)), draws)


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    return tmp_path / "cache"


# --- Facts ----------------------------------------------------------------------------------


def test_facts_are_in_presentation_units(df: pd.DataFrame, context: DataContext) -> None:
    facts = context.facts
    expected = round(df[config.TARGET_COL].sum() / config.INR_PER_CRORE, 2)
    assert facts["totals_over_data_period"]["revenue_cr"] == expected
    assert set(facts["channels"]) == {c.name for c in BRAND.channels}
    assert set(facts["channels"]["tv"]["roi"]) == {"best_estimate", "low", "high"}
    recommendation = facts["main_recommendation"]
    assert recommendation["window_weeks"] == config.OPTIMIZER_WEEKS
    assert set(recommendation["uplift_vs_current_cr"]) == {"best_estimate", "low", "high"}
    assert "ground_truth_check_synthetic_data_only" in recommendation
    assert facts["validation"]["data"]["readiness_score_out_of_100"] <= 100
    assert "divergences" in facts["validation"]["sampler"]
    assert facts["about"]["interval_pct"] == 94


def test_facts_render_deterministically(context: DataContext) -> None:
    rendered = render_facts(context.facts)
    assert rendered == render_facts(json.loads(json.dumps(context.facts)))
    assert rendered.startswith("<facts>") and rendered.endswith("</facts>")


# --- Number check ---------------------------------------------------------------------------


def values(text_value: str) -> list[float]:
    return [number.value for number in extract_numbers(text_value)]


def test_extract_numbers_handles_currency_signs_and_ranges() -> None:
    assert values("₹15.8 crore, up 7.2% (likely between −0.05 and +0.88)") == [
        15.8,
        7.2,
        0.05,
        0.88,
    ]
    assert values("a range of 0.44–0.88 crore") == [0.44, 0.88]
    assert values("revenue of 1,234.5 and the 10th percentile") == [1234.5, 10.0]


def test_extract_numbers_ignores_list_markers_and_labels() -> None:
    assert values("1. First point\n2) Second point\n## 3. Third") == []
    assert values("Q4 was strong and H1 was weak") == []
    assert extract_numbers("ROI of 2.05 overall")[0].decimals == 2


def test_collect_numbers_walks_nested_structures() -> None:
    source = {"roi": -1.25, "weeks_until_90pct": [3, {"note": "from 2023-01-02"}], "ok": True}
    assert collect_numbers(source) == {1.25, 90.0, 3.0, 2023.0, 1.0, 2.0}


def test_rounding_to_fewer_decimals_is_allowed_but_extra_precision_is_not() -> None:
    def number(raw: str) -> FoundNumber:
        return extract_numbers(raw)[0]

    allowed = {15.77}
    assert is_supported(number("15.77"), allowed)
    assert is_supported(number("15.8"), allowed)
    assert is_supported(number("16"), allowed)
    assert not is_supported(number("15.9"), allowed)
    assert not is_supported(number("15.771"), allowed)


def test_check_numbers_flags_invented_values() -> None:
    check = check_numbers("Uplift of 0.44 crore, ROI 9.99.", [{"uplift_cr": 0.44}])
    assert check.n_numbers == 2 and check.n_verified == 1
    assert [n.raw for n in check.unverified] == ["9.99"]
    assert not check.passed and "NOT FOUND: 9.99" in check.summary()
    assert check_numbers("Uplift of 0.44 crore.", [{"uplift_cr": 0.44}]).passed


# --- Prompts and tones ----------------------------------------------------------------------


def test_tones_share_a_cacheable_prefix_and_differ_after_it() -> None:
    blocks = {tone: system_blocks("<facts>{}</facts>", tone) for tone in TONE_PRESETS}
    assert set(blocks) == {"cmo", "analyst", "founder"}
    assert len({b[0]["text"] for b in blocks.values()}) == 1
    assert len({b[1]["text"] for b in blocks.values()}) == 3
    for stable, _reader in blocks.values():
        assert stable["cache_control"] == {"type": "ephemeral"}
        assert "likely between" in stable["text"]  # uncertainty rule applies to every tone
        assert "do not calculate new numbers" in stable["text"]
    with pytest.raises(ValueError, match="Unknown tone"):
        system_blocks("", "pirate")  # type: ignore[arg-type]


# --- Executive brief ------------------------------------------------------------------------


def grounded_brief(context: DataContext) -> str:
    uplift = context.facts["main_recommendation"]["uplift_vs_current_cr"]
    return (
        "## Headline finding\nReallocating is likely worth between "
        f"{uplift['low']} and {uplift['high']} crore.\n\n1. Keep TV."
    )


def test_brief_request_is_grounded_and_correctly_configured(
    context: DataContext, cache: Path
) -> None:
    client = FakeClient([reply(thinking(), text(grounded_brief(context)))])
    brief = generate_brief(context, "cmo", client=client, cache_dir=cache)

    (request,) = client.requests
    assert request["model"] == config.AI_MODEL == "claude-opus-5-5"
    assert request["thinking"] == {"type": "adaptive"}
    assert request["output_config"] == {"effort": config.AI_EFFORT_BRIEF}
    assert request["fallbacks"] == "default" and request["betas"] == [config.AI_FALLBACK_BETA]
    assert "tool_choice" not in request and "tools" not in request
    assert render_facts(context.facts) in request["system"][0]["text"]
    assert request["system"][1]["text"] == TONE_PRESETS["cmo"]
    assert request["messages"] == [{"role": "user", "content": BRIEF_TASK}]

    assert brief.kind == "brief" and brief.tone == "cmo" and not brief.cached
    assert brief.text == grounded_brief(context)
    assert brief.number_check.passed
    assert brief.usage["output_tokens"] == 50
    assert "numbers found in the input" in render(brief)


def test_brief_flags_a_number_that_is_not_in_the_facts(context: DataContext, cache: Path) -> None:
    invented = grounded_brief(context) + " Revenue will rise 87654.3 crore."
    brief = generate_brief(
        context, "founder", client=FakeClient([reply(text(invented))]), cache_dir=cache
    )
    assert [n.raw for n in brief.number_check.unverified] == ["87654.3"]
    assert "NOT FOUND" in render(brief)


def test_responses_are_cached(context: DataContext, cache: Path) -> None:
    client = FakeClient([reply(text("First.")), reply(text("Second.")), reply(text("Third."))])
    first = generate_brief(context, "cmo", client=client, cache_dir=cache)
    again = generate_brief(context, "cmo", client=client, cache_dir=cache)
    assert len(client.requests) == 1
    assert again.cached and again.text == first.text == "First."

    other_tone = generate_brief(context, "analyst", client=client, cache_dir=cache)
    assert len(client.requests) == 2 and other_tone.text == "Second."

    fresh = generate_brief(context, "cmo", client=client, use_cache=False, cache_dir=cache)
    assert len(client.requests) == 3 and fresh.text == "Third." and not fresh.cached


def test_fallback_can_be_switched_off(
    context: DataContext, cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "AI_REFUSAL_FALLBACK", False)
    client = FakeClient([reply(text("Fine."))])
    client.beta = None  # the plain endpoint must be used
    generate_brief(context, "cmo", client=client, cache_dir=cache)
    assert "fallbacks" not in client.requests[0] and "betas" not in client.requests[0]


# --- Tools ----------------------------------------------------------------------------------


def test_tool_schemas_match_the_real_functions(context: DataContext) -> None:
    tools = tool_definitions(context.channels)
    assert {tool["name"] for tool in tools} == set(TOOL_FUNCTIONS)
    for tool in tools:
        schema = tool["input_schema"]
        assert schema["additionalProperties"] is False
        assert schema["required"] == list(schema["properties"])
    by_name = {tool["name"]: tool for tool in tools}
    channel_schema = by_name["get_channel_metrics"]["input_schema"]["properties"]["channel"]
    assert channel_schema["enum"] == context.channels


def test_run_scenario_pause_cut_and_shift(context: DataContext) -> None:
    paused = run_scenario(context, [{"channel": "tv", "kind": "percent_change", "value": -100}], [])
    assert paused["spend_by_channel_lakh"]["tv"]["scenario"] == 0
    assert paused["revenue_change_vs_last_quarter_cr"]["best_estimate"] <= 0
    assert paused["total_spend_cr"]["change"] < 0

    shifted = run_scenario(
        context, [], [{"from_channel": "meta_ads", "to_channel": "youtube", "pct_of_from": 15}]
    )
    assert shifted["total_spend_cr"]["change"] == 0
    spend = shifted["spend_by_channel_lakh"]
    assert spend["meta_ads"]["scenario"] < spend["meta_ads"]["last_quarter"]
    assert spend["youtube"]["scenario"] > spend["youtube"]["last_quarter"]

    fixed = run_scenario(context, [{"channel": "email", "kind": "set_spend_lakh", "value": 20}], [])
    assert fixed["spend_by_channel_lakh"]["email"]["scenario"] == 20
    with pytest.raises(ValueError, match="negative"):
        run_scenario(context, [{"channel": "tv", "kind": "percent_change", "value": -150}], [])


def test_response_curve_rises_with_spend(context: DataContext) -> None:
    curve = get_response_curve(context, "google_search")
    spend = [point["weekly_spend_lakh"] for point in curve["points"]]
    revenue = [point["weekly_incremental_revenue_lakh"] for point in curve["points"]]
    assert spend == sorted(spend) and len(spend) == len(config.AI_RESPONSE_CURVE_MULTIPLES)
    assert all(
        a["best_estimate"] <= b["best_estimate"] for a, b in zip(revenue, revenue[1:], strict=False)
    )
    assert all(point["low"] <= point["best_estimate"] <= point["high"] for point in revenue)


def test_tool_errors_are_returned_not_raised(context: DataContext) -> None:
    assert execute_tool(context, "delete_everything", {}).is_error
    bad_channel = execute_tool(context, "get_channel_metrics", {"channel": "radio"})
    assert bad_channel.is_error and "Unknown channel" in bad_channel.output
    without_model = DataContext(context.facts, draws=None)
    assert execute_tool(without_model, "get_response_curve", {"channel": "tv"}).is_error
    assert not execute_tool(without_model, "get_model_health", {}).is_error


# --- Ask your data --------------------------------------------------------------------------


def test_ask_runs_the_real_tools_and_grounds_the_answer(context: DataContext, cache: Path) -> None:
    pause_tv = {
        "changes": [{"channel": "tv", "kind": "percent_change", "value": -100}],
        "shifts": [],
    }
    expected = run_scenario(context, **pause_tv)
    change = expected["revenue_change_vs_last_quarter_cr"]
    answer = (
        f"Pausing TV likely changes revenue by between {change['low']} and {change['high']} "
        "crore. I would also expect 424242 new customers."
    )
    first = reply(
        thinking(),
        tool_use("call_1", "run_scenario", pause_tv),
        tool_use("call_2", "get_channel_metrics", {"channel": "tv"}),
        stop_reason="tool_use",
    )
    client = FakeClient([first, reply(text(answer))])

    result = ask(
        context, "What happens if we pause TV for a quarter?", "founder", client, cache_dir=cache
    )

    opening, follow_up = client.requests
    assert {tool["name"] for tool in opening["tools"]} == set(TOOL_FUNCTIONS)
    assert "tool_choice" not in opening
    assert opening["output_config"] == {"effort": config.AI_EFFORT_QA}
    assert opening["messages"] == [
        {"role": "user", "content": "What happens if we pause TV for a quarter?"}
    ]

    _question, assistant_turn, tool_turn = follow_up["messages"]
    assert assistant_turn == {"role": "assistant", "content": first.content}  # thinking kept
    results = tool_turn["content"]
    assert [r["tool_use_id"] for r in results] == ["call_1", "call_2"]
    assert json.loads(results[0]["content"]) == expected  # the real function's output
    assert json.loads(results[1]["content"])["channel"] == "tv"
    assert not any(r["is_error"] for r in results)

    assert [call.name for call in result.tool_calls] == ["run_scenario", "get_channel_metrics"]
    assert result.text == answer and result.kind == "answer"
    assert [n.raw for n in result.number_check.unverified] == ["424242"]
    assert result.usage["input_tokens"] == 200
    assert "Tools called: run_scenario, get_channel_metrics" in render(result)

    cached = ask(
        context, "What happens if we pause TV for a quarter?", "founder", client, cache_dir=cache
    )
    assert cached.cached and len(client.requests) == 2


def test_ask_reports_tool_errors_back_to_claude(context: DataContext, cache: Path) -> None:
    client = FakeClient(
        [
            reply(
                tool_use("c1", "get_channel_metrics", {"channel": "radio"}), stop_reason="tool_use"
            ),
            reply(text("There is no radio channel in this model.")),
        ]
    )
    result = ask(context, "How is radio doing?", client=client, cache_dir=cache)
    error_result = client.requests[1]["messages"][2]["content"][0]
    assert error_result["is_error"] and "Unknown channel" in error_result["content"]
    assert result.tool_calls[0].is_error and result.number_check.passed


def test_ask_stops_after_too_many_tool_rounds(context: DataContext, cache: Path) -> None:
    looping = [
        reply(tool_use(f"c{i}", "get_model_health", {}), stop_reason="tool_use")
        for i in range(config.AI_MAX_TOOL_ROUNDS)
    ]
    with pytest.raises(ExplainerError, match="rounds of tool calls"):
        ask(context, "Loop forever", client=FakeClient(looping), cache_dir=cache)


# --- Failure handling and credentials -------------------------------------------------------


def test_refusal_is_surfaced(context: DataContext, cache: Path) -> None:
    client = FakeClient([reply(stop_reason="refusal")])
    with pytest.raises(ExplainerError, match="declined"):
        generate_brief(context, client=client, cache_dir=cache)


def test_missing_credentials_give_setup_instructions(context: DataContext, cache: Path) -> None:
    error = TypeError("Could not resolve authentication method. Expected one of api_key ...")
    with pytest.raises(ExplainerError, match=r"\.env"):
        generate_brief(context, client=FakeClient([error]), cache_dir=cache)


def test_truncated_response_is_marked(context: DataContext, cache: Path) -> None:
    client = FakeClient([reply(text("Cut off mid"), stop_reason="max_tokens")])
    brief = generate_brief(context, client=client, cache_dir=cache)
    assert brief.truncated and "token limit" in render(brief)


def test_api_key_is_read_from_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("ANTHROPIC_API_KEY=dummy-key-for-tests\n")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(config, "ENV_FILE", env_file)
    try:
        assert make_client().api_key == "dummy-key-for-tests"
    finally:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def test_module_exposes_three_tones() -> None:
    assert sorted(ai_explainer.TONE_PRESETS) == ["analyst", "cmo", "founder"]


def test_template_brief_is_grounded_and_exports_to_pdf(context: DataContext) -> None:
    from mixlab.ai_explainer import brief_to_pdf, template_brief

    brief = template_brief(context.facts, name=str.title)
    assert brief.startswith("## Headline finding") and "likely between" in brief
    assert check_numbers(brief, [context.facts]).passed
    pdf = brief_to_pdf(brief + "\n\n---\n₹ and “quotes” render", "MixLab brief")
    assert pdf.startswith(b"%PDF") and len(pdf) > 5000
