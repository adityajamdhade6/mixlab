"""Claude-powered layer that explains model results to a non-technical CMO.

Two entry points:

* ``generate_brief`` writes a one-page executive brief from a compact FACTS document built
  from the insights summary, the optimizer output and the validation results.
* ``ask`` answers a free-form question. Claude is given tools wired to the real analysis
  functions (channel metrics, scenario simulation, response curves) and calls them instead of
  guessing.

Grounding: the model is told to use only numbers it was given, and ``check_numbers`` then
extracts every number from the response and verifies it against the facts and tool results.
The check confirms a number exists in the input (allowing rounding to fewer decimals); it
cannot confirm the number is attached to the right claim, so flagged numbers are a hard
signal and a clean check is a necessary, not sufficient, condition.

Responses are cached on disk, keyed by the full request, so repeated calls cost nothing.
The API key is read from ``.env`` (``ANTHROPIC_API_KEY``).

Run ``python -m mixlab.ai_explainer examples`` to write one brief per tone.
"""

import argparse
import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import anthropic
import matplotlib
import pandas as pd
from dotenv import load_dotenv
from fpdf import FPDF
from pydantic import BaseModel, Field

from mixlab import config
from mixlab.evaluate import divergence_limit, is_converged
from mixlab.insights import PosteriorDraws, extract_draws, hdi, response_curve_draws
from mixlab.model import MixLabModel
from mixlab.optimizer import compare_scenarios, last_quarter_spend, what_if
from mixlab.validate import Severity, ValidationReport, validate

Tone = Literal["cmo", "analyst", "founder"]


class ExplainerError(RuntimeError):
    """A problem the user can act on: missing key, rejected request, refusal."""


# --- Prompts --------------------------------------------------------------------------------

SHARED_RULES = """\
You are the explanation layer of MixLab, a marketing mix model built for a direct-to-consumer \
skincare brand in India. A statistical model has already estimated how much revenue each \
marketing channel drives, and an optimizer has proposed how to reallocate the budget. Your job \
is to explain those results to a specific reader. You do not redo the analysis; you report it \
faithfully.

The reader will move real budget based on what you write, so two things matter more than \
anything else.

First, every number you write must come from the material you are given: the FACTS document, \
or the result of a tool call. Copy numbers as they appear. You may round to fewer decimals, \
but do not calculate new numbers: no sums, differences, ratios or unit conversions. If a \
number you want is not available, say so rather than estimating it. An automated check \
compares every number in your answer against the material and flags any that do not match, \
and a flagged answer is not shown to the reader.

Second, every modelled quantity is uncertain and the reader needs to see that. A modelled \
quantity is given as `best_estimate`, `low` and `high`: the model's central estimate and the \
range it considers 94% likely. When you cite one, give the range in words, for example \
"likely between 2 and 34 crore". Do not present a best estimate on its own as if it were a \
measurement, and do not add decimals the material does not have. Where a probability is \
given (a field ending `_pct` that describes a chance), use it. Spend and observed revenue are \
measured, not modelled, and can be stated plainly.

Units and terms. Money is in Indian rupees: fields ending `_cr` are crore and fields ending \
`_lakh` are lakh. Fields ending `_pct` are percentages. ROI is revenue per rupee spent; it is \
revenue-based, with no product margin applied, so an ROI of 1 means a rupee of revenue per \
rupee spent, not break-even on profit. Marginal ROI is what the next rupee earns, which is \
what a budget decision actually changes. Profit ROI is gross profit per rupee spent (ROI \
times the product margin); above 1 the spend pays for itself. When you name a range, call it \
the 94% range. Report a chance below 1% as "almost no chance (<1%)" and above 99% as "almost \
certain (>99%)" rather than 0% or 100%. "Incremental revenue" is revenue caused by marketing, \
excluding the baseline that would have happened anyway.

Write counts that are not data as words ("three insights", "two risks") and do not number \
your headings, so that every digit in your answer is a real data point.\
"""

TONE_PRESETS: dict[str, str] = {
    "cmo": """\
The reader is the Chief Marketing Officer. They are commercially sharp and not a \
statistician. Write in business language: "likely range" rather than "credible interval", \
"carryover" rather than "adstock", "diminishing returns" rather than "saturation". Do not use \
statistical terms such as posterior, HDI, r-hat, MCMC or prior. Lead with the decision and \
what it is worth. Use everyday channel names (Meta, Google Search, YouTube, influencers, \
email, TV). Keep it to one page: about 300 to 350 words.""",
    "analyst": """\
The reader is a marketing analyst who will have to defend these numbers. Be technical and \
precise. Report each 94% HDI explicitly as [low, high], state how a metric is defined when it \
matters, and report the sampler diagnostics (r-hat, effective sample size, divergences) with \
their values. Call out identification problems directly: wide intervals, allocations pinned \
at a bound, spend beyond the historical range, and any gap between what the model expected \
and what the ground-truth check found. Keep it to one page: up to about 450 words.""",
    "founder": """\
The reader is the founder, who wants the answer in under a minute. Be short and blunt: about \
150 words at most, short sentences, no preamble and no softening beyond the ranges \
themselves. Say what to do, what it is likely worth as a range, and the main way it could be \
wrong.""",
}

BRIEF_TASK = """\
Write a one-page brief for this reader from the FACTS. Use Markdown with a short heading for \
each of these four parts, in this order:

- Headline finding: the single most important thing, in one or two sentences, including \
whether marketing pays for itself after the product margin.
- Top insights: the three findings that most change how the budget should be thought about.
- Recommended budget shift: which channels go up and down under the main recommendation, the \
expected uplift as a range, the realistic uplift beside it with one sentence on why it is \
lower, and the chance it beats the current plan. Where a channel carries a caveat in FACTS, \
state that caveat right next to its change.
- Risks and caveats: the two things most likely to make this recommendation wrong.

Start directly with the headline; do not add an introduction or a sign-off.\
"""

QA_GUIDANCE = """\
You are answering a question from this reader. You have tools that return the model's actual \
results; use them to get the numbers you need rather than relying on the context below, which \
only tells you what exists. If the question asks what would happen under a different spend \
plan, run that scenario. Call several tools at once when the question needs more than one. \
Answer the question that was asked, directly, and stop.\
"""

NO_CREDENTIALS = (
    "No Claude API credentials found. Copy .env.example to .env and set ANTHROPIC_API_KEY, "
    "then run again."
)


def system_blocks(material: str, tone: Tone, extra: str = "") -> list[dict[str, Any]]:
    """Return the system prompt as blocks: stable rules and material first, then the tone.

    The first block is identical across tones, so it is marked for prompt caching and the
    three tone variants share one cached prefix.
    """
    if tone not in TONE_PRESETS:
        raise ValueError(f"Unknown tone '{tone}'. Choose from {sorted(TONE_PRESETS)}.")
    stable = f"{SHARED_RULES}\n\n{material}"
    reader = f"{extra}\n\n{TONE_PRESETS[tone]}" if extra else TONE_PRESETS[tone]
    return [
        {"type": "text", "text": stable, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": reader},
    ]


# --- Facts: one compact, pre-rounded document -----------------------------------------------


def _round(value: float, decimals: int) -> float:
    """Round and normalise negative zero."""
    return round(float(value), decimals) + 0.0


def crore(value: float) -> float:
    """Convert INR to crore at presentation precision."""
    return _round(value / config.INR_PER_CRORE, config.CRORE_DECIMALS)


def lakh(value: float) -> float:
    """Convert INR to lakh at presentation precision."""
    return _round(value / config.INR_PER_LAKH, config.LAKH_DECIMALS)


def pct(value: float) -> float:
    """Round a percentage to presentation precision."""
    return _round(value, config.PCT_DECIMALS)


def ratio(value: float) -> float:
    """Round a ratio such as ROI to presentation precision."""
    return _round(value, config.RATIO_DECIMALS)


def estimate(raw: dict[str, float], convert: Any = ratio) -> dict[str, float]:
    """Convert an ``Estimate`` dump to the best_estimate / low / high shape the prompts use."""
    return {
        "best_estimate": convert(raw["mean"]),
        "low": convert(raw["hdi_low"]),
        "high": convert(raw["hdi_high"]),
    }


def scaled(raw: dict[str, float], factor: float) -> dict[str, float]:
    """Return an ``Estimate`` dump with every value multiplied by ``factor``."""
    return {key: value * factor for key, value in raw.items()}


def chance_text(chance_pct: float) -> str:
    """Return a chance in words, avoiding false certainty at the extremes.

    A posterior share of 0% or 100% only means no sampled draw disagreed, so those are
    reported as "almost no chance (<1%)" and "almost certain (>99%)".
    """
    if chance_pct < config.CHANCE_FLOOR_PCT:
        return f"almost no chance (<{config.CHANCE_FLOOR_PCT:g}%)"
    if chance_pct > config.CHANCE_CEILING_PCT:
        return f"an almost certain chance (>{config.CHANCE_CEILING_PCT:g}%)"
    return f"a {chance_pct:.0f}% chance"


def channel_facts(metrics: dict[str, Any], margin: float = config.DEFAULT_MARGIN) -> dict[str, Any]:
    """Return one channel's metrics in presentation units."""
    return {
        "profit_roi": estimate(scaled(metrics["roi"], margin)),
        "marginal_profit_roi": estimate(scaled(metrics["marginal_roi"], margin)),
        "total_spend_cr": crore(metrics["total_spend"]),
        "weeks_active": metrics["active_weeks"],
        "average_weekly_spend_lakh": lakh(metrics["current_weekly_spend"]),
        "attributed_revenue_cr": estimate(metrics["attributed_revenue"], crore),
        "share_of_total_revenue_pct": estimate(metrics["contribution_pct"], pct),
        "roi": estimate(metrics["roi"]),
        "marginal_roi": estimate(metrics["marginal_roi"]),
        "chance_next_rupee_returns_more_than_a_rupee_pct": pct(
            100 * metrics["prob_marginal_roi_above_1"]
        ),
        "weekly_spend_where_next_rupee_stops_paying_back_lakh": {
            "typical": lakh(metrics["saturation_weekly_spend"]["median"]),
            "low": lakh(metrics["saturation_weekly_spend"]["hdi_low"]),
            "high": lakh(metrics["saturation_weekly_spend"]["hdi_high"]),
        },
        "weeks_until_90pct_of_effect_lands": estimate(metrics["weeks_to_90pct_effect"], pct),
        "naive_same_week_attribution": {
            "roi_it_would_report": ratio(metrics["last_click"]["naive_roi"]),
            "naive_roi_as_multiple_of_model_roi": ratio(
                metrics["last_click"]["naive_over_mmm_ratio"]["median"]
            ),
        },
    }


def recommendation_facts(result: dict[str, Any]) -> dict[str, Any]:
    """Return one optimizer result in presentation units, with the truth check if present."""
    current, recommended = result["current"], result["recommended"]
    allocation = {
        channel: {
            "current": lakh(spend),
            "recommended": lakh(recommended["spend"][channel]),
            "change_pct": pct(100 * (recommended["spend"][channel] / spend - 1)) if spend else None,
            **(
                {"caveat_to_state_with_this_change": result["caveats"][channel]}
                if channel in result.get("caveats", {})
                else {}
            ),
        }
        for channel, spend in current["spend"].items()
    }
    facts: dict[str, Any] = {
        "objective": result["objective"],
        "window_weeks": current["n_weeks"],
        "budget_cr": crore(recommended["total_spend"]),
        "spend_by_channel_lakh": allocation,
        "incremental_revenue_cr": {
            "current_plan": estimate(current["incremental_revenue"], crore),
            "recommended_plan": estimate(recommended["incremental_revenue"], crore),
        },
        "uplift_vs_current_cr": estimate(result["uplift"], crore),
        "uplift_vs_current_pct": estimate(result["uplift_pct"], pct),
        "realistic_uplift_vs_current_cr": crore(result.get("realistic_uplift", 0.0)),
        "realistic_uplift_vs_current_pct": pct(result.get("realistic_uplift_pct", 0.0)),
        "why_realistic_is_lower": "Optimizers move money to the channels the model happens to "
        f"overestimate. On synthetic brands with known truth, about "
        f"{round(100 * config.UPLIFT_SHRINKAGE)}% of the expected uplift was delivered, so the "
        "expected uplift is scaled down by that share.",
        "chance_it_beats_current_plan_pct": pct(100 * result["prob_recommended_beats_current"]),
        "channels_pushed_beyond_any_historical_weekly_spend": result["extrapolated_channels"],
    }
    check = result.get("truth_check")
    if check:
        facts["ground_truth_check_synthetic_data_only"] = {
            "uplift_the_model_expected_cr": crore(check["model_expected_uplift"]),
            "uplift_it_would_really_have_delivered_cr": crore(check["true_uplift"]),
            "uplift_the_model_expected_pct": pct(check["model_expected_uplift_pct"]),
            "uplift_it_would_really_have_delivered_pct": pct(check["true_uplift_pct"]),
        }
    return facts


def scenario_facts(row: dict[str, Any]) -> dict[str, Any]:
    """Return one pre-built scenario in presentation units."""
    facts: dict[str, Any] = {
        "scenario": row["scenario"],
        "spend_change_cr": crore(row["spend_change"]),
        "revenue_change_cr": {
            "best_estimate": crore(row["revenue_change_mean"]),
            "low": crore(row["revenue_change_hdi_low"]),
            "high": crore(row["revenue_change_hdi_high"]),
        },
        "chance_revenue_rises_pct": pct(100 * row["prob_revenue_up"]),
    }
    if row.get("true_revenue_change") is not None:
        facts["true_revenue_change_cr_synthetic_data_only"] = crore(row["true_revenue_change"])
    return facts


def validation_facts(
    model_meta: dict[str, Any], data_report: ValidationReport | None
) -> dict[str, Any]:
    """Return data-quality and sampler-health results."""
    facts: dict[str, Any] = {}
    if data_report is not None:
        facts["data"] = {
            "readiness_score_out_of_100": data_report.readiness_score,
            "issues": [
                issue.message for issue in data_report.issues if issue.severity != Severity.INFO
            ],
        }
    diagnostics = model_meta.get("diagnostics")
    if diagnostics:
        facts["sampler"] = {
            "chains": diagnostics["chains"],
            "draws_per_chain": diagnostics["draws_per_chain"],
            "worst_r_hat": _round(diagnostics["max_r_hat"], 3),
            "r_hat_limit": config.MAX_RHAT,
            "smallest_effective_sample_size": round(
                min(diagnostics["min_ess_bulk"], diagnostics["min_ess_tail"])
            ),
            "effective_sample_size_minimum": config.MIN_ESS,
            "divergences": diagnostics["divergences"],
            "divergences_tolerated": divergence_limit(diagnostics),
            "all_checks_passed": is_converged(diagnostics),
        }
    return facts


def build_facts(
    insights: dict[str, Any],
    optimizer: dict[str, Any],
    model_meta: dict[str, Any] | None = None,
    data_report: ValidationReport | None = None,
    margin: float = config.DEFAULT_MARGIN,
) -> dict[str, Any]:
    """Build the single FACTS document the brief is written from.

    ``margin`` is the product gross margin (0 to 1) used for profit ROI: gross profit earned
    per rupee of spend, which must exceed 1 for the spend to pay for itself.

    Every value is already in the unit and precision it should be quoted in, so the model
    never has to convert, and the number check can match what it writes.
    """
    totals = insights["totals"]
    curve = optimizer["budget_curve"]
    nearest = min(curve, key=lambda row: abs(row["budget"] - optimizer["current_budget"]))
    payback = optimizer["payback_budget"]
    return {
        "about": {
            "brand": "Fictional Indian direct-to-consumer skincare brand (synthetic data).",
            "data_period": insights["period"],
            "currency": "INR",
            "interval_pct": round(100 * insights["hdi_prob"]),
            "roi_that_returns_one_rupee_of_revenue_per_rupee": config.MARGINAL_ROI_BREAKEVEN,
            "gross_margin_pct": pct(100 * margin),
            "profit_roi_meaning": "Gross profit per rupee spent (ROI x margin). Above 1 the "
            "spend pays for itself; below 1 it does not.",
            "chances_below_this_pct_are_reported_as_less_than": config.CHANCE_FLOOR_PCT,
            "chances_above_this_pct_are_reported_as_more_than": config.CHANCE_CEILING_PCT,
            "channels": list(insights["channels"]),
        },
        "totals_over_data_period": {
            "revenue_cr": crore(totals["revenue"]),
            "media_spend_cr": crore(totals["media_spend"]),
            "revenue_caused_by_media_cr": estimate(totals["media_revenue"], crore),
            "media_share_of_revenue_pct": estimate(totals["media_pct_of_revenue"], pct),
            "blended_media_roi": estimate(totals["blended_media_roi"]),
            "blended_profit_roi": estimate(scaled(totals["blended_media_roi"], margin)),
            "marketing_pays_for_itself_after_margin": bool(
                totals["blended_media_roi"]["mean"] * margin >= config.PROFIT_BREAKEVEN
            ),
        },
        "share_of_revenue_by_driver_pct": {
            name: estimate(values["pct_of_revenue"], pct)
            for name, values in insights["decomposition"].items()
        },
        "channels": {
            name: channel_facts(metrics, margin) for name, metrics in insights["channels"].items()
        },
        "main_recommendation": {
            "rule": f"Same budget as last quarter; no channel moves more than "
            f"{round(100 * optimizer['max_change'])}% from last quarter.",
            **recommendation_facts(optimizer["expected_revenue"]),
        },
        "conservative_recommendation": recommendation_facts(optimizer["conservative"]),
        "recommendation_with_no_limits_on_channel_changes": recommendation_facts(
            optimizer["unconstrained"]
        ),
        "scenarios_vs_last_quarter": [scenario_facts(row) for row in optimizer["scenarios"]],
        "budget_level": {
            "last_quarter_budget_cr": crore(optimizer["current_budget"]),
            "quarterly_budget_beyond_which_an_extra_rupee_returns_under_a_rupee_cr": (
                None if payback is None else crore(payback)
            ),
            "return_per_extra_rupee_near_last_quarter_budget": (
                ratio(nearest["marginal_return"]) if pd.notna(nearest["marginal_return"]) else None
            ),
        },
        "validation": validation_facts(model_meta or {}, data_report),
    }


def render_facts(facts: dict[str, Any]) -> str:
    """Return the FACTS block for the prompt, serialised deterministically."""
    return f"<facts>\n{json.dumps(facts, indent=1, sort_keys=True)}\n</facts>"


# --- Number check ---------------------------------------------------------------------------

NUMBER_PATTERN = re.compile(r"(?<![A-Za-z\d.,])\d+(?:,\d{2,3})*(?:\.\d+)?")
LIST_MARKER_PATTERN = re.compile(r"^[ \t]*(?:[#>*-]+[ \t]*)*\d+[.)][ \t]", re.MULTILINE)


class FoundNumber(BaseModel):
    """A number found in text, with its surrounding words."""

    raw: str
    value: float
    decimals: int
    context: str


class NumberCheck(BaseModel):
    """Result of verifying every number in a response against its source material."""

    n_numbers: int
    n_verified: int
    unverified: list[FoundNumber] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Return whether every number was found in the source material."""
        return not self.unverified

    def summary(self) -> str:
        """Return a one-line description, listing any flagged numbers."""
        line = f"Number check: {self.n_verified}/{self.n_numbers} numbers found in the input"
        if self.passed:
            return line + "."
        flagged = "; ".join(f'{n.raw} ("...{n.context}...")' for n in self.unverified)
        return f"{line}. NOT FOUND: {flagged}"


def extract_numbers(text: str) -> list[FoundNumber]:
    """Return every number in ``text``, ignoring ordered-list markers and signs.

    Thousands separators are removed. Digits glued to a letter (``Q4``) are not numbers.
    """
    cleaned = LIST_MARKER_PATTERN.sub(lambda m: " " * len(m.group()), text)
    numbers = []
    for match in NUMBER_PATTERN.finditer(cleaned):
        raw = match.group()
        plain = raw.replace(",", "")
        start = max(match.start() - config.NUMBER_CONTEXT_CHARS, 0)
        end = match.end() + config.NUMBER_CONTEXT_CHARS
        numbers.append(
            FoundNumber(
                raw=raw,
                value=float(plain),
                decimals=len(plain.split(".")[1]) if "." in plain else 0,
                context=" ".join(cleaned[start:end].split()),
            )
        )
    return numbers


def collect_numbers(source: Any) -> set[float]:
    """Return every number (as an absolute value) anywhere in a nested structure.

    Numbers inside strings and dictionary keys count too, so dates and field names such as
    ``weeks_until_90pct`` are quotable.
    """
    if isinstance(source, bool) or source is None:
        return set()
    if isinstance(source, int | float):
        return {abs(float(source))}
    if isinstance(source, str):
        return {number.value for number in extract_numbers(source.replace("_", " "))}
    if isinstance(source, dict):
        return collect_numbers(list(source.keys())) | collect_numbers(list(source.values()))
    if isinstance(source, list | tuple | set):
        return set().union(*(collect_numbers(item) for item in source)) if source else set()
    return set()


def is_supported(number: FoundNumber, allowed: set[float]) -> bool:
    """Return whether a quoted number equals a source number rounded to the same decimals."""
    tolerance = 0.5 * 10 ** (-number.decimals) + 1e-9
    return any(abs(candidate - number.value) <= tolerance for candidate in allowed)


def check_numbers(text: str, sources: list[Any]) -> NumberCheck:
    """Verify every number in ``text`` appears in ``sources`` and flag the ones that do not."""
    allowed = collect_numbers(sources)
    found = extract_numbers(text)
    unverified = [number for number in found if not is_supported(number, allowed)]
    return NumberCheck(
        n_numbers=len(found), n_verified=len(found) - len(unverified), unverified=unverified
    )


# --- Data context and tools -----------------------------------------------------------------


@dataclass(frozen=True)
class DataContext:
    """Everything the explainer can draw on.

    Attributes:
        facts: The FACTS document.
        draws: Posterior draws, needed by the scenario and response-curve tools.
        fingerprint: Hash of the facts, so cached answers are tied to the data they used.

    """

    facts: dict[str, Any]
    draws: PosteriorDraws | None = None

    @property
    def fingerprint(self) -> str:
        """Return a short hash identifying the underlying results."""
        return hashlib.sha256(json.dumps(self.facts, sort_keys=True).encode()).hexdigest()[:16]

    @property
    def channels(self) -> list[str]:
        """Return the channel names."""
        return list(self.facts["channels"])

    @property
    def n_weeks(self) -> int:
        """Return the planning window used by the optimizer."""
        return int(self.facts["main_recommendation"]["window_weeks"])


def load_context(
    reports_dir: Path = config.REPORTS_DIR,
    models_dir: Path = config.MODELS_DIR,
    data_path: Path = config.SYNTHETIC_DIR / config.WEEKLY_DATA_FILENAME,
    with_draws: bool = False,
) -> DataContext:
    """Load saved results from disk. Loading posterior draws is slower and only needed for Q&A."""
    insights = json.loads((reports_dir / config.INSIGHTS_SUMMARY_FILENAME).read_text())
    optimizer = json.loads((reports_dir / config.OPTIMIZER_SUMMARY_FILENAME).read_text())
    meta_path = models_dir / config.MODEL_META_FILENAME
    model_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    df = pd.read_csv(data_path, parse_dates=[config.DATE_COL])
    facts = build_facts(insights, optimizer, model_meta, validate(df))
    draws = extract_draws(MixLabModel.load(models_dir), df) if with_draws else None
    return DataContext(facts=facts, draws=draws)


def load_brand_context(brand_dir: Path, with_draws: bool = False) -> DataContext:
    """Load a demo brand whose data, model and reports all live in one folder."""
    data_path = brand_dir / config.WEEKLY_DATA_FILENAME
    return load_context(brand_dir, brand_dir, data_path, with_draws)


def _require_channel(context: DataContext, channel: str) -> None:
    """Raise a clear error for a channel that does not exist."""
    if channel not in context.channels:
        raise ValueError(f"Unknown channel '{channel}'. Available: {context.channels}")


def _require_draws(context: DataContext) -> PosteriorDraws:
    """Return the posterior draws or explain that this tool needs the fitted model loaded."""
    if context.draws is None:
        raise ValueError("This tool needs the fitted model; load the context with_draws=True.")
    return context.draws


def get_channel_metrics(context: DataContext, channel: str) -> dict[str, Any]:
    """Return a channel's metrics and how the main recommendation changes its spend."""
    _require_channel(context, channel)
    change = context.facts["main_recommendation"]["spend_by_channel_lakh"][channel]
    return {
        "channel": channel,
        "basis": "Full data period.",
        **context.facts["channels"][channel],
        "main_recommendation_spend_lakh": change,
    }


def run_scenario(
    context: DataContext, changes: list[dict[str, Any]], shifts: list[dict[str, Any]]
) -> dict[str, Any]:
    """Simulate a spend plan defined as changes to last quarter's plan.

    Args:
        context: Loaded results including posterior draws.
        changes: Items of ``{"channel", "kind", "value"}`` where kind is ``percent_change``
            (value -100 pauses the channel) or ``set_spend_lakh`` (total for the window).
        shifts: Items of ``{"from_channel", "to_channel", "pct_of_from"}`` moving budget.

    """
    draws = _require_draws(context)
    n_weeks = context.n_weeks
    current = last_quarter_spend(draws, n_weeks)
    plan = dict(current)
    for change in changes:
        _require_channel(context, change["channel"])
        if change["kind"] == "percent_change":
            plan[change["channel"]] = current[change["channel"]] * (1 + change["value"] / 100)
        elif change["kind"] == "set_spend_lakh":
            plan[change["channel"]] = change["value"] * config.INR_PER_LAKH
        else:
            raise ValueError(f"Unknown change kind '{change['kind']}'.")
    for shift in shifts:
        _require_channel(context, shift["from_channel"])
        _require_channel(context, shift["to_channel"])
        moved = current[shift["from_channel"]] * shift["pct_of_from"] / 100
        plan[shift["from_channel"]] -= moved
        plan[shift["to_channel"]] += moved
    if any(value < 0 for value in plan.values()):
        raise ValueError("The scenario makes a channel's spend negative.")

    table = compare_scenarios(draws, {"current": current, "scenario": plan}, n_weeks)
    row = table.loc["scenario"]
    peak = dict(zip(draws.channels, draws.spend.max(axis=0), strict=True))
    return {
        "window_weeks": n_weeks,
        "spend_by_channel_lakh": {
            channel: {"last_quarter": lakh(current[channel]), "scenario": lakh(plan[channel])}
            for channel in draws.channels
        },
        "total_spend_cr": {
            "last_quarter": crore(sum(current.values())),
            "scenario": crore(sum(plan.values())),
            "change": crore(row["spend_change"]),
        },
        "incremental_revenue_cr": {
            "last_quarter_plan": estimate(
                what_if(draws, current, n_weeks).incremental_revenue.model_dump(), crore
            ),
            "scenario": estimate(
                what_if(draws, plan, n_weeks).incremental_revenue.model_dump(), crore
            ),
        },
        "revenue_change_vs_last_quarter_cr": {
            "best_estimate": crore(row["revenue_change_mean"]),
            "low": crore(row["revenue_change_hdi_low"]),
            "high": crore(row["revenue_change_hdi_high"]),
        },
        "chance_revenue_rises_pct": pct(100 * row["prob_revenue_up"]),
        "revenue_change_minus_spend_change_cr": crore(row["net_change_mean"]),
        "channels_pushed_beyond_any_historical_weekly_spend": [
            channel for channel in draws.channels if plan[channel] / n_weeks > peak[channel]
        ],
        "assumptions": "Spend is spread evenly over the window. Revenue is incremental media "
        "revenue including carryover after the window; baseline revenue is unaffected.",
    }


def get_response_curve(context: DataContext, channel: str) -> dict[str, Any]:
    """Return a channel's response curve as points around last quarter's weekly spend."""
    _require_channel(context, channel)
    draws = _require_draws(context)
    index = draws.channels.index(channel)
    weekly = last_quarter_spend(draws, context.n_weeks)[channel] / context.n_weeks
    base = weekly if weekly > 0 else float(draws.spend[draws.spend[:, index] > 0, index].mean())
    points = []
    for multiple in config.AI_RESPONSE_CURVE_MULTIPLES:
        curve = response_curve_draws(draws, index, [multiple * base])[:, 0]
        low, high = hdi(curve)
        points.append(
            {
                "weekly_spend_lakh": lakh(multiple * base),
                "weekly_incremental_revenue_lakh": {
                    "best_estimate": lakh(curve.mean()),
                    "low": lakh(low),
                    "high": lakh(high),
                },
            }
        )
    metrics = context.facts["channels"][channel]
    return {
        "channel": channel,
        "last_quarter_weekly_spend_lakh": lakh(weekly),
        "points": points,
        "weekly_spend_where_next_rupee_stops_paying_back_lakh": metrics[
            "weekly_spend_where_next_rupee_stops_paying_back_lakh"
        ],
        "marginal_roi": metrics["marginal_roi"],
        "assumptions": "Steady weekly spend, after carryover has built up.",
    }


def get_budget_recommendation(context: DataContext, objective: str) -> dict[str, Any]:
    """Return one of the optimizer's recommendations plus the budget-level findings."""
    keys = {
        "expected_revenue": "main_recommendation",
        "conservative": "conservative_recommendation",
        "unconstrained": "recommendation_with_no_limits_on_channel_changes",
    }
    if objective not in keys:
        raise ValueError(f"Unknown objective '{objective}'. Choose from {sorted(keys)}.")
    return {
        "recommendation": context.facts[keys[objective]],
        "budget_level": context.facts["budget_level"],
    }


def get_model_health(context: DataContext) -> dict[str, Any]:
    """Return data-quality and sampler diagnostics."""
    return context.facts["validation"]


TOOL_FUNCTIONS: dict[str, Any] = {
    "get_channel_metrics": get_channel_metrics,
    "run_scenario": run_scenario,
    "get_response_curve": get_response_curve,
    "get_budget_recommendation": get_budget_recommendation,
    "get_model_health": get_model_health,
}


def tool_definitions(channels: list[str]) -> list[dict[str, Any]]:
    """Return the tool schemas offered to Claude for the given channels."""
    channel = {"type": "string", "enum": channels}

    def tool(name: str, description: str, properties: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": name,
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        }

    return [
        tool(
            "get_channel_metrics",
            "Get one channel's results over the full data period: spend, attributed revenue, "
            "ROI, marginal ROI and the chance the next rupee pays back, carryover, how a naive "
            "same-week attribution would misreport it, and how the main recommendation changes "
            "its spend. Call this for any question about how a channel is performing or why "
            "its budget should move.",
            {"channel": channel},
        ),
        tool(
            "run_scenario",
            "Simulate a what-if spend plan for the next quarter and compare it with last "
            "quarter's plan. Returns incremental revenue with a likely range, the change "
            "versus last quarter, and the chance revenue rises. Describe the plan as changes "
            "to last quarter; channels you do not mention stay as they were. Call this for any "
            "'what happens if' question instead of reasoning from ROI figures.",
            {
                "changes": {
                    "type": "array",
                    "description": "Per-channel changes. Use an empty list if only shifting.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "channel": channel,
                            "kind": {
                                "type": "string",
                                "enum": ["percent_change", "set_spend_lakh"],
                                "description": "percent_change: value is a percentage of last "
                                "quarter's spend (-100 pauses the channel, 20 raises it 20%). "
                                "set_spend_lakh: value is the total spend for the quarter in "
                                "lakh.",
                            },
                            "value": {"type": "number"},
                        },
                        "required": ["channel", "kind", "value"],
                        "additionalProperties": False,
                    },
                },
                "shifts": {
                    "type": "array",
                    "description": "Budget moved between channels. Use an empty list if none.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "from_channel": channel,
                            "to_channel": channel,
                            "pct_of_from": {
                                "type": "number",
                                "description": "Percentage of the source channel's last-quarter "
                                "spend to move.",
                            },
                        },
                        "required": ["from_channel", "to_channel", "pct_of_from"],
                        "additionalProperties": False,
                    },
                },
            },
        ),
        tool(
            "get_response_curve",
            "Get a channel's response curve: weekly incremental revenue at several steady "
            "weekly spend levels around last quarter's, each with a likely range, plus the "
            "spend level beyond which the next rupee returns less than a rupee. Call this for "
            "questions about diminishing returns, saturation or headroom.",
            {"channel": channel},
        ),
        tool(
            "get_budget_recommendation",
            "Get an optimizer recommendation for next quarter at last quarter's budget: spend "
            "per channel, expected uplift with a range, the chance it beats the current plan, "
            "and how it scored against ground truth. Also returns where extra total budget "
            "stops paying back. 'expected_revenue' and 'conservative' keep every channel "
            "within the allowed change; 'unconstrained' has no limits.",
            {
                "objective": {
                    "type": "string",
                    "enum": ["expected_revenue", "conservative", "unconstrained"],
                }
            },
        ),
        tool(
            "get_model_health",
            "Get data-quality findings and sampler diagnostics (r-hat, effective sample size, "
            "divergences). Call this when asked how far the results can be trusted.",
            {},
        ),
    ]


def qa_context(context: DataContext) -> str:
    """Return the small orientation block for Q&A: what exists, not the results themselves."""
    about = context.facts["about"]
    orientation = {
        "brand": about["brand"],
        "data_period": about["data_period"],
        "channels": about["channels"],
        "interval_pct": about["interval_pct"],
        "next_quarter_weeks": context.n_weeks,
        "last_quarter_spend_by_channel_lakh": {
            channel: values["current"]
            for channel, values in context.facts["main_recommendation"][
                "spend_by_channel_lakh"
            ].items()
        },
    }
    return f"<context>\n{json.dumps(orientation, indent=1, sort_keys=True)}\n</context>"


# --- Claude API calls -----------------------------------------------------------------------


class ToolCall(BaseModel):
    """One tool call Claude made and what it got back."""

    name: str
    input: dict[str, Any]
    output: Any
    is_error: bool = False


class Explanation(BaseModel):
    """A generated brief or answer, with its grounding check."""

    kind: Literal["brief", "answer"]
    tone: str
    model: str
    text: str
    number_check: NumberCheck
    tool_calls: list[ToolCall] = Field(default_factory=list)
    stop_reason: str | None = None
    truncated: bool = False
    usage: dict[str, int] = Field(default_factory=dict)
    cached: bool = False


def make_client() -> anthropic.Anthropic:
    """Return a Claude API client, reading ``ANTHROPIC_API_KEY`` from ``.env`` if present."""
    load_dotenv(config.ENV_FILE)
    return anthropic.Anthropic()


def build_request(
    system: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    effort: str,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return the Messages API request shared by the brief and Q&A."""
    request: dict[str, Any] = {
        "model": config.AI_MODEL,
        "max_tokens": config.AI_MAX_TOKENS,
        "system": system,
        "messages": messages,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": effort},
    }
    if tools:
        request["tools"] = tools
    if config.AI_REFUSAL_FALLBACK:
        request["betas"] = [config.AI_FALLBACK_BETA]
        request["fallbacks"] = "default"
    return request


def create_message(client: Any, request: dict[str, Any]) -> Any:
    """Send one request and turn API failures into messages the user can act on."""
    endpoint = client.beta.messages if config.AI_REFUSAL_FALLBACK else client.messages
    try:
        response = endpoint.create(**request)
    except TypeError as error:  # the SDK raises this when no credentials can be resolved
        if "authentication" in str(error).lower():
            raise ExplainerError(NO_CREDENTIALS) from error
        raise
    except anthropic.AuthenticationError as error:
        raise ExplainerError(
            "The Claude API rejected the key. Check ANTHROPIC_API_KEY in .env."
        ) from error
    except anthropic.BadRequestError as error:
        raise ExplainerError(
            f"The Claude API rejected the request: {error.message} If this mentions "
            "'fallbacks' or a beta, set AI_REFUSAL_FALLBACK = False in config.py."
        ) from error
    except anthropic.RateLimitError as error:
        raise ExplainerError("Rate limited by the Claude API. Wait and try again.") from error
    except anthropic.APIStatusError as error:
        raise ExplainerError(f"Claude API error {error.status_code}: {error.message}") from error
    except anthropic.APIConnectionError as error:
        raise ExplainerError("Could not reach the Claude API. Check the network.") from error
    if response.stop_reason == "refusal":
        raise ExplainerError("Claude declined this request, so no explanation was produced.")
    return response


def response_text(response: Any) -> str:
    """Return the visible text of a response, skipping thinking and tool blocks."""
    return "".join(block.text for block in response.content if block.type == "text").strip()


def response_usage(response: Any) -> dict[str, int]:
    """Return token counts, including prompt-cache reads, as plain integers."""
    fields = ("input_tokens", "output_tokens", "cache_read_input_tokens")
    usage = getattr(response, "usage", None)
    return {field: int(getattr(usage, field, 0) or 0) for field in fields}


def cache_key(payload: dict[str, Any]) -> str:
    """Return a stable hash of everything that determines a response."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def read_cache(key: str, directory: Path) -> Explanation | None:
    """Return a cached explanation, or ``None`` if there is none."""
    path = directory / f"{key}.json"
    if not path.exists():
        return None
    return Explanation.model_validate_json(path.read_text()).model_copy(update={"cached": True})


def write_cache(key: str, explanation: Explanation, directory: Path) -> None:
    """Store an explanation on disk."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{key}.json").write_text(explanation.model_dump_json(indent=2))


def brief_request(context: DataContext, tone: Tone) -> dict[str, Any]:
    """Return the request that produces the brief for this data and tone."""
    return build_request(
        system_blocks(render_facts(context.facts), tone),
        [{"role": "user", "content": BRIEF_TASK}],
        config.AI_EFFORT_BRIEF,
    )


def cached_brief(
    context: DataContext, tone: Tone, cache_dir: Path = config.AI_CACHE_DIR
) -> Explanation | None:
    """Return a previously generated brief without calling the API, or ``None``."""
    return read_cache(
        cache_key({"kind": "brief", "request": brief_request(context, tone)}), cache_dir
    )


def generate_brief(
    context: DataContext,
    tone: Tone = "cmo",
    client: Any | None = None,
    use_cache: bool = True,
    cache_dir: Path = config.AI_CACHE_DIR,
) -> Explanation:
    """Write the one-page executive brief in the given tone and check its numbers."""
    request = brief_request(context, tone)
    key = cache_key({"kind": "brief", "request": request})
    if use_cache and (cached := read_cache(key, cache_dir)) is not None:
        return cached

    response = create_message(client or make_client(), request)
    text = response_text(response)
    explanation = Explanation(
        kind="brief",
        tone=tone,
        model=response.model,
        text=text,
        number_check=check_numbers(text, [context.facts]),
        stop_reason=response.stop_reason,
        truncated=response.stop_reason == "max_tokens",
        usage=response_usage(response),
    )
    write_cache(key, explanation, cache_dir)
    return explanation


def execute_tool(context: DataContext, name: str, tool_input: dict[str, Any]) -> ToolCall:
    """Run one tool call against the real analysis functions.

    Failures are returned to Claude as error results rather than raised, so it can correct
    the call or explain what is unavailable.
    """
    try:
        if name not in TOOL_FUNCTIONS:
            raise ValueError(f"Unknown tool '{name}'.")
        output = TOOL_FUNCTIONS[name](context, **tool_input)
        return ToolCall(name=name, input=tool_input, output=output)
    except (ValueError, KeyError, TypeError) as error:
        return ToolCall(name=name, input=tool_input, output=f"Error: {error}", is_error=True)


def ask(
    context: DataContext,
    question: str,
    tone: Tone = "cmo",
    client: Any | None = None,
    use_cache: bool = True,
    cache_dir: Path = config.AI_CACHE_DIR,
) -> Explanation:
    """Answer a question about the results, letting Claude call the analysis tools.

    Runs the tool-use loop by hand (rather than the SDK's tool runner) so that every tool
    result is captured for the number check and the whole exchange can be cached.
    """
    orientation = qa_context(context)
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    request = build_request(
        system_blocks(orientation, tone, extra=QA_GUIDANCE),
        messages,
        config.AI_EFFORT_QA,
        tool_definitions(context.channels),
    )
    key = cache_key({"kind": "answer", "request": request, "data": context.fingerprint})
    if use_cache and (cached := read_cache(key, cache_dir)) is not None:
        return cached

    client = client or make_client()
    calls: list[ToolCall] = []
    usage = dict.fromkeys(("input_tokens", "output_tokens", "cache_read_input_tokens"), 0)
    for _ in range(config.AI_MAX_TOOL_ROUNDS):
        response = create_message(client, request)
        for field, count in response_usage(response).items():
            usage[field] += count
        if response.stop_reason != "tool_use":
            break
        # Keep the full content (thinking and tool_use blocks included) in the history.
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            call = execute_tool(context, block.name, dict(block.input))
            calls.append(call)
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(call.output, sort_keys=True),
                    "is_error": call.is_error,
                }
            )
        messages.append({"role": "user", "content": results})
    else:
        raise ExplainerError(
            f"No answer after {config.AI_MAX_TOOL_ROUNDS} rounds of tool calls; "
            "try a narrower question."
        )

    text = response_text(response)
    sources = [orientation, *(call.output for call in calls)]
    explanation = Explanation(
        kind="answer",
        tone=tone,
        model=response.model,
        text=text,
        number_check=check_numbers(text, sources),
        tool_calls=calls,
        stop_reason=response.stop_reason,
        truncated=response.stop_reason == "max_tokens",
        usage=usage,
    )
    write_cache(key, explanation, cache_dir)
    return explanation


# --- Keyless brief and PDF export -----------------------------------------------------------


def range_text(
    estimate_: dict[str, float], prefix: str = "", suffix: str = "", digits: str = ".2f"
) -> str:
    """Return the standard range phrase, e.g. "94% range: ₹0.27 to ₹0.70"."""
    interval = round(100 * config.HDI_PROB)
    low, high = (
        f"{'-' if estimate_[key] < 0 else ''}{prefix}{abs(estimate_[key]):{digits}}{suffix}"
        for key in ("low", "high")
    )
    return f"{interval}% range: {low} to {high}"


def template_brief(facts: dict[str, Any], name: Callable[[str], str] = str) -> str:
    """Return a fixed-wording brief built directly from FACTS, with no AI involved.

    Used when no API key is available, so the app and the PDF export still work. Every
    number is copied from the facts (percentages rounded to whole numbers), so it always
    passes the number check. ``name`` maps a channel id to the label shown to the reader.
    """
    about = facts["about"]
    totals = facts["totals_over_data_period"]
    recommendation = facts["main_recommendation"]
    spend = recommendation["spend_by_channel_lakh"]
    channels = facts["channels"]
    chance = "chance_next_rupee_returns_more_than_a_rupee_pct"
    strongest = max(channels, key=lambda c: channels[c][chance])
    weakest = min(channels, key=lambda c: channels[c][chance])
    widest = max(channels, key=lambda c: channels[c]["roi"]["high"] - channels[c]["roi"]["low"])

    def moves(direction: int) -> str:
        chosen = []
        for channel, values in spend.items():
            if (values["change_pct"] or 0) * direction <= 0:
                continue
            caveat = values.get("caveat_to_state_with_this_change")
            note = f"; {caveat}" if caveat else ""
            chosen.append(f"{name(channel)} ({values['change_pct']:+.0f}%{note})")
        return ", ".join(chosen) or "nothing"

    def percent(estimate_: dict[str, float]) -> str:
        whole = {key: round(value) for key, value in estimate_.items()}
        return range_text(whole, suffix="%", digits="g")

    uplift = recommendation["uplift_vs_current_cr"]
    share = totals["media_share_of_revenue_pct"]
    profit = totals["blended_profit_roi"]
    pays = totals["marketing_pays_for_itself_after_margin"]
    baseline = facts["share_of_revenue_by_driver_pct"]["baseline"]
    lines = [
        "## Headline finding",
        f"Marketing drives about {share['best_estimate']:.0f}% of revenue ({percent(share)}). "
        f"At a {about['gross_margin_pct']:.0f}% product margin it "
        f"{'pays for itself' if pays else 'does not pay for itself'}: each ₹1 of spend returns "
        f"about ₹{profit['best_estimate']:.2f} of gross profit ({range_text(profit, '₹')}).",
        "",
        "## Top insights",
        f"- About {baseline['best_estimate']:.0f}% of revenue would have happened without "
        f"marketing ({percent(baseline)}).",
        f"- {name(strongest)} has the strongest case for more budget: "
        f"{chance_text(channels[strongest][chance])} that the next ₹1 returns more than ₹1 of "
        "revenue.",
        f"- {name(weakest)} has the weakest case: {chance_text(channels[weakest][chance])}.",
        "",
        "## Recommended budget shift",
        f"Increase {moves(1)}. Reduce {moves(-1)}.",
        "",
        f"Expected uplift over the next {recommendation['window_weeks']} weeks: about "
        f"₹{uplift['best_estimate']:.2f} crore ({range_text(uplift, '₹', ' crore')}). Realistic "
        f"uplift: about ₹{recommendation['realistic_uplift_vs_current_cr']:.2f} crore, because "
        "optimizers favour the channels a model happens to overestimate. "
        f"There is {chance_text(recommendation['chance_it_beats_current_plan_pct'])} that this "
        "plan beats the current one.",
        "",
        "## Risks and caveats",
        f"- {name(widest)} is the least certain channel: each ₹1 returns between "
        f"₹{channels[widest]['roi']['low']:.2f} and ₹{channels[widest]['roi']['high']:.2f} of "
        "revenue (94% range).",
    ]
    check = recommendation.get("ground_truth_check_synthetic_data_only")
    if check:
        lines.append(
            f"- Scored against the known truth for this synthetic brand, the plan delivers "
            f"₹{check['uplift_it_would_really_have_delivered_cr']:.2f} crore against the "
            f"₹{check['uplift_the_model_expected_cr']:.2f} crore the model expected."
        )
    else:
        lines.append("- ROI here is revenue per rupee, before product margin.")
    return "\n".join(lines)


def brief_to_pdf(text: str, title: str) -> bytes:
    """Render a Markdown brief as a one-column A4 PDF and return the file's bytes.

    Handles headings, bullets, bold and horizontal rules, which is what the briefs contain.
    """
    fonts = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    pdf = FPDF(format="A4")
    for style, filename in (
        ("", "DejaVuSans.ttf"),
        ("B", "DejaVuSans-Bold.ttf"),
        ("I", "DejaVuSans-Oblique.ttf"),
        ("BI", "DejaVuSans-BoldOblique.ttf"),
    ):
        pdf.add_font("DejaVu", style, str(fonts / filename))
    pdf.set_margins(config.PDF_MARGIN_MM, config.PDF_MARGIN_MM, config.PDF_MARGIN_MM)
    pdf.set_auto_page_break(auto=True, margin=config.PDF_MARGIN_MM)
    pdf.add_page()
    size = config.PDF_FONT_SIZE
    line = size * 0.55

    def write(content: str, style: str = "", points: int = size, markdown: bool = True) -> None:
        pdf.set_font("DejaVu", style, points)
        pdf.multi_cell(0, points * 0.55, content, markdown=markdown, new_x="LMARGIN", new_y="NEXT")

    write(title, "B", size + 6, markdown=False)
    pdf.ln(line)
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            pdf.ln(line / 2)
        elif set(stripped) <= {"-", "_", "*"} and len(stripped) >= 3:
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
            pdf.ln(line / 2)
        elif stripped.startswith("#"):
            pdf.ln(line / 2)
            write(stripped.lstrip("# ").strip("*"), "B", size + 2, markdown=False)
        elif stripped.startswith(("- ", "* ")):
            write("\u2022  " + stripped[2:])
        else:
            write(stripped)
    return bytes(pdf.output())


# --- Command line ---------------------------------------------------------------------------


def render(explanation: Explanation) -> str:
    """Return an explanation as Markdown with its grounding check as a footer."""
    footer = [explanation.number_check.summary()]
    if explanation.tool_calls:
        footer.append("Tools called: " + ", ".join(call.name for call in explanation.tool_calls))
    if explanation.truncated:
        footer.append("WARNING: the response hit the token limit and may be cut off.")
    footer.append(
        f"Model: {explanation.model} | tone: {explanation.tone}"
        + (" | served from cache" if explanation.cached else "")
    )
    return explanation.text + "\n\n---\n" + "\n".join(f"_{line}_" for line in footer) + "\n"


def write_examples(context: DataContext, directory: Path = config.AI_EXAMPLES_DIR) -> list[Path]:
    """Generate one brief per tone and save each as Markdown."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for tone in TONE_PRESETS:
        path = directory / f"brief_{tone}.md"
        path.write_text(render(generate_brief(context, tone)))  # type: ignore[arg-type]
        paths.append(path)
    return paths


def main() -> None:
    """Command-line entry point: ``facts``, ``brief``, ``ask`` or ``examples``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("facts", help="Print the FACTS document (no API call).")
    brief = commands.add_parser("brief", help="Write the executive brief.")
    question = commands.add_parser("ask", help="Ask a question about the results.")
    question.add_argument("question")
    for sub in (brief, question):
        sub.add_argument("--tone", choices=sorted(TONE_PRESETS), default="cmo")
        sub.add_argument("--no-cache", action="store_true")
    commands.add_parser("examples", help="Write one brief per tone to reports/ai_examples/.")
    args = parser.parse_args()

    try:
        if args.command == "facts":
            print(render_facts(load_context().facts))
        elif args.command == "brief":
            result = generate_brief(load_context(), args.tone, use_cache=not args.no_cache)
            print(render(result))
        elif args.command == "ask":
            context = load_context(with_draws=True)
            print(render(ask(context, args.question, args.tone, use_cache=not args.no_cache)))
        else:
            for path in write_examples(load_context()):
                print(f"Saved {path}\n\n{path.read_text()}")
    except ExplainerError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
