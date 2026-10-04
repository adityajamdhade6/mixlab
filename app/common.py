"""Shared app plumbing: brand selection, cached loading, formatting and plain-English errors."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError

import mixlab  # noqa: F401  (sets the PyTensor backend before PyMC is imported)
from mixlab import config
from mixlab.ai_explainer import DataContext, build_facts
from mixlab.insights import PosteriorDraws, extract_draws, marginal_roi_draws
from mixlab.model import MixLabModel
from mixlab.optimizer import BudgetAllocator
from mixlab.validate import validate

BUILD_COMMAND = "uv run python scripts/build_demo.py"
CHANNEL_LABELS = {"tv": "TV", "meta_ads": "Meta Ads", "youtube": "YouTube"}
COMPONENT_LABELS = {"other_controls": "Other controls"}


@dataclass(frozen=True)
class Runtime:
    """The heavy objects: fitted model, posterior draws and a compiled budget allocator."""

    model: MixLabModel
    draws: PosteriorDraws
    allocator: BudgetAllocator


def use_hosted_secret() -> None:
    """Copy the API key from Streamlit secrets into the environment, when deployed.

    Locally the key comes from ``.env``; on Streamlit Community Cloud it is set in the app's
    Secrets settings and never stored in the repository.
    """
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY")
    except (FileNotFoundError, StreamlitSecretNotFoundError):
        return
    if key:
        os.environ.setdefault("ANTHROPIC_API_KEY", key)


def label(name: str) -> str:
    """Return a display name for a channel or revenue component."""
    return CHANNEL_LABELS.get(name) or COMPONENT_LABELS.get(name) or name.replace("_", " ").title()


def money(value: float, unit_size: float, unit: str, decimals: int) -> str:
    """Format INR in a unit, with the minus sign before the rupee symbol."""
    return f"{'-' if value < 0 else ''}₹{abs(value) / unit_size:,.{decimals}f} {unit}"


def crore(value: float, decimals: int = 1) -> str:
    """Format INR as crore."""
    return money(value, config.INR_PER_CRORE, "Cr", decimals)


def lakh(value: float, decimals: int = 0) -> str:
    """Format INR as lakh."""
    return money(value, config.INR_PER_LAKH, "L", decimals)


def range_text(low: str, high: str) -> str:
    """Return the one phrase used for every uncertainty range in the app."""
    return f"{round(100 * config.HDI_PROB)}% range: {low} to {high}"


def plural(count: int, noun: str) -> str:
    """Return "1 warning" or "2 warnings"."""
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def chance(share: float) -> str:
    """Format a probability (0 to 1) without claiming certainty at the extremes."""
    percent = 100 * share
    if percent < config.CHANCE_FLOOR_PCT:
        return f"<{config.CHANCE_FLOOR_PCT:g}%"
    if percent > config.CHANCE_CEILING_PCT:
        return f">{config.CHANCE_CEILING_PCT:g}%"
    return f"{percent:.0f}%"


def per_rupee(value: float) -> str:
    """Format a return per rupee of spend, with its unit."""
    return f"₹{value:.2f}"


def margin() -> float:
    """Return the product margin chosen in the sidebar, as a share between 0 and 1."""
    return st.session_state.get("margin_pct", round(100 * config.DEFAULT_MARGIN)) / 100


def has_api_key() -> bool:
    """Return whether Claude credentials are available for the AI features."""
    load_dotenv(config.ENV_FILE)
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def header(brand: str | None = None) -> None:
    """Render the line at the top of every page: which brand is shown, and who built this."""
    showing = f" · {config.BRAND_LABELS.get(brand, label(brand))} demo brand" if brand else ""
    st.caption(
        f"**MixLab**{showing} · built by {config.AUTHOR_NAME} · [GitHub]({config.REPO_URL}) · "
        f"[Case study]({config.CASE_STUDY_URL})"
    )


def brand_dir(brand: str) -> Path:
    """Return the folder holding a brand's saved results."""
    return config.ARTIFACTS_DIR / brand


def available_brands() -> list[str]:
    """Return brands that have saved results, in the order of the presets."""
    return [
        name
        for name in config.BRAND_PRESETS
        if (brand_dir(name) / config.INSIGHTS_SUMMARY_FILENAME).exists()
    ]


@st.cache_data(show_spinner="Loading saved results…")
def load_results(brand: str) -> dict[str, Any]:
    """Load a brand's data and saved summaries (fast; no model needed)."""
    folder = brand_dir(brand)

    def read(name: str) -> Any:
        path = folder / name
        return json.loads(path.read_text()) if path.exists() else None

    df = pd.read_csv(folder / config.WEEKLY_DATA_FILENAME, parse_dates=[config.DATE_COL])
    insights = read(config.INSIGHTS_SUMMARY_FILENAME)
    optimizer = read(config.OPTIMIZER_SUMMARY_FILENAME)
    meta = read(config.MODEL_META_FILENAME) or {}
    report = validate(df)
    return {
        "df": df,
        "insights": insights,
        "optimizer": optimizer,
        "meta": meta,
        "truth": read(config.GROUND_TRUTH_FILENAME),
        "report": report,
        "backtest": read(config.BACKTEST_FILENAME),
        "weekly": pd.read_csv(
            folder / config.DECOMPOSITION_FILENAME, parse_dates=[config.DATE_COL]
        ).set_index(config.DATE_COL),
    }


@st.cache_resource(show_spinner="Loading the fitted model (a few seconds, once per brand)…")
def load_runtime(brand: str) -> Runtime:
    """Load the fitted model and prepare the optimizer. Nothing is refitted."""
    df = load_results(brand)["df"]
    model = MixLabModel.load(brand_dir(brand))
    return Runtime(model, extract_draws(model, df), BudgetAllocator(model))


@st.cache_data(show_spinner=False)
def load_facts(brand: str, margin_share: float) -> dict[str, Any]:
    """Build the facts document for a brand at the chosen product margin."""
    results = load_results(brand)
    return build_facts(
        results["insights"], results["optimizer"], results["meta"], results["report"], margin_share
    )


@st.cache_data(show_spinner="Working out what the next rupee earns…")
def load_marginal_roi(brand: str) -> Any:
    """Return marginal ROI draws (draws x channels) for a brand, computed once."""
    return marginal_roi_draws(load_runtime(brand).draws)


def ai_context(brand: str, with_model: bool = False) -> DataContext:
    """Return what the AI layer needs; the model is only loaded for Q&A tools."""
    draws = load_runtime(brand).draws if with_model else None
    return DataContext(load_facts(brand, margin()), draws)


def setup(title: str, intro: str) -> tuple[str, dict[str, Any]]:
    """Render the sidebar brand selector and page header; return the brand and its results.

    Stops the page with setup instructions if no demo brand has been built yet.
    """
    use_hosted_secret()
    brands = available_brands()
    if not brands:
        st.title("MixLab")
        st.error(
            "No fitted demo brands were found, so there is nothing to show yet. Build them "
            "once with the command below (about five minutes), then reload this page."
        )
        st.code(BUILD_COMMAND, language="bash")
        st.stop()
    with st.sidebar:
        brand = st.selectbox(
            "Demo brand",
            brands,
            format_func=lambda name: config.BRAND_LABELS.get(name, label(name)),
            key="brand",
            help="Three synthetic skincare brands with different media mixes. Each has its "
            "own pre-fitted model.",
        )
        st.slider(
            "Product margin",
            min_value=5,
            max_value=95,
            value=round(100 * config.DEFAULT_MARGIN),
            step=5,
            format="%d%%",
            key="margin_pct",
            help="Share of each rupee of revenue left after the cost of the product. Profit "
            "ROI is ROI times this margin; above ₹1 per ₹1 the spend pays for itself.",
        )
        st.caption("Synthetic data. All money in INR (₹1 Cr = ₹100 L).")
    header(brand)
    st.title(title)
    st.caption(intro)
    return brand, load_results(brand)


def show(figure: Any) -> None:
    """Render a plotly figure at full width without the mode bar."""
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
