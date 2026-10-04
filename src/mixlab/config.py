"""Central configuration: paths, seeds, and data-contract constants. No magic numbers elsewhere."""

from dataclasses import dataclass, replace
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
SYNTHETIC_DIR: Path = DATA_DIR / "synthetic"
FIGURES_DIR: Path = PROJECT_ROOT / "reports" / "figures"

RANDOM_SEED: int = 42

DATE_COL: str = "date"
TARGET_COL: str = "revenue"
SPEND_PREFIX: str = "spend_"
CURRENCY: str = "INR"

# --- Calendar -------------------------------------------------------------------------------
DAYS_PER_WEEK: int = 7
DAYS_PER_YEAR: float = 365.25
WEEKS_PER_YEAR: float = DAYS_PER_YEAR / DAYS_PER_WEEK

# --- Synthetic data outputs -----------------------------------------------------------------
WEEKLY_DATA_FILENAME: str = "mmm_weekly.csv"
GROUND_TRUTH_FILENAME: str = "ground_truth.json"
TRUE_CONTRIBUTIONS_FILENAME: str = "true_contributions.csv"
HOLIDAY_PREFIX: str = "holiday_"
PROMO_COL: str = "promo_flag"
PRICE_COL: str = "price_index"
SEASON_PREFIX: str = "season_"
MIN_DEMAND_INDEX: float = 0.1

# --- Plot styling ---------------------------------------------------------------------------
INR_PER_LAKH: float = 100_000.0
FIGURE_SIZE: tuple[float, float] = (10.0, 5.5)
FIGURE_DPI: int = 150
COLOR_SPEND: str = "#2a78d6"
COLOR_CONTRIBUTION: str = "#eb6834"
COLOR_SURFACE: str = "#fcfcfb"
COLOR_TEXT: str = "#0b0b0b"
COLOR_TEXT_MUTED: str = "#52514e"
COLOR_GRID: str = "#e4e3df"


@dataclass(frozen=True, kw_only=True)
class ChannelConfig:
    """Spend pattern and true response parameters for one media channel.

    Attributes:
        name: Channel name; the spend column is ``spend_<name>``.
        base_spend: Typical weekly spend in INR at the start of the series.
        annual_growth: Yearly spend growth rate (0.2 means +20% per year).
        festive_sensitivity: How strongly spend rises with the festive index.
        spend_noise: Sigma of the multiplicative lognormal noise on weekly spend.
        adstock_decay: True geometric adstock decay (share of effect carried to next week).
        half_saturation: Adstocked weekly spend in INR at which response is half of ``beta``.
        hill_slope: Hill curve steepness (1 is concave; above 1 is S-shaped).
        beta: Maximum incremental weekly revenue in INR at full saturation.
        demand_elasticity: Exponent linking spend to latent demand (creates confounding).
        flight_events: Event names the channel flights around; empty means always on.
        flight_weeks: Length of each flight in weeks, ending in the event week.

    """

    name: str
    base_spend: float
    annual_growth: float
    festive_sensitivity: float
    spend_noise: float
    adstock_decay: float
    half_saturation: float
    hill_slope: float
    beta: float
    demand_elasticity: float = 0.0
    flight_events: tuple[str, ...] = ()
    flight_weeks: int = 0


@dataclass(frozen=True, kw_only=True)
class EventConfig:
    """A recurring calendar event that lifts demand and pulls marketing spend up.

    Attributes:
        name: Event name; the control column is ``holiday_<name>``.
        dates: ISO dates of each occurrence.
        revenue_lift: Organic revenue lift in the event week, as a fraction of baseline.
        spend_peak: Peak of the festive spend index at the event date.
        ramp_weeks: Width (in weeks) of the spend build-up around the event.

    """

    name: str
    dates: tuple[str, ...]
    revenue_lift: float
    spend_peak: float
    ramp_weeks: float


@dataclass(frozen=True, kw_only=True)
class BrandConfig:
    """Everything needed to generate one synthetic brand, including the seed.

    Attributes:
        name: Brand preset name.
        channels: Media channels with their true response parameters.
        events: Calendar events (festivals, sale seasons).
        seed: Random seed; same config and seed give identical data.
        n_weeks: Number of weekly rows.
        start_date: ISO date of the first week (a Monday).
        baseline_revenue: Organic weekly revenue in INR with no media.
        annual_trend: Yearly organic growth as a fraction of baseline.
        seasonality: Fourier (sin, cos) coefficient pairs as fractions of baseline.
        promo_probability: Chance that any given week has a brand promotion.
        promo_lift: Revenue lift in a promo week, as a fraction of baseline.
        price_annual_increase: Step increase in the price index each calendar year.
        price_noise: Standard deviation of week-to-week price index noise.
        price_sensitivity: Revenue change per unit of price index, as a fraction of baseline.
        revenue_noise: Standard deviation of revenue noise, as a fraction of baseline.
        demand_noise: Standard deviation of noise on the latent demand index.
        adstock_l_max: Number of weeks over which adstock carries over.

    """

    name: str
    channels: tuple[ChannelConfig, ...]
    events: tuple[EventConfig, ...]
    seed: int = RANDOM_SEED
    n_weeks: int = 156
    start_date: str = "2023-01-02"
    baseline_revenue: float = 6_000_000.0
    annual_trend: float = 0.12
    seasonality: tuple[tuple[float, float], ...] = ((0.06, -0.04), (0.02, 0.03))
    promo_probability: float = 0.12
    promo_lift: float = 0.15
    price_annual_increase: float = 0.04
    price_noise: float = 0.01
    price_sensitivity: float = -0.6
    revenue_noise: float = 0.05
    demand_noise: float = 0.05
    adstock_l_max: int = 12


INDIA_EVENTS: tuple[EventConfig, ...] = (
    EventConfig(
        name="diwali",
        dates=("2023-11-12", "2024-11-01", "2025-10-20"),
        revenue_lift=0.35,
        spend_peak=1.0,
        ramp_weeks=2.5,
    ),
    EventConfig(
        name="sale_season",
        dates=("2023-10-08", "2024-09-27", "2025-09-23"),
        revenue_lift=0.25,
        spend_peak=0.8,
        ramp_weeks=1.5,
    ),
    EventConfig(
        name="new_year",
        dates=("2023-01-01", "2024-01-01", "2025-01-01", "2026-01-01"),
        revenue_lift=0.10,
        spend_peak=0.3,
        ramp_weeks=1.5,
    ),
)

PERFORMANCE_HEAVY_BRAND: BrandConfig = BrandConfig(
    name="performance_heavy",
    events=INDIA_EVENTS,
    channels=(
        ChannelConfig(
            name="meta_ads",
            base_spend=900_000.0,
            annual_growth=0.20,
            festive_sensitivity=0.6,
            spend_noise=0.15,
            adstock_decay=0.35,
            half_saturation=1_000_000.0,
            hill_slope=1.0,
            beta=2_800_000.0,
        ),
        ChannelConfig(
            name="google_search",
            base_spend=400_000.0,
            annual_growth=0.15,
            festive_sensitivity=0.1,
            spend_noise=0.10,
            adstock_decay=0.15,
            half_saturation=350_000.0,
            hill_slope=1.0,
            beta=1_500_000.0,
            demand_elasticity=1.0,
        ),
        ChannelConfig(
            name="youtube",
            base_spend=300_000.0,
            annual_growth=0.25,
            festive_sensitivity=0.5,
            spend_noise=0.20,
            adstock_decay=0.60,
            half_saturation=500_000.0,
            hill_slope=1.3,
            beta=900_000.0,
        ),
        ChannelConfig(
            name="influencers",
            base_spend=250_000.0,
            annual_growth=0.30,
            festive_sensitivity=0.8,
            spend_noise=0.35,
            adstock_decay=0.45,
            half_saturation=300_000.0,
            hill_slope=1.0,
            beta=700_000.0,
        ),
        ChannelConfig(
            name="email",
            base_spend=40_000.0,
            annual_growth=0.10,
            festive_sensitivity=0.4,
            spend_noise=0.10,
            adstock_decay=0.20,
            half_saturation=50_000.0,
            hill_slope=1.0,
            beta=350_000.0,
        ),
        ChannelConfig(
            name="tv",
            base_spend=1_500_000.0,
            annual_growth=0.10,
            festive_sensitivity=0.3,
            spend_noise=0.10,
            adstock_decay=0.70,
            half_saturation=1_200_000.0,
            hill_slope=1.5,
            beta=2_000_000.0,
            flight_events=("diwali",),
            flight_weeks=6,
        ),
    ),
)


def _retune(channel: ChannelConfig, **changes: float | int | tuple[str, ...]) -> ChannelConfig:
    """Return a copy of ``channel`` with some fields replaced."""
    return replace(channel, **changes)


_TV_HEAVY_OVERRIDES: dict[str, dict[str, float | int | tuple[str, ...]]] = {
    "meta_ads": {"base_spend": 350_000.0, "beta": 1_400_000.0},
    "google_search": {"base_spend": 250_000.0},
    "influencers": {"base_spend": 120_000.0},
    "tv": {
        "base_spend": 4_000_000.0,
        "half_saturation": 3_000_000.0,
        "beta": 6_500_000.0,
        "flight_events": ("diwali", "new_year"),
    },
}

TV_HEAVY_BRAND: BrandConfig = replace(
    PERFORMANCE_HEAVY_BRAND,
    name="tv_heavy",
    channels=tuple(
        _retune(c, **_TV_HEAVY_OVERRIDES.get(c.name, {})) for c in PERFORMANCE_HEAVY_BRAND.channels
    ),
)

_INFLUENCER_LED_OVERRIDES: dict[str, dict[str, float | int | tuple[str, ...]]] = {
    "meta_ads": {"base_spend": 450_000.0, "beta": 1_800_000.0},
    "influencers": {
        "base_spend": 900_000.0,
        "half_saturation": 900_000.0,
        "beta": 2_600_000.0,
        "spend_noise": 0.30,
    },
    "tv": {"base_spend": 800_000.0, "beta": 1_200_000.0},
}

INFLUENCER_LED_BRAND: BrandConfig = replace(
    PERFORMANCE_HEAVY_BRAND,
    name="influencer_led",
    channels=tuple(
        _retune(c, **_INFLUENCER_LED_OVERRIDES.get(c.name, {}))
        for c in PERFORMANCE_HEAVY_BRAND.channels
    ),
)

BRAND_PRESETS: dict[str, BrandConfig] = {
    PERFORMANCE_HEAVY_BRAND.name: PERFORMANCE_HEAVY_BRAND,
    TV_HEAVY_BRAND.name: TV_HEAVY_BRAND,
    INFLUENCER_LED_BRAND.name: INFLUENCER_LED_BRAND,
}

# --- Validation thresholds ------------------------------------------------------------------
MIN_WEEKS_HISTORY: int = 104
MIN_SPEND_CV: float = 0.10
MAX_CHANNEL_CORRELATION: float = 0.80
DUPLICATE_CHANNEL_CORRELATION: float = 0.95
MAX_VIF: float = 5.0
MIN_SPEND_SHARE: float = 0.02
FLIGHTED_ZERO_SHARE: float = 0.50
OUTLIER_IQR_MULTIPLIER: float = 1.5
MIN_POINTS_FOR_OUTLIERS: int = 8
MAX_READINESS_SCORE: int = 100
SEVERITY_PENALTY: dict[str, int] = {"critical": 30, "warning": 8, "info": 1}
MAX_DATES_IN_MESSAGE: int = 5

# --- EDA styling ----------------------------------------------------------------------------
# Categorical colours are assigned to channels in column order and never reused for rank.
CHANNEL_COLORS: tuple[str, ...] = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
)
COLOR_DIVERGING: tuple[str, str, str] = (
    "#e34948",
    "#f0efec",
    "#2a78d6",
)  # negative, zero, positive
COLOR_FESTIVE_SHADE: str = "#eda100"
FESTIVE_SHADE_ALPHA: float = 0.18
EDA_LAGS: tuple[int, ...] = (0, 1, 2)

# --- Model: backend and files ---------------------------------------------------------------
# PyTensor's C backend fails to link on recent macOS toolchains, so models compile with Numba.
PYTENSOR_FLAGS: str = "mode=NUMBA,cxx="
MODELS_DIR: Path = PROJECT_ROOT / "models"
MODEL_FILENAME: str = "mixlab_mmm.nc"
MODEL_META_FILENAME: str = "model_meta.json"
# A thinned copy small enough to commit and deploy; loaded when the full file is absent.
MODEL_SLIM_FILENAME: str = "mixlab_mmm_slim.nc"
SLIM_DRAWS_PER_CHAIN: int = 250
SLIM_DROP_VARIABLES: tuple[str, ...] = (
    "channel_contribution_original_scale",
    "y_original_scale",
    "total_media_contribution_original_scale",
    "fourier_contribution",
)
SLIM_KEEP_GROUPS: tuple[str, ...] = (
    "posterior",
    "sample_stats",
    "observed_data",
    "constant_data",
    "fit_data",
)
PRIOR_PREDICTIVE_FIGURE: str = "prior_predictive.png"
TREND_PREFIX: str = "trend"
PRICE_INDEX_BASE: float = 1.0

# --- Model: diagnostics and intervals -------------------------------------------------------
MAX_RHAT: float = 1.01
MIN_ESS: int = 400
# A handful of divergent draws is normal; more than this share suggests biased sampling.
MAX_DIVERGENCE_SHARE: float = 0.001
INTERVAL_OUTER: tuple[float, float] = (0.03, 0.97)
INTERVAL_INNER: tuple[float, float] = (0.25, 0.75)
PRIOR_MAX_REVENUE_MULTIPLE: float = 5.0
PRIOR_MAX_NEGATIVE_SHARE: float = 0.25


class BetaParams(BaseModel):
    """Parameters of a Beta prior (values between 0 and 1; mean is alpha / (alpha + beta))."""

    model_config = ConfigDict(extra="forbid")
    alpha: float = Field(gt=0)
    beta: float = Field(gt=0)


class GammaParams(BaseModel):
    """Parameters of a Gamma prior (positive values; mean is alpha / beta)."""

    model_config = ConfigDict(extra="forbid")
    alpha: float = Field(gt=0)
    beta: float = Field(gt=0)


class HalfNormalParams(BaseModel):
    """Parameter of a HalfNormal prior (positive values, most mass below 2 x sigma)."""

    model_config = ConfigDict(extra="forbid")
    sigma: float = Field(gt=0)


class ChannelPriorOverride(BaseModel):
    """Optional per-channel prior beliefs; anything left out uses the default prior."""

    model_config = ConfigDict(extra="forbid")
    adstock_alpha: BetaParams | None = None
    saturation_lam: GammaParams | None = None
    saturation_beta: HalfNormalParams | None = None


class PriorSettings(BaseModel):
    """Default weakly-informative priors.

    All values are in the model's scaled units: revenue is divided by its peak week and each
    channel's spend by its own peak week, so 1.0 means "as large as the biggest week".
    """

    model_config = ConfigDict(extra="forbid")
    intercept_mu: float = 0.5
    intercept_sigma: float = Field(default=0.2, gt=0)
    adstock_alpha: BetaParams = BetaParams(alpha=1.0, beta=3.0)
    saturation_lam: GammaParams = GammaParams(alpha=3.0, beta=1.0)
    saturation_beta: HalfNormalParams = HalfNormalParams(sigma=0.3)
    control_sigma: float = Field(default=0.3, gt=0)
    fourier_scale: float = Field(default=0.1, gt=0)
    noise_sigma: float = Field(default=0.1, gt=0)
    trend_scale: float = Field(default=0.2, gt=0)


class SamplerSettings(BaseModel):
    """MCMC sampler settings."""

    model_config = ConfigDict(extra="forbid")
    chains: int = Field(default=4, ge=1)
    draws: int = Field(default=1000, ge=1)
    tune: int = Field(default=1000, ge=1)
    target_accept: float = Field(default=0.9, gt=0, lt=1)
    seed: int = RANDOM_SEED
    nuts_sampler: str = "nutpie"


class ModelSettings(BaseModel):
    """Everything that defines one MixLab model run; loadable from a YAML file."""

    model_config = ConfigDict(extra="forbid")
    adstock_l_max: int = Field(default=8, ge=1)
    yearly_seasonality_order: int = Field(default=2, ge=1)
    trend_changepoints: int = Field(default=2, ge=2)
    control_columns: list[str] | None = None
    spend_share_prior: bool = False
    prior_predictive_samples: int = Field(default=500, ge=10)
    priors: PriorSettings = PriorSettings()
    channel_priors: dict[str, ChannelPriorOverride] = Field(default_factory=dict)
    sampler: SamplerSettings = SamplerSettings()

    @classmethod
    def from_yaml(cls, path: Path) -> "ModelSettings":
        """Load settings from a YAML file; unknown keys are rejected."""
        return cls.model_validate(yaml.safe_load(Path(path).read_text()) or {})


# --- Insights -------------------------------------------------------------------------------
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
INSIGHTS_SUMMARY_FILENAME: str = "insights_summary.json"
DECOMPOSITION_FILENAME: str = "decomposition_weekly.csv"
HDI_PROB: float = 0.94
INR_PER_CRORE: float = 10_000_000.0
MARGINAL_STEP_INR: float = INR_PER_LAKH
MARGINAL_ROI_BREAKEVEN: float = 1.0
CARRYOVER_SHARE: float = 0.90
RESPONSE_CURVE_POINTS: int = 80
RESPONSE_CURVE_MAX_MULTIPLE: float = 1.5
ORGANIC_COMPONENTS: tuple[str, ...] = (
    "baseline",
    "trend",
    "seasonality",
    "holidays",
    "promos",
    "price",
    "other_controls",
)
# Neutral ramp for non-media components, so the six channel hues stay reserved for channels.
ORGANIC_COLORS: dict[str, str] = {
    "baseline": "#d9d8d3",
    "trend": "#bfbdb6",
    "seasonality": "#a5a39b",
    "holidays": "#8b8980",
    "promos": "#716f67",
    "price": "#57554f",
    "other_controls": "#3d3c38",
}
COLOR_NAIVE: str = "#eb6834"
ROI_AXIS_LIMITS: tuple[float, float] = (0.03, 60.0)

# --- Optimizer ------------------------------------------------------------------------------
OPTIMIZER_WEEKS: int = 13
RISK_PERCENTILE: float = 10.0
DEFAULT_MAX_CHANGE: float = 0.30
SCENARIO_BUDGET_CHANGE: float = 0.20
SCENARIO_SHIFT_SHARE: float = 0.15
SCENARIO_SHIFT_FROM: str = "meta_ads"
SCENARIO_SHIFT_TO: str = "youtube"
SCENARIO_PAUSE_CHANNEL: str = "tv"
BUDGET_CURVE_MULTIPLES: tuple[float, float] = (0.05, 2.5)
BUDGET_CURVE_POINTS: int = 15
OPTIMIZER_SUMMARY_FILENAME: str = "optimizer_summary.json"
BUDGET_CURVE_FIGURE: str = "optimizer_budget_curve.png"
ALLOCATION_FIGURE: str = "optimizer_allocation.png"
BOUND_TOLERANCE: float = 1e-6
# Safety cap: on flat regions the solver can wander for minutes before giving up.
OPTIMIZER_FTOL: float = 1e-9
OPTIMIZER_MAX_ITERATIONS: int = 300

# --- AI explainer ---------------------------------------------------------------------------
AI_MODEL: str = "claude-opus-5-5"
AI_MAX_TOKENS: int = 16_000
AI_EFFORT_BRIEF: str = "high"
AI_EFFORT_QA: str = "medium"
# Re-run a request on Anthropic's recommended fallback model if a safety classifier declines
# it. Uses a beta header; set to False to call the plain Messages endpoint instead.
AI_REFUSAL_FALLBACK: bool = True
AI_FALLBACK_BETA: str = "server-side-fallback-2026-07-01"
AI_MAX_TOOL_ROUNDS: int = 6
AI_CACHE_DIR: Path = PROJECT_ROOT / ".cache" / "ai_explainer"
AI_EXAMPLES_DIR: Path = REPORTS_DIR / "ai_examples"
ENV_FILE: Path = PROJECT_ROOT / ".env"
AI_RESPONSE_CURVE_MULTIPLES: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
CRORE_DECIMALS: int = 2
LAKH_DECIMALS: int = 1
PCT_DECIMALS: int = 1
RATIO_DECIMALS: int = 2
NUMBER_CONTEXT_CHARS: int = 30

# --- Demo artefacts and app -----------------------------------------------------------------
# One folder per demo brand holding its data, fitted model and every saved result, so the
# app can switch brands without refitting.
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
BRAND_LABELS: dict[str, str] = {
    "performance_heavy": "Performance-heavy",
    "tv_heavy": "TV-heavy",
    "influencer_led": "Influencer-led",
}
WIDE_INTERVAL_RATIO: float = 0.25
PDF_FONT_SIZE: int = 11
PDF_MARGIN_MM: float = 18.0

# --- Real-data onboarding -------------------------------------------------------------------
TEMPLATES_DIR: Path = PROJECT_ROOT / "templates"
DATA_TEMPLATE_FILENAME: str = "mmm_data_template.xlsx"
EXPERIMENTS_TEMPLATE_FILENAME: str = "experiments_template.csv"
PUBLIC_DATA_DIR: Path = DATA_DIR / "public"
TEMPLATE_DATA_SHEET: str = "Data"
# Pandas period for weeks that end on Sunday, i.e. start on Monday.
WEEK_PERIOD: str = "W-SUN"
ANONYMIZE_FACTOR_RANGE: tuple[float, float] = (0.3, 3.0)
ANONYMIZE_SECRET_ENV: str = "MIXLAB_ANON_SECRET"
# Google Ads campaign types mapped to channel names; anything else becomes google_other.
GOOGLE_CAMPAIGN_CHANNELS: dict[str, str] = {
    "search": "google_search",
    "video": "youtube",
    "shopping": "google_shopping",
    "performance max": "google_pmax",
    "display": "google_display",
}
SHOPIFY_EXCLUDED_STATUSES: tuple[str, ...] = ("voided", "refunded")

# --- Lift-test calibration ------------------------------------------------------------------
CALIBRATION_SUMMARY_FILENAME: str = "calibration_summary.json"
CALIBRATION_FIGURE: str = "calibration_before_after.png"
LIFT_LIKELIHOOD_NAME: str = "lift_measurements"
SIMULATED_TEST_RELATIVE_SE: float = 0.10
# Test planner: 80% power at 5% significance, and the noise inflation of a 50/50 geo split.
POWER_Z: float = 2.8
GEO_DESIGN_FACTOR: float = 2.0
MIN_TEST_WEEKS: int = 2
MAX_TEST_WEEKS: int = 26

# --- Demo review fixes ----------------------------------------------------------------------
# Confidence gate: channels the health checks cannot measure well get tighter default bounds.
GATED_MAX_CHANGE: float = 0.10
CAVEAT_BURSTS: str = "ran in bursts, so its effect is tangled with the season"
CAVEAT_SMALL: str = "too small a share of spend to measure precisely"
# Optimizer's curse: share of the model's expected uplift that the truth delivered, averaged
# over the three synthetic demo brands (see `optimizer.measured_shrinkage`).
UPLIFT_SHRINKAGE: float = 0.40
DEFAULT_MARGIN: float = 0.40
PROFIT_BREAKEVEN: float = 1.0
BACKTEST_HORIZON_WEEKS: int = 12
BACKTEST_FOLDS: int = 3
BACKTEST_DRAWS: int = 500
BACKTEST_FILENAME: str = "backtest.json"
CHANCE_FLOOR_PCT: float = 1.0
CHANCE_CEILING_PCT: float = 99.0
AUTHOR_NAME: str = "Aditya Jamdhade"
REPO_URL: str = "https://github.com/adityajamdhade6/mixlab"
CASE_STUDY_URL: str = f"{REPO_URL}/blob/main/docs/case_study.md"
