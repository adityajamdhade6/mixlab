# MixLab architecture

## Data flow
```
exports / template ──► onboarding ──► weekly contract CSV ──► validate ──► model ──► insights ──► optimizer
                                                                 ▲            │           │
                                             experiments ──► calibration      └──► ai_explainer ◄──┘
                                                                                        │
                                              artifacts/<brand>/ (saved results) ──► app (Streamlit)
```

1. **Data** (`data_gen`, `onboarding`). Synthetic brands with known truth, or real exports
   (Meta Ads, Google Ads, Shopify) mapped to the contract: one row per week, `date`,
   `revenue`, one `spend_<channel>` per channel, controls, money in INR.
2. **Validation** (`validate`). Schema, quality and MMM-readiness checks; a score out of 100
   and issues ranked by severity. Nothing is fixed or deleted automatically.
3. **Model** (`model`). `MixLabModel` wraps `pymc_marketing.mmm.multidimensional.MMM`.
4. **Insights** (`insights`). `PosteriorDraws` holds every draw; all metrics are computed per
   draw and summarised as mean, median and 94% range.
5. **Optimizer** (`optimizer`). Budget allocation, scenarios, budget-level curve, truth check.
6. **Calibration** (`calibration`). Lift tests added to the model's likelihood; test planner.
7. **AI layer** (`ai_explainer`). Grounded brief and tool-calling Q&A.
8. **App** (`app/`). Loads `artifacts/<brand>/`; never fits a model.

`scripts/build_demo.py` produces the artefacts in stages: `data`, `train` (fit, insights,
optimizer, backtest), or `optimize` / `backtest` alone to refresh without refitting.

## Model specification
```
revenue = baseline + linear trend + yearly Fourier seasonality + controls
          + Σ_c beta_c · hill_saturation(geometric_adstock(spend_c)) + noise
```
- Adstock: geometric, 8 weeks, normalised (`adstock: delayed` is available; it did worse).
- Saturation: **Hill** curve (`saturation: hill`). The v1 logistic curve is still available
  (`saturation: logistic`) and is what `ModelSettings()` gives with no settings file.
- Zero-spend weeks are modelled as one rupee: the Hill gradient is undefined at exactly zero
  and the sampler reports that as divergences.
- Controls: `holiday_*`, `promo_flag`, `price_index` (centred on 1.0).
- Scaling: revenue and each channel's spend divided by their peak week.
- Priors: stated on **channel ROI** (`roi_prior`), converted to the model's coefficient with
  `model.roi_scale`. `independent` (default): each channel LogNormal around a median ROI of 1,
  or around a per-channel benchmark. `pooled`: a learned shared ROI (partial pooling). Shape
  priors and per-channel overrides come from `config.PriorSettings` and YAML.
- Options kept for comparison: `time_varying_intercept`, `time_varying_media`.
- Sampling: nutpie, 4 chains x (1,000 tune + 1,000 draws), Numba backend, fixed seed.
- Checks: prior predictive before fitting; r-hat, effective sample size and divergences
  after; rolling backtest (three 12-week windows); ROI recovery against ground truth;
  `scripts/compare_models.py` (LOO, holdout, recovery across seeds) for model form.

See [model_card.md](model_card.md) for assumptions and [model_log.md](model_log.md) for why
the model has this form.

## Optimizer
`optimizer.py` scores every candidate plan across posterior draws with the project's own
response maths (`insights.simulate_contributions`, tested equal to the fitted model) and
searches with SciPy (`solve_allocation`, several starting points because a Hill surface is not
concave). PyMC-Marketing's `BudgetOptimizer` is kept only as a cross-check in the tests.

- **Plan:** a total per channel over 13 weeks, with carryover after the window counted.
- **Objectives:** expected revenue, expected profit (margin x revenue - spend), risk-adjusted
  (mean - lambda x standard deviation), or the 10th percentile.
- **Uncertainty-aware limits** (`channel_limits`): ±10% for channels that ran in bursts or are
  under 2% of spend, ±15% where the ROI range is wide relative to the estimate, ±30%
  otherwise; never above a channel's highest week on record. Each limit carries its reason.
- **Corner check** (`corner_check`): flags plans where most channels sit on a limit.
- **Never worse:** a plan the model rates below the current one is not recommended.
- **Realistic uplift:** expected uplift x a haircut estimated per brand by
  `estimate_optimism`: treat a posterior draw as the real world, simulate revenue, refit,
  re-optimize, and score the chosen plan under that world. No ground truth is used.
- **Business goals** (`optimize_goal`): lowest spend for a revenue target, or the largest
  plan that keeps a target ROI. CAC is not supported (the data has no customer counts).
- **Timing** (`weekly_plan`): each channel's total is spread evenly or put into one burst,
  whichever earns more; burst channels keep a minimum weekly spend. The model is additive, so
  it cannot say which calendar weeks are better.
- **Rollout** (`rollout_plan`): stepped changes with a checkpoint each, including whether the
  step is large enough to see in revenue at all.
- Scenario simulator (`what_if`, `compare_scenarios`) and the budget-level curve.
- `scripts/optimizer_benchmark.py` compares a naive and the robust optimizer on 20 random
  brands against the truth (`reports/optimizer_benchmark.json`).

## AI layer
- **Facts:** `build_facts` produces one pre-rounded JSON document (crore, lakh, percentages,
  profit ROI at the chosen margin, caveats, realistic uplift, validation results).
- **Brief:** one request to `claude-opus-5-5`; the system prompt is rules + facts (cached
  prefix) then a tone block (CMO, Analyst, Founder).
- **Q&A:** manual tool loop over `get_channel_metrics`, `run_scenario`, `get_response_curve`,
  `get_budget_recommendation`, `get_model_health`, all wired to the real functions.
- **Grounding:** `check_numbers` extracts every number in a response and verifies it against
  the facts and tool results; unmatched numbers are flagged.
- **Keyless:** `template_brief` fills a fixed-wording summary from the same facts.
- Responses are cached on disk; the key comes from `.env` or Streamlit secrets.

## App pages
| Page | Shows | Loads the model? |
|---|---|---|
| Overview | Profit verdict at the chosen margin, KPIs, expected vs. realistic uplift, brief, decomposition | No |
| Channel performance | ROI and profit ROI with ranges, response curves, model vs. naive attribution | Yes (curves) |
| Budget optimizer | Budget, four objectives, per-channel limits with reasons, allocation, corner warning, rollout plan, weekly schedule, goal search | Yes |
| Scenario planner | Per-channel sliders, live revenue and profit change, saved scenarios | Yes |
| Ask MixLab | Chat over the Q&A tools; says upfront when no API key is set | On first question |
| Model health | Out-of-sample accuracy and rolling backtest, trust notes, ground-truth recovery, diagnostics, data checks | No |
| Upload data | Validator on an uploaded CSV | No |
| How it works | Pipeline steps and MMM in five lines | No |

Shared pieces: `app/common.py` (brand selector, margin, cached loaders, formatting) and
`app/charts.py` (Plotly figures on one theme). Every page is tested headlessly in
`tests/test_app.py`.
