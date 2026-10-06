# Model log

What was tried on the model, what the ground truth said about it, and why the default is what
it is. All results are for the synthetic `performance_heavy` brand unless stated; raw numbers
are in `reports/model_comparison.json`, `reports/meridian_benchmark.json` and
`reports/evaluation_summary.json` (the v1 report is kept in `reports/history/`).

## 1. The influencers bias: root cause

**Symptom.** True ROI 0.89; the v1 model said 0.22 with a 94% range of 0.00 to 0.50, in all
three demo brands, while every sampler check passed.

**What it was not.**

| Suspect | Test | Result |
|---|---|---|
| Confounding with Meta | Correlation of influencer and Meta spend | 0.65 raw, but 0.17 once both are detrended. They share a growth trend and festive peaks, not week-to-week movement. |
| Wrong adstock shape (delayed peak) | Refit with delayed adstock | Worse: 0.18 (0.00 to 0.45), and Meta and TV were then missed too. |
| Prior on its contribution | Spend-share prior; ROI priors | No change: 0.22 to 0.27, upper bound at most 0.58. |
| Noise | Regenerate with revenue noise cut from 5% to 1% | Still wrong and more confident: 0.32 (0.12 to 0.58). A bias, not bad luck. |

**What it was.** The saturation curve. Two tests isolate it:

| Test | Influencers ROI | Recovered |
|---|---|---|
| v1 as is | 0.22 (0.00 to 0.50) | 5/6 |
| Same model, influencer spend made smooth | 0.96 (0.00 to 2.55), truth 0.97 | 6/6 |
| Same data, Hill curve instead of logistic | 1.07 (0.00 to 4.06), truth 0.89 | 6/6 |

Influencer spend is spiky (coefficient of variation 0.61, with weeks several times the
typical level). The model scales spend by its peak week, so a typical week sits near 0.1 on
the scaled axis. The true response is already strongly saturated there; a logistic curve with
any plausible speed is still close to a straight line at that point. A near-linear curve
predicts a large revenue jump in every spike week. The data shows no such jump, so the only
way to fit is a small coefficient, which also drags down the estimate for ordinary weeks. A
Hill curve can bend early, so it does not have to make that trade.

**Why it looked like "one unlucky seed" at first.** In the v1 evaluation the influencers range
contained the truth in 5 of 5 other seeds, and I initially concluded the miss was noise shared
by three brands with one seed. That was wrong: the estimate was biased low in every seed and
the range sometimes still reached the truth. Test D (less noise, still wrong) is what
separates a bias from noise.

## 2. A trap on the way: Hill and zero spend

The first Hill fits recovered 6/6 with about 3,400 divergences in 4,000 draws. Cause: the
Hill curve computes `x ** slope`, whose gradient with respect to the slope is `x ** slope *
log(x)`, undefined at exactly zero spend. TV is off in 138 of 156 weeks. The sampler counts
each undefined gradient as a divergence. Changing the priors made no difference; flooring
spend at one rupee (`config.HILL_MIN_SPEND`) removed every divergence and left the estimates
unchanged.

## 3. Priors

**ROI-based priors.** `roi_prior` states the prior on revenue per rupee and converts it to the
model's coefficient (`model.roi_scale`). The default is a LogNormal with median 1 and
log-scale spread 1.0 per channel. Per-channel medians can be loaded from a benchmarks file
(`configs/benchmarks/illustrative.yaml` shows the format; its values are placeholders, not
real benchmarks).

**Partial pooling was tested and rejected.** `mode: pooled` learns a shared typical ROI and a
between-channel spread. It gives the narrowest ranges and the worst recovery: 4/6 on the demo
seed and 11/18 on other seeds. Pooling assumes channels are roughly alike. Here true ROIs run
from 0.75 to 3.47, and email, the channel pooling was meant to help, is the genuine outlier,
so it is pulled toward the others and its range (0.01 to 2.14) no longer contains 3.47.

**Prior sensitivity (pooled prior, v1 curve).** Each cell is the ROI estimate and 94% range.

| Spread | meta_ads | google_search | youtube | influencers | email | tv | Recovered |
|---|---|---|---|---|---|---|---|
| 0.25 | 0.91 (0.32 to 1.46) | 0.79 (0.19 to 1.48) | 0.86 (0.16 to 1.69) | 0.35 (0.03 to 0.68) | 0.75 (0.08 to 1.40) | 0.77 (0.58 to 0.98) | 3/6 |
| 0.5 | 0.92 (0.29 to 1.61) | 0.97 (0.12 to 2.06) | 0.94 (0.15 to 1.85) | 0.34 (0.04 to 0.71) | 0.84 (0.07 to 1.72) | 0.76 (0.57 to 0.98) | 4/6 |
| 1.0 | 0.92 (0.27 to 1.65) | 1.10 (0.08 to 2.49) | 0.95 (0.07 to 1.93) | 0.31 (0.01 to 0.65) | 0.95 (0.02 to 2.30) | 0.75 (0.54 to 0.96) | 4/6 |
| 2.0 | 0.88 (0.11 to 1.54) | 1.18 (0.10 to 2.82) | 0.97 (0.01 to 2.09) | 0.29 (0.00 to 0.61) | 1.15 (0.00 to 2.96) | 0.76 (0.54 to 0.95) | 4/6 |

Meta, YouTube and TV barely move across an eightfold change in prior strength: the data
determines them. Google Search and email move with the prior: the data says little about
them. Those are the channels a lift test would help most.

## 4. Adstock

Delayed adstock was worse on every measure. Weibull adstock cannot be sampled on this
project's Numba backend (its gradient is unavailable), so it is not offered. The library
applies one adstock form to all channels, so "choose per channel" is not possible without a
custom model; the comparison is at model level.

## 5. Time-varying effects

- **Time-varying baseline** (`time_varying_intercept`): 41 divergences, no better fit, wider
  ranges. Rejected.
- **Time-varying media effectiveness** (`time_varying_media`): the library estimates one
  multiplier shared by all channels, not one per channel, so it cannot show Meta alone. On
  this data the multiplier stays between 0.98 and 1.02 for all three years, which is correct:
  the synthetic effects are constant. See `reports/figures/model_time_varying_media.png`.
  It is left off by default because it adds parameters the data does not need.

## 6. Model comparison

| Variant | LOO elpd | Holdout MAPE | Recovered (demo seed) | Recovered (3 other seeds) | Mean ROI range width | Email range (truth 3.47) | Divergences |
|---|---|---|---|---|---|---|---|
| v1: logistic curve, coefficient priors | 368.3 | 1.43% | 5/6 | 17/18 | 7.29 | 0.00 to 34.63 | 1 |
| Pooled ROI prior (partial pooling) | 368.9 | 1.56% | 4/6 | 11/18 | 1.48 | 0.01 to 2.14 | 4 |
| Independent ROI prior | 369.5 | 1.49% | 5/6 | 16/18 | 2.01 | 0.03 to 4.18 | 0 |
| Delayed adstock | 367.4 | 1.60% | 3/6 | 16/18 | 4.85 | 0.07 to 22.10 | 0 |
| Time-varying baseline | 367.6 | 1.50% | 5/6 | 17/18 | 10.13 | 0.00 to 50.30 | 41 |
| Hill curve | 368.0 | 1.39% | 6/6 | 18/18 | 8.10 | 0.00 to 35.42 | 0 |
| **Hill curve + independent ROI prior (chosen)** | 369.0 | 1.44% | 6/6 | 18/18 | 2.59 | 0.02 to 4.37 | 0 |

**Decision: Hill curve with an independent ROI prior.**

- LOO and holdout cannot separate the variants: elpd differs by about 1 with a standard error
  of about 8, and holdout error is 1.4% to 1.6% throughout. Predictive fit says nothing
  about which model splits credit correctly. Only ground truth does.
- The Hill curve is what fixes recovery (6/6, and 18/18 on other seeds).
- The ROI prior is what makes the ranges usable: with the Hill curve alone the mean range
  width is 8.1 and email is 0 to 35; with the ROI
  prior it is 2.6 and email is
  0.02 to 4.37, at the same recovery.

**After the change, on all three demo brands:** recovery 18 of 18 (was 15 of 18); 30 of 30
across five other seeds (was 27 of 30); 6 of 6 without the TV and YouTube adstock priors
(was 5 of 6); holdout MAPE 1.4% to 1.5% (unchanged); the optimizer's true uplift is 65% of
what it expects (was 38%).

**Is this just fitting the generator?** The synthetic truth is a Hill curve, so a Hill model
has an unfair advantage. To check, the data was regenerated with a *logistic* true curve
(`true_saturation: logistic`) on three seeds. The Hill model recovered 18 of 18 channel ROIs;
the v1 logistic model recovered 16 of 18 on data that matches its own curve family. The Hill
curve with a free slope is the more flexible of the two, which is what matters, not that it
shares a name with the generator.

**What it costs.** Email's best estimate (1.5) is biased well below its truth (3.47), because
a channel with 1.6% of spend is mostly prior. A real brand's response need not be Hill-shaped either; the check above shows the model is
robust to one alternative, not to all.

## 7. Benchmark against Google Meridian

Meridian 2.1 (national model, default priors, 4 chains, 500 kept draws), run by
`scripts/benchmark_meridian.py` in its own environment. Ranges are 3rd to 97th percentile for
Meridian and 94% HDI for MixLab. ✗ marks a range that misses the truth.

| Channel | True ROI | MixLab v1 | MixLab v2 | Meridian |
|---|---|---|---|---|
| meta_ads | 1.18 | 0.75 (0.01 to 1.44) | 1.05 (0.08 to 1.99) | 0.89 (0.20 to 2.07) |
| google_search | 1.71 | 2.05 (0.04 to 4.21) | 2.02 (0.10 to 4.47) | 4.45 (2.15 to 7.88) ✗ |
| youtube | 0.89 | 1.21 (0.00 to 2.56) | 1.46 (0.04 to 3.30) | 2.52 (0.62 to 5.28) |
| influencers | 0.89 | 0.22 (0.00 to 0.51) | 0.47 (0.01 to 1.16) | 1.37 (0.30 to 3.32) |
| email | 3.47 | 11.84 (0.00 to 34.63) | 1.52 (0.02 to 4.37) | 1.60 (0.21 to 5.43) |
| tv | 0.75 | 0.71 (0.50 to 0.92) | 0.76 (0.54 to 0.99) | 0.34 (0.18 to 0.52) ✗ |

Meridian recovers 4 of 6. My best explanations for the two misses, which I have not
tested by changing Meridian's settings:

- **Google Search (Meridian 4.45, truth 1.71).** Search spend in this data follows demand, so
  it rises in weeks that would have sold more anyway. If the default national model does not
  absorb that seasonal and festive demand, the lift is credited to search. MixLab includes
  yearly seasonality and holiday controls, which take it.
- **TV (Meridian 0.34, truth 0.75).** TV runs only in bursts on top of Diwali. MixLab's
  slow-decay adstock prior for TV spreads its effect over the following weeks; without such
  a prior more of the burst is attributed to the festival control.
- **Email.** Both tools rely on the prior for a channel with 1.6% of spend.

This is one brand, default settings, and Meridian is built around geo-level data, which this
test does not give it. It shows the two tools make different default assumptions, not that
one is better.

## 8. Effects downstream of the model change

- **Optimizer.** With moves gated and capped, the model now expects +2.9% and the truth
  delivers +2.0% (v1: +5.6% against +1.9%). With no cap: +6.8% against +4.0% (v1: +16.4%
  against -0.1%). The haircut applied for "realistic uplift" moved from 0.38 to 0.65. It is
  measured on the same three brands it is then applied to, so it is a calibration, not a
  validated forecast; Phase 3 replaces it.
- **A local-optimum bug.** A Hill response surface is not concave. On the TV-heavy brand the
  unconstrained solver returned a plan the model itself rated below the current one.
  `optimize_budget` now retries from the current plan and, failing that, recommends no change.
- **Calibration.** The test planner now picks Meta first. A simulated 8-week lift test narrows
  Meta's ROI range by 79% (0.08 to 1.99 becomes 0.88 to 1.29; truth 1.18).

## 9. The robust optimizer (Phase 3)

**What changed.** Plans are scored across posterior draws with the project's own response
maths and searched with SciPy from several starting points. Limits per channel come from the
evidence (±10% burst or tiny channels, ±15% where the ROI range is wide, ±30% otherwise, never
above the channel's peak week). Objectives: expected revenue, expected profit, risk-adjusted,
10th percentile. Goals: a revenue target or a target ROI. Output includes a rollout plan, a
week-by-week schedule and a corner-solution warning.

**The optimizer's-curse correction uses no ground truth.** For each of six replicates a
posterior draw is treated as the real world; revenue is simulated from it, the model is refit
on that history, the plan is re-optimized, and the chosen plan is scored under the world that
generated the data. Delivered over expected uplift, pooled, is the haircut.

| Brand | Expected uplift | Haircut (80% range) | Realistic uplift | True uplift |
|---|---|---|---|---|
| performance_heavy | +1.94% | 0.59 (0.24 to 0.94) | +1.15% | +1.60% |
| tv_heavy | +1.15% | 0.88 (0.40 to 1.00) | +1.01% | +0.92% |
| influencer_led | +3.00% | 0.92 (0.76 to 1.00) | +2.78% | +2.09% |

The corrected figure is within about 0.7 percentage points of the truth on all three brands,
on the right side of the expected figure for two and too cautious for one. Six replicates make
the haircut itself uncertain, which the range shows.

**Twenty random brands, naive against robust** (`reports/optimizer_benchmark.json`). Naive is
the v1 behaviour: maximise expected revenue with no limits and no haircut. Robust uses the
limits, the risk-adjusted objective and one haircut (0.59, estimated on the main
demo brand and applied unchanged to all twenty).

| | Promised uplift | True uplift | Promised / true | Brands made worse | Worst case | Mean gap |
|---|---|---|---|---|---|---|
| Naive | +6.6% | +2.6% | 2.6x | 20% | -28.8% | 6.1 pts |
| Robust | +1.4% | +1.7% | 0.8x | 10% | -1.6% | 1.0 pts |

**Reading it.** The robust optimizer keeps its promises (it under-promises slightly) and its
worst outcome is a 1.6% loss against the naive optimizer's 28.8%. It pays for that with less
uplift on average: the naive optimizer's large moves are right more often than not and deliver
more when they are. Which one to use is a judgment about how much a bad quarter costs, not a
statistical question.

**Limits of this result.**
- The model is additive, so the weekly schedule cannot say which calendar weeks suit media.
- No rollout step is large enough to see in topline revenue within four weeks; the checkpoints
  say so rather than pretending otherwise.
- All three demo recommendations are corner solutions: the limits decide them.
- Customer acquisition cost goals are not supported; the data has revenue, not customers.

## 10. The geo-level model (Phase 4)

**What changed.** A hierarchical model over a regional panel (`geo_model.GeoMixLabModel`,
`configs/geo.yaml`). The performance-heavy brand is split across ten Indian regions with their
own size, demand, media prices, spend mix, festivals and true channel effects (drawn around the
national ones, LogNormal sigma 0.3). Each region has its own baseline and channel ROI; ROI is
partially pooled, `log ROI[g, c] = log ROI_national[c] + tau[c] * z[g, c]`. Carryover, curve
shape, seasonality, trend and control effects are shared. The comparison baseline is the v2
national model (`configs/demo.yaml`) fitted on the same panel summed to national weeks.

**Sampling.** Geo model: 4 x (1,000 + 1,000), target_accept 0.95, 14 minutes, worst r-hat
1.010, smallest ESS 602, 1 divergence. National model: 3.5 minutes, 0 divergences.

**National ROI: national model against geo model** (`artifacts/india_regions/geo_comparison.json`)

| Channel | True ROI | National model (94% range) | Geo model (94% range) | Width ratio |
|---|---|---|---|---|
| meta_ads | 1.34 | 0.85 (0.09 to 1.68) | 0.84 (0.40 to 1.36) | 0.61 |
| google_search | 1.57 | 3.43 (0.25 to 6.82) | 1.82 (0.60 to 3.16) | 0.39 |
| youtube | 0.86 | 0.81 (0.03 to 1.92) | 0.92 (0.30 to 1.67) | 0.73 |
| influencers | 0.81 | 0.87 (0.07 to 1.94) | 0.48 (0.14 to 0.88) | 0.39 |
| email | 3.91 | 1.73 (0.03 to 5.00) | 1.47 (0.02 to 4.17) | 0.84 |
| tv | 0.70 | 0.68 (0.53 to 0.83) | 0.74 (0.69 to 0.79) | 0.32 |

- Both models recover 6 of 6. The geo ranges are **50% narrower** (median width ratio 0.50) and
  the mean absolute error falls from 0.78 to 0.60. Recovery did not get worse, so it merges.
- Why: ten regions whose spend moved differently give the model far more independent variation
  than one national series where channels rise and fall together.
- Email barely improves (ratio 0.84): it is about 1.6% of spend in every region, so no region
  measures it well and pooling has little to pool. It still needs a lift test (Phase 5).

**Regional optimizer.** At the same budget, moving money between regions as well as channels
gives a true uplift of **+3.4%**, inside the model's 94% range of +2.7% to +6.9% (expected
+4.7%, so 1.4x over-promised, against 2x for the v1 national optimizer). The best national
plan, spread across regions in today's proportions (all a national model can recommend),
delivers +2.6% in truth.

**Senior-practitioner review: the top three, and what was done.**
1. *Regional limits ignored the evidence.* Every region-channel pair could move ±30%. Fixed:
   `geo_bounds` now applies `optimizer.channel_limits` per region (±10% for bursts and tiny
   shares, ±15% for wide ROI ranges), as the national optimizer does.
2. *"Under/over-invested" compared medians only.* Fixed: a region is flagged only when the gap
   is over 1.2x AND at least 80% of posterior draws agree on its direction
   (`GEO_STATUS_CONFIDENCE`).
3. *Regional coverage is below nominal and rests on one seed.* Regional ranges contain the
   truth for 51 of 60 pairs (85%, against 94% nominal). The likely cause is the shared curve
   shape: regional media prices move the true half-saturation point, which a shared kappa
   cannot follow. Not fixed here: freeing kappa per region and repeating over seeds is a
   refit study (about 20 minutes per seed) that belongs with the Phase 7 benchmark. Treat
   regional ranges as slightly overconfident.

**Not built.** Reach and frequency inputs (optional). In real data national TV usually cannot
be bought or measured by region, which this synthetic panel does not reproduce, so TV's very
tight geo range is optimistic.
