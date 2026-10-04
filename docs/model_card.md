# MixLab model card

## What the model is
A Bayesian marketing mix model fitted to weekly data with PyMC-Marketing
(`pymc_marketing.mmm.multidimensional.MMM`, version 0.19). This card describes the v2 default;
[model_log.md](model_log.md) records how it was chosen.

```
revenue = baseline + linear trend + yearly seasonality (Fourier) + controls
          + sum over channels of  beta_c * hill_saturation(geometric_adstock(spend_c))
          + noise
```

- **Carryover:** geometric adstock over 8 weeks, normalised.
- **Diminishing returns:** Hill saturation, one curve per channel.
- **Priors:** stated on each channel's ROI (median 1, wide), not on raw coefficients.
- **Controls:** holiday flags, promo flag, price index (centred on 1.0).
- **Scaling:** revenue and each channel's spend are divided by their peak week.
- **Sampler:** nutpie NUTS, 4 chains x (1,000 tune + 1,000 draws), fixed seed.

## Assumptions, and where they bend
| Assumption | Why it matters |
|---|---|
| Effects are constant over time | A channel that improved or decayed over three years gets one average answer. |
| Channels act independently | TV creating demand that Search then captures is credited to Search. |
| Spend is exogenous given the controls | If spend chases demand the controls do not capture, that channel is over-credited. |
| One straight-line trend | Growth that bends is partly absorbed by channels whose spend grew in step. |
| Carryover is at most 8 weeks | Brand effects longer than two months are cut off. |
| ROI is revenue per rupee | No margin is applied; profit break-even is above an ROI of 1. |

## Match and mismatch with the synthetic data
The data generator uses a Hill saturation curve and 12-week adstock. The v2 model also uses a
Hill curve (v1 used a logistic curve, which is what caused its systematic miss on influencers)
but only 8 weeks of adstock. Recovery on synthetic data is therefore a favourable test of the
curve shape and an unfavourable one of carryover length. Real response curves need not be
Hill-shaped.

## Priors
ROI priors: each channel LogNormal with median 1.0 and log-scale spread 1.0 (about 0.14 to 7
covers 95%). TV and YouTube also carry slow-decay adstock priors that agree with the synthetic
truth; `scripts/evaluate_models.py` refits without them.

## Evidence (synthetic brands, `reports/evaluation_summary.json`)
| Check | v1 (logistic) | v2 (Hill + ROI priors) |
|---|---|---|
| True ROI inside the 94% range, three demo brands | 15 of 18 | **18 of 18** |
| Same, across 5 other random seeds | 27 of 30 | **30 of 30** |
| Same, without the TV/YouTube priors | 5 of 6 | **6 of 6** |
| Average width of the ROI range | 7.3 | **2.6** |
| Email ROI range (truth 3.47) | 0.00 to 34.4 | **0.02 to 4.35** |
| Holdout MAPE, last 12 to 13 weeks | 1.3% to 1.5% | 1.4% to 1.5% |
| Optimizer: true uplift as a share of expected | 0.38 | **0.65** |
| Public pymc-marketing dataset (v1 only) | Both decay values recovered; MAPE 3.8% | not re-run |

All 30 of 30 is slightly more than a 94% range should contain, so the v2 ranges are, if
anything, a little wide.

## Known failure modes
- **Ranges are still wide.** A typical channel's ROI spans roughly 0.05 to 3. Only channels
  with distinctive spend patterns (TV flights) are pinned down without an experiment.
- **Small channels are pulled toward the prior.** Email's estimate (1.5) sits well below its
  true ROI (3.47): with 1.6% of spend the data says little and the prior median of 1 dominates.
  The truth is inside the range, but the best estimate is biased low.
- **Curve-shape dependence.** The v1 logistic curve passed every sampler check while
  underestimating a channel with spiky spend in every brand.
- **Optimizer optimism.** Expected uplift is about 1.5 times the true uplift (v1: 2.6 times).
- **Forecast accuracy is uneven.** Rolling backtest error is 1.2% to 4.6% and R-squared 0.29
  to 0.98 depending on the window.
- **Lift calibration ignores carryover.** Experiments are matched to the saturation curve at
  average weekly spend.

## Not suitable for
Daily or campaign-level decisions, creative testing, channels with under about a year of
history, or any use where a single ROI number is read without its range.
