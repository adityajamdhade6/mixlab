# MixLab model card

## What the model is
A Bayesian marketing mix model fitted to weekly data with PyMC-Marketing
(`pymc_marketing.mmm.multidimensional.MMM`, version 0.19).

```
revenue = baseline + linear trend + yearly seasonality (Fourier) + controls
          + sum over channels of  beta_c * logistic_saturation(geometric_adstock(spend_c))
          + noise
```

- **Carryover:** geometric adstock over 8 weeks, normalised.
- **Diminishing returns:** logistic saturation, one curve per channel.
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

## Deliberate mismatch with the synthetic data
The data generator uses a **Hill** saturation curve and **12-week** adstock; the model uses a
**logistic** curve and **8 weeks**. Recovery results therefore include the cost of a wrong
functional form, as they would with real data.

## Priors
Weakly informative, set in `config.PriorSettings` and explained line by line in
`model.build_priors`. `configs/default.yaml` adds slow-decay priors for TV and YouTube; these
happen to agree with the synthetic truth. `scripts/evaluate_models.py` refits without them.

## Evidence (synthetic brands, `reports/evaluation_summary.json`)
| Check | Result |
|---|---|
| True ROI inside the 94% range, demo brands | 5 of 6 channels in each of 3 brands |
| Same, across 5 other random seeds | 27 of 30 (90%) |
| Same, without the TV/YouTube priors | 5 of 6 |
| Holdout MAPE, last 13 weeks | 1.3% to 1.5% (all weeks inside the 94% predictive range) |
| Public pymc-marketing dataset | Both documented decay values recovered; in-sample MAPE 3.8% |

The three demo brands share one random seed, so their misses are correlated: all three miss
influencers. Across other seeds influencers is recovered every time.

## Known failure modes
- **Wide ranges.** Most channel ROIs span "poor" to "good". Only channels with distinctive
  spend patterns (TV flights) are pinned down.
- **Small smooth channels are overestimated.** Email (1.6% of spend, low variation) has a
  posterior mean about three times the truth; the spend-share prior option corrects this.
- **Optimizer optimism.** Expected uplift is roughly double the true uplift, and with no
  bounds on channel moves the true uplift is zero or negative.
- **Lift calibration ignores carryover.** Experiments are matched to the saturation curve at
  average weekly spend.

## Not suitable for
Daily or campaign-level decisions, creative testing, channels with under about a year of
history, or any use where a single ROI number is read without its range.
