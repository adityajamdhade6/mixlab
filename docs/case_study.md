# MixLab: a marketing mix model that knows when it is wrong

## Problem
A direct-to-consumer brand spending a few crore a quarter across Meta, Google, YouTube,
influencers, email and TV has one recurring question: where should the next rupee go? The
numbers usually used to answer it come from the ad platforms themselves or from last-click
analytics. Both credit a sale to an ad that was touched, which means they cannot see the sales
that would have happened anyway, and they cannot see an ad whose effect lands three weeks
later.

The damage is specific. Search looks brilliant because people search when they have already
decided to buy. TV looks useless because nobody clicks a television. Every channel's reported
ROI is inflated by baseline demand it did not create, so the budget drifts toward whatever is
closest to the checkout and already saturated.

A marketing mix model (MMM) asks a different question: given three years of weekly spend and
revenue, how much revenue would have been lost without each channel? It is the right question,
but MMMs have their own reputation problem. They produce confident-looking numbers that nobody
can check, because with real data nobody ever observes what revenue would have been without
Meta. I wanted to build one that could be checked, and that reported its own uncertainty and
failures as prominently as its recommendations.

## Approach
Three decisions shaped the project.

**Start from a known answer.** Before modelling anything I wrote a data generator for a
fictional Indian skincare brand in which every channel's true effect is set by me: its
carryover, its saturation point, its ROI. The generator includes the traps real data has.
Search spend rises when demand is already high. TV runs in six-week bursts that always sit on
top of Diwali. Spend on every channel peaks in the festive season. If the model gets the wrong
answer on this data, I can see it, measure it and say so.

**Report distributions, never single numbers.** The model is Bayesian (PyMC-Marketing), so
each quantity is a distribution. Every metric downstream, from ROI to the optimizer's expected
uplift, carries a 94% range, and the dashboard and the AI layer are both required to show it.

**Make the model small and the checks large.** The model itself is conventional: geometric
carryover, a saturation curve per channel, trend, seasonality, holidays, promotions and price.
Most of the code is around it: validation before, diagnostics and ground-truth scoring after.

## What I built
- **A validator** that scores a dataset out of 100 for MMM readiness and explains each issue
  in plain English, for example that the model cannot learn from a channel whose spend never
  changes.
- **The model wrapper**, with every prior explained in a comment, prior predictive checks run
  before fitting, and saved models the dashboard can load without refitting.
- **An insights layer** computing ROI, marginal ROI (what the next rupee earns), saturation
  points and carryover per channel, plus a comparison against naive same-week attribution.
- **A budget optimizer** with per-channel bounds, a cautious objective that maximises the
  10th-percentile outcome, a scenario simulator and a curve showing where extra total budget
  stops paying back.
- **Lift-test calibration**: experiment results can be added to the model, and a planner
  suggests which channel to test next and for roughly how long.
- **An AI explanation layer** using Claude. It writes a one-page brief in three tones and
  answers questions by calling the real analysis functions rather than guessing. Every number
  in its output is extracted and checked against the inputs; anything not found is flagged.
- **Onboarding** for real data: an Excel template, mappers for Meta, Google Ads and Shopify
  exports, and an option to scale money by a secret factor so data can be shared.
- **A seven-page Streamlit dashboard** over three pre-fitted demo brands.

It is covered by more than 150 tests at 94% coverage, with lint and tests run on every push.

## Results
**The model recovers the truth about as often as it claims to.** Across five independently
generated versions of the main brand, the true ROI fell inside the model's 94% range for 27 of
30 channel estimates, or 90%. Removing the two priors that encoded "TV and YouTube decay
slowly", which happened to agree with the truth, left recovery at 5 of 6 channels.

**It predicts weeks it has not seen.** Fitted on the first 143 weeks and scored on the last 13,
error was 1.3% to 1.5% across the three brands, with every held-out week inside the predictive
range.

**Naive attribution is badly wrong in a predictable direction.** Same-week attribution gave
almost every channel an ROI near 3.5. Against the model it overstated Meta about five-fold,
YouTube 3.5-fold and TV 2.8-fold, because it credits channels with the 59% of revenue that is
baseline.

**The optimizer is right about direction and wrong about size.** With each channel allowed to
move at most 30%, the model expected a 7.4% uplift in incremental revenue from reallocating
last quarter's budget. Scored against the true data-generating process, that plan delivers
3.3%.

**One experiment is worth more than more modelling.** The test planner ranked Google Search
first by uncertainty times spend. An eight-week lift test simulated from ground truth narrowed
Search's ROI range by 85%, from 0.02 to 4.19 down to 1.33 to 1.94, around a true value of
1.71.

**It works on data I did not generate.** On the public example dataset from pymc-marketing the
model recovered both documented carryover rates (0.40 and 0.19 against 0.4 and 0.2).

## What surprised me
**Unconstrained optimization was worth nothing, or worse.** With no limit on channel moves the
model expected a 16% uplift for the main brand and the truth delivered roughly zero. For the
influencer-led brand it expected 24% and the truth was a 20% loss. The optimizer does exactly
what it is told: it moves money to wherever the model's estimate is highest, and the highest
estimates are disproportionately the overestimates. The "no channel moves more than 30%" rule,
which I had added as a practical nicety, turned out to be the thing that makes the
recommendation safe.

**A healthy sampler says nothing about a correct answer.** Convergence diagnostics passed
while the model put email's ROI at three times its true value. Diagnostics tell you the model's
answer was computed properly, not that the model is right.

**I nearly diagnosed a flaw that was not there.** All three demo brands missed the same
channel, influencers, which looked like a structural bias. I formed a theory about spiky spend
and saturation priors, tested a fix, and it did not help. The real reason was mundane: the
three brands shared one random seed, so their noise was identical and their misses were
correlated. Across five other seeds influencers was recovered every time.

**Calibration is local.** The lift test fixed Search and left the other five channels'
ranges essentially unchanged. An experiment buys certainty about the channel tested and little
else.

## Limitations
Everything above is on synthetic data. The generator is deliberately different from the model
(a different saturation curve, longer carryover), but it is still a world I designed, with
constant effects and channels that do not influence each other. No real brand's data has been
fitted.

ROI is revenue per rupee, not profit. Most channel ranges are too wide to act on without an
experiment; only TV, with its distinctive flights, is pinned down. The optimizer's expected
uplift should be roughly halved. The AI layer has been tested against a mocked client only, and
its number check confirms a figure exists in the inputs, not that it is attached to the right
claim.

## What I'd do next
1. **Fit a real brand** and put its results beside the platform-reported numbers.
2. **Add a profit view**, so break-even means break-even.
3. **Correct the optimizer's optimism** by shrinking expected uplift using the measured gap,
   and report the corrected figure by default.
4. **Model at geo level**, so lift tests and the MMM share one structure and calibration can
   use carryover properly.
5. **Run the test planner's first recommendation for real.** The clearest finding of the
   project is that the model and experiments are complements: the model says where to look and
   what is at stake, and an experiment settles it.
