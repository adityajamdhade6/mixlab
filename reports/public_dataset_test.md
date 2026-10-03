# Public dataset test

Source: [mmm_example.csv](https://github.com/pymc-labs/pymc-marketing/blob/main/data/mmm_example.csv) from the pymc-marketing repository (simulated data; values are unitless, not INR).

## Mapping
`date_week` to `date`, `y` to `revenue`, `x1`/`x2` to `spend_x1`/`spend_x2`, and `event_1`/`event_2` to `holiday_event_1`/`holiday_event_2`. `t` and `dayofyear` were dropped because the model builds its own trend and seasonality.

## Validation
Readiness score **97/100** over 179 weeks and 2 channels.

- info: 'spend_x2' is zero in 82% of weeks. That is fine for a channel run in bursts, but its estimate rests on few active weeks and will be less certain.
- info: 'revenue' has 2 unusual week(s): 2019-12-09, 2019-12-23. Check whether they are real (festive peaks, launches) or errors. Nothing has been removed.
- info: 'spend_x1' has 16 unusual week(s): 2018-06-04, 2018-07-16, 2018-07-30, 2018-11-12, 2018-12-24 and 11 more. Check whether they are real (festive peaks, launches) or errors. Nothing has been removed.

## Fit
Default priors, no channel-specific beliefs, target_accept 0.95.

- PASS r_hat: worst 1.010 (intercept_contribution), limit 1.01. The independent chains agree with each other.
- PASS effective sample size: smallest 693 (saturation_lam[spend_x2]), minimum 400. There are enough independent draws for stable averages and ranges.
- PASS divergences: 0 (tolerated: 4). The sampler explored the space without meaningful trouble.
- In-sample error (MAPE): **3.8%**. 96% of weeks fall inside the model's 94% predictive range.

## Results

| Channel | Adstock decay (94% range) | Documented decay | Share of revenue (94% range) |
|---|---|---|---|
| x1 | 0.40 (0.34 to 0.45) | 0.4 (inside range) | 29.3% (26.0% to 32.8%) |
| x2 | 0.19 (0.11 to 0.26) | 0.2 (inside range) | 7.2% (6.6% to 7.8%) |

Media drives 36% of the target (33% to 40%).

The documented decay values are the ones the pymc-marketing example notebook says were used to simulate the data. ROI is not reported because `x1` and `x2` are scaled activity levels, not spend in currency.
