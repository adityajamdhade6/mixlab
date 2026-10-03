# Resume bullets (draft)

- Built a Bayesian marketing mix model (PyMC-Marketing) with synthetic ground-truth
  validation that recovered true channel ROI within its 94% range for 27 of 30 estimates and
  predicted a 13-week holdout at 1.4% MAPE.
- Built a constrained budget optimizer and scenario simulator and scored its recommendations
  against known truth, showing that capping channel moves at 30% turned a 0% true uplift
  (unconstrained) into +3.3% on the same budget.
- Built lift-test calibration and a test planner that picked the highest-value channel to
  test, where one simulated 8-week experiment narrowed that channel's ROI range by 85%.

Alternates:

- Built an LLM explanation layer (Claude, tool calling) that writes CMO briefs from model
  output and verifies every number against its inputs, covered by 150+ tests at 94% coverage.
- Built export mappers for Meta Ads, Google Ads and Shopify plus a data validator that scores
  MMM readiness out of 100, cutting data preparation for a new brand to one command.
