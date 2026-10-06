"""Value of information: how much a lift test on each channel is worth before running it.

For one channel, the decision is to shift budget into or out of it at the same total budget
(``config.VOI_ACTIONS``): the channel moves by that share and every other channel moves the
opposite way in proportion. Each action's value in a posterior draw is the gross profit it
adds, ``margin x extra revenue`` (spend is unchanged). Today's decision is the action with the
best average value.

- **Expected loss now**: the average, over draws, of how much better the best action for that
  draw would have done than today's choice. It is the money at risk from being uncertain.
- **Expected loss after a test**: simulate a test result for a channel (its true ROI plus
  measurement noise), re-weight the posterior by how well each draw explains that result,
  decide again, and score the decision in the world that produced the result.

The difference is the value of the test. Rank channels by it to choose what to test first.
"""

from typing import Any

import numpy as np

from mixlab import config
from mixlab.insights import PosteriorDraws
from mixlab.optimizer import Plan, last_quarter_spend, plan_revenue_draws, thin
from mixlab.transforms import FloatArray


def action_values(
    draws: PosteriorDraws,
    channel: str,
    current: Plan,
    n_weeks: int,
    margin: float,
    actions: tuple[float, ...] = config.VOI_ACTIONS,
) -> FloatArray:
    """Return the gross profit each budget shift adds versus today, per draw (S, A)."""
    base = plan_revenue_draws(draws, current, n_weeks).sum(axis=1)
    values = []
    for change in actions:
        values.append(
            margin
            * (
                plan_revenue_draws(draws, shift(current, channel, change), n_weeks).sum(axis=1)
                - base
            )
        )
    return np.stack(values, axis=1)


def shift(current: Plan, channel: str, change: float) -> Plan:
    """Return ``current`` with ``channel`` moved by ``change`` and the others rebalanced.

    The total budget is unchanged: what the channel gains or loses is taken from or given to
    the other channels in proportion to their spend.
    """
    moved = current[channel] * change
    others = sum(v for c, v in current.items() if c != channel)
    return {
        c: (v + moved) if c == channel else v * (1 - moved / others if others else 1.0)
        for c, v in current.items()
    }


def expected_loss(values: FloatArray, weights: FloatArray | None = None) -> float:
    """Return the expected regret of choosing the action with the best weighted average."""
    weights = np.full(len(values), 1 / len(values)) if weights is None else weights
    chosen = int(np.argmax(weights @ values))
    return float(weights @ (values.max(axis=1) - values[:, chosen]))


def loss_after_test(values: FloatArray, signal: FloatArray, se: float, seed: int) -> float:
    """Return the expected regret after observing a noisy measurement of ``signal``.

    Each draw in turn is taken as the true world; a test result is simulated from it; the
    posterior is re-weighted by the likelihood of that result; the decision is remade and
    scored in the true world.
    """
    rng = np.random.default_rng(seed)
    observed = signal + rng.normal(0.0, se, len(signal))
    log_like = -0.5 * ((observed[:, None] - signal[None, :]) / se) ** 2
    weights = np.exp(log_like - log_like.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    chosen = np.argmax(weights @ values, axis=1)
    rows = np.arange(len(values))
    return float(np.mean(values.max(axis=1) - values[rows, chosen]))


def value_of_information(
    draws: PosteriorDraws,
    margin: float = config.DEFAULT_MARGIN,
    n_weeks: int = config.OPTIMIZER_WEEKS,
    relative_se: float = config.SIMULATED_TEST_RELATIVE_SE,
    seed: int = config.RANDOM_SEED,
) -> list[dict[str, Any]]:
    """Rank channels by how much a lift test would reduce the cost of a wrong budget call.

    The test measures the channel's ROI over the window with a standard error of
    ``relative_se`` times its posterior median.
    """
    sample = thin(draws, config.VOI_DRAWS)
    current = last_quarter_spend(sample, n_weeks)
    roi = plan_revenue_draws(sample, current, n_weeks) / np.array(
        [max(current[c], 1.0) for c in sample.channels]
    )
    rows = []
    for index, channel in enumerate(sample.channels):
        values = action_values(sample, channel, current, n_weeks, margin)
        signal = roi[:, index]
        se = relative_se * max(float(np.median(signal)), 1e-9)
        before = expected_loss(values)
        after = loss_after_test(values, signal, se, seed + index)
        best = config.VOI_ACTIONS[int(np.argmax(values.mean(axis=0)))]
        rows.append(
            {
                "channel": channel,
                "decision_now": best,
                "expected_loss_now": before,
                "expected_loss_after_test": after,
                "value_of_test": max(before - after, 0.0),
                "roi_median": float(np.median(signal)),
            }
        )
    rows.sort(key=lambda row: row["value_of_test"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["priority"] = rank
    return rows


METHOD_NOTE: str = (
    "For each channel we ask: if we moved 20% of its budget to or from the other channels, "
    "how much profit "
    "would we expect to lose by picking the wrong option, given what the model is unsure "
    "about? Then we imagine running a test that measures the channel's return, update the "
    "model with each possible result, and ask the same question again. The drop in expected "
    "loss is what the test is worth. Test the channel with the biggest drop first."
)
