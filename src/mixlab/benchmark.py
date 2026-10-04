"""Random synthetic brands for benchmarking: many worlds with known truth, not one.

``random_brand`` perturbs the base brand's media mix, response curves and noise so that
conclusions about the model and the optimizer do not rest on a single hand-built example.
"""

from dataclasses import replace

import numpy as np

from mixlab import config
from mixlab.config import BrandConfig, ChannelConfig

SPEND_SPREAD = 0.35
EFFECT_SPREAD = 0.40
SATURATION_SPREAD = 0.30
DECAY_JITTER = 0.10
NOISE_RANGE = (0.03, 0.08)
BASE_SEED = 1000


def _vary(channel: ChannelConfig, rng: np.random.Generator) -> ChannelConfig:
    """Return a channel with its spend level, effect size, saturation and decay perturbed."""
    return replace(
        channel,
        base_spend=channel.base_spend * float(rng.lognormal(0.0, SPEND_SPREAD)),
        beta=channel.beta * float(rng.lognormal(0.0, EFFECT_SPREAD)),
        half_saturation=channel.half_saturation * float(rng.lognormal(0.0, SATURATION_SPREAD)),
        adstock_decay=float(
            np.clip(channel.adstock_decay + rng.uniform(-DECAY_JITTER, DECAY_JITTER), 0.05, 0.85)
        ),
    )


def random_brand(index: int, base: BrandConfig = config.PERFORMANCE_HEAVY_BRAND) -> BrandConfig:
    """Return the ``index``-th random brand; the same index always gives the same brand."""
    rng = np.random.default_rng(BASE_SEED + index)
    return replace(
        base,
        name=f"random_{index:03d}",
        seed=BASE_SEED + index,
        revenue_noise=float(rng.uniform(*NOISE_RANGE)),
        channels=tuple(_vary(channel, rng) for channel in base.channels),
    )
