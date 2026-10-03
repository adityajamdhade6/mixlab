"""MixLab: Bayesian Marketing Mix Model and budget optimizer."""

import os

from mixlab.config import PYTENSOR_FLAGS

# Must be set before PyTensor is first imported; an existing PYTENSOR_FLAGS value wins.
os.environ.setdefault("PYTENSOR_FLAGS", PYTENSOR_FLAGS)

__version__ = "0.1.0"
