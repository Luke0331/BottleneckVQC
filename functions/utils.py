"""Shared utilities: seeds, TF logging, constants."""
from __future__ import annotations

import logging
import os
import random

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

MISSING_VALUE = -999.0


def set_seeds(seed: int = 7) -> None:
    """Seed Python, NumPy, and TensorFlow RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def silence_tf_warnings(enabled: bool = True) -> None:
    """Silence TensorFlow WARNING logs (disable when debugging)."""
    if not enabled:
        return
    tf.get_logger().setLevel(logging.ERROR)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
