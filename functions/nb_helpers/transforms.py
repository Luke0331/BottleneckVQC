"""Signed log1p transforms used by the paper experiments."""
from __future__ import annotations

import numpy as np


def signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def signed_expm1(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.expm1(np.abs(x))


def transform_y_signed_log1p(y: np.ndarray, missing_value: float) -> np.ndarray:
    y_out = y.copy()
    valid = y_out != missing_value
    y_out[valid] = signed_log1p(y_out[valid])
    return y_out.astype(np.float32)


def inverse_transform_y_signed_log1p(y_log: np.ndarray, missing_value: float) -> np.ndarray:
    y_out = y_log.copy()
    valid = y_out != missing_value
    y_out[valid] = signed_expm1(y_out[valid])
    return y_out.astype(np.float32)
