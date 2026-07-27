"""Masked losses and evaluation metrics for UV wind fields."""
from __future__ import annotations

import math
from typing import Dict

import numpy as np
import tensorflow as tf

from .utils import MISSING_VALUE

def masked_mse(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    y_true: (...,H,W,2) with building=MISSING_VALUE
    """
    mask = tf.cast(tf.not_equal(y_true[..., 0], MISSING_VALUE), tf.float32)  # (B,H,W)
    mask2 = mask[..., tf.newaxis]  # (B,H,W,1)
    diff2 = tf.square((y_pred - y_true) * mask2)
    denom = tf.reduce_sum(mask) * 2.0 + 1e-6
    return tf.reduce_sum(diff2) / denom


def masked_weighted_charbonnier(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    eps: float = 1e-3,
    high_value_gain: float = 1.0,
    weight_clip: float = 3.0,
) -> tf.Tensor:
    """
    Weighted robust pixel loss (Charbonnier) on valid (non-building) pixels.
    Weights are larger on high-speed regions to emphasize wake/high-value structure.
    """
    mask = tf.cast(tf.not_equal(y_true[..., 0], MISSING_VALUE), tf.float32)  # (B,H,W)
    mask2 = mask[..., tf.newaxis]  # (B,H,W,1)

    # Speed magnitude from ground truth (normalized space); invalid pixels are masked out.
    speed = tf.sqrt(tf.reduce_sum(tf.square(y_true), axis=-1) + 1e-12) * mask  # (B,H,W)
    speed_mean = tf.reduce_sum(speed) / (tf.reduce_sum(mask) + 1e-6)
    speed_norm = speed / (speed_mean + 1e-6)

    # Base weight=1, add extra emphasis on high-speed area.
    w = 1.0 + high_value_gain * tf.clip_by_value(speed_norm, 0.0, weight_clip)
    w = w * mask  # keep invalid pixels at zero weight
    w2 = w[..., tf.newaxis]

    diff = y_pred - y_true
    charbonnier = tf.sqrt(tf.square(diff) + eps * eps)
    weighted = charbonnier * w2 * mask2

    denom = tf.reduce_sum(w) * 2.0 + 1e-6
    return tf.reduce_sum(weighted) / denom


def masked_grad_mse(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Gradient consistency loss on valid (non-building) pixels only.
    Uses simple finite differences in x/y directions for both channels.
    """
    mask = tf.cast(tf.not_equal(y_true[..., 0], MISSING_VALUE), tf.float32)  # (B,H,W)

    # x-direction gradient: (B,H,W-1,2)
    dy_true_x = y_true[:, :, 1:, :] - y_true[:, :, :-1, :]
    dy_pred_x = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
    mask_x = mask[:, :, 1:] * mask[:, :, :-1]
    mask_x2 = mask_x[..., tf.newaxis]

    # y-direction gradient: (B,H-1,W,2)
    dy_true_y = y_true[:, 1:, :, :] - y_true[:, :-1, :, :]
    dy_pred_y = y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]
    mask_y = mask[:, 1:, :] * mask[:, :-1, :]
    mask_y2 = mask_y[..., tf.newaxis]

    diff_x2 = tf.square((dy_pred_x - dy_true_x) * mask_x2)
    diff_y2 = tf.square((dy_pred_y - dy_true_y) * mask_y2)

    denom_x = tf.reduce_sum(mask_x) * 2.0 + 1e-6
    denom_y = tf.reduce_sum(mask_y) * 2.0 + 1e-6
    loss_x = tf.reduce_sum(diff_x2) / denom_x
    loss_y = tf.reduce_sum(diff_y2) / denom_y
    return 0.5 * (loss_x + loss_y)


def masked_mse_with_grad(y_true: tf.Tensor, y_pred: tf.Tensor, grad_weight: float = 0.1) -> tf.Tensor:
    return masked_mse(y_true, y_pred) + grad_weight * masked_grad_mse(y_true, y_pred)


def masked_mae_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    mask = tf.cast(tf.not_equal(y_true[..., 0], MISSING_VALUE), tf.float32)
    mask2 = mask[..., tf.newaxis]
    diff = tf.abs((y_pred - y_true) * mask2)
    denom = tf.reduce_sum(mask) * 2.0 + 1e-6
    return tf.reduce_sum(diff) / denom


def masked_rmse_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return tf.sqrt(masked_mse(y_true, y_pred) + 1e-12)


class MaskedR2(tf.keras.metrics.Metric):
    """
    R2 = 1 - SSE/SST
    Compute over non-building pixels only; pool U/V into one R2.
    """

    def __init__(self, name: str = "masked_r2", **kwargs):
        super().__init__(name=name, **kwargs)
        self.sum_y = self.add_weight(name="sum_y", initializer="zeros", dtype=tf.float32)
        self.sum_y_sq = self.add_weight(name="sum_y_sq", initializer="zeros", dtype=tf.float32)
        self.sum_err_sq = self.add_weight(name="sum_err_sq", initializer="zeros", dtype=tf.float32)
        self.count = self.add_weight(name="count", initializer="zeros", dtype=tf.float32)

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None):
        mask = tf.cast(tf.not_equal(y_true[..., 0], MISSING_VALUE), tf.float32)  # (B,H,W)
        mask2 = mask[..., tf.newaxis]  # (B,H,W,1)

        y_t = tf.reshape(y_true * mask2, [-1, 2])
        y_p = tf.reshape(y_pred * mask2, [-1, 2])
        m = tf.reshape(mask2, [-1, 1])

        # count valid scalars (2 channels)
        valid = tf.reduce_sum(m) * 2.0
        self.count.assign_add(valid)

        self.sum_y.assign_add(tf.reduce_sum(y_t))
        self.sum_y_sq.assign_add(tf.reduce_sum(tf.square(y_t)))
        self.sum_err_sq.assign_add(tf.reduce_sum(tf.square(y_p - y_t)))

    def result(self):
        eps = tf.constant(1e-6, dtype=tf.float32)
        mean = self.sum_y / (self.count + eps)
        sst = self.sum_y_sq - self.count * tf.square(mean)
        return 1.0 - (self.sum_err_sq / (sst + eps))

    def reset_state(self):
        self.sum_y.assign(0.0)
        self.sum_y_sq.assign(0.0)
        self.sum_err_sq.assign(0.0)
        self.count.assign(0.0)


def evaluate_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mask = y_true[..., 0] != MISSING_VALUE  # (N,H,W)
    mask2 = mask[..., None]
    diff = (y_pred - y_true) * mask2
    mae = float(np.sum(np.abs(diff)) / (np.sum(mask) * 2.0 + 1e-6))
    mse = float(np.sum(diff**2) / (np.sum(mask) * 2.0 + 1e-6))
    rmse = float(math.sqrt(mse))
    # R2 over valid pixels only; U/V pooled
    y_t = y_true[mask].reshape(-1)  # (num_valid,2) -> flatten
    y_p = y_pred[mask].reshape(-1)
    y_mean = float(y_t.mean()) if y_t.size > 0 else 0.0
    sse = float(np.sum((y_p - y_t) ** 2))
    sst = float(np.sum((y_t - y_mean) ** 2)) + 1e-6
    r2 = float(1.0 - sse / sst)
    return {"mae": mae, "rmse": rmse, "mse": mse, "r2": r2}

