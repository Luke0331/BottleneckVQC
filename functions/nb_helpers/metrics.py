"""Evaluation helpers used by reproduction notebooks."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ..data import denormalize_y
from ..losses import evaluate_numpy
from ..utils import MISSING_VALUE
from .transforms import inverse_transform_y_signed_log1p


def evaluate_split_physical(
    model,
    name: str,
    Xs: np.ndarray,
    Cs: np.ndarray,
    Ys: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    missing_value: float = MISSING_VALUE,
    verbose: bool = True,
):
    """Predict in normalized space, invert log1p, evaluate in physical m/s."""
    pred_norm = model.predict({"mask_img": Xs, "cond": Cs}, verbose=0).astype(np.float32)
    pred_log = denormalize_y(pred_norm, y_mean, y_std)
    y_log = denormalize_y(Ys, y_mean, y_std)
    pred_denorm = inverse_transform_y_signed_log1p(pred_log, missing_value)
    y_denorm = inverse_transform_y_signed_log1p(y_log, missing_value)
    metrics = evaluate_numpy(y_denorm, pred_denorm)
    if verbose:
        print(name, metrics)
    return y_denorm, pred_denorm, metrics


def evaluate_region_numpy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    region_mask: np.ndarray,
    missing_value: float = MISSING_VALUE,
) -> Dict[str, float]:
    valid = (y_true[..., 0] != missing_value) & region_mask[None, ...]
    valid2 = valid[..., None]
    diff = (y_pred - y_true) * valid2
    denom = np.sum(valid) * 2.0 + 1e-6
    mae = float(np.sum(np.abs(diff)) / denom)
    mse = float(np.sum(diff**2) / denom)
    rmse = float(np.sqrt(mse))

    y_t = y_true[valid].reshape(-1)
    y_p = y_pred[valid].reshape(-1)
    if y_t.size == 0:
        return {"mae": np.nan, "rmse": np.nan, "mse": np.nan, "r2": np.nan, "n_points": 0}

    y_m = float(y_t.mean())
    sse = float(np.sum((y_p - y_t) ** 2))
    sst = float(np.sum((y_t - y_m) ** 2)) + 1e-6
    r2 = float(1.0 - sse / sst)
    return {"mae": mae, "rmse": rmse, "mse": mse, "r2": r2, "n_points": int(np.sum(valid))}


def print_region_metrics(
    split_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tl_h: int = 75,
    tl_w: int = 90,
    missing_value: float = MISSING_VALUE,
) -> None:
    h, w = y_true.shape[1], y_true.shape[2]
    roi_topleft = np.zeros((h, w), dtype=bool)
    roi_topleft[:tl_h, :tl_w] = True
    roi_rest = ~roi_topleft
    m_tl = evaluate_region_numpy(y_true, y_pred, roi_topleft, missing_value=missing_value)
    m_rest = evaluate_region_numpy(y_true, y_pred, roi_rest, missing_value=missing_value)
    print(f"[{split_name}] top-left({tl_h}x{tl_w}) -> {m_tl}")
    print(f"[{split_name}] remaining area -> {m_rest}")


def _valid_inner_mask(valid_mask: np.ndarray) -> np.ndarray:
    inner = valid_mask.copy()
    inner[1:, :] &= valid_mask[:-1, :]
    inner[:-1, :] &= valid_mask[1:, :]
    inner[:, 1:] &= valid_mask[:, :-1]
    inner[:, :-1] &= valid_mask[:, 1:]
    return inner


def evaluate_gradient_numpy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    missing_value: float = MISSING_VALUE,
    dx: float = 10.0,
    dy: float = 10.0,
) -> Dict[str, Any]:
    comp_names = ["U", "V"]
    true_all, pred_all = [], []
    detail: Dict[str, Any] = {}

    for ci, cname in enumerate(comp_names):
        true_c_all, pred_c_all = [], []
        for i in range(y_true.shape[0]):
            t = y_true[i, ..., ci].astype(np.float32)
            p = y_pred[i, ..., ci].astype(np.float32)
            valid = t != missing_value
            inner = _valid_inner_mask(valid)
            if not np.any(inner):
                continue
            t_fill = np.where(valid, t, 0.0)
            p_fill = np.where(valid, p, 0.0)
            gt_y, gt_x = np.gradient(t_fill, dy, dx)
            gp_y, gp_x = np.gradient(p_fill, dy, dx)
            true_c_all.append(np.concatenate([gt_x[inner], gt_y[inner]], axis=0))
            pred_c_all.append(np.concatenate([gp_x[inner], gp_y[inner]], axis=0))

        if len(true_c_all) == 0:
            detail[cname] = {"grad_mae": np.nan, "grad_rmse": np.nan, "grad_r2": np.nan, "n_grad": 0}
            continue

        t_comp = np.concatenate(true_c_all)
        p_comp = np.concatenate(pred_c_all)
        diff_comp = p_comp - t_comp
        mae_comp = float(np.mean(np.abs(diff_comp)))
        rmse_comp = float(np.sqrt(np.mean(diff_comp**2)))
        sse_comp = float(np.sum(diff_comp**2))
        sst_comp = float(np.sum((t_comp - np.mean(t_comp)) ** 2)) + 1e-6
        detail[cname] = {
            "grad_mae": mae_comp,
            "grad_rmse": rmse_comp,
            "grad_r2": float(1.0 - sse_comp / sst_comp),
            "n_grad": int(t_comp.size),
        }
        true_all.append(t_comp)
        pred_all.append(p_comp)

    if len(true_all) == 0:
        return {"grad_mae": np.nan, "grad_rmse": np.nan, "grad_r2": np.nan, "n_grad": 0, "detail": detail}

    t_all = np.concatenate(true_all)
    p_all = np.concatenate(pred_all)
    diff_all = p_all - t_all
    sse_all = float(np.sum(diff_all**2))
    sst_all = float(np.sum((t_all - np.mean(t_all)) ** 2)) + 1e-6
    return {
        "grad_mae": float(np.mean(np.abs(diff_all))),
        "grad_rmse": float(np.sqrt(np.mean(diff_all**2))),
        "grad_r2": float(1.0 - sse_all / sst_all),
        "n_grad": int(t_all.size),
        "detail": detail,
    }


def collect_grad_rows(
    model_name: str,
    split_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    missing_value: float = MISSING_VALUE,
    dx: float = 10.0,
    dy: float = 10.0,
) -> List[Dict[str, Any]]:
    g = evaluate_gradient_numpy(y_true, y_pred, missing_value=missing_value, dx=dx, dy=dy)
    rows = [
        {
            "model": model_name,
            "split": split_name,
            "component": "ALL",
            "grad_MAE": g["grad_mae"],
            "grad_RMSE": g["grad_rmse"],
            "grad_R2": g["grad_r2"],
            "n_grad": g["n_grad"],
        }
    ]
    for comp in ("U", "V"):
        rows.append(
            {
                "model": model_name,
                "split": split_name,
                "component": comp,
                "grad_MAE": g["detail"][comp]["grad_mae"],
                "grad_RMSE": g["detail"][comp]["grad_rmse"],
                "grad_R2": g["detail"][comp]["grad_r2"],
                "n_grad": g["detail"][comp]["n_grad"],
            }
        )
    return rows


def predict_denorm(
    model,
    X_in: np.ndarray,
    C_in: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    missing_value: float = MISSING_VALUE,
) -> np.ndarray:
    """Predict then invert normalization + signed-log1p to physical m/s."""
    pred_norm = model.predict({"mask_img": X_in, "cond": C_in}, verbose=0).astype(np.float32)
    pred_log = denormalize_y(pred_norm, y_mean, y_std)
    return inverse_transform_y_signed_log1p(pred_log, missing_value)


def sample_r2_uv_together(y_true_i, y_pred_i, missing_value: float = MISSING_VALUE) -> float:
    """Per-sample R2 with U/V pooled (same convention as evaluate_numpy)."""
    mask = y_true_i[..., 0] != missing_value
    y_t = y_true_i[mask].reshape(-1)
    y_p = y_pred_i[mask].reshape(-1)
    if y_t.size == 0:
        return float("nan")
    y_mean_i = float(np.mean(y_t))
    sse = float(np.sum((y_p - y_t) ** 2))
    sst = float(np.sum((y_t - y_mean_i) ** 2)) + 1e-6
    return float(1.0 - sse / sst)
