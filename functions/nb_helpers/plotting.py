"""Plotting helpers for UV field comparisons."""
from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from ..utils import MISSING_VALUE


def masked_array(arr: np.ndarray, missing_value: float = MISSING_VALUE) -> np.ma.MaskedArray:
    return np.ma.array(arr, mask=(arr == missing_value))


def robust_range(a, lo: float = 1.0, hi: float = 99.0):
    if hasattr(a, "compressed"):
        data = a.compressed()
    elif isinstance(a, (list, tuple)):
        vals = []
        for x in a:
            c = x.compressed() if hasattr(x, "compressed") else np.asarray(x).ravel()
            if c.size:
                vals.append(c)
        data = np.concatenate(vals) if vals else np.array([])
    else:
        data = np.asarray(a).ravel()
    if data.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(data, lo))
    vmax = float(np.percentile(data, hi))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6
    return vmin, vmax


def plot_uv_threeway(
    y_true, y_pred_vqc, y_pred_mlp, idx, title=None, missing_value: float = MISSING_VALUE
):
    true_u, true_v = y_true[idx, ..., 0], y_true[idx, ..., 1]
    building_mask = true_u == missing_value
    u_true = np.ma.array(true_u, mask=building_mask)
    v_true = np.ma.array(true_v, mask=building_mask)
    u_vqc = np.ma.array(y_pred_vqc[idx, ..., 0], mask=building_mask)
    v_vqc = np.ma.array(y_pred_vqc[idx, ..., 1], mask=building_mask)
    u_mlp = np.ma.array(y_pred_mlp[idx, ..., 0], mask=building_mask)
    v_mlp = np.ma.array(y_pred_mlp[idx, ..., 1], mask=building_mask)
    u_min, u_max = robust_range(u_true)
    v_min, v_max = robust_range(v_true)
    cmap = plt.cm.coolwarm.copy()
    cmap.set_bad("white")
    fig, axes = plt.subplots(
        2, 4, figsize=(6.6, 4.5),
        gridspec_kw={"width_ratios": [1, 1, 1, 0.05], "wspace": 0.03, "hspace": 0.18},
    )
    axes[0, 0].imshow(u_true, origin="lower", cmap=cmap, vmin=u_min, vmax=u_max)
    axes[0, 0].set_title("U True", pad=6)
    axes[0, 1].imshow(u_mlp, origin="lower", cmap=cmap, vmin=u_min, vmax=u_max)
    axes[0, 1].set_title("U Pred (C-UNet)", pad=6)
    im_u = axes[0, 2].imshow(u_vqc, origin="lower", cmap=cmap, vmin=u_min, vmax=u_max)
    axes[0, 2].set_title("U Pred (C-QB-UNet)", pad=6)
    axes[1, 0].imshow(v_true, origin="lower", cmap=cmap, vmin=v_min, vmax=v_max)
    axes[1, 0].set_title("V True", pad=6)
    axes[1, 1].imshow(v_mlp, origin="lower", cmap=cmap, vmin=v_min, vmax=v_max)
    axes[1, 1].set_title("V Pred (C-UNet)", pad=6)
    im_v = axes[1, 2].imshow(v_vqc, origin="lower", cmap=cmap, vmin=v_min, vmax=v_max)
    axes[1, 2].set_title("V Pred (C-QB-UNet)", pad=6)
    u_ticks = np.arange(np.floor(u_min), np.ceil(u_max) + 1, 1)
    v_ticks = np.arange(np.floor(v_min), np.ceil(v_max) + 1, 1)
    fig.colorbar(im_u, cax=axes[0, 3], ticks=u_ticks).set_label("m/s")
    fig.colorbar(im_v, cax=axes[1, 3], ticks=v_ticks).set_label("m/s")
    for ax in [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]]:
        ax.set_xticks([])
        ax.set_yticks([])
    if title:
        fig.suptitle(title, y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.88])
    else:
        fig.tight_layout()
    plt.show()


def plot_uv_qubits_compare(
    y_true, pred_dict: Dict[int, np.ndarray], idx: int = 0, title=None, missing_value: float = MISSING_VALUE
):
    _plot_uv_param_compare(
        y_true, pred_dict, idx=idx, title=title, missing_value=missing_value, unit_label="qubits"
    )


def plot_uv_layers_compare(
    y_true, pred_dict: Dict[int, np.ndarray], idx: int = 0, title=None, missing_value: float = MISSING_VALUE
):
    _plot_uv_param_compare(
        y_true, pred_dict, idx=idx, title=title, missing_value=missing_value, unit_label="layers"
    )


def _plot_uv_param_compare(
    y_true,
    pred_dict: Dict[int, np.ndarray],
    idx: int = 0,
    title=None,
    missing_value: float = MISSING_VALUE,
    unit_label: str = "qubits",
):
    keys = sorted(pred_dict.keys())
    true_u, true_v = y_true[idx, ..., 0], y_true[idx, ..., 1]
    building_mask = true_u == missing_value
    u_true = np.ma.array(true_u, mask=building_mask)
    v_true = np.ma.array(true_v, mask=building_mask)
    u_min, u_max = robust_range(u_true)
    v_min, v_max = robust_range(v_true)
    cmap = plt.cm.coolwarm.copy()
    cmap.set_bad("white")
    n_cols = 1 + len(keys)
    fig, axes = plt.subplots(2, n_cols, figsize=(2.5 * n_cols, 5), gridspec_kw={"wspace": 0.04, "hspace": 0.20})
    im_u = axes[0, 0].imshow(u_true, origin="lower", cmap=cmap, vmin=u_min, vmax=u_max)
    im_v = axes[1, 0].imshow(v_true, origin="lower", cmap=cmap, vmin=v_min, vmax=v_max)
    axes[0, 0].set_title("U True", pad=6)
    axes[1, 0].set_title("V True", pad=6)
    for col, k in enumerate(keys, start=1):
        pu = np.ma.array(pred_dict[k][idx, ..., 0], mask=building_mask)
        pv = np.ma.array(pred_dict[k][idx, ..., 1], mask=building_mask)
        axes[0, col].imshow(pu, origin="lower", cmap=cmap, vmin=u_min, vmax=u_max)
        axes[1, col].imshow(pv, origin="lower", cmap=cmap, vmin=v_min, vmax=v_max)
        axes[0, col].set_title(f"U Pred ({k} {unit_label})", pad=6)
        axes[1, col].set_title(f"V Pred ({k} {unit_label})", pad=6)
    for r in range(2):
        for c in range(n_cols):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
    fig.colorbar(im_u, ax=axes[0, :].ravel().tolist(), fraction=0.02, pad=0.01).set_label("m/s")
    fig.colorbar(im_v, ax=axes[1, :].ravel().tolist(), fraction=0.02, pad=0.01).set_label("m/s")
    if title:
        fig.suptitle(title, y=0.99, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def extract_valid_points_by_comp(y_true, y_pred, comp_idx: int, missing_value: float = MISSING_VALUE):
    """Return 1D (true, pred) arrays for one component over all valid pixels."""
    t = y_true[..., comp_idx]
    p = y_pred[..., comp_idx]
    valid = t != missing_value
    return t[valid].reshape(-1), p[valid].reshape(-1)


def r2_score_np(y_true_1d, y_pred_1d) -> float:
    sse = float(np.sum((y_pred_1d - y_true_1d) ** 2))
    sst = float(np.sum((y_true_1d - np.mean(y_true_1d)) ** 2)) + 1e-6
    return float(1.0 - sse / sst)


def shared_axis_range(*arrays, pad_ratio: float = 0.05):
    vals = [np.asarray(a).reshape(-1) for a in arrays if np.asarray(a).size > 0]
    if len(vals) == 0:
        return (-1.0, 1.0)
    allv = np.concatenate(vals)
    lo = float(np.min(allv))
    hi = float(np.max(allv))
    span = hi - lo
    pad = pad_ratio * (span + 1e-6)
    return (lo - pad, hi + pad)


def plot_scatter_reg(
    ax,
    y_true_1d,
    y_pred_1d,
    title: str,
    axis_range,
    point_color: str = "tab:blue",
    max_points: int = 120000,
):
    """Scatter + linear fit; R2/fit use all points, scatter may be subsampled."""
    n = y_true_1d.size
    if n > max_points:
        idx = np.random.default_rng(7).choice(n, size=max_points, replace=False)
        xt = y_true_1d[idx]
        yp = y_pred_1d[idx]
    else:
        xt = y_true_1d
        yp = y_pred_1d

    a, b = np.polyfit(y_true_1d, y_pred_1d, 1)
    r2 = r2_score_np(y_true_1d, y_pred_1d)
    lo, hi = axis_range

    ax.scatter(xt, yp, s=3, alpha=0.15, color=point_color, edgecolors="none")
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2, label="y=x")
    x_line = np.array([lo, hi], dtype=np.float32)
    ax.plot(x_line, a * x_line + b, color="crimson", linewidth=1.6, label="fit")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Truth (m/s)")
    ax.set_ylabel("Prediction (m/s)")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", fontsize=9)
    return {"n": int(n), "r2": r2, "slope": float(a), "intercept": float(b)}
