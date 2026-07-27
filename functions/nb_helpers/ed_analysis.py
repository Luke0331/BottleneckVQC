"""Effective-dimension / spectral feature analysis helpers (ED notebook)."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import tensorflow as tf

from ..models_classical import build_unet_cond
from ..train import build_unet_cond_mlp_bottleneck_vqc


def build_models(
    input_shape,
    cond_dim: int,
    cond_emb_dim: int = 16,
    n_qubits: int = 5,
    n_layers: int = 2,
):
    """Build C-QB-UNet and C-UNet pair for ED / multi-seed analysis."""
    model_vqc = build_unet_cond_mlp_bottleneck_vqc(
        input_shape=input_shape,
        cond_dim=cond_dim,
        cond_emb_dim=cond_emb_dim,
        n_qubits=n_qubits,
        n_layers=n_layers,
    )
    model_mlp = build_unet_cond(
        input_shape=input_shape,
        cond_dim=cond_dim,
    )
    return model_vqc, model_mlp


def spectrum_and_metrics_from_Z(Z: np.ndarray, eps: float = 1e-12) -> Dict[str, Any]:
    Z = np.asarray(Z, dtype=np.float64)
    n = Z.shape[0]
    empty = {
        "eigvals": np.array([], dtype=np.float64),
        "explained_ratio": np.array([], dtype=np.float64),
        "cum_explained_ratio": np.array([], dtype=np.float64),
        "d_pr": np.nan,
        "d_erank": np.nan,
        "k90": np.nan,
        "rank_eff": 0,
    }
    if n < 2:
        return empty

    zc = Z - Z.mean(axis=0, keepdims=True)
    s = np.linalg.svd(zc, full_matrices=False, compute_uv=False)
    eig = np.clip((s**2) / max(n - 1, 1), 0.0, None)
    tot = eig.sum()
    if tot <= 1e-12:
        explained = np.zeros_like(eig)
        cum = np.zeros_like(eig)
        d_pr, d_erank, k90 = 0.0, 0.0, 0
    else:
        explained = eig / tot
        cum = np.cumsum(explained)
        d_pr = (tot * tot) / (np.sum(eig * eig) + eps)
        p = explained[explained > 0]
        d_erank = float(np.exp(-(p * np.log(p)).sum()))
        k90 = int(np.searchsorted(cum, 0.9) + 1)
    return {
        "eigvals": eig,
        "explained_ratio": explained,
        "cum_explained_ratio": cum,
        "d_pr": float(d_pr),
        "d_erank": float(d_erank),
        "k90": int(k90),
        "rank_eff": int(np.sum(eig > 1e-12)),
    }


def existing_layers(model, names: Sequence[str]) -> List[str]:
    ok = []
    for n in names:
        try:
            model.get_layer(n)
            ok.append(n)
        except Exception:
            pass
    return ok


def build_multi_output_model(model, layer_names: Sequence[str], tag: str):
    outs, used = [], []
    for ln in layer_names:
        try:
            t = model.get_layer(ln).output
        except Exception:
            continue
        if len(t.shape) == 4:
            outs.append(t)
            used.append(ln)
    if not outs:
        raise RuntimeError(f"{tag}: no available 4D layers")
    return tf.keras.Model(model.inputs, outs, name=f"multi_out_{tag}"), used


def spatial_flatten(feat_4d: np.ndarray) -> np.ndarray:
    n, h, w, c = feat_4d.shape
    return feat_4d.reshape(n * h * w, c)


def hierarchical_outer_ci(
    seed_to_inner_arr: Dict[Any, Any],
    b_outer: int = 5000,
    seed: int = 123,
    ci_level: float = 0.95,
    metric: str | None = None,
) -> Tuple[float, float, float]:
    """Hierarchical bootstrap CI.

    ``seed_to_inner_arr`` maps seed -> array, OR seed -> dict[metric]->array
    when ``metric`` is provided (legacy notebook signature).
    """
    rng = np.random.default_rng(seed)
    seeds = list(seed_to_inner_arr.keys())
    s = len(seeds)
    vals = np.empty(b_outer, dtype=np.float64)
    for i in range(b_outer):
        sampled_seeds = rng.choice(seeds, size=s, replace=True)
        draws = []
        for ss in sampled_seeds:
            entry = seed_to_inner_arr[ss]
            arr = entry[metric] if metric is not None else entry
            j = rng.integers(0, len(arr))
            draws.append(arr[j])
        vals[i] = np.mean(draws)
    alpha = 1.0 - ci_level
    lo, hi = np.percentile(vals, [100 * (alpha / 2.0), 100 * (1.0 - alpha / 2.0)])
    return float(np.mean(vals)), float(lo), float(hi)


def inner_bootstrap_metrics(A_4d, B_4d, b_inner: int = 200, seed: int = 0) -> Dict[str, Dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    n = A_4d.shape[0]
    out = {
        "MLP": {"d_pr": np.empty(b_inner), "d_erank": np.empty(b_inner), "k90": np.empty(b_inner)},
        "VQC": {"d_pr": np.empty(b_inner), "d_erank": np.empty(b_inner), "k90": np.empty(b_inner)},
        "DELTA": {"d_pr": np.empty(b_inner), "d_erank": np.empty(b_inner), "k90": np.empty(b_inner)},
    }
    for i in range(b_inner):
        idx = rng.integers(0, n, size=n)
        ma = spectrum_and_metrics_from_Z(spatial_flatten(A_4d[idx]))
        mb = spectrum_and_metrics_from_Z(spatial_flatten(B_4d[idx]))
        for m in ["d_pr", "d_erank", "k90"]:
            out["MLP"][m][i] = ma[m]
            out["VQC"][m][i] = mb[m]
            out["DELTA"][m][i] = mb[m] - ma[m]
    return out
