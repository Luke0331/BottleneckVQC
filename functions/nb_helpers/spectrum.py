"""Spectral analysis helpers for wind-field predictions."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def component_energy_spectrum(
    uv: np.ndarray,
    valid_mask: np.ndarray,
    comp: int = 0,
    dx: float = 1.0,
    dy: float = 1.0,
    n_bins: int = 80,
) -> Tuple[np.ndarray, np.ndarray]:
    uv = np.asarray(uv, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    n, h, w, c = uv.shape
    assert c == 2

    kx = 2.0 * np.pi * np.fft.fftfreq(w, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(h, d=dy)
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k_mag = np.sqrt(kx_grid**2 + ky_grid**2)

    k_nonzero = k_mag[k_mag > 0]
    edges = np.linspace(k_nonzero.min(), k_nonzero.max(), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dk = np.diff(edges)

    acc = np.zeros(n_bins, dtype=np.float64)
    used = 0
    for i in range(n):
        m = valid_mask[i]
        if not np.any(m):
            continue
        x = uv[i, ..., comp]
        x0 = np.where(m, x - x[m].mean(), 0.0)
        xh = np.fft.fft2(x0)
        e_mode = 0.5 * (np.abs(xh) ** 2) / (h * w) ** 2
        idx = np.digitize(k_mag.ravel(), edges) - 1
        shell_sum = np.zeros(n_bins, dtype=np.float64)
        ok = (idx >= 0) & (idx < n_bins)
        np.add.at(shell_sum, idx[ok], e_mode.ravel()[ok])
        acc += shell_sum / np.maximum(dk, 1e-12)
        used += 1
    return centers, acc / max(used, 1)


def _band_mask(k: np.ndarray, kmin: float, kmax: float) -> np.ndarray:
    return (k >= kmin) & (k < kmax)


def band_energy(k: np.ndarray, e: np.ndarray, kmin: float, kmax: float) -> float:
    m = _band_mask(k, kmin, kmax)
    if np.sum(m) < 2:
        return float("nan")
    return float(np.trapz(e[m], k[m]))


def band_energy_ratios(
    k: np.ndarray,
    e: np.ndarray,
    band_defs: Sequence[Tuple[str, float, float]],
    eps: float = 1e-20,
) -> Dict[str, float]:
    total = float(np.trapz(e, k))
    out = {"E_total": total}
    for name, kmin, kmax in band_defs:
        eb = band_energy(k, e, kmin, kmax)
        out[f"ratio_{name}"] = eb / (total + eps)
        out[f"E_{name}"] = eb
    return out


def make_band_table(
    k: np.ndarray,
    e_gt: np.ndarray,
    e_mlp: np.ndarray,
    e_vqc: np.ndarray,
    band_defs: Sequence[Tuple[str, float, float]],
    region_name: str = "region",
) -> pd.DataFrame:
    rows = []
    for model_name, ee in [("GT", e_gt), ("MLP", e_mlp), ("VQC", e_vqc)]:
        rr = band_energy_ratios(k, ee, band_defs)
        row = {"region": region_name, "model": model_name, "E_total": rr["E_total"]}
        for name, _, _ in band_defs:
            row[f"ratio_{name}"] = rr[f"ratio_{name}"]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_spectral_error(
    k: np.ndarray,
    e_gt: np.ndarray,
    e_mlp: np.ndarray,
    e_vqc: np.ndarray,
    title: str = "Spectral Error",
    eps: float = 1e-20,
) -> None:
    ratio_mlp = e_mlp / (e_gt + eps)
    ratio_vqc = e_vqc / (e_gt + eps)
    logerr_mlp = np.log10(ratio_mlp + eps)
    logerr_vqc = np.log10(ratio_vqc + eps)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.2))
    axes[0].semilogx(k, ratio_mlp, label="C-UNet / Truth", lw=2)
    axes[0].semilogx(k, ratio_vqc, label="C-QB-UNet / Truth", lw=2)
    axes[0].axhline(1.0, color="k", ls="--", lw=1)
    axes[0].set_xlabel("Wavenumber k")
    axes[0].set_ylabel("Spectrum ratio")
    axes[0].set_title(f"Total Spectrum Ratio ({title})")
    axes[0].legend(frameon=False)

    axes[1].semilogx(k, logerr_mlp, label="C-UNet / Truth", lw=2)
    axes[1].semilogx(k, logerr_vqc, label="C-QB-UNet / Truth", lw=2)
    axes[1].axhline(0.0, color="k", ls="--", lw=1)
    axes[1].set_xlabel("Wavenumber k")
    axes[1].set_ylabel("Log error")
    axes[1].set_title(f"Total Spectrum Logarithmic Error ({title})")
    axes[1].legend(frameon=False)
    plt.tight_layout()
    plt.show()
