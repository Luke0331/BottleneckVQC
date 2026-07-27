"""Data loading and preprocessing for PALM UV NetCDF cases."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import xarray as xr

from .utils import MISSING_VALUE

@dataclass(frozen=True)
class Case:
    path: str
    speed: int
    d_code: int  # d03..d10 (or negative angle for deg-named files)
    angle_deg: float  # 0=north, clockwise


def parse_case_from_filename(path: str) -> Case:
    """
    Supported filenames:
      - extracted_w04_d03_3d.000.nc          (direction code)
      - extracted_w07_deg120_3d.000.nc       (explicit angle in degrees)
    """
    base = os.path.basename(path)
    parts = base.split("_")
    w = next(p for p in parts if p.startswith("w") and p[1:].isdigit())
    speed = int(w[1:])

    deg_tok = next(
        (p for p in parts if p.startswith("deg") and p[3:].replace(".", "", 1).isdigit()),
        None,
    )
    if deg_tok is not None:
        angle_deg = float(deg_tok[3:])
        # Synthetic code: negative angle so it never collides with d03..d28
        d_code = -int(round(angle_deg))
        return Case(path=path, speed=speed, d_code=d_code, angle_deg=angle_deg)

    d = next(p for p in parts if p.startswith("d") and p[1:].isdigit())
    d_code = int(d[1:])
    # per user: d03=north, clockwise; d05=east
    # => d03:0°, d04:45° ... d10:315°
    angle_deg = ((d_code - 3) % 8) * 45.0
    return Case(path=path, speed=speed, d_code=d_code, angle_deg=angle_deg)


def list_cases(extracted_uv_dir: str, *, include_deg_named: bool = False) -> List[Case]:
    """List NetCDF cases under ``extracted_uv_dir``.

    By default, skip ``*_deg*`` files (e.g. OOD ``extracted_w07_deg120_...``)
    so the main train set stays the standard speed/direction grid.
    """
    cases: List[Case] = []
    for fn in sorted(os.listdir(extracted_uv_dir)):
        if not fn.endswith(".nc"):
            continue
        if (not include_deg_named) and ("_deg" in fn):
            continue
        cases.append(parse_case_from_filename(os.path.join(extracted_uv_dir, fn)))
    if len(cases) == 0:
        raise FileNotFoundError(f"No .nc files found in: {extracted_uv_dir}")
    return cases


def build_splits(
    cases: List[Case],
    train_dirs: List[int],
    test_dirs: List[int],
    val_speed: int,
    val_dirs: List[int] | None = None,
) -> Dict[str, List[Case]]:
    train: List[Case] = []
    val: List[Case] = []
    test: List[Case] = []

    for c in cases:
        if c.d_code in test_dirs:
            test.append(c)
        elif val_dirs is not None and c.d_code in val_dirs:
            val.append(c)
        elif c.d_code in train_dirs:
            if val_dirs is None and c.speed == val_speed:
                val.append(c)
            else:
                train.append(c)
        else:
            raise ValueError(f"Unknown d_code={c.d_code}， not in train/test lists: {c.path}")

    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise ValueError(
            f"Empty split: train={len(train)}, val={len(val)}, test={len(test)}. "
            f"Check train_dirs/test_dirs/val_speed."
        )
    return {"train": train, "val": val, "test": test}


def cond_vector(speed: int, angle_deg: float) -> np.ndarray:
    theta = math.radians(angle_deg)
    return np.array([float(speed), math.sin(theta), math.cos(theta)], dtype=np.float32)


def _nanmean_last_k(arr: np.ndarray, k: int) -> np.ndarray:
    if arr.shape[0] < k:
        raise ValueError(f"time length ({arr.shape[0]}) < k({k})")
    a = arr[-k:].astype(np.float32)
    a = np.where(a == MISSING_VALUE, np.nan, a)
    return np.nanmean(a, axis=0)


def load_uv_steady_mean(
    path: str,
    height_m: float = 15.0,
    last_k: int = 12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      - mask_building: (H,W)  1=building, 0=fluid
      - u_mean: (H,W) with building= MISSING_VALUE
      - v_mean: (H,W) with building= MISSING_VALUE

    NOTE:
      PALM fields are on a staggered grid: u(time,zu_3d,y,xu), v(time,zu_3d,yv,x)
      For the baseline we treat both as aligned 200x200 images,
      without regridding (proper centering can be added later).
    """
    ds = xr.open_dataset(path)
    if "u" not in ds.data_vars or "v" not in ds.data_vars:
        raise KeyError(f"Missing u/v variables: {path}，vars={list(ds.data_vars)}")

    u = ds["u"].sel(zu_3d=height_m, method="nearest")  # (time,y,xu)
    v = ds["v"].sel(zu_3d=height_m, method="nearest")  # (time,yv,x)
    u_np = np.asarray(u.values)
    v_np = np.asarray(v.values)

    u_mean = _nanmean_last_k(u_np, k=last_k)
    v_mean = _nanmean_last_k(v_np, k=last_k)

    # building mask: NaN in either component marks a building pixel
    mask_building = np.isnan(u_mean) | np.isnan(v_mean)
    u_mean = np.where(mask_building, MISSING_VALUE, u_mean).astype(np.float32)
    v_mean = np.where(mask_building, MISSING_VALUE, v_mean).astype(np.float32)
    return mask_building.astype(np.float32), u_mean, v_mean


def compute_y_norm_stats(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    y: (N,H,W,2) with building=MISSING_VALUE
    returns per-channel (mean,std), ignoring missing.
    """
    means = []
    stds = []
    for ch in range(2):
        a = y[..., ch]
        a = a[a != MISSING_VALUE]
        mu = float(a.mean())
        sigma = float(a.std() + 1e-6)
        means.append(mu)
        stds.append(sigma)
    return np.array(means, dtype=np.float32), np.array(stds, dtype=np.float32)


def normalize_y(y: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    y2 = y.copy().astype(np.float32)
    for ch in range(2):
        m = y2[..., ch] != MISSING_VALUE
        y2[..., ch][m] = (y2[..., ch][m] - mean[ch]) / std[ch]
    return y2


def denormalize_y(y: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    y2 = y.copy().astype(np.float32)
    for ch in range(2):
        m = y2[..., ch] != MISSING_VALUE
        y2[..., ch][m] = y2[..., ch][m] * std[ch] + mean[ch]
    return y2


def parse_float01(x: float, name: str) -> float:
    if not (0.0 < x < 1.0):
        raise ValueError(f"{name} must be in (0,1), got {x}")
    return float(x)


def random_split_indices(n: int, train_frac: float, val_frac: float, seed: int) -> Dict[str, List[int]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))

    n_train = max(1, min(n_train, n - 2))
    n_val = max(1, min(n_val, n - n_train - 1))
    n_test = n - n_train - n_val
    if n_test <= 0:
        n_test = 1
        if n_val > 1:
            n_val -= 1
        else:
            n_train -= 1

    tr = idx[:n_train].tolist()
    va = idx[n_train : n_train + n_val].tolist()
    te = idx[n_train + n_val :].tolist()
    return {"train": tr, "val": va, "test": te}



# Backwards-compatible aliases used by notebooks/scripts.
_parse_float01 = parse_float01
_random_split_indices = random_split_indices
