"""Path helpers for notebooks and scripts."""
from __future__ import annotations

import os
from pathlib import Path


def repo_root(start: str | Path | None = None) -> Path:
    """Locate BottleneckVQC repo root (directory containing package + configs)."""
    cand = Path.cwd().resolve() if start is None else Path(start).resolve()
    for p in [cand, *cand.parents]:
        if (
            (p / "functions" / "__init__.py").is_file()
            and (p / "configs").is_dir()
            and (p / "requirements.txt").is_file()
        ):
            return p
    # Fallbacks when cwd is notebooks/ or scripts/
    for fb in (Path(__file__).resolve().parents[2], Path.cwd().resolve().parent):
        if (fb / "functions" / "__init__.py").is_file():
            return fb
    return Path(__file__).resolve().parents[2]


def resolve_data_dir(*candidates: str | Path) -> Path:
    for p in candidates:
        path = Path(p)
        if path.is_dir():
            return path.resolve()
    raise FileNotFoundError(
        "Data directory not found. Download NetCDF from Zenodo "
        "(DOI 10.5281/zenodo.21500592) and unpack to data/extracted_uv/. "
        "See data/README.md or run: python scripts/download_assets.py --zenodo\n"
        f"tried: {candidates}"
    )


def resolve_extracted_uv_dir(root: str | Path | None = None) -> Path:
    root = Path(root) if root is not None else repo_root()
    return resolve_data_dir(root / "data" / "extracted_uv", root / "extracted_uv")


def resolve_ood_w07_120_path(root: str | Path | None = None) -> Path:
    """Path to the 7 m/s, 120° OOD NetCDF under extracted_uv."""
    root = Path(root) if root is not None else repo_root()
    candidates = [
        root / "data" / "extracted_uv" / "extracted_w07_deg120_3d.000.nc",
        root / "extracted_uv" / "extracted_w07_deg120_3d.000.nc",
    ]
    for p in candidates:
        if p.is_file():
            return p.resolve()
    raise FileNotFoundError(
        "OOD file extracted_w07_deg120_3d.000.nc not found under data/extracted_uv/. "
        "Download the NetCDF deposit (DOI 10.5281/zenodo.21500592) and unpack it. "
        f"Tried: {candidates}"
    )


def resolve_weights_path(root_dir: str | Path, stem: str) -> str:
    root_dir = str(root_dir)
    candidates = [
        os.path.join(root_dir, stem),
        os.path.join(root_dir, f"{stem}.h5"),
        os.path.join(root_dir, f"{stem}.weights.h5"),
        os.path.join(root_dir, stem, "best_val_weights.h5"),
        os.path.join(root_dir, stem, "best.weights.h5"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Weights not found for stem={stem}. Tried: {candidates}")
