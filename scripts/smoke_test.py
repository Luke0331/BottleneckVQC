#!/usr/bin/env python3
"""Smoke test: load a few NetCDF cases and run 1 training epoch with tiny VQC."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def main() -> int:
    from functions.data import list_cases
    from functions.train import _resolve_data_dir, main as train_main

    try:
        data_dir = _resolve_data_dir(None)
    except FileNotFoundError as exc:
        print(exc)
        print(
            "Hint: download NetCDF from Zenodo (DOI 10.5281/zenodo.21500592) "
            "into data/extracted_uv/. See: python scripts/download_assets.py --zenodo"
        )
        return 2

    cases = list_cases(str(data_dir))
    print(f"Found {len(cases)} cases under {data_dir}")
    if len(cases) < 3:
        print("Need at least 3 NetCDF cases for a train/val/test split.")
        return 2

    train_main(
        [
            "--config",
            str(ROOT / "configs" / "smoke.yaml"),
            "--data_dir",
            str(data_dir),
            "--out_dir",
            str(ROOT / "artifacts" / "smoke"),
        ]
    )
    print("Smoke test finished OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
