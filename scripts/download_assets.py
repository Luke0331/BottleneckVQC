#!/usr/bin/env python3
"""Print Zenodo URLs for NetCDF data and checkpoints.

Examples
--------
  python scripts/download_assets.py --zenodo

  python scripts/download_assets.py \\
    --zenodo-data-doi 10.5281/zenodo.21500592 \\
    --zenodo-weights-doi 10.5281/zenodo.21603668
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_DOI = "10.5281/zenodo.21500592"
DEFAULT_WEIGHTS_DOI = "10.5281/zenodo.21603668"


def _zenodo_record_url(doi: str) -> str:
    record = doi.split("zenodo.")[-1]
    return f"https://zenodo.org/api/records/{record}"


def from_zenodo(data_doi: str | None, weights_doi: str | None) -> None:
    """Point users at Zenodo records; files must be unzipped into data/ and checkpoints/."""
    pairs = []
    if data_doi:
        pairs.append(
            (
                "data (NetCDF → data/extracted_uv/)",
                data_doi,
                ROOT / "data" / "extracted_uv",
            )
        )
    if weights_doi:
        pairs.append(("weights (checkpoints/)", weights_doi, ROOT / "checkpoints"))
    if not pairs:
        print("Provide --zenodo-data-doi and/or --zenodo-weights-doi", file=sys.stderr)
        sys.exit(2)

    print(
        "Download each Zenodo record below, then unpack so paths match this repository:\n"
        f"  NetCDF files → {ROOT / 'data' / 'extracted_uv'}\n"
        f"  Checkpoints  → {ROOT / 'checkpoints'}\n"
    )
    ok = True
    for label, doi, dest in pairs:
        url = _zenodo_record_url(doi)
        landing = f"https://doi.org/{doi}"
        print(f"[{label}]")
        print(f"  DOI:     {doi}")
        print(f"  Landing: {landing}")
        print(f"  API:     {url}")
        print(f"  Unpack:  {dest}")
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                print(f"  Record HTTP status: {resp.status}")
        except Exception as exc:  # noqa: BLE001
            print(f"  Could not query Zenodo ({exc})", file=sys.stderr)
            ok = False
        print()
    if not ok:
        sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--zenodo-doi",
        type=str,
        default=None,
        help="Deprecated single DOI; prefer --zenodo-data-doi / --zenodo-weights-doi",
    )
    p.add_argument(
        "--zenodo-data-doi",
        type=str,
        default=None,
        help=f"Zenodo DOI for NetCDF data (default with --zenodo: {DEFAULT_DATA_DOI})",
    )
    p.add_argument(
        "--zenodo-weights-doi",
        type=str,
        default=None,
        help=f"Zenodo DOI for checkpoints (default with --zenodo: {DEFAULT_WEIGHTS_DOI})",
    )
    p.add_argument(
        "--zenodo",
        action="store_true",
        help="Use default data + weights Zenodo DOIs",
    )
    args = p.parse_args()

    if args.zenodo or args.zenodo_doi or args.zenodo_data_doi or args.zenodo_weights_doi:
        data_doi = args.zenodo_data_doi
        weights_doi = args.zenodo_weights_doi
        if args.zenodo and data_doi is None and weights_doi is None and args.zenodo_doi is None:
            data_doi = DEFAULT_DATA_DOI
            weights_doi = DEFAULT_WEIGHTS_DOI
        if args.zenodo_doi and data_doi is None and weights_doi is None:
            print(
                f"[warn] --zenodo-doi is deprecated; showing record for {args.zenodo_doi} only.\n"
                "       Prefer --zenodo-data-doi / --zenodo-weights-doi or --zenodo."
            )
            from_zenodo(args.zenodo_doi, None)
        else:
            from_zenodo(data_doi, weights_doi)
    else:
        # Default for reviewers: show both Zenodo records
        from_zenodo(DEFAULT_DATA_DOI, DEFAULT_WEIGHTS_DOI)


if __name__ == "__main__":
    main()
