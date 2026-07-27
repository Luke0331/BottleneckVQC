# Data layout for BottleneckVQC

Processed PALM LES horizontal wind fields (`u`, `v`) as NetCDF files are **not**
stored in git. Download them from Zenodo and place them under this directory.

## Zenodo

| Asset | DOI | Link |
|-------|-----|------|
| NetCDF data (`extracted_uv/`) | `10.5281/zenodo.21500592` | https://doi.org/10.5281/zenodo.21500592 |
| Pretrained checkpoints | `10.5281/zenodo.21603668` | https://doi.org/10.5281/zenodo.21603668 |

```bash
# From the repository root: print landing / API URLs
python scripts/download_assets.py --zenodo
```

Unpack the data archive so that NetCDF files live at:

```text
data/extracted_uv/*.nc
```

Unpack the checkpoints archive so that weight folders live at:

```text
checkpoints/bottleneckvqc_unet_log1p/
checkpoints/mlp_unet_log1p/
checkpoints/hier_boot_block_1000epochs/   # used by notebooks/04_dimension.ipynb
```

Optional override for the NetCDF directory:

```bash
export BOTTLENECKVQC_DATA_DIR=/path/to/extracted_uv
```

## Expected data layout

The Zenodo data deposit provides a single folder `extracted_uv/` containing all
cases used by the notebooks (main grid + OOD):

```text
data/
└── extracted_uv/
    ├── extracted_w04_d03_3d.000.nc
    ├── ...
    └── extracted_w07_deg120_3d.000.nc   # OOD: 7 m/s, 120°
```

Filename patterns:

```text
extracted_w{speed:02d}_d{dir:02d}_3d.000.nc      # main grid (d03–d10)
extracted_w{speed:02d}_deg{angle}_3d.000.nc      # explicit angle (OOD)
```

Each file contains staggered-grid `u`/`v` with time and height (`zu_3d` ≈ 5/15/25 m).
Building cells are marked with missing values (`-999` after loading).

By default, `list_cases()` loads the main speed/direction grid and skips
`*_deg*` files; the OOD notebook loads `extracted_w07_deg120_3d.000.nc`
explicitly.

## Checkpoints

Paper weights (from the checkpoints Zenodo record, not git):

- `checkpoints/bottleneckvqc_unet_log1p/best_val_weights*.h5`
- `checkpoints/mlp_unet_log1p/best_val_weights*.h5`
- `checkpoints/hier_boot_block_1000epochs/` (multi-seed ED / spectral analyses)
