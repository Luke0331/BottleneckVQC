# BottleneckVQC

Hybrid **quantum–classical** UNet for conditional urban wind field reconstruction.
A PennyLane variational quantum circuit (VQC) replaces the UNet bottleneck. Inflow
conditions (wind speed and direction) are encoded and injected via FiLM into every
convolutional block in the encoder and decoder.
In the paper, the conditional quantum-bottleneck model is denoted as **C-QB-UNet**, 
and the classical conditional UNet baseline is called **C-UNet**.

This repository is the code availability package for *Communications AI & Computing*.

## Quick start

1. Clone this repository and install dependencies.
2. Download **data** and **checkpoints** from Zenodo (links below).
3. Unpack them into `data/` and `checkpoints/` at the repository root.
4. Open the notebooks under `notebooks/` (or run the training CLI).

```bash
git clone https://github.com/Luke0331/BottleneckVQC.git
cd BottleneckVQC
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Download assets

| Asset | DOI | Unpack to |
|-------|-----|-----------|
| NetCDF data (`extracted_uv/`) | [10.5281/zenodo.21500592](https://doi.org/10.5281/zenodo.21500592) | `data/extracted_uv/` |
| Model checkpoints | [10.5281/zenodo.21603668](https://doi.org/10.5281/zenodo.21603668) | `checkpoints/` |

Expected layout after unpacking:

```text
BottleneckVQC/
├── data/
│   └── extracted_uv/          # all NetCDF cases used by the notebooks
│       ├── extracted_w04_d03_3d.000.nc
│       ├── ...
│       └── extracted_w07_deg120_3d.000.nc   # OOD case (7 m/s, 120°)
└── checkpoints/
    ├── bottleneckvqc_unet_log1p/
    ├── mlp_unet_log1p/
    └── hier_boot_block_1000epochs/          # multi-seed ED / spectral analyses
```

Print Zenodo record URLs:

```bash
python scripts/download_assets.py --zenodo
```

See [data/README.md](data/README.md) for file naming and details.

## Requirements

- Python 3.10+
- TensorFlow 2.13.1, PennyLane 0.42.3 (versions used in the paper experiments)

GPU (PennyLane Lightning / CUDA) is optional. CPU uses `default.qubit`.

## Repository layout

```text
BottleneckVQC/
├── functions/          # Python package (data, models, train, nb_helpers)
├── configs/            # YAML experiment configs
├── notebooks/          # Reproduction notebooks
├── scripts/            # download_assets, smoke_test
├── data/               # Place Zenodo NetCDF here
├── checkpoints/        # Place Zenodo weights here
├── requirements.txt
└── environment.yml
```

## Reproduce results

**Main config:** `configs/main_log1p_seed7.yaml`  
(`seed=7`, `train_frac=0.8`, `val_frac=0.1`, `n_qubits=5`, `n_layers=2`)

```bash
# Optional: train from scratch
python -m functions.train --config configs/main_log1p_seed7.yaml

# Or evaluate / plot with pretrained weights
jupyter notebook notebooks/01_main.ipynb
```

Smoke test (1 epoch, small VQC):

```bash
python scripts/smoke_test.py
```

### Notebooks

| Notebook | Role |
|----------|------|
| `01_main.ipynb` | Main IID experiment (C-QB-UNet vs C-UNet) |
| `02_ood_w07.ipynb` | OOD extrapolation (7 m/s, 120°) |
| `02_ood_w12.ipynb` | OOD extrapolation (12 m/s) |
| `03_spectrum.ipynb` | Spectral analysis |
| `04_dimension.ipynb` | Effective-dimension / channel-wise analysis |

Shared helpers live in `functions/nb_helpers/`. Notebook outputs are cleared.

## Code / data availability

- **Code:** this repository (MIT).
- **Data:** [10.5281/zenodo.21500592](https://doi.org/10.5281/zenodo.21500592)
- **Weights:** [10.5281/zenodo.21603668](https://doi.org/10.5281/zenodo.21603668)

## Citation

See `CITATION.cff`.
