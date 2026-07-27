"""BottleneckVQC-UNet models and training CLI."""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import yaml

from . import losses
from .models_classical import build_cond_encoder_mlp, conv_block, dump_cond_embeddings
from .models_vqc import build_vqc_layer
from .data import (
    compute_y_norm_stats,
    cond_vector,
    list_cases,
    load_uv_steady_mean,
    normalize_y,
    denormalize_y,
    parse_float01,
    random_split_indices,
)
from .utils import MISSING_VALUE, set_seeds, silence_tf_warnings

def _flatten_hw(x: tf.Tensor) -> tf.Tensor:
    shape = tf.shape(x)
    b, h, w, c = shape[0], shape[1], shape[2], shape[3]
    return tf.reshape(x, [b * h * w, c])


def _unflatten_hw(args: List[tf.Tensor]) -> tf.Tensor:
    flat, ref = args
    shape = tf.shape(ref)
    b, h, w = shape[0], shape[1], shape[2]
    c = tf.shape(flat)[1]
    return tf.reshape(flat, [b, h, w, c])


def bottleneck_vqc(
    x: tf.Tensor,
    name: str,
    n_qubits: int = 5,
    n_layers: int = 2,
    out_channels: int = 512,
) -> tf.Tensor:
    q_inputs = tf.keras.layers.Conv2D(n_qubits, 1, activation="tanh", name=f"{name}_vqc_in")(x)
    q_inputs = tf.keras.layers.Lambda(lambda t: t * math.pi, name=f"{name}_vqc_in_scale")(q_inputs)
    flat = tf.keras.layers.Lambda(_flatten_hw, name=f"{name}_vqc_flat")(q_inputs)
    qlayer = build_vqc_layer(n_qubits=n_qubits, n_layers=n_layers)
    q_out = qlayer(flat)
    q_out = tf.keras.layers.Activation(tf.keras.activations.linear, name=f"{name}_vqc_out")(q_out)
    q_out = tf.keras.layers.Lambda(_unflatten_hw, name=f"{name}_vqc_unflat")([q_out, q_inputs])
    out = tf.keras.layers.Conv2D(out_channels, 1, padding="same", activation="relu", name=f"{name}_proj")(q_out)
    return out


def build_unet_cond_mlp_bottleneck_vqc(
    input_shape: Tuple[int, int, int] = (200, 200, 1),
    cond_dim: int = 3,
    cond_emb_dim: int = 16,
    n_qubits: int = 5,
    n_layers: int = 2,
) -> tf.keras.Model:
    img_in = tf.keras.Input(shape=input_shape, name="mask_img")
    cond_in = tf.keras.Input(shape=(cond_dim,), name="cond")

    cond_encoder = build_cond_encoder_mlp(cond_dim=cond_dim, cond_emb_dim=cond_emb_dim)
    cond_emb = cond_encoder(cond_in)

    # 200 is not a power-of-two multiple; pad to 208 then crop back to 200
    x0 = tf.keras.layers.ZeroPadding2D(((4, 4), (4, 4)), name="pad_to_208")(img_in)

    c1 = conv_block(x0, cond_emb, 32, "down1")
    p1 = tf.keras.layers.MaxPool2D(2, name="pool1")(c1)

    c2 = conv_block(p1, cond_emb, 64, "down2")
    p2 = tf.keras.layers.MaxPool2D(2, name="pool2")(c2)

    c3 = conv_block(p2, cond_emb, 128, "down3")
    p3 = tf.keras.layers.MaxPool2D(2, name="pool3")(c3)

    c4 = conv_block(p3, cond_emb, 256, "down4")
    p4 = tf.keras.layers.MaxPool2D(2, name="pool4")(c4)

    bn = bottleneck_vqc(p4, "bottleneck", n_qubits=n_qubits, n_layers=n_layers, out_channels=512)

    u4 = tf.keras.layers.Conv2DTranspose(256, 2, strides=2, padding="same", name="up4")(bn)
    u4 = tf.keras.layers.Concatenate(name="cat4")([u4, c4])
    c5 = conv_block(u4, cond_emb, 256, "up_block4")

    u3 = tf.keras.layers.Conv2DTranspose(128, 2, strides=2, padding="same", name="up3")(c5)
    u3 = tf.keras.layers.Concatenate(name="cat3")([u3, c3])
    c6 = conv_block(u3, cond_emb, 128, "up_block3")

    u2 = tf.keras.layers.Conv2DTranspose(64, 2, strides=2, padding="same", name="up2")(c6)
    u2 = tf.keras.layers.Concatenate(name="cat2")([u2, c2])
    c7 = conv_block(u2, cond_emb, 64, "up_block2")

    u1 = tf.keras.layers.Conv2DTranspose(32, 2, strides=2, padding="same", name="up1")(c7)
    u1 = tf.keras.layers.Concatenate(name="cat1")([u1, c1])
    c8 = conv_block(u1, cond_emb, 32, "up_block1")

    out = tf.keras.layers.Conv2D(2, 1, padding="same", activation="linear", name="uv_out")(c8)
    out = tf.keras.layers.Cropping2D(((4, 4), (4, 4)), name="crop_to_200")(out)
    return tf.keras.Model(inputs=[img_in, cond_in], outputs=out, name="UNetFiLM_MLP_BottleneckVQC")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_data_dir(explicit: str | None = None) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
    else:
        env = os.environ.get("BOTTLENECKVQC_DATA_DIR")
        if env:
            p = Path(env).expanduser().resolve()
        else:
            p = _repo_root() / "data" / "extracted_uv"
    if not p.is_dir():
        raise FileNotFoundError(
            f"Data directory not found: {p}\n"
            "Download NetCDF from Zenodo (DOI 10.5281/zenodo.21500592) and unpack to "
            "data/extracted_uv/, or set BOTTLENECKVQC_DATA_DIR. "
            "See data/README.md or run: python scripts/download_assets.py --zenodo"
        )
    return p


def load_config(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main(argv: list[str] | None = None) -> None:
    start_ts = time.time()
    parser = argparse.ArgumentParser(
        description="Train UNet+MLP with VQC bottleneck (random split)."
    )
    parser.add_argument("--config", type=str, default=None, help="YAML config (overrides defaults).")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory of extracted_*.nc files.")
    parser.add_argument("--height_m", type=float, default=15.0)
    parser.add_argument("--last_k", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--cond_emb_dim", type=int, default=16)
    parser.add_argument("--n_qubits", type=int, default=5)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dump_cond_emb", action="store_true")
    parser.add_argument("--sweep_step_deg", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7, help="Seed for split AND weight init.")
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use_log1p", action="store_true", help="Apply signed log1p to U/V before norm.")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--no_silence_tf_warnings", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    for k, v in cfg.items():
        if hasattr(args, k) and k != "config":
            setattr(args, k, v)

    set_seeds(args.seed)
    silence_tf_warnings(enabled=not args.no_silence_tf_warnings)

    train_frac = parse_float01(args.train_frac, "train_frac")
    val_frac = parse_float01(args.val_frac, "val_frac")
    if train_frac + val_frac >= 1.0:
        raise ValueError(f"train_frac+val_frac must be < 1. Got {train_frac + val_frac}")

    extracted_uv_dir = str(_resolve_data_dir(args.data_dir))
    cases = list_cases(extracted_uv_dir)

    def signed_log1p(x: np.ndarray) -> np.ndarray:
        return np.sign(x) * np.log1p(np.abs(x))

    xs, cs, ys, meta = [], [], [], []
    for c in cases:
        m_building, u_mean, v_mean = load_uv_steady_mean(
            c.path, height_m=args.height_m, last_k=args.last_k
        )
        x = m_building[..., None].astype(np.float32)
        y = np.stack([u_mean, v_mean], axis=-1).astype(np.float32)
        if args.use_log1p:
            m = y != MISSING_VALUE
            y = y.copy()
            y[m] = signed_log1p(y[m])
        xs.append(x)
        cs.append(cond_vector(c.speed, c.angle_deg))
        ys.append(y)
        meta.append(
            {
                "file": os.path.basename(c.path),
                "speed": c.speed,
                "d_code": c.d_code,
                "angle_deg": c.angle_deg,
            }
        )

    X = np.stack(xs, axis=0)
    C = np.stack(cs, axis=0)
    Y = np.stack(ys, axis=0)

    split_idx = random_split_indices(
        n=len(cases), train_frac=train_frac, val_frac=val_frac, seed=args.seed
    )
    tr, va, te = split_idx["train"], split_idx["val"], split_idx["test"]

    c_scaler = StandardScaler()
    C_tr = c_scaler.fit_transform(C[tr])
    C_va = c_scaler.transform(C[va])
    C_te = c_scaler.transform(C[te])

    y_mean, y_std = compute_y_norm_stats(Y[tr])
    Y_tr = normalize_y(Y[tr], y_mean, y_std)
    Y_va = normalize_y(Y[va], y_mean, y_std)
    Y_te = normalize_y(Y[te], y_mean, y_std)
    X_tr, X_va, X_te = X[tr], X[va], X[te]

    model = build_unet_cond_mlp_bottleneck_vqc(
        input_shape=X_tr.shape[1:],
        cond_dim=C_tr.shape[-1],
        cond_emb_dim=args.cond_emb_dim,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss=losses.masked_mse_with_grad,
        metrics=[losses.masked_mae_metric, losses.masked_rmse_metric, losses.MaskedR2()],
    )

    root = _repo_root()
    run_dir = Path(args.out_dir) if args.out_dir else root / "artifacts" / f"bottleneckvqc_{int(start_ts)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "height_m": args.height_m,
                "last_k": args.last_k,
                "missing_value": MISSING_VALUE,
                "seed": args.seed,
                "train_frac": train_frac,
                "val_frac": val_frac,
                "split_idx": split_idx,
                "meta": meta,
                "y_mean": y_mean.tolist(),
                "y_std": y_std.tolist(),
                "cond_emb_dim": args.cond_emb_dim,
                "n_qubits": args.n_qubits,
                "n_layers": args.n_layers,
                "use_log1p": bool(args.use_log1p),
                "data_dir": extracted_uv_dir,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10, min_lr=1e-5, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(run_dir / "best.weights.h5"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
        ),
    ]

    history = model.fit(
        x={"mask_img": X_tr, "cond": C_tr},
        y=Y_tr,
        validation_data=({"mask_img": X_va, "cond": C_va}, Y_va),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    with open(run_dir / "history.json", "w", encoding="utf-8") as f:
        hist = {k: [float(x) for x in v] for k, v in history.history.items()}
        json.dump(hist, f, ensure_ascii=False, indent=2)

    if args.dump_cond_emb:
        encoder = tf.keras.Model(
            inputs=model.get_layer("CondEncoderMLP").input,
            outputs=model.get_layer("CondEncoderMLP").output,
            name="CondEncoderMLP_export",
        )
        dump_cond_embeddings(
            run_dir=str(run_dir),
            encoder=encoder,
            c_scaler=c_scaler,
            meta=meta,
            split_idx=split_idx,
            speeds=sorted({m["speed"] for m in meta}),
            sweep_step_deg=args.sweep_step_deg,
        )

    pred_te = model.predict({"mask_img": X_te, "cond": C_te}, verbose=0).astype(np.float32)
    pred_te_denorm = denormalize_y(pred_te, y_mean, y_std)
    y_te_denorm = denormalize_y(Y_te, y_mean, y_std)
    metrics_te = losses.evaluate_numpy(y_te_denorm, pred_te_denorm)

    with open(run_dir / "metrics_test.json", "w", encoding="utf-8") as f:
        json.dump(metrics_te, f, ensure_ascii=False, indent=2)

    print("Test metrics:", metrics_te)
    print(f"Run dir: {run_dir}")
    print(f"Total Runtime: {time.time() - start_ts:.2f}s")


if __name__ == "__main__":
    main()
