"""Classical FiLM-UNet baselines and helpers."""
from __future__ import annotations

import csv
import json
import math
import os
from typing import Dict, List

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from .utils import MISSING_VALUE

def build_cond_encoder_mlp(cond_dim: int = 3, cond_emb_dim: int = 16) -> tf.keras.Model:
    """
    Condition encoder: map (speed, sinθ, cosθ) to cond_emb for FiLM.
    Exposed separately to export cond_emb and analyze angular smoothness.
    """
    cond_in = tf.keras.Input(shape=(cond_dim,), name="cond")
    x = tf.keras.layers.Dense(32, activation="relu", name="cond_d1")(cond_in)
    x = tf.keras.layers.Dense(cond_emb_dim, activation="relu", name="cond_d2")(x)
    return tf.keras.Model(inputs=cond_in, outputs=x, name="CondEncoderMLP")


def film(x: tf.Tensor, cond_emb: tf.Tensor, channels: int, name: str) -> tf.Tensor:
    gamma_beta = tf.keras.layers.Dense(2 * channels, name=f"{name}_gb")(cond_emb)
    gamma, beta = tf.split(gamma_beta, num_or_size_splits=2, axis=-1)
    gamma = tf.keras.layers.Reshape((1, 1, channels), name=f"{name}_gamma_r")(gamma)
    beta = tf.keras.layers.Reshape((1, 1, channels), name=f"{name}_beta_r")(beta)
    return tf.keras.layers.Add(name=f"{name}_add")(
        [tf.keras.layers.Multiply(name=f"{name}_mul")([x, 1.0 + gamma]), beta]
    )


def conv_block(
    x: tf.Tensor,
    cond_emb: tf.Tensor,
    channels: int,
    name: str,
) -> tf.Tensor:
    x = tf.keras.layers.Conv2D(channels, 3, padding="same", activation="relu", name=f"{name}_c1")(x)
    x = film(x, cond_emb, channels, name=f"{name}_film1")
    x = tf.keras.layers.Conv2D(channels, 3, padding="same", activation="relu", name=f"{name}_c2")(x)
    x = film(x, cond_emb, channels, name=f"{name}_film2")
    return x


def build_unet_cond(
    input_shape: Tuple[int, int, int] = (200, 200, 1),
    cond_dim: int = 3,
) -> tf.keras.Model:
    img_in = tf.keras.Input(shape=input_shape, name="mask_img")
    cond_in = tf.keras.Input(shape=(cond_dim,), name="cond")  # normalized condition vector
    cond_encoder = build_cond_encoder_mlp(cond_dim=cond_dim, cond_emb_dim=16)
    cond_emb = cond_encoder(cond_in)

    # 200 is not a power-of-two multiple; naive UNet up/downsampling breaks skip sizes.
    # Pad input to 208=16*13, then crop back to 200 to keep concatenations valid.
    x0 = tf.keras.layers.ZeroPadding2D(((4, 4), (4, 4)), name="pad_to_208")(img_in)

    c1 = conv_block(x0, cond_emb, 32, "down1")
    p1 = tf.keras.layers.MaxPool2D(2, name="pool1")(c1)

    c2 = conv_block(p1, cond_emb, 64, "down2")
    p2 = tf.keras.layers.MaxPool2D(2, name="pool2")(c2)

    c3 = conv_block(p2, cond_emb, 128, "down3")
    p3 = tf.keras.layers.MaxPool2D(2, name="pool3")(c3)

    c4 = conv_block(p3, cond_emb, 256, "down4")
    p4 = tf.keras.layers.MaxPool2D(2, name="pool4")(c4)

    bn = conv_block(p4, cond_emb, 512, "bottleneck")

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
    return tf.keras.Model(inputs=[img_in, cond_in], outputs=out, name="UNetFiLM")
