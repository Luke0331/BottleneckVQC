"""Compatibility shim mirroring `run_unet_mlp_bottleneckvqc_uv15_randomsplit`."""
from functions.train import *  # noqa: F401,F403
from functions.data import random_split_indices as _random_split_indices  # noqa: F401
from functions.data import parse_float01 as _parse_float01  # noqa: F401
