"""Shared JAX runtime configuration."""

import os

# Use CPU for normal project runs. Explicit launch configurations, such as
# run_msf_gpu.sh setting JAX_PLATFORMS=cuda, take precedence over this default.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from jax import config


def enable_x64():
    """Enable float64 support before JAX arrays or compiled functions are built."""

    config.update("jax_enable_x64", True)
