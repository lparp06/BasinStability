"""Shared JAX runtime configuration."""

from jax import config


def enable_x64():
    """Enable float64 support before JAX arrays or compiled functions are built."""

    config.update("jax_enable_x64", True)
