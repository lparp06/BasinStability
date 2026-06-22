"""Inner-coupling helpers for MSF calculations."""

from network_dynamics.core.jax_config import enable_x64

enable_x64()

import jax.numpy as jnp


def inner_coupling_matrix_jax(
    dimension: int = 3,
    target: int = 0,
    source: int = 0,
) -> jnp.ndarray:
    """Build one-component inner coupling matrix H.

    Paper notation is source+1 -> target+1. Example: target=0, source=0
    means 1->1 coupling.
    """

    H = jnp.zeros((dimension, dimension), dtype=jnp.float64)
    return H.at[target, source].set(1.0)
