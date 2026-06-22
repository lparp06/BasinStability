"""Oscillator-specific synchronized RHS and Jacobian functions for MSF.

To add a new dynamics family:

1. Add ``<name>_rhs_jax(state, params)``.
2. Add ``<name>_jacobian_jax(state, params)``.
3. Register the pair in ``MSF_DYNAMICS``.
4. Pass ``dynamics='<name>'`` into ``MSFConfig`` or the MSF scan CLI.

The basin simulation code already supports selecting dynamics independently.
This module is the MSF-specific swap point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from network_dynamics.core.jax_config import enable_x64

enable_x64()

import jax.numpy as jnp


@dataclass(frozen=True)
class MSFDynamics:
    rhs: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    jacobian: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


def rossler_rhs_jax(state: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    """Rössler vector field F(s)."""

    x, y, z = state
    a, b, c = params
    return jnp.array(
        [
            -y - z,
            x + a * y,
            b + z * (x - c),
        ],
        dtype=state.dtype,
    )


def rossler_jacobian_jax(state: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    """Analytical Jacobian DF(s) for the Rössler vector field."""

    x, _y, z = state
    a, _b, c = params
    return jnp.array(
        [
            [0.0, -1.0, -1.0],
            [1.0, a, 0.0],
            [z, 0.0, x - c],
        ],
        dtype=state.dtype,
    )


def lorenz_rhs_jax(state: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    """Lorenz vector field F(s), with params=(sigma, beta, rho)."""

    x, y, z = state
    a, b, c = params  # a=sigma, b=beta, c=rho

    return jnp.array(
        [
            a * (y - x),
            x * (c - z) - y,
            x * y - b * z,
        ],
        dtype=state.dtype,
    )


def lorenz_jacobian_jax(state: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    """Analytical Jacobian DF(s) for the Lorenz vector field."""

    x, y, z = state
    a, b, c = params  # a=sigma, b=beta, c=rho
    return jnp.array(
        [
            [-a, a, 0.0],
            [c - z, -1.0, -x],
            [y, x, -b],
        ],
        dtype=state.dtype,
    )


MSF_DYNAMICS = {
    "rossler": MSFDynamics(
        rhs=rossler_rhs_jax,
        jacobian=rossler_jacobian_jax,
    ),
    "lorenz": MSFDynamics(
        rhs=lorenz_rhs_jax,
        jacobian=lorenz_jacobian_jax,
    ),
}

# TODO: Add future oscillator MSF implementations above and register them here.


def normalize_msf_dynamics(dynamics: str) -> str:
    aliases = {
        "rossler": "rossler",
        "rössler": "rossler",
        "roessler": "rossler",
        "lorenz": "lorenz",
    }

    normalized = aliases.get(dynamics.lower())

    if normalized is None or normalized not in MSF_DYNAMICS:
        raise NotImplementedError(
            "MSF dynamics is not implemented for "
            f"{dynamics!r}. Add its RHS and Jacobian in "
            "network_dynamics/core/msf/dynamics.py."
        )

    return normalized


def get_msf_dynamics(dynamics: str) -> MSFDynamics:
    return MSF_DYNAMICS[normalize_msf_dynamics(dynamics)]
