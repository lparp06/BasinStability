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


def chen_rhs_jax(state: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    """Chen vector field F(s), with params=(a, beta, c)."""

    x, y, z = state
    a, b, c = params  # a=35, b=beta=8/3, c=25

    return jnp.array(
        [
            a * (y - x),
            (c - a - z) * x + c * y,
            x * y - b * z,
        ],
        dtype=state.dtype,
    )


def chen_jacobian_jax(state: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    """Analytical Jacobian DF(s) for the Chen vector field."""

    x, y, z = state
    a, b, c = params  # a=35, b=beta=8/3, c=25
    return jnp.array(
        [
            [-a, a, 0.0],
            [c - a - z, c, -x],
            [y, x, -b],
        ],
        dtype=state.dtype,
    )


def chua_rhs_jax(state: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    """Chua's circuit vector field F(s).

    params = (alpha, beta, gamma, a_nl, b_nl)
    """

    x, y, z = state
    alpha, beta, gamma, a_nl, b_nl = params

    f = jnp.where(
        jnp.abs(x) <= 1.0,
        -a_nl * x,
        jnp.where(x > 1.0, -b_nl * x - a_nl + b_nl, -b_nl * x + a_nl - b_nl),
    )

    return jnp.array(
        [
            alpha * (y - x + f),
            x - y + z,
            -beta * y - gamma * z,
        ],
        dtype=state.dtype,
    )


def chua_jacobian_jax(state: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    """Analytical Jacobian DF(s) for Chua's circuit.

    The piecewise f'(x) = -b_nl for |x|>1, -a_nl for |x|<1.
    Row 0: d/dx[alpha*(y - x + f(x))] = alpha*(-1 + f'(x)) = -alpha*(1 + f'_coeff)
    """

    x, _y, _z = state
    alpha, beta, gamma, a_nl, b_nl = params

    f_prime_coeff = jnp.where(jnp.abs(x) > 1.0, b_nl, a_nl)
    df_dx = -alpha * (1.0 + f_prime_coeff)

    return jnp.array(
        [
            [df_dx,   alpha,  0.0   ],
            [1.0,    -1.0,    1.0   ],
            [0.0,    -beta,  -gamma ],
        ],
        dtype=state.dtype,
    )


def hr_rhs_jax(state: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    """Hindmarsh-Rose neuron vector field F(s), with params=(I, r, s).

    Eq. (16) in PhysRevE.80.036204.  Structural constants 3, 5, 1.6 are fixed.
    """

    x, y, z = state
    I, r, s = params
    return jnp.array(
        [
            y + 3.0 * x**2 - x**3 - z + I,
            1.0 - 5.0 * x**2 - y,
            -r * z + r * s * (x + 1.6),
        ],
        dtype=state.dtype,
    )


def hr_jacobian_jax(state: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    """Analytical Jacobian DF(s) for the Hindmarsh-Rose vector field.

    Eq. (17) in PhysRevE.80.036204.
    """

    x, _y, _z = state
    _I, r, s = params
    return jnp.array(
        [
            [6.0 * x - 3.0 * x**2,  1.0, -1.0],
            [-10.0 * x,             -1.0,  0.0],
            [r * s,                  0.0,  -r ],
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
    "chen": MSFDynamics(
        rhs=chen_rhs_jax,
        jacobian=chen_jacobian_jax,
    ),
    "chua": MSFDynamics(
        rhs=chua_rhs_jax,
        jacobian=chua_jacobian_jax,
    ),
    "hr": MSFDynamics(
        rhs=hr_rhs_jax,
        jacobian=hr_jacobian_jax,
    ),
}


def normalize_msf_dynamics(dynamics: str) -> str:
    aliases = {
        "rossler": "rossler",
        "rössler": "rossler",
        "roessler": "rossler",
        "lorenz": "lorenz",
        "chen": "chen",
        "chua": "chua",
        "hr": "hr",
        "hindmarsh_rose": "hr",
        "hindmarsh-rose": "hr",
        "hindmarshrose": "hr",
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
