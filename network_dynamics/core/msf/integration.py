"""RK4 integration pieces used by MSF calculations."""

from __future__ import annotations

from network_dynamics.core.jax_config import enable_x64

enable_x64()

import jax.numpy as jnp
from jax import lax

from network_dynamics.core.msf.dynamics import get_msf_dynamics


def msf_rhs_jax(
    state: jnp.ndarray,
    tangent_matrix: jnp.ndarray,
    K: float | jnp.ndarray,
    H: jnp.ndarray,
    params: jnp.ndarray,
    dynamics: str = "rossler",
):
    """Combined RHS: ds/dt = F(s), dY/dt = [DF(s) - K H]Y."""

    msf_dynamics = get_msf_dynamics(dynamics)
    dsdt = msf_dynamics.rhs(state, params)
    A = msf_dynamics.jacobian(state, params) - K * H
    dYdt = A @ tangent_matrix
    return dsdt, dYdt


def rk4_step_state_jax(
    state: jnp.ndarray,
    dt: float,
    params: jnp.ndarray,
    dynamics: str = "rossler",
) -> jnp.ndarray:
    """One RK4 step for the isolated synchronized oscillator."""

    rhs = get_msf_dynamics(dynamics).rhs
    k1 = rhs(state, params)
    k2 = rhs(state + 0.5 * dt * k1, params)
    k3 = rhs(state + 0.5 * dt * k2, params)
    k4 = rhs(state + dt * k3, params)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk4_step_msf_jax(
    state: jnp.ndarray,
    tangent_matrix: jnp.ndarray,
    dt: float,
    K: float | jnp.ndarray,
    H: jnp.ndarray,
    params: jnp.ndarray,
    dynamics: str = "rossler",
):
    """One RK4 step for the synchronized state and tangent matrix."""

    k1_s, k1_Y = msf_rhs_jax(state, tangent_matrix, K, H, params, dynamics)
    k2_s, k2_Y = msf_rhs_jax(
        state + 0.5 * dt * k1_s,
        tangent_matrix + 0.5 * dt * k1_Y,
        K,
        H,
        params,
        dynamics,
    )
    k3_s, k3_Y = msf_rhs_jax(
        state + 0.5 * dt * k2_s,
        tangent_matrix + 0.5 * dt * k2_Y,
        K,
        H,
        params,
        dynamics,
    )
    k4_s, k4_Y = msf_rhs_jax(
        state + dt * k3_s,
        tangent_matrix + dt * k3_Y,
        K,
        H,
        params,
        dynamics,
    )

    next_state = state + (dt / 6.0) * (k1_s + 2.0 * k2_s + 2.0 * k3_s + k4_s)
    next_tangent = tangent_matrix + (
        (dt / 6.0) * (k1_Y + 2.0 * k2_Y + 2.0 * k3_Y + k4_Y)
    )
    return next_state, next_tangent


def run_transient_jax(
    initial_state: jnp.ndarray,
    transient_steps: int,
    dt: float,
    params: jnp.ndarray,
    dynamics: str = "rossler",
) -> jnp.ndarray:
    """Integrate the isolated oscillator before measuring exponents."""

    def body_fun(_step, state):
        return rk4_step_state_jax(state, dt, params, dynamics)

    return lax.fori_loop(0, transient_steps, body_fun, initial_state)
