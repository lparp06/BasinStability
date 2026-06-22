"""Lyapunov-exponent and MSF scan kernels."""

from __future__ import annotations

from network_dynamics.core.jax_config import enable_x64

enable_x64()

import jax
import jax.numpy as jnp
from jax import lax

from network_dynamics.core.msf.integration import rk4_step_msf_jax, run_transient_jax


def qr_update_jax(tangent_matrix: jnp.ndarray, log_stretch: jnp.ndarray):
    """QR-renormalize tangent matrix and accumulate log stretching."""

    Q, R = jnp.linalg.qr(tangent_matrix)
    diagonal = jnp.diag(R)
    log_stretch = log_stretch + jnp.log(jnp.abs(diagonal) + 1e-30)
    return Q, log_stretch


def msf_value_jax_impl(
    K: float | jnp.ndarray,
    initial_state: jnp.ndarray,
    H: jnp.ndarray,
    params: jnp.ndarray,
    transient_steps: int,
    measurement_steps: int,
    dt: float,
    qr_interval_steps: int,
    dynamics: str = "rossler",
):
    """Compute Psi(K) and full Lyapunov exponent vector."""

    state = run_transient_jax(initial_state, transient_steps, dt, params, dynamics)
    return msf_value_from_state_jax(
        K,
        state,
        H,
        params,
        measurement_steps,
        dt,
        qr_interval_steps,
        dynamics,
    )


def msf_value_from_state_jax_impl(
    K: float | jnp.ndarray,
    transient_state: jnp.ndarray,
    H: jnp.ndarray,
    params: jnp.ndarray,
    measurement_steps: int,
    dt: float,
    qr_interval_steps: int,
    dynamics: str = "rossler",
):
    """Compute Psi(K) after the K-independent transient has already run."""

    dimension = transient_state.shape[0]
    state = transient_state
    tangent_matrix = jnp.eye(dimension, dtype=transient_state.dtype)
    log_stretch = jnp.zeros(dimension, dtype=transient_state.dtype)

    def body_fun(step, carry):
        state, tangent_matrix, log_stretch = carry
        state, tangent_matrix = rk4_step_msf_jax(
            state,
            tangent_matrix,
            dt,
            K,
            H,
            params,
            dynamics,
        )

        do_qr = ((step + 1) % qr_interval_steps) == 0
        tangent_matrix, log_stretch = lax.cond(
            do_qr,
            lambda args: qr_update_jax(*args),
            lambda args: args,
            operand=(tangent_matrix, log_stretch),
        )
        return state, tangent_matrix, log_stretch

    state, tangent_matrix, log_stretch = lax.fori_loop(
        0,
        measurement_steps,
        body_fun,
        (state, tangent_matrix, log_stretch),
    )

    lyapunov_exponents = log_stretch / (measurement_steps * dt)
    psi = jnp.max(lyapunov_exponents)
    return psi, lyapunov_exponents


msf_value_from_state_jax = jax.jit(
    msf_value_from_state_jax_impl,
    static_argnames=(
        "measurement_steps",
        "qr_interval_steps",
        "dynamics",
    ),
)


msf_value_jax = jax.jit(
    msf_value_jax_impl,
    static_argnames=(
        "transient_steps",
        "measurement_steps",
        "qr_interval_steps",
        "dynamics",
    ),
)


def scan_msf_jax_impl(
    K_values: jnp.ndarray,
    initial_state: jnp.ndarray,
    H: jnp.ndarray,
    params: jnp.ndarray,
    transient_steps: int,
    measurement_steps: int,
    dt: float,
    qr_interval_steps: int,
    dynamics: str = "rossler",
):
    """Vectorized MSF scan over a batch of K values."""

    transient_state = run_transient_jax(
        initial_state,
        transient_steps,
        dt,
        params,
        dynamics,
    )

    def one_K(K):
        return msf_value_from_state_jax(
            K,
            transient_state,
            H,
            params,
            measurement_steps,
            dt,
            qr_interval_steps,
            dynamics,
        )

    return jax.vmap(one_K)(K_values)


scan_msf_jax = jax.jit(
    scan_msf_jax_impl,
    static_argnames=(
        "transient_steps",
        "measurement_steps",
        "qr_interval_steps",
        "dynamics",
    ),
)
