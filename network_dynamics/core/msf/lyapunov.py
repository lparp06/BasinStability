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
    log_stretch = log_stretch + jnp.log(jnp.fmax(jnp.abs(diagonal), 1e-15))
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
    """Compute Psi(K) after the K-independent transient has already run.

    The loop is structured as n_qr outer iterations, each running exactly
    qr_interval_steps RK4 steps then one unconditional QR.  This eliminates
    lax.cond from the inner loop, which was the dominant cause of 2-hour XLA
    compilation times when vmapped over large K batches on GPU.
    """

    dimension = transient_state.shape[0]
    state = transient_state
    tangent_matrix = jnp.eye(dimension, dtype=transient_state.dtype)
    log_stretch = jnp.zeros(dimension, dtype=transient_state.dtype)

    # Both static args, so n_qr is a compile-time constant.
    n_qr = measurement_steps // qr_interval_steps

    def outer_scan_body(carry, _):
        state, tangent_matrix, log_stretch = carry

        # Python for-loop: unrolled at trace time into qr_interval_steps
        # sequential RK4 calls baked into one XLA kernel body.
        # This avoids a nested while_loop which is slow on GPU.
        for _ in range(qr_interval_steps):
            state, tangent_matrix = rk4_step_msf_jax(
                state, tangent_matrix, dt, K, H, params, dynamics
            )

        tangent_matrix, log_stretch = qr_update_jax(tangent_matrix, log_stretch)
        return (state, tangent_matrix, log_stretch), None

    (state, tangent_matrix, log_stretch), _ = lax.scan(
        outer_scan_body,
        (state, tangent_matrix, log_stretch),
        xs=None,
        length=n_qr,
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


def scan_msf_from_transient_state_jax_impl(
    K_values: jnp.ndarray,
    transient_state: jnp.ndarray,
    H: jnp.ndarray,
    params: jnp.ndarray,
    measurement_steps: int,
    dt: float,
    qr_interval_steps: int,
    dynamics: str = "rossler",
):
    """Vectorized MSF scan starting from a pre-computed transient state.

    Use this instead of ``scan_msf_jax`` when you want to validate the
    transient state before committing to the expensive measurement phase.
    """

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


scan_msf_from_transient_state_jax = jax.jit(
    scan_msf_from_transient_state_jax_impl,
    static_argnames=(
        "measurement_steps",
        "qr_interval_steps",
        "dynamics",
    ),
)
