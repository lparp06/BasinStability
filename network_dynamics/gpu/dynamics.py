"""
JAX dynamics kernels for coupled oscillator networks.

These functions are intentionally small and array-oriented so they can be
composed by both full-trajectory GPU integration and fast on-device basin
metric scans.
"""

from network_dynamics.core.jax_config import enable_x64

enable_x64()

import jax.numpy as jnp


DYNAMICS_CODES = {
    "rossler": 0,
    "lorenz": 1,
}


def dynamics_code(dynamics):
    try:
        return DYNAMICS_CODES[dynamics]
    except KeyError as error:
        raise ValueError(
            "Unknown dynamics. Supported GPU dynamics are: "
            + ", ".join(DYNAMICS_CODES)
        ) from error


def rossler_batch_jax(state_batch, coupling_matrix, parameters):
    """
    Evaluate the coupled Rössler right-hand side for a batch of states.

    state_batch has shape ``(n_trials, state_dimension)``. The state layout is
    ``[x0, y0, z0, x1, y1, z1, ...]`` for each row.
    """

    a, b, c = parameters

    x = state_batch[:, 0::3]
    y = state_batch[:, 1::3]
    z = state_batch[:, 2::3]

    # Interleave dx/dy/dz back into flat order without scatter ops:
    # stack along last axis → (n_trials, n_nodes, 3) → reshape to (n_trials, state_dim)
    dx = jnp.stack([-y - z, x + a * y, b + z * (x - c)], axis=-1)
    derivative = dx.reshape(state_batch.shape)

    # coupling_matrix is symmetric for undirected graphs; state_batch @ C.T ≡ state_batch @ C
    coupling_term = state_batch @ coupling_matrix.T

    return derivative - coupling_term


def lorenz_batch_jax(state_batch, coupling_matrix, parameters):
    """
    Evaluate the coupled Lorenz right-hand side for a batch of states.
    """

    sigma, beta, rho = parameters

    x = state_batch[:, 0::3]
    y = state_batch[:, 1::3]
    z = state_batch[:, 2::3]

    dx = jnp.stack([sigma * (y - x), x * (rho - z) - y, x * y - beta * z], axis=-1)
    derivative = dx.reshape(state_batch.shape)

    coupling_term = state_batch @ coupling_matrix.T

    return derivative - coupling_term


def oscillator_batch_jax(state_batch, coupling_matrix, parameters, dynamics_code_value):
    if dynamics_code_value == DYNAMICS_CODES["rossler"]:
        return rossler_batch_jax(
            state_batch=state_batch,
            coupling_matrix=coupling_matrix,
            parameters=parameters,
        )

    if dynamics_code_value == DYNAMICS_CODES["lorenz"]:
        return lorenz_batch_jax(
            state_batch=state_batch,
            coupling_matrix=coupling_matrix,
            parameters=parameters,
        )

    raise ValueError(f"Unsupported dynamics code: {dynamics_code_value}")


def rk4_step_batch_jax(
    state_batch,
    dt,
    coupling_matrix,
    parameters,
    dynamics_code_value=DYNAMICS_CODES["rossler"],
):
    """
    Advance a batch of coupled oscillator states by one fixed RK4 step.
    """

    k1 = oscillator_batch_jax(
        state_batch=state_batch,
        coupling_matrix=coupling_matrix,
        parameters=parameters,
        dynamics_code_value=dynamics_code_value,
    )
    k2 = oscillator_batch_jax(
        state_batch=state_batch + 0.5 * dt * k1,
        coupling_matrix=coupling_matrix,
        parameters=parameters,
        dynamics_code_value=dynamics_code_value,
    )
    k3 = oscillator_batch_jax(
        state_batch=state_batch + 0.5 * dt * k2,
        coupling_matrix=coupling_matrix,
        parameters=parameters,
        dynamics_code_value=dynamics_code_value,
    )
    k4 = oscillator_batch_jax(
        state_batch=state_batch + dt * k3,
        coupling_matrix=coupling_matrix,
        parameters=parameters,
        dynamics_code_value=dynamics_code_value,
    )

    return state_batch + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
