"""
JAX dynamics kernels for coupled Rössler networks.

These functions are intentionally small and array-oriented so they can be
composed by both full-trajectory GPU integration and fast on-device basin
metric scans.
"""

import jax.numpy as jnp


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

    derivative = jnp.zeros_like(state_batch)
    derivative = derivative.at[:, 0::3].set(-y - z)
    derivative = derivative.at[:, 1::3].set(x + a * y)
    derivative = derivative.at[:, 2::3].set(b + z * (x - c))

    coupling_term = state_batch @ coupling_matrix.T

    return derivative - coupling_term


def rk4_step_batch_jax(state_batch, dt, coupling_matrix, parameters):
    """
    Advance a batch of coupled Rössler states by one fixed RK4 step.
    """

    k1 = rossler_batch_jax(
        state_batch=state_batch,
        coupling_matrix=coupling_matrix,
        parameters=parameters,
    )
    k2 = rossler_batch_jax(
        state_batch=state_batch + 0.5 * dt * k1,
        coupling_matrix=coupling_matrix,
        parameters=parameters,
    )
    k3 = rossler_batch_jax(
        state_batch=state_batch + 0.5 * dt * k2,
        coupling_matrix=coupling_matrix,
        parameters=parameters,
    )
    k4 = rossler_batch_jax(
        state_batch=state_batch + dt * k3,
        coupling_matrix=coupling_matrix,
        parameters=parameters,
    )

    return state_batch + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
