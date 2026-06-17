"""
On-device basin-stability metrics for JAX/GPU runs.

The fast backend uses this module to avoid materializing full trajectories on
the CPU. It scans the RK4 integration and keeps only compact per-trial metrics.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from network_dynamics.gpu.dynamics import rk4_step_batch_jax


class BasinMetricInputs(NamedTuple):
    """
    Dynamic JAX inputs used by the metric scan.
    """

    coupling_matrix: jnp.ndarray
    parameters: jnp.ndarray
    dt: jnp.ndarray
    sync_tol: jnp.ndarray
    max_abs_threshold: jnp.ndarray


def max_pairwise_distance_batch(state_batch, dimension=3):
    """
    Compute each trial's maximum pairwise node distance.
    """

    n_trials = state_batch.shape[0]
    state_by_node = state_batch.reshape(n_trials, -1, dimension)

    differences = state_by_node[:, :, None, :] - state_by_node[:, None, :, :]
    squared_distances = jnp.sum(differences * differences, axis=-1)
    max_squared_distance = jnp.max(squared_distances, axis=(1, 2))

    return jnp.sqrt(max_squared_distance)


def choose_success_code(success_definition):
    """
    Convert a success-definition string into a static JAX branch code.
    """

    if success_definition == "final_success":
        return 0

    if success_definition == "window_success":
        return 1

    if success_definition == "first_crossing":
        return 2

    raise ValueError(
        "success_definition must be one of: "
        "'final_success', 'window_success', or 'first_crossing'."
    )


@partial(
    jax.jit,
    static_argnames=(
        "n_steps",
        "window_start",
        "success_code",
        "dimension",
    ),
)
def run_basin_metrics_jax(
    initial_states,
    inputs,
    n_steps,
    window_start,
    success_code,
    dimension,
):
    """
    Integrate a batch and classify basin trials entirely in JAX.

    success_code values are:
        0 -> final_success
        1 -> window_success
        2 -> first_crossing
    """

    initial_distance = max_pairwise_distance_batch(
        state_batch=initial_states,
        dimension=dimension,
    )
    initial_max_abs = jnp.max(jnp.abs(initial_states), axis=1)
    initial_finite = jnp.all(jnp.isfinite(initial_states), axis=1)
    initial_window_max = jnp.where(window_start <= 0, initial_distance, -jnp.inf)
    initial_ever_synchronized = initial_distance < inputs.sync_tol
    initial_sync_time = jnp.where(initial_ever_synchronized, 0.0, jnp.inf)

    initial_carry = (
        initial_states,
        initial_max_abs,
        initial_finite,
        initial_window_max,
        initial_sync_time,
        initial_ever_synchronized,
        initial_distance,
    )

    def step_function(carry, step_index):
        (
            state_batch,
            max_abs_seen,
            finite_seen,
            window_max_distance,
            sync_time,
            ever_synchronized,
            min_distance,
        ) = carry

        next_state_batch = rk4_step_batch_jax(
            state_batch=state_batch,
            dt=inputs.dt,
            coupling_matrix=inputs.coupling_matrix,
            parameters=inputs.parameters,
        )

        distance = max_pairwise_distance_batch(
            state_batch=next_state_batch,
            dimension=dimension,
        )
        min_distance = jnp.minimum(min_distance, distance)

        step_max_abs = jnp.max(jnp.abs(next_state_batch), axis=1)
        step_finite = jnp.all(jnp.isfinite(next_state_batch), axis=1)
        max_abs_seen = jnp.maximum(max_abs_seen, step_max_abs)
        finite_seen = finite_seen & step_finite

        in_final_window = step_index >= window_start
        window_max_distance = jnp.where(
            in_final_window,
            jnp.maximum(window_max_distance, distance),
            window_max_distance,
        )

        currently_synchronized = distance < inputs.sync_tol
        newly_synchronized = (~ever_synchronized) & currently_synchronized
        sync_time = jnp.where(newly_synchronized, step_index * inputs.dt, sync_time)
        ever_synchronized = ever_synchronized | currently_synchronized

        next_carry = (
            next_state_batch,
            max_abs_seen,
            finite_seen,
            window_max_distance,
            sync_time,
            ever_synchronized,
            min_distance,
        )

        return next_carry, None

    step_indices = jnp.arange(1, n_steps, dtype=jnp.float32)
    final_carry, _ = lax.scan(step_function, initial_carry, step_indices)

    (
        final_states,
        max_abs_seen,
        finite_seen,
        window_max_distance,
        sync_time,
        ever_synchronized,
        min_distance,
    ) = final_carry

    final_distance = max_pairwise_distance_batch(
        state_batch=final_states,
        dimension=dimension,
    )

    final_success = final_distance < inputs.sync_tol
    window_success = window_max_distance < inputs.sync_tol
    first_crossing_success = ever_synchronized
    health_failed = (~finite_seen) | (
        max_abs_seen > inputs.max_abs_threshold
    )
    integration_failed = jnp.where(
        success_code == 2,
        health_failed & (~first_crossing_success),
        health_failed,
    )

    chosen_success = jnp.where(
        success_code == 0,
        final_success,
        jnp.where(
            success_code == 1,
            window_success,
            first_crossing_success,
        ),
    )
    success = chosen_success & (~integration_failed)
    sync_time = jnp.where(integration_failed, jnp.inf, sync_time)

    return {
        "success": success,
        "final_success": final_success,
        "window_success": window_success,
        "first_crossing_success": first_crossing_success,
        "integration_failed": integration_failed,
        "final_distance": final_distance,
        "window_max_distance": window_max_distance,
        "min_distance": min_distance,
        "sync_time": sync_time,
    }
