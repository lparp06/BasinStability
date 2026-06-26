"""
On-device basin-stability metrics for JAX/GPU runs.

The fast backend uses this module to avoid materializing full trajectories on
the CPU. It scans the RK4 integration and keeps only compact per-trial metrics.
"""

from functools import partial
from typing import NamedTuple

from network_dynamics.core.jax_config import enable_x64

enable_x64()

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

    Uses the identity ||xi-xj||^2 = ||xi||^2 + ||xj||^2 - 2*xi·xj so the
    n×n distance matrix is built via a batched matmul (cuBLAS) rather than a
    broadcast subtraction that would materialise a (T, N, N, D) tensor at every
    scan step.
    """

    n_trials = state_batch.shape[0]
    x = state_batch.reshape(n_trials, -1, dimension)   # (T, N, D)

    sq_norms = jnp.sum(x * x, axis=-1)                # (T, N)
    gram     = x @ jnp.swapaxes(x, -1, -2)            # (T, N, N)  — cuBLAS path

    sq_dists = sq_norms[:, :, None] + sq_norms[:, None, :] - 2.0 * gram
    sq_dists = jnp.maximum(sq_dists, 0.0)             # clip floating-point negatives

    return jnp.sqrt(jnp.max(sq_dists, axis=(1, 2)))


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
        "dimension",
        "dynamics_code_value",
    ),
)
def run_basin_metrics_first_crossing_jax(
    initial_states,
    inputs,
    n_steps,
    dimension,
    dynamics_code_value,
):
    """
    Integrate a batch for first_crossing basin stability using lax.while_loop.

    Exits early once every trial has either synchronized or health-failed,
    which can give orders-of-magnitude speedup when trajectories synchronize
    well before tmax.
    """

    initial_distance = max_pairwise_distance_batch(
        state_batch=initial_states,
        dimension=dimension,
    )
    initial_ever_synced = initial_distance < inputs.sync_tol
    initial_sync_time = jnp.where(initial_ever_synced, 0.0, jnp.inf)
    initial_health_ok = jnp.all(jnp.isfinite(initial_states), axis=1)
    initial_max_abs = jnp.max(jnp.abs(initial_states), axis=1)

    init_carry = (
        initial_states,
        jnp.array(1, dtype=jnp.int32),
        initial_ever_synced,
        initial_sync_time,
        initial_health_ok,
        initial_max_abs,
    )

    def cond(carry):
        _, step, ever_synced, _, health_ok, _ = carry
        resolved = ever_synced | ~health_ok
        return (step < n_steps) & ~jnp.all(resolved)

    def body(carry):
        state_batch, step, ever_synced, sync_time, health_ok, max_abs_seen = carry

        next_state_batch = rk4_step_batch_jax(
            state_batch=state_batch,
            dt=inputs.dt,
            coupling_matrix=inputs.coupling_matrix,
            parameters=inputs.parameters,
            dynamics_code_value=dynamics_code_value,
        )

        distance = max_pairwise_distance_batch(
            state_batch=next_state_batch,
            dimension=dimension,
        )

        currently_synced = distance < inputs.sync_tol
        newly_synced = (~ever_synced) & currently_synced
        sync_time = jnp.where(newly_synced, step * inputs.dt, sync_time)
        ever_synced = ever_synced | currently_synced

        step_max_abs = jnp.max(jnp.abs(next_state_batch), axis=1)
        step_finite = jnp.all(jnp.isfinite(next_state_batch), axis=1)
        max_abs_seen = jnp.maximum(max_abs_seen, step_max_abs)
        health_ok = health_ok & step_finite

        return (next_state_batch, step + 1, ever_synced, sync_time, health_ok, max_abs_seen)

    final_carry = lax.while_loop(cond, body, init_carry)
    final_states, _, ever_synced, sync_time, health_ok, max_abs_seen = final_carry

    health_failed = ~health_ok | (max_abs_seen > inputs.max_abs_threshold)
    # Post-sync instability is forgiven for first_crossing
    integration_failed = health_failed & ~ever_synced
    success = ever_synced & ~integration_failed
    sync_time = jnp.where(integration_failed, jnp.inf, sync_time)

    final_distance = max_pairwise_distance_batch(
        state_batch=final_states,
        dimension=dimension,
    )

    n_trials = initial_states.shape[0]
    placeholder = jnp.full((n_trials,), jnp.inf, dtype=jnp.float64)

    return {
        "success": success,
        "final_success": final_distance < inputs.sync_tol,
        "window_success": jnp.zeros(n_trials, dtype=jnp.bool_),
        "first_crossing_success": ever_synced,
        "integration_failed": integration_failed,
        "final_distance": final_distance,
        "window_max_distance": placeholder,
        "min_distance": placeholder,
        "sync_time": sync_time,
    }


@partial(
    jax.jit,
    static_argnames=(
        "n_steps",
        "window_start",
        "success_code",
        "dimension",
        "dynamics_code_value",
    ),
)
def run_basin_metrics_jax(
    initial_states,
    inputs,
    n_steps,
    window_start,
    success_code,
    dimension,
    dynamics_code_value,
):
    """
    Integrate a batch and classify basin trials entirely in JAX.

    success_code values are:
        0 -> final_success
        1 -> window_success
        2 -> first_crossing  (dispatches to while_loop early-exit kernel)
    """

    if success_code == 2:
        return run_basin_metrics_first_crossing_jax(
            initial_states=initial_states,
            inputs=inputs,
            n_steps=n_steps,
            dimension=dimension,
            dynamics_code_value=dynamics_code_value,
        )

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
            dynamics_code_value=dynamics_code_value,
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

    step_indices = jnp.arange(1, n_steps, dtype=jnp.int32)
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
    integration_failed = health_failed

    chosen_success = jnp.where(
        success_code == 0,
        final_success,
        window_success,
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
