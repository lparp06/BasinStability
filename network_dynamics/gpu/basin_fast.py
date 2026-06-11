"""
gpu/basin_fast.py

Fully GPU/JAX basin-stability experiment.

This version avoids returning full trajectories to the CPU.

It does on GPU:
- sample initial conditions
- integrate trajectories with RK4
- compute maximum pairwise synchronization distance
- compute final_success and window_success
- compute sync_time
- classify success/failure

The CPU only receives compact per-trial metrics at the end.
"""

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from network_dynamics.core.results import TrialResult, BasinSummary
from network_dynamics.core.sampling import trial_seeds
from network_dynamics.core.graphs import graph_laplacian
from network_dynamics.core.coupling import build_coupling_matrix


def sample_initial_conditions_jax(config):
    """
    Sample all initial conditions using JAX.

    This is production mode: sampling happens with JAX random numbers, not NumPy.

    Important:
    These samples will not match NumPy's random samples exactly, even with
    the same base_seed. That is fine for production basin-stability runs.
    """

    low, high = config.sampling_bounds

    key = jax.random.PRNGKey(config.base_seed)

    initial_states = jax.random.uniform(
        key=key,
        shape=(config.n_trials, config.state_dimension),
        minval=low,
        maxval=high,
        dtype=jnp.float32,
    )

    return initial_states


def rossler_batch_jax(state_batch, coupling_matrix, parameters):
    """
    Batched coupled Rössler right-hand side.

    state_batch shape:
        (n_trials, state_dimension)

    coupling_matrix shape:
        (state_dimension, state_dimension)
    """

    a, b, c = parameters

    X = state_batch[:, 0::3]
    Y = state_batch[:, 1::3]
    Z = state_batch[:, 2::3]

    derivative = jnp.zeros_like(state_batch)

    derivative = derivative.at[:, 0::3].set(-Y - Z)
    derivative = derivative.at[:, 1::3].set(X + a * Y)
    derivative = derivative.at[:, 2::3].set(b + Z * (X - c))

    coupling_term = state_batch @ coupling_matrix.T

    return derivative - coupling_term


def rk4_step_batch_jax(state_batch, dt, coupling_matrix, parameters):
    """
    One batched RK4 step.
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

    next_state_batch = state_batch + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return next_state_batch


def max_pairwise_distance_batch(state_batch, dimension=3):
    """
    Compute max pairwise node distance for each trajectory in a batch.

    Parameters
    ----------
    state_batch : jnp.ndarray
        Shape: (n_trials, state_dimension)

    Returns
    -------
    distances : jnp.ndarray
        Shape: (n_trials,)
    """

    n_trials = state_batch.shape[0]

    state_by_node = state_batch.reshape(
        n_trials,
        -1,
        dimension,
    )

    differences = state_by_node[:, :, None, :] - state_by_node[:, None, :, :]

    squared_distances = jnp.sum(
        differences * differences,
        axis=-1,
    )

    max_squared_distance = jnp.max(
        squared_distances,
        axis=(1, 2),
    )

    distances = jnp.sqrt(max_squared_distance)

    return distances


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
    coupling_matrix,
    parameters,
    dt,
    sync_tol,
    tol_max,
    max_abs_threshold,
    n_steps,
    window_start,
    success_code,
    dimension,
):
    """
    Integrate and classify basin trials entirely in JAX.

    success_code:
        0 -> final_success
        1 -> window_success
    """

    initial_distance = max_pairwise_distance_batch(
        state_batch=initial_states,
        dimension=dimension,
    )

    initial_max_abs = jnp.max(
        jnp.abs(initial_states),
        axis=1,
    )

    initial_finite = jnp.all(
        jnp.isfinite(initial_states),
        axis=1,
    )

    initial_window_max = jnp.where(
        window_start <= 0,
        initial_distance,
        -jnp.inf,
    )

    initial_sync_time = jnp.where(
        initial_distance < sync_tol,
        0.0,
        jnp.inf,
    )

    initial_carry = (
        initial_states,
        initial_max_abs,
        initial_finite,
        initial_window_max,
        initial_sync_time,
    )

    def step_function(carry, step_index):
        (
            state_batch,
            max_abs_seen,
            finite_seen,
            window_max_distance,
            sync_time,
        ) = carry

        next_state_batch = rk4_step_batch_jax(
            state_batch=state_batch,
            dt=dt,
            coupling_matrix=coupling_matrix,
            parameters=parameters,
        )

        distance = max_pairwise_distance_batch(
            state_batch=next_state_batch,
            dimension=dimension,
        )

        time = step_index * dt

        step_max_abs = jnp.max(
            jnp.abs(next_state_batch),
            axis=1,
        )

        step_finite = jnp.all(
            jnp.isfinite(next_state_batch),
            axis=1,
        )

        max_abs_seen = jnp.maximum(
            max_abs_seen,
            step_max_abs,
        )

        finite_seen = finite_seen & step_finite

        in_final_window = step_index >= window_start

        window_max_distance = jnp.where(
            in_final_window,
            jnp.maximum(window_max_distance, distance),
            window_max_distance,
        )

        newly_synchronized = (
            (sync_time == jnp.inf) & (distance < sync_tol) & (distance <= tol_max)
        )

        sync_time = jnp.where(
            newly_synchronized,
            time,
            sync_time,
        )

        next_carry = (
            next_state_batch,
            max_abs_seen,
            finite_seen,
            window_max_distance,
            sync_time,
        )

        return next_carry, None

    step_indices = jnp.arange(
        1,
        n_steps,
        dtype=jnp.float32,
    )

    final_carry, _ = lax.scan(
        step_function,
        initial_carry,
        step_indices,
    )

    (
        final_states,
        max_abs_seen,
        finite_seen,
        window_max_distance,
        sync_time,
    ) = final_carry

    final_distance = max_pairwise_distance_batch(
        state_batch=final_states,
        dimension=dimension,
    )

    final_success = final_distance < sync_tol
    window_success = window_max_distance < sync_tol

    integration_failed = (~finite_seen) | (max_abs_seen > max_abs_threshold)

    chosen_success = jnp.where(
        success_code == 0,
        final_success,
        window_success,
    )

    success = chosen_success & (~integration_failed)

    sync_time = jnp.where(
        integration_failed,
        jnp.inf,
        sync_time,
    )

    return {
        "success": success,
        "final_success": final_success,
        "window_success": window_success,
        "integration_failed": integration_failed,
        "final_distance": final_distance,
        "window_max_distance": window_max_distance,
        "sync_time": sync_time,
    }


def make_coupling_matrix_jax(config):
    """
    Build the coupling matrix and move it to JAX.

    This setup step is tiny. The expensive simulation still runs on GPU.
    """

    L = graph_laplacian(config.G)

    coupling_matrix_np = build_coupling_matrix(
        L=L,
        H=config.H,
        strength=config.coupling_strength,
    )

    coupling_matrix = jnp.asarray(
        coupling_matrix_np,
        dtype=jnp.float32,
    )

    return coupling_matrix


def choose_success_code(success_definition):
    """
    Convert success_definition into a small static integer for JAX.
    """

    if success_definition == "final_success":
        return 0

    if success_definition == "window_success":
        return 1

    raise ValueError(
        "success_definition must be either " "'final_success' or 'window_success'."
    )


def basin_stability_gpu_fast(config):
    """
    Run a GPU/JAX basin-stability experiment with on-GPU classification.

    Returns the usual BasinSummary object, but only compact metrics are copied
    back to the CPU.
    """

    config.validate()

    if config.integrator != "RK4":
        raise ValueError(
            "GPU fast basin currently supports only integrator='RK4'. "
            "Use CPU for LSODA."
        )

    if config.dimension != 3:
        raise ValueError("GPU fast basin currently assumes Rössler dimension=3.")

    t = np.arange(
        0.0,
        config.tmax,
        config.dt,
    )

    n_steps = len(t)

    window_start = int((1.0 - config.window_fraction) * n_steps)

    success_code = choose_success_code(
        config.success_definition,
    )

    seeds = trial_seeds(
        base_seed=config.base_seed,
        n_trials=config.n_trials,
    )

    initial_states = sample_initial_conditions_jax(config)

    coupling_matrix = make_coupling_matrix_jax(config)

    parameters = jnp.asarray(
        config.parameters,
        dtype=jnp.float32,
    )

    dt = jnp.asarray(
        config.dt,
        dtype=jnp.float32,
    )

    sync_tol = jnp.asarray(
        config.sync_tol,
        dtype=jnp.float32,
    )

    tol_max = jnp.asarray(
        config.tol_max,
        dtype=jnp.float32,
    )

    max_abs_threshold = jnp.asarray(
        config.max_abs_threshold,
        dtype=jnp.float32,
    )

    metrics = run_basin_metrics_jax(
        initial_states=initial_states,
        coupling_matrix=coupling_matrix,
        parameters=parameters,
        dt=dt,
        sync_tol=sync_tol,
        tol_max=tol_max,
        max_abs_threshold=max_abs_threshold,
        n_steps=n_steps,
        window_start=window_start,
        success_code=success_code,
        dimension=config.dimension,
    )

    # Force GPU work to finish before converting results to NumPy.
    metrics["success"].block_until_ready()

    metrics_np = {key: np.asarray(value) for key, value in metrics.items()}

    results = []

    for trial_index, seed in enumerate(seeds):
        integration_failed = bool(metrics_np["integration_failed"][trial_index])

        if integration_failed:
            error = "nonfinite values or max_abs_threshold exceeded"
        else:
            error = None

        result = TrialResult(
            trial_seed=seed,
            success=bool(metrics_np["success"][trial_index]),
            final_success=bool(metrics_np["final_success"][trial_index]),
            window_success=bool(metrics_np["window_success"][trial_index]),
            integration_failed=integration_failed,
            final_distance=float(metrics_np["final_distance"][trial_index]),
            window_max_distance=float(metrics_np["window_max_distance"][trial_index]),
            sync_time=float(metrics_np["sync_time"][trial_index]),
            error=error,
        )

        results.append(result)

    summary = BasinSummary.from_results(
        config=config,
        seeds=seeds,
        results=results,
    )

    return summary


def basin_stability_gpu_fast_from_initial_conditions(
    config,
    initial_conditions_batch,
    seeds=None,
):
    """
    Run the fast GPU/JAX basin-stability experiment from a fixed batch of
    initial conditions.

    Use this for CPU/GPU validation, because CPU and GPU can then use the
    exact same initial conditions.

    Parameters
    ----------
    config : BasinConfig
        Experiment configuration.

    initial_conditions_batch : array-like
        Shape: (n_trials, state_dimension). These are copied to the GPU once.

    seeds : sequence of int or None
        Trial labels. If None, uses trial_seeds(config.base_seed, config.n_trials).

    Returns
    -------
    BasinSummary
        Summary object with compact per-trial metrics.
    """

    config.validate()

    if config.integrator != "RK4":
        raise ValueError(
            "GPU fast basin currently supports only integrator='RK4'. "
            "Use CPU for LSODA."
        )

    if config.dimension != 3:
        raise ValueError("GPU fast basin currently assumes Rössler dimension=3.")

    initial_conditions_batch = np.asarray(
        initial_conditions_batch,
        dtype=np.float32,
    )

    if initial_conditions_batch.ndim != 2:
        raise ValueError(
            "initial_conditions_batch must have shape " "(n_trials, state_dimension)."
        )

    if initial_conditions_batch.shape[0] != config.n_trials:
        raise ValueError(
            "initial_conditions_batch has the wrong number of trials. "
            f"Expected {config.n_trials}, got {initial_conditions_batch.shape[0]}."
        )

    if initial_conditions_batch.shape[1] != config.state_dimension:
        raise ValueError(
            "initial_conditions_batch has the wrong state dimension. "
            f"Expected {config.state_dimension}, got {initial_conditions_batch.shape[1]}."
        )

    t = np.arange(
        0.0,
        config.tmax,
        config.dt,
    )

    n_steps = len(t)

    window_start = int((1.0 - config.window_fraction) * n_steps)

    success_code = choose_success_code(
        config.success_definition,
    )

    if seeds is None:
        seeds = trial_seeds(
            base_seed=config.base_seed,
            n_trials=config.n_trials,
        )
    else:
        seeds = list(seeds)

    initial_states = jnp.asarray(
        initial_conditions_batch,
        dtype=jnp.float32,
    )

    coupling_matrix = make_coupling_matrix_jax(config)

    parameters = jnp.asarray(
        config.parameters,
        dtype=jnp.float32,
    )

    dt = jnp.asarray(
        config.dt,
        dtype=jnp.float32,
    )

    sync_tol = jnp.asarray(
        config.sync_tol,
        dtype=jnp.float32,
    )

    tol_max = jnp.asarray(
        config.tol_max,
        dtype=jnp.float32,
    )

    max_abs_threshold = jnp.asarray(
        config.max_abs_threshold,
        dtype=jnp.float32,
    )

    metrics = run_basin_metrics_jax(
        initial_states=initial_states,
        coupling_matrix=coupling_matrix,
        parameters=parameters,
        dt=dt,
        sync_tol=sync_tol,
        tol_max=tol_max,
        max_abs_threshold=max_abs_threshold,
        n_steps=n_steps,
        window_start=window_start,
        success_code=success_code,
        dimension=config.dimension,
    )

    metrics["success"].block_until_ready()

    metrics_np = {key: np.asarray(value) for key, value in metrics.items()}

    results = []

    for trial_index, seed in enumerate(seeds):
        integration_failed = bool(metrics_np["integration_failed"][trial_index])

        if integration_failed:
            error = "nonfinite values or max_abs_threshold exceeded"
        else:
            error = None

        result = TrialResult(
            trial_seed=seed,
            success=bool(metrics_np["success"][trial_index]),
            final_success=bool(metrics_np["final_success"][trial_index]),
            window_success=bool(metrics_np["window_success"][trial_index]),
            integration_failed=integration_failed,
            final_distance=float(metrics_np["final_distance"][trial_index]),
            window_max_distance=float(metrics_np["window_max_distance"][trial_index]),
            sync_time=float(metrics_np["sync_time"][trial_index]),
            error=error,
        )

        results.append(result)

    summary = BasinSummary.from_results(
        config=config,
        seeds=seeds,
        results=results,
    )

    return summary


# Optional alias so benchmark scripts can import this as the GPU basin function.
basin_stability_gpu = basin_stability_gpu_fast
