"""
Fast JAX/GPU basin-stability backend.

This backend is intended for cluster runs. It samples or accepts a full batch
of initial conditions, integrates all trials with fixed-step RK4 on the GPU,
and copies back only compact per-trial metrics.
"""

import numpy as np
import jax
import jax.numpy as jnp

from network_dynamics.core.basin_common import validate_initial_conditions_batch
from network_dynamics.core.graphs import graph_laplacian
from network_dynamics.core.results import BasinSummary, TrialResult
from network_dynamics.core.sampling import trial_seeds
from network_dynamics.core.coupling import build_coupling_matrix
from network_dynamics.gpu.dynamics import rossler_batch_jax, rk4_step_batch_jax
from network_dynamics.gpu.metrics import (
    BasinMetricInputs,
    choose_success_code,
    max_pairwise_distance_batch,
    run_basin_metrics_jax,
)


def sample_initial_conditions_jax(config):
    """
    Sample all initial conditions using JAX random numbers.

    JAX and NumPy use different random streams, so these samples are
    reproducible for JAX runs but are not expected to match CPU samples.
    """

    low, high = config.sampling_bounds
    key = jax.random.PRNGKey(config.base_seed)

    return jax.random.uniform(
        key=key,
        shape=(config.n_trials, config.state_dimension),
        minval=low,
        maxval=high,
        dtype=jnp.float32,
    )


def make_coupling_matrix_jax(config):
    """
    Build the network coupling matrix and move it to a JAX array.
    """

    laplacian = graph_laplacian(config.G)
    coupling_matrix = build_coupling_matrix(
        L=laplacian,
        H=config.H,
        strength=config.coupling_strength,
    )

    return jnp.asarray(coupling_matrix, dtype=jnp.float32)


def validate_fast_gpu_config(config):
    """
    Validate config constraints specific to the fast JAX backend.
    """

    config.validate()

    if config.integrator != "RK4":
        raise ValueError(
            "GPU fast basin currently supports only integrator='RK4'. "
            "Use CPU for LSODA."
        )

    if config.dimension != 3:
        raise ValueError("GPU fast basin currently assumes Rössler dimension=3.")


def _make_metric_inputs(config):
    return BasinMetricInputs(
        coupling_matrix=make_coupling_matrix_jax(config),
        parameters=jnp.asarray(config.parameters, dtype=jnp.float32),
        dt=jnp.asarray(config.dt, dtype=jnp.float32),
        sync_tol=jnp.asarray(config.sync_tol, dtype=jnp.float32),
        max_abs_threshold=jnp.asarray(
            config.max_abs_threshold,
            dtype=jnp.float32,
        ),
    )


def _run_fast_metrics(config, initial_states):
    n_steps = config.n_time_points
    window_start = int((1.0 - config.window_fraction) * n_steps)

    return run_basin_metrics_jax(
        initial_states=initial_states,
        inputs=_make_metric_inputs(config),
        n_steps=n_steps,
        window_start=window_start,
        success_code=choose_success_code(config.success_definition),
        dimension=config.dimension,
    )


def _metrics_to_summary(config, seeds, metrics):
    """
    Convert compact JAX metrics into TrialResult objects and BasinSummary.
    """

    metrics["success"].block_until_ready()
    metrics_np = {key: np.asarray(value) for key, value in metrics.items()}

    results = [
        TrialResult(
            trial_seed=seed,
            success=bool(metrics_np["success"][trial_index]),
            final_success=bool(metrics_np["final_success"][trial_index]),
            window_success=bool(metrics_np["window_success"][trial_index]),
            integration_failed=bool(metrics_np["integration_failed"][trial_index]),
            final_distance=float(metrics_np["final_distance"][trial_index]),
            window_max_distance=float(
                metrics_np["window_max_distance"][trial_index]
            ),
            min_distance=float(metrics_np["min_distance"][trial_index]),
            sync_time=float(metrics_np["sync_time"][trial_index]),
            error=(
                "nonfinite values or max_abs_threshold exceeded"
                if bool(metrics_np["integration_failed"][trial_index])
                else None
            ),
        )
        for trial_index, seed in enumerate(seeds)
    ]

    return BasinSummary.from_results(
        config=config,
        seeds=seeds,
        results=results,
    )


def _summary_from_initial_states(config, initial_states, seeds):
    metrics = _run_fast_metrics(
        config=config,
        initial_states=initial_states,
    )

    return _metrics_to_summary(
        config=config,
        seeds=seeds,
        metrics=metrics,
    )


def basin_stability_gpu_fast(config):
    """
    Run a production GPU/JAX basin-stability experiment.
    """

    validate_fast_gpu_config(config)

    seeds = trial_seeds(
        base_seed=config.base_seed,
        n_trials=config.n_trials,
    )
    initial_states = sample_initial_conditions_jax(config)

    return _summary_from_initial_states(
        config=config,
        initial_states=initial_states,
        seeds=seeds,
    )


def basin_stability_gpu_fast_from_initial_conditions(
    config,
    initial_conditions_batch,
    seeds=None,
):
    """
    Run the fast GPU/JAX backend from fixed initial conditions.

    Use this for CPU/GPU validation because both backends can consume the same
    sampled initial-condition array.
    """

    validate_fast_gpu_config(config)

    initial_conditions_batch = validate_initial_conditions_batch(
        config=config,
        initial_conditions_batch=initial_conditions_batch,
        dtype=np.float32,
    )

    if seeds is None:
        seeds = trial_seeds(
            base_seed=config.base_seed,
            n_trials=config.n_trials,
        )
    else:
        seeds = list(seeds)

    initial_states = jnp.asarray(initial_conditions_batch, dtype=jnp.float32)

    return _summary_from_initial_states(
        config=config,
        initial_states=initial_states,
        seeds=seeds,
    )


basin_stability_gpu = basin_stability_gpu_fast
