"""
Shared helpers for basin-stability experiments.

The CPU and GPU backends differ in where they integrate trajectories, but
they should agree on sampling, success definitions, input validation, and
how trajectory diagnostics become TrialResult objects.
"""

import numpy as np

from network_dynamics.core.diagnostics import (
    format_health_message,
    is_solution_valid,
    solution_health,
)
from network_dynamics.core.results import TrialResult
from network_dynamics.core.sampling import sample_uniform_initial_condition
from network_dynamics.core.sync import analyze_synchronization


SUCCESS_DEFINITIONS = (
    "final_success",
    "window_success",
    "first_crossing",
)


def choose_success(sync_metrics, success_definition):
    """
    Select the synchronization condition that counts as basin success.
    """

    if success_definition == "final_success":
        return sync_metrics["final_success"]

    if success_definition == "window_success":
        return sync_metrics["window_success"]

    if success_definition == "first_crossing":
        return sync_metrics["first_crossing_success"]

    raise ValueError(
        "success_definition must be one of: "
        "'final_success', 'window_success', or 'first_crossing'."
    )


def sample_initial_condition(config, trial_seed):
    """
    Generate one random initial condition for one basin trial.
    """

    if config.sampler != "uniform":
        raise ValueError(f"Unknown sampler: {config.sampler}")

    low, high = config.sampling_bounds
    rng = np.random.default_rng(trial_seed)

    return sample_uniform_initial_condition(
        rng=rng,
        n_nodes=config.n_nodes,
        dimension=config.dimension,
        low=low,
        high=high,
    )


def sample_initial_conditions_batch(config, seeds, dtype=np.float32):
    """
    Generate a batch of reproducible initial conditions from trial seeds.
    """

    initial_conditions = [
        sample_initial_condition(
            config=config,
            trial_seed=seed,
        )
        for seed in seeds
    ]

    return np.asarray(initial_conditions, dtype=dtype)


def validate_initial_conditions_batch(config, initial_conditions_batch, dtype=float):
    """
    Convert and validate fixed initial conditions for CPU/GPU comparison runs.
    """

    initial_conditions_batch = np.asarray(initial_conditions_batch, dtype=dtype)

    if initial_conditions_batch.ndim != 2:
        raise ValueError(
            "initial_conditions_batch must have shape "
            "(n_trials, state_dimension)."
        )

    expected_shape = (config.n_trials, config.state_dimension)

    if initial_conditions_batch.shape != expected_shape:
        raise ValueError(
            "initial_conditions_batch has the wrong shape. "
            f"Expected {expected_shape}, got {initial_conditions_batch.shape}."
        )

    return initial_conditions_batch


def failed_trial_result(trial_seed, error):
    """
    Build a TrialResult for integration or diagnostic failure.
    """

    return TrialResult(
        trial_seed=trial_seed,
        success=False,
        final_success=False,
        window_success=False,
        integration_failed=True,
        final_distance=None,
        window_max_distance=None,
        min_distance=None,
        sync_time=None,
        error=error,
    )


def classify_solution(config, trial_seed, sol, t):
    """
    Convert one integrated trajectory into a basin TrialResult.
    """

    health = solution_health(
        sol,
        max_abs_threshold=config.max_abs_threshold,
    )

    if not is_solution_valid(health):
        return failed_trial_result(
            trial_seed=trial_seed,
            error=format_health_message(health),
        )

    sync_metrics = analyze_synchronization(
        sol=sol,
        t=t,
        dimension=config.dimension,
        tol=config.sync_tol,
        tol_max=config.tol_max,
        win_frac=config.window_fraction,
    )

    success = choose_success(
        sync_metrics=sync_metrics,
        success_definition=config.success_definition,
    )

    return TrialResult(
        trial_seed=trial_seed,
        success=bool(success),
        final_success=bool(sync_metrics["final_success"]),
        window_success=bool(sync_metrics["window_success"]),
        integration_failed=False,
        final_distance=sync_metrics["final_distance"],
        window_max_distance=sync_metrics["window_max_distance"],
        min_distance=sync_metrics["min_distance"],
        sync_time=sync_metrics["sync_time"],
        error=None,
    )
