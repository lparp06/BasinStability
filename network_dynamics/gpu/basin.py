"""
gpu/basin.py

GPU/JAX basin-stability experiment.

For now:
- samples initial conditions on CPU using NumPy
- integrates trajectories on GPU using JAX RK4
- processes trials in chunks to avoid GPU memory pressure
- brings trajectories back to NumPy
- reuses CPU/core diagnostics and synchronization code
- returns the same BasinSummary class as CPU basin.py
"""

import numpy as np

from network_dynamics.core.results import TrialResult, BasinSummary
from network_dynamics.core.sampling import sample_uniform_initial_condition, trial_seeds
from network_dynamics.core.diagnostics import (
    solution_health,
    is_solution_valid,
    format_health_message,
)
from network_dynamics.core.sync import analyze_synchronization
from network_dynamics.gpu.integration import integrate_rk4_batch_jax


def sample_initial_conditions_batch(config, seeds):
    """
    Generate a batch of initial conditions.

    Parameters
    ----------
    config : BasinConfig
        Experiment settings.

    seeds : sequence of int
        Trial seeds for this batch.

    Returns
    -------
    initial_conditions_batch : np.ndarray
        Shape: (batch_size, state_dimension)
    """

    if config.sampler != "uniform":
        raise ValueError(f"Unknown sampler: {config.sampler}")

    low, high = config.sampling_bounds

    initial_conditions = []

    for seed in seeds:
        rng = np.random.default_rng(seed)

        initial_condition = sample_uniform_initial_condition(
            rng=rng,
            n_nodes=config.n_nodes,
            dimension=config.dimension,
            low=low,
            high=high,
        )

        initial_conditions.append(initial_condition)

    initial_conditions_batch = np.asarray(
        initial_conditions,
        dtype=np.float32,
    )

    return initial_conditions_batch


def choose_success(sync_metrics, success_definition):
    """
    Choose which synchronization condition counts as basin success.
    """

    if success_definition == "final_success":
        return sync_metrics["final_success"]

    if success_definition == "window_success":
        return sync_metrics["window_success"]

    raise ValueError(
        "success_definition must be either " "'final_success' or 'window_success'."
    )


def classify_single_trajectory(config, trial_seed, sol, t):
    """
    Classify one already-integrated trajectory.

    This does the same post-integration work as CPU basin.py:
    - check numerical health
    - compute synchronization metrics
    - return a TrialResult
    """

    health = solution_health(
        sol,
        max_abs_threshold=config.max_abs_threshold,
    )

    if not is_solution_valid(health):
        return TrialResult(
            trial_seed=trial_seed,
            success=False,
            final_success=False,
            window_success=False,
            integration_failed=True,
            final_distance=None,
            window_max_distance=None,
            sync_time=None,
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
        sync_time=sync_metrics["sync_time"],
        error=None,
    )


def basin_stability_gpu(config, batch_size=25, verbose=True):
    """
    Run a GPU/JAX basin-stability experiment.

    Parameters
    ----------
    config : BasinConfig
        Experiment configuration.

    batch_size : int
        Number of trials to integrate on the GPU at once.

        Smaller batch_size uses less GPU memory.
        Larger batch_size may be faster, but can overwhelm Apple MPS memory.

    verbose : bool
        If True, print chunk progress.

    Returns
    -------
    BasinSummary
        Summary object containing counts and trial results.
    """

    config.validate()

    if config.integrator != "RK4":
        raise ValueError(
            "GPU basin currently supports only integrator='RK4'. " "Use CPU for LSODA."
        )

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    seeds = trial_seeds(
        base_seed=config.base_seed,
        n_trials=config.n_trials,
    )

    results = []

    for start in range(0, len(seeds), batch_size):
        end = min(start + batch_size, len(seeds))
        seed_chunk = seeds[start:end]

        if verbose:
            print(f"GPU chunk: trials {start} to {end - 1}")

        initial_conditions_batch = sample_initial_conditions_batch(
            config=config,
            seeds=seed_chunk,
        )

        sol_batch, t = integrate_rk4_batch_jax(
            G=config.G,
            initial_conditions_batch=initial_conditions_batch,
            parameters=config.parameters,
            coupling_strength=config.coupling_strength,
            H=config.H,
            tmax=config.tmax,
            dt=config.dt,
            dimension=config.dimension,
            return_numpy=True,
        )

        for trial_index, seed in enumerate(seed_chunk):
            sol = sol_batch[trial_index]

            result = classify_single_trajectory(
                config=config,
                trial_seed=seed,
                sol=sol,
                t=t,
            )

            results.append(result)

        # Drop references to big arrays before the next chunk.
        del initial_conditions_batch
        del sol_batch

    summary = BasinSummary.from_results(
        config=config,
        seeds=seeds,
        results=results,
    )

    return summary
