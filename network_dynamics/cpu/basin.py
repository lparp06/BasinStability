"""
basin.py

Runs basin-stability experiments on the CPU.

This file coordinates basin-stability trials:
1. sample an initial condition
2. integrate the trajectory
3. check solution health
4. check synchronization
5. summarize results

Supports:
- serial CPU runs
- multiprocessing CPU runs
- BasinConfig
- TrialResult
- BasinSummary
"""

import numpy as np
import networkx as nx
from concurrent.futures import ProcessPoolExecutor

from network_dynamics.core.config import BasinConfig
from network_dynamics.core.results import TrialResult, BasinSummary
from network_dynamics.core.sampling import sample_uniform_initial_condition, trial_seeds
from network_dynamics.core.diagnostics import (
    solution_health,
    is_solution_valid,
    format_health_message,
)
from network_dynamics.core.sync import analyze_synchronization
from network_dynamics.cpu.integration import integrate


def sample_initial_condition(config, trial_seed):
    """
    Generate one random initial condition for one basin trial.
    """

    rng = np.random.default_rng(trial_seed)

    if config.sampler != "uniform":
        raise ValueError(f"Unknown sampler: {config.sampler}")

    low, high = config.sampling_bounds

    initial_condition = sample_uniform_initial_condition(
        rng=rng,
        n_nodes=config.n_nodes,
        dimension=config.dimension,
        low=low,
        high=high,
    )

    return initial_condition


def choose_success(sync_metrics, success_definition):
    """
    Choose which synchronization condition counts as basin success.

    Options:
    - "final_success"
    - "window_success"
    """

    if success_definition == "final_success":
        return sync_metrics["final_success"]

    if success_definition == "window_success":
        return sync_metrics["window_success"]

    raise ValueError(
        "success_definition must be either "
        "'final_success' or 'window_success'."
    )


def run_single_trial(config, trial_seed):
    """
    Run one basin-stability trial.

    Returns
    -------
    TrialResult
        Result object for one trial.
    """

    try:
        initial_condition = sample_initial_condition(
            config=config,
            trial_seed=trial_seed,
        )

        sol, t = integrate(
            G=config.G,
            initial_conditions=initial_condition,
            parameters=config.parameters,
            coupling_strength=config.coupling_strength,
            H=config.H,
            tmax=config.tmax,
            dt=config.dt,
            dimension=config.dimension,
            integrator=config.integrator
        )

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
            sync_time=sync_metrics["sync_time"],
            error=None,
        )

    except Exception as error:
        return TrialResult(
            trial_seed=trial_seed,
            success=False,
            final_success=False,
            window_success=False,
            integration_failed=True,
            final_distance=None,
            sync_time=None,
            error=str(error),
        )


def _run_trial_from_settings(settings):
    """
    Helper for multiprocessing.

    ProcessPoolExecutor needs a top-level function that can be imported
    by worker processes.
    """

    return run_single_trial(
        config=settings["config"],
        trial_seed=settings["trial_seed"],
    )


def basin_stability_serial(config):
    """
    Run a serial basin-stability experiment.

    Parameters
    ----------
    config : BasinConfig
        Experiment configuration.

    Returns
    -------
    BasinSummary
        Summary object containing counts and trial results.
    """

    config.validate()

    seeds = trial_seeds(
        base_seed=config.base_seed,
        n_trials=config.n_trials,
    )

    results = []

    for seed in seeds:
        result = run_single_trial(
            config=config,
            trial_seed=seed,
        )
        results.append(result)

    summary = BasinSummary.from_results(
        config=config,
        seeds=seeds,
        results=results,
    )

    return summary


def basin_stability_cpu(config):
    """
    Run a parallel CPU basin-stability experiment.

    Each basin trial is independent, so multiprocessing can distribute
    trials across CPU worker processes.
    """

    config.validate()

    seeds = trial_seeds(
        base_seed=config.base_seed,
        n_trials=config.n_trials,
    )

    trial_settings = [
        {
            "config": config,
            "trial_seed": seed,
        }
        for seed in seeds
    ]

    with ProcessPoolExecutor(max_workers=config.n_workers) as executor:
        results = list(
            executor.map(
                _run_trial_from_settings,
                trial_settings,
            )
        )

    summary = BasinSummary.from_results(
        config=config,
        seeds=seeds,
        results=results,
    )

    return summary


def print_basin_summary(summary):
    """
    Print a readable basin-stability summary.
    """

    print("Basin stability summary")
    print("-" * 40)
    print(f"Success definition:   {summary.success_definition}")
    print(f"Basin stability:      {summary.basin_stability}")
    print(f"Number of trials:    {summary.n_trials}")
    print(f"Successes:           {summary.successes}")
    print(f"Sync failures:       {summary.sync_failures}")
    print(f"Integration failures:{summary.integration_failures}")
    print(f"Mean sync time:      {summary.sync_time_mean}")
    print(f"Base seed:           {summary.base_seed}")


def print_trial_results(summary):
    """
    Print one line per trial.
    Useful while debugging.
    """

    print()
    print("Trial results")
    print("-" * 40)

    for result in summary.results:
        print(
            f"seed={result.trial_seed} | "
            f"success={result.success} | "
            f"final_success={result.final_success} | "
            f"window_success={result.window_success} | "
            f"integration_failed={result.integration_failed} | "
            f"final_distance={result.final_distance} | "
            f"error={result.error}"
        )

