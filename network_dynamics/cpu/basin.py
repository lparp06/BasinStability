"""
CPU basin-stability experiments.

This module coordinates trial sampling, CPU integration, optional
multiprocessing, and result summaries. Shared basin concepts live in
``network_dynamics.core.basin_common`` so the CPU and GPU backends classify
trials the same way.
"""

import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from network_dynamics.core.basin_common import (
    classify_solution,
    failed_trial_result,
    sample_initial_condition,
    validate_initial_conditions_batch,
)
from network_dynamics.core.results import BasinSummary
from network_dynamics.core.sampling import trial_seeds
from network_dynamics.cpu.integration import integrate_from_config


def run_single_trial(config, trial_seed):
    """
    Sample, integrate, and classify one CPU basin-stability trial.
    """

    try:
        initial_condition = sample_initial_condition(
            config=config,
            trial_seed=trial_seed,
        )

        return run_single_trial_from_initial_condition(
            config=config,
            trial_seed=trial_seed,
            initial_condition=initial_condition,
        )

    except Exception as error:
        return failed_trial_result(
            trial_seed=trial_seed,
            error=str(error),
        )


def run_single_trial_from_initial_condition(config, trial_seed, initial_condition):
    """
    Integrate and classify one CPU trial from a fixed initial condition.
    """

    try:
        sol, t = integrate_from_config(
            config=config,
            initial_conditions=initial_condition,
        )

        return classify_solution(
            config=config,
            trial_seed=trial_seed,
            sol=sol,
            t=t,
        )

    except Exception as error:
        return failed_trial_result(
            trial_seed=trial_seed,
            error=str(error),
        )


def _run_trial_from_settings(settings):
    """
    Top-level multiprocessing adapter for sampled trials.
    """

    return run_single_trial(
        config=settings["config"],
        trial_seed=settings["trial_seed"],
    )


def _run_trial_from_initial_condition_settings(settings):
    """
    Top-level multiprocessing adapter for fixed initial-condition trials.
    """

    return run_single_trial_from_initial_condition(
        config=settings["config"],
        trial_seed=settings["trial_seed"],
        initial_condition=settings["initial_condition"],
    )


def _print_progress(
    completed,
    total,
    start_time,
    label,
    progress_stream,
):
    elapsed = time.perf_counter() - start_time
    rate = completed / elapsed if elapsed > 0 else 0.0
    remaining = total - completed
    eta = remaining / rate if rate > 0 else float("inf")

    print(
        f"{label}: {completed}/{total} trials "
        f"({100.0 * completed / total:.1f}%) | "
        f"{rate:.2f} trials/s | ETA {eta / 60.0:.1f} min",
        file=progress_stream,
        flush=True,
    )


def _map_trials(
    trial_settings,
    worker_function,
    n_workers=None,
    progress_label=None,
    progress_interval=100,
    progress_stream=None,
):
    total = len(trial_settings)
    progress_stream = progress_stream or sys.stdout
    progress_interval = max(1, int(progress_interval))
    start_time = time.perf_counter()

    def maybe_print_progress(completed):
        if progress_label is None:
            return

        if completed == total or completed % progress_interval == 0:
            _print_progress(
                completed=completed,
                total=total,
                start_time=start_time,
                label=progress_label,
                progress_stream=progress_stream,
            )

    if n_workers is None or n_workers <= 1:
        results = []

        for completed, settings in enumerate(trial_settings, start=1):
            results.append(worker_function(settings))
            maybe_print_progress(completed)

        return results

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_index = {
            executor.submit(worker_function, settings): trial_index
            for trial_index, settings in enumerate(trial_settings)
        }
        results = [None] * total

        for completed, future in enumerate(as_completed(future_to_index), start=1):
            trial_index = future_to_index[future]
            results[trial_index] = future.result()
            maybe_print_progress(completed)

    return results


def basin_stability_cpu_from_initial_conditions(
    config,
    initial_conditions_batch,
    seeds=None,
    progress_label=None,
    progress_interval=100,
    progress_stream=None,
):
    """
    Run CPU basin stability from fixed initial conditions.

    This is the validation path for exact CPU/GPU comparisons.
    """

    config.validate()

    initial_conditions_batch = validate_initial_conditions_batch(
        config=config,
        initial_conditions_batch=initial_conditions_batch,
        dtype=float,
    )

    if seeds is None:
        seeds = trial_seeds(
            base_seed=config.base_seed,
            n_trials=config.n_trials,
        )
    else:
        seeds = list(seeds)

    trial_settings = [
        {
            "config": config,
            "trial_seed": seeds[trial_index],
            "initial_condition": initial_conditions_batch[trial_index],
        }
        for trial_index in range(config.n_trials)
    ]

    results = _map_trials(
        trial_settings=trial_settings,
        worker_function=_run_trial_from_initial_condition_settings,
        n_workers=config.n_workers,
        progress_label=progress_label,
        progress_interval=progress_interval,
        progress_stream=progress_stream,
    )

    return BasinSummary.from_results(
        config=config,
        seeds=seeds,
        results=results,
    )


def basin_stability_serial(config):
    """
    Run a serial CPU basin-stability experiment.
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

    results = _map_trials(
        trial_settings=trial_settings,
        worker_function=_run_trial_from_settings,
        n_workers=1,
    )

    return BasinSummary.from_results(
        config=config,
        seeds=seeds,
        results=results,
    )


def basin_stability_cpu(config):
    """
    Run a multiprocessing CPU basin-stability experiment.
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

    results = _map_trials(
        trial_settings=trial_settings,
        worker_function=_run_trial_from_settings,
        n_workers=config.n_workers,
    )

    return BasinSummary.from_results(
        config=config,
        seeds=seeds,
        results=results,
    )


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
    Print one line per trial for debugging.
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
            f"min_distance={result.min_distance} | "
            f"error={result.error}"
        )
