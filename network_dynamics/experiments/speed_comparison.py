"""
validate_fast_gpu_against_cpu.py

Validate fast GPU basin calculations against CPU RK4 using the exact same
initial conditions.

Run from project root:

    python -m network_dynamics.experiments.validate_fast_gpu_against_cpu
"""

import time

import numpy as np
import networkx as nx

from network_dynamics.core.config import BasinConfig
from network_dynamics.core.sampling import sample_uniform_initial_condition, trial_seeds
from network_dynamics.cpu.integration import integrate
from network_dynamics.core.diagnostics import (
    solution_health,
    is_solution_valid,
    format_health_message,
)
from network_dynamics.core.sync import analyze_synchronization
from network_dynamics.core.results import TrialResult, BasinSummary
from network_dynamics.cpu.basin import print_basin_summary
from network_dynamics.gpu.basin_fast import (
    basin_stability_gpu_fast_from_initial_conditions,
)


def make_config(backend):
    return BasinConfig(
        G=nx.path_graph(5),
        n_trials=10000,
        base_seed=42,
        parameters=(0.2, 0.2, 7.0),
        coupling_strength=1.0,
        H=None,
        tmax=150.0,
        dt=0.05,
        dimension=3,
        sampling_bounds=(-5.0, 5.0),
        sync_tol=1e-2,
        tol_max=1e6,
        window_fraction=0.2,
        max_abs_threshold=1e6,
        success_definition="window_success",
        integrator="RK4",
        backend=backend,
    ).validate()


def make_initial_conditions(config, seeds):
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

    return np.asarray(initial_conditions, dtype=np.float32)


def choose_success(sync_metrics, success_definition):
    if success_definition == "final_success":
        return sync_metrics["final_success"]

    if success_definition == "window_success":
        return sync_metrics["window_success"]

    raise ValueError("Unknown success_definition.")


def run_cpu_from_initial_conditions(config, initial_conditions_batch, seeds):
    results = []

    for trial_index, seed in enumerate(seeds):
        initial_condition = initial_conditions_batch[trial_index]

        try:
            sol, t = integrate(
                G=config.G,
                initial_conditions=initial_condition,
                parameters=config.parameters,
                coupling_strength=config.coupling_strength,
                H=config.H,
                tmax=config.tmax,
                dt=config.dt,
                dimension=config.dimension,
                integrator=config.integrator,
            )

            health = solution_health(
                sol,
                max_abs_threshold=config.max_abs_threshold,
            )

            if not is_solution_valid(health):
                result = TrialResult(
                    trial_seed=seed,
                    success=False,
                    final_success=False,
                    window_success=False,
                    integration_failed=True,
                    final_distance=None,
                    window_max_distance=None,
                    sync_time=None,
                    error=format_health_message(health),
                )
            else:
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

                result = TrialResult(
                    trial_seed=seed,
                    success=bool(success),
                    final_success=bool(sync_metrics["final_success"]),
                    window_success=bool(sync_metrics["window_success"]),
                    integration_failed=False,
                    final_distance=sync_metrics["final_distance"],
                    window_max_distance=sync_metrics["window_max_distance"],
                    sync_time=sync_metrics["sync_time"],
                    error=None,
                )

        except Exception as error:
            result = TrialResult(
                trial_seed=seed,
                success=False,
                final_success=False,
                window_success=False,
                integration_failed=True,
                final_distance=None,
                window_max_distance=None,
                sync_time=None,
                error=str(error),
            )

        results.append(result)

    return BasinSummary.from_results(
        config=config,
        seeds=seeds,
        results=results,
    )


def time_call(function, *args, **kwargs):
    start = time.perf_counter()
    value = function(*args, **kwargs)
    end = time.perf_counter()

    return value, end - start


def compare(cpu_summary, gpu_summary):
    print()
    print("Comparison")
    print("-" * 40)

    print("CPU basin stability:", cpu_summary.basin_stability)
    print("GPU basin stability:", gpu_summary.basin_stability)
    print(
        "Absolute difference:",
        abs(cpu_summary.basin_stability - gpu_summary.basin_stability),
    )

    matches = [
        cpu_result.success == gpu_result.success
        for cpu_result, gpu_result in zip(cpu_summary.results, gpu_summary.results)
    ]

    print("Per-trial matches:", sum(matches), "/", len(matches))

    print()
    print("Mismatches")
    print("-" * 40)

    mismatch_count = 0

    for cpu_result, gpu_result in zip(cpu_summary.results, gpu_summary.results):
        if cpu_result.success != gpu_result.success:
            mismatch_count += 1

            print("seed:", cpu_result.trial_seed)
            print("  CPU success:", cpu_result.success)
            print("  GPU success:", gpu_result.success)
            print("  CPU final_distance:", cpu_result.final_distance)
            print("  GPU final_distance:", gpu_result.final_distance)
            print("  CPU window_max_distance:", cpu_result.window_max_distance)
            print("  GPU window_max_distance:", gpu_result.window_max_distance)
            print()

    if mismatch_count == 0:
        print("No mismatches.")
    else:
        print("Total mismatches:", mismatch_count)


def main():
    cpu_config = make_config(backend="serial")
    gpu_config = make_config(backend="gpu")

    seeds = trial_seeds(
        base_seed=cpu_config.base_seed,
        n_trials=cpu_config.n_trials,
    )

    initial_conditions_batch = make_initial_conditions(
        config=cpu_config,
        seeds=seeds,
    )

    print("Running CPU validation...")
    cpu_summary, cpu_time = time_call(
        run_cpu_from_initial_conditions,
        cpu_config,
        initial_conditions_batch,
        seeds,
    )

    print_basin_summary(cpu_summary)
    print("CPU time:", cpu_time)

    print()
    print("Running fast GPU validation using same initial conditions...")
    gpu_summary, gpu_time = time_call(
        basin_stability_gpu_fast_from_initial_conditions,
        gpu_config,
        initial_conditions_batch,
        seeds,
    )

    print_basin_summary(gpu_summary)
    print("GPU time:", gpu_time)

    compare(
        cpu_summary=cpu_summary,
        gpu_summary=gpu_summary,
    )


if __name__ == "__main__":
    main()
