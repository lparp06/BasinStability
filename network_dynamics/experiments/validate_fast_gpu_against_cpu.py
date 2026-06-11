"""
validate_fast_gpu_against_cpu.py

Validate fast GPU basin calculations against CPU RK4 using the exact same
initial conditions.

This version is designed for the first_crossing synchronization definition:

    success = True if max_pairwise_distance < sync_tol at any sampled time.

For first_crossing, the most important diagnostics are:
- sync_time
- min_distance over the full trajectory

Run from project root:

    python -m network_dynamics.experiments.validate_fast_gpu_against_cpu
"""

import time
import sys

import numpy as np
import networkx as nx

from network_dynamics.core.config import BasinConfig
from network_dynamics.core.sampling import sample_uniform_initial_condition, trial_seeds
from network_dynamics.cpu.basin import (
    basin_stability_cpu_from_initial_conditions,
    print_basin_summary,
)
from network_dynamics.experiments.experiment_io import (
    OUTPUT_PATH,
    run_with_output_file,
)
from network_dynamics.gpu.basin_fast import (
    basin_stability_gpu_fast_from_initial_conditions,
)


def make_config(backend):
    return BasinConfig(
        G=nx.path_graph(5),
        n_trials=5000,
        base_seed=42,
        parameters=(0.2, 0.2, 7.0),
        coupling_strength=1.0,
        H=None,
        tmax=5000.0,
        dt=0.05,
        dimension=3,
        sampling_bounds=(-5.0, 5.0),
        sync_tol=1e-3,
        tol_max=1e6,
        window_fraction=0.2,
        max_abs_threshold=1e6,
        success_definition="first_crossing",
        integrator="RK4",
        backend=backend,
        n_workers=4,
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


def time_call(function, *args, **kwargs):
    start = time.perf_counter()
    value = function(*args, **kwargs)
    end = time.perf_counter()

    return value, end - start


def extract_min_distances(summary):
    return np.asarray(
        [
            np.inf if result.min_distance is None else result.min_distance
            for result in summary.results
        ],
        dtype=float,
    )


def summarize_borderline(cpu_min_distances, gpu_min_distances, sync_tol):
    border_width = 1e-3

    cpu_borderline = np.abs(cpu_min_distances - sync_tol) < border_width
    gpu_borderline = np.abs(gpu_min_distances - sync_tol) < border_width

    either_borderline = cpu_borderline | gpu_borderline

    print()
    print("Borderline first-crossing diagnostics")
    print("-" * 40)
    print("sync_tol:", sync_tol)
    print("border_width:", border_width)
    print("CPU borderline count:", int(np.sum(cpu_borderline)))
    print("GPU borderline count:", int(np.sum(gpu_borderline)))
    print("Either borderline count:", int(np.sum(either_borderline)))


def compare(cpu_summary, gpu_summary, sync_tol):
    cpu_min_distances = extract_min_distances(cpu_summary)
    gpu_min_distances = extract_min_distances(gpu_summary)

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

    summarize_borderline(
        cpu_min_distances=cpu_min_distances,
        gpu_min_distances=gpu_min_distances,
        sync_tol=sync_tol,
    )

    mismatch_mask = np.asarray(
        [
            cpu_result.success != gpu_result.success
            for cpu_result, gpu_result in zip(cpu_summary.results, gpu_summary.results)
        ],
        dtype=bool,
    )
    mismatch_count = int(np.sum(mismatch_mask))

    print()
    print("Mismatch summary")
    print("-" * 40)
    print("Total mismatches:", mismatch_count)

    if mismatch_count:
        min_distance_differences = np.abs(
            cpu_min_distances[mismatch_mask] - gpu_min_distances[mismatch_mask]
        )
        print(
            "Mean mismatched min-distance difference:",
            float(np.mean(min_distance_differences)),
        )
        print(
            "Max mismatched min-distance difference:",
            float(np.max(min_distance_differences)),
        )


def print_first_five_errors(summary, label):
    error_results = [
        result
        for result in summary.results
        if result.integration_failed or result.error is not None
    ]

    if not error_results:
        return

    print()
    print(f"First 5 {label} errors")
    print("-" * 40)

    for result in error_results[:5]:
        print("seed:", result.trial_seed)
        print("integration_failed:", result.integration_failed)
        print("error:", result.error)
        print()


def run_validation():
    cpu_config = make_config(backend="cpu")
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
        basin_stability_cpu_from_initial_conditions,
        cpu_config,
        initial_conditions_batch,
        seeds,
        progress_label="CPU validation",
        progress_interval=100,
        progress_stream=sys.__stdout__,
    )

    print_basin_summary(cpu_summary)
    print("CPU time:", cpu_time)
    print_first_five_errors(cpu_summary, "CPU")

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
    print_first_five_errors(gpu_summary, "GPU")

    compare(
        cpu_summary=cpu_summary,
        gpu_summary=gpu_summary,
        sync_tol=cpu_config.sync_tol,
    )


def main():
    run_with_output_file(run_validation)
    print(f"Wrote validation output to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
