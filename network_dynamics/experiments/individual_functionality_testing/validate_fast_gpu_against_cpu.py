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
import os
import tempfile
from dataclasses import replace
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "matplotlib-generate-dynamics"),
)

import numpy as np
import networkx as nx

from network_dynamics.core.basin_common import sample_initial_conditions_batch
from network_dynamics.core.config import BasinConfig
from network_dynamics.core.sampling import trial_seeds
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


PLOT_DIR = Path(__file__).resolve().parent / "benchmark_outputs"


def make_config(backend):
    return BasinConfig(
        G=nx.path_graph(5),
        n_trials=10000,
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


def time_call(function, *args, **kwargs):
    start = time.perf_counter()
    value = function(*args, **kwargs)
    end = time.perf_counter()

    return value, end - start


def terminal_print(message):
    print(message, file=sys.__stdout__, flush=True)


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


def plot_timing_results(timing_rows, output_dir=PLOT_DIR):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    labels = [row["label"] for row in timing_rows]
    runtimes = [row["runtime"] for row in timing_rows]
    throughputs = [row["n_trials"] / row["runtime"] for row in timing_rows]

    runtime_path = output_dir / "validation_runtime_by_backend.png"
    throughput_path = output_dir / "validation_throughput_by_backend.png"

    plt.figure(figsize=(8, 5))
    plt.bar(labels, runtimes)
    plt.title("Validation Runtime by Backend")
    plt.xlabel("Backend")
    plt.ylabel("Runtime (seconds)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(runtime_path, dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(labels, throughputs)
    plt.title("Validation Throughput by Backend")
    plt.xlabel("Backend")
    plt.ylabel("Trials per second")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(throughput_path, dpi=200)
    plt.close()

    return runtime_path, throughput_path


def run_validation():
    cpu_config = make_config(backend="cpu")
    serial_config = replace(
        cpu_config,
        backend="serial",
        n_workers=1,
    )
    gpu_config = make_config(backend="gpu")

    seeds = trial_seeds(
        base_seed=cpu_config.base_seed,
        n_trials=cpu_config.n_trials,
    )

    initial_conditions_batch = sample_initial_conditions_batch(
        config=cpu_config,
        seeds=seeds,
    )

    terminal_print("Starting serial CPU validation...")
    print("Running serial CPU validation...")
    serial_summary, serial_time = time_call(
        basin_stability_cpu_from_initial_conditions,
        serial_config,
        initial_conditions_batch,
        seeds,
        progress_label="Serial CPU validation",
        progress_interval=10,
        progress_stream=sys.__stdout__,
    )

    terminal_print(f"Finished serial CPU validation in {serial_time:.3f} seconds.")
    print_basin_summary(serial_summary)
    print("Serial CPU time:", serial_time)
    print_first_five_errors(serial_summary, "serial CPU")

    print()
    terminal_print("Starting parallel CPU validation...")
    print("Running parallel CPU validation...")

    cpu_summary, cpu_time = time_call(
        basin_stability_cpu_from_initial_conditions,
        cpu_config,
        initial_conditions_batch,
        seeds,
        progress_label="Parallel CPU validation",
        progress_interval=100,
        progress_stream=sys.__stdout__,
    )

    terminal_print(f"Finished parallel CPU validation in {cpu_time:.3f} seconds.")
    print_basin_summary(cpu_summary)
    print("Parallel CPU time:", cpu_time)
    print_first_five_errors(cpu_summary, "parallel CPU")

    print()
    terminal_print("Starting fast GPU validation...")
    print("Running fast GPU validation using same initial conditions...")

    gpu_summary, gpu_time = time_call(
        basin_stability_gpu_fast_from_initial_conditions,
        gpu_config,
        initial_conditions_batch,
        seeds,
    )

    terminal_print(f"Finished fast GPU validation in {gpu_time:.3f} seconds.")
    print_basin_summary(gpu_summary)
    print("GPU time:", gpu_time)
    print_first_five_errors(gpu_summary, "GPU")

    print()
    print("Serial CPU vs fast GPU")
    print("=" * 40)
    compare(
        cpu_summary=serial_summary,
        gpu_summary=gpu_summary,
        sync_tol=cpu_config.sync_tol,
    )

    print()
    print("Parallel CPU vs fast GPU")
    print("=" * 40)
    compare(
        cpu_summary=cpu_summary,
        gpu_summary=gpu_summary,
        sync_tol=cpu_config.sync_tol,
    )

    timing_rows = [
        {
            "label": "Serial CPU",
            "runtime": serial_time,
            "n_trials": serial_summary.n_trials,
        },
        {
            "label": "Parallel CPU",
            "runtime": cpu_time,
            "n_trials": cpu_summary.n_trials,
        },
        {
            "label": "Fast GPU",
            "runtime": gpu_time,
            "n_trials": gpu_summary.n_trials,
        },
    ]
    runtime_path, throughput_path = plot_timing_results(timing_rows)

    print()
    print("Timing plots")
    print("-" * 40)
    print("Runtime plot:", runtime_path)
    print("Throughput plot:", throughput_path)
    terminal_print(f"Saved timing plots to {PLOT_DIR}")


def main():
    run_with_output_file(run_validation)
    print(f"Wrote validation output to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
