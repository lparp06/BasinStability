"""
compare_cpu_gpu_basin.py

Compare CPU RK4 basin stability and GPU/JAX RK4 basin stability.

This is a full basin-level comparison, not a tiny trajectory correctness test.

For this laptop/MPS setup:
- use dt=0.05 for the full basin comparison
- use smaller dt only in small correctness scripts
- accept small CPU/GPU basin-stability differences because CPU uses float64
  and GPU/MPS usually uses float32

Run from project root:

    python -m network_dynamics.experiments.compare_cpu_gpu_basin
"""

import time

import networkx as nx

from network_dynamics.core.config import BasinConfig
from network_dynamics.cpu.basin import (
    basin_stability_serial,
    print_basin_summary,
)
from network_dynamics.gpu.basin import basin_stability_gpu


def make_config(backend):
    """
    Create one shared experiment configuration.

    The only thing that changes between CPU and GPU is the backend label.
    Both use RK4 so that we are comparing similar integrators.
    """

    return BasinConfig(
        G=nx.path_graph(5),
        n_trials=2000,
        base_seed=42,
        parameters=(0.2, 0.2, 7.0),
        coupling_strength=1.0,
        H=None,
        tmax=150.0,
        dt=0.025,
        dimension=3,
        sampling_bounds=(-5.0, 5.0),
        sync_tol=1e-2,
        tol_max=1e6,
        window_fraction=0.1,
        max_abs_threshold=1e6,
        success_definition="window_success",
        integrator="RK4",
        backend=backend,
    ).validate()


def print_config(config):
    """
    Print the main experiment settings so we can verify what was run.
    """

    print()
    print("Experiment config")
    print("-" * 40)
    print("n_trials:", config.n_trials)
    print("base_seed:", config.base_seed)
    print("tmax:", config.tmax)
    print("dt:", config.dt)
    print("sampling_bounds:", config.sampling_bounds)
    print("sync_tol:", config.sync_tol)
    print("window_fraction:", config.window_fraction)
    print("success_definition:", config.success_definition)
    print("integrator:", config.integrator)


def time_run(function, config):
    """
    Time one basin-stability function call.
    """

    start = time.perf_counter()
    summary = function(config)
    end = time.perf_counter()

    elapsed_seconds = end - start

    return summary, elapsed_seconds


def compare_summaries(cpu_summary, gpu_summary):
    """
    Compare CPU and GPU basin summaries.

    With RK4, CPU and GPU should be close, but not necessarily identical.
    CPU NumPy usually uses float64. JAX on MPS usually uses float32.
    Tiny trajectory differences can flip strict window-success classification.

    Therefore, for this full run, we check:
    - basin stability difference
    - per-trial success matches
    - integration failures
    """

    allowed_basin_difference = 0.05

    print()
    print("Comparison")
    print("-" * 40)

    print("CPU basin stability:", cpu_summary.basin_stability)
    print("GPU basin stability:", gpu_summary.basin_stability)

    basin_difference = abs(
        cpu_summary.basin_stability - gpu_summary.basin_stability
    )

    print("Absolute basin stability difference:", basin_difference)
    print("Allowed basin stability difference:", allowed_basin_difference)

    print()
    print("CPU successes:", cpu_summary.successes)
    print("GPU successes:", gpu_summary.successes)

    print("CPU sync failures:", cpu_summary.sync_failures)
    print("GPU sync failures:", gpu_summary.sync_failures)

    print("CPU integration failures:", cpu_summary.integration_failures)
    print("GPU integration failures:", gpu_summary.integration_failures)

    cpu_successes = [
        result.success
        for result in cpu_summary.results
    ]

    gpu_successes = [
        result.success
        for result in gpu_summary.results
    ]

    matches = [
        cpu_success == gpu_success
        for cpu_success, gpu_success in zip(cpu_successes, gpu_successes)
    ]

    n_matches = sum(matches)
    n_trials = len(matches)

    print()
    print("Per-trial success matches:", n_matches, "/", n_trials)

    if basin_difference <= allowed_basin_difference:
        print("CPU and GPU basin stability are close enough for this RK4 comparison.")
    else:
        print("CPU and GPU basin stability differ more than expected.")

    if all(matches):
        print("CPU and GPU classifications match exactly.")
    else:
        print("Some CPU and GPU classifications differ.")
        print("That is acceptable if the basin-stability difference is small.")


def print_mismatched_trials(cpu_summary, gpu_summary):
    """
    Print trials where CPU and GPU success classifications differ.
    """

    print()
    print("Mismatched trials")
    print("-" * 40)

    mismatch_count = 0

    for cpu_result, gpu_result in zip(cpu_summary.results, gpu_summary.results):
        if cpu_result.success != gpu_result.success:
            mismatch_count += 1

            print(f"seed={cpu_result.trial_seed}")
            print(f"  CPU success: {cpu_result.success}")
            print(f"  GPU success: {gpu_result.success}")
            print(f"  CPU final_success: {cpu_result.final_success}")
            print(f"  GPU final_success: {gpu_result.final_success}")
            print(f"  CPU window_success: {cpu_result.window_success}")
            print(f"  GPU window_success: {gpu_result.window_success}")
            print(f"  CPU final_distance: {cpu_result.final_distance}")
            print(f"  GPU final_distance: {gpu_result.final_distance}")
            print(f"  CPU window_max_distance: {cpu_result.window_max_distance}")
            print(f"  GPU window_max_distance: {gpu_result.window_max_distance}")
            print()

    if mismatch_count == 0:
        print("No mismatches.")
    else:
        print(f"Total mismatches: {mismatch_count}")


def main():
    cpu_config = make_config(backend="serial")
    gpu_config = make_config(backend="gpu")

    print_config(cpu_config)

    print()
    print("=" * 70)
    print("CPU RK4 basin")
    print("=" * 70)

    cpu_summary, cpu_time = time_run(
        basin_stability_serial,
        cpu_config,
    )

    print_basin_summary(cpu_summary)
    print("CPU runtime:", cpu_time)

    print()
    print("First 5 CPU trial errors")
    print("-" * 40)

    for result in cpu_summary.results[:5]:
        print("seed:", result.trial_seed)
        print("integration_failed:", result.integration_failed)
        print("error:", result.error)
        print()

    print()
    print("=" * 70)
    print("GPU/JAX RK4 basin")
    print("=" * 70)

    gpu_summary, gpu_time = time_run(
        basin_stability_gpu,
        gpu_config,
    )

    print_basin_summary(gpu_summary)
    print("GPU runtime:", gpu_time)

    compare_summaries(
        cpu_summary=cpu_summary,
        gpu_summary=gpu_summary,
    )

    print_mismatched_trials(
        cpu_summary=cpu_summary,
        gpu_summary=gpu_summary,
    )

    


if __name__ == "__main__":
    main()