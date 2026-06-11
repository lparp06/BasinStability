"""
run_fast_gpu_basin.py

Run the fast GPU/JAX basin-stability implementation.

This version uses:

    network_dynamics.gpu.basin_fast

The fast GPU version:
- samples initial conditions using JAX
- integrates on the GPU
- computes synchronization metrics on the GPU
- returns only compact results to the CPU

Run from project root:

    python -m network_dynamics.experiments.run_fast_gpu_basin
"""

import time

import networkx as nx

from network_dynamics.core.config import BasinConfig
from network_dynamics.cpu.basin import print_basin_summary
from network_dynamics.gpu.basin_fast import basin_stability_gpu_fast


def make_config():
    """
    Create the fast GPU basin-stability configuration.
    """

    return BasinConfig(
        G=nx.path_graph(5),
        n_trials=2000,
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
        backend="gpu",
    ).validate()


def print_config(config):
    """
    Print the main experiment settings.
    """

    print()
    print("Fast GPU experiment config")
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
    print("backend:", config.backend)


def time_run(config):
    """
    Time the fast GPU basin run.
    """

    start = time.perf_counter()

    summary = basin_stability_gpu_fast(config)

    end = time.perf_counter()

    elapsed_seconds = end - start

    return summary, elapsed_seconds


def print_first_trials(summary, n=10):
    """
    Print a few compact trial results.
    """

    print()
    print(f"First {n} trial results")
    print("-" * 40)

    for result in summary.results[:n]:
        print(
            f"seed={result.trial_seed} | "
            f"success={result.success} | "
            f"final_success={result.final_success} | "
            f"window_success={result.window_success} | "
            f"integration_failed={result.integration_failed} | "
            f"final_distance={result.final_distance} | "
            f"window_max_distance={result.window_max_distance} | "
            f"sync_time={result.sync_time} | "
            f"error={result.error}"
        )


def main():
    config = make_config()

    print_config(config)

    print()
    print("=" * 70)
    print("Fast GPU/JAX basin")
    print("=" * 70)

    summary, runtime = time_run(config)

    print_basin_summary(summary)
    print("Fast GPU runtime:", runtime)

    print_first_trials(
        summary=summary,
        n=10,
    )


if __name__ == "__main__":
    main()
