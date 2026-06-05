"""
test_basin_timing.py

Compare serial vs CPU-parallel runtime for basin stability.

Run from project root:

    python3 -m tests.test_basin_timing
"""

import time
import networkx as nx

from network_dynamics.basin import basin_stability_serial, basin_stability_cpu


def time_run(label, function, **kwargs):
    start = time.perf_counter()
    summary = function(**kwargs)
    elapsed = time.perf_counter() - start

    print()
    print(label)
    print("-" * len(label))
    print("Elapsed seconds:", elapsed)
    print("Basin stability:", summary["basin_stability"])
    print("Successes:", summary["successes"])
    print("Sync failures:", summary["sync_failures"])
    print("Integration failures:", summary["integration_failures"])

    return summary, elapsed


def main():
    G = nx.path_graph(5)

    base_settings = {
        "G": G,
        "base_seed": 42,
        "parameters": [0.2, 0.2, 7],
        "coupling_strength": 1.0,
        "H": None,
        "tmax": 150,
        "tstep": 0.05,
        "dimension": 3,
        "sync_tol": 1e-2,
        "tol_max": 1e6,
        "window_fraction": 0.1,
        "sampler": "uniform",
        "sampling_bounds": [-2, 2],
        "store_initial_conditions": False,
        "max_abs_threshold": 1e6,
    }

    trial_counts = [250]
    worker_counts = [6]

    print("=" * 70)
    print("BASIN STABILITY TIMING TEST")
    print("=" * 70)

    for n_trials in trial_counts:
        print()
        print("=" * 70)
        print(f"n_trials = {n_trials}")
        print("=" * 70)

        settings = {
            **base_settings,
            "n_trials": n_trials,
        }

        serial_summary, serial_time = time_run(
            "Serial",
            basin_stability_serial,
            **settings,
        )

        for n_workers in worker_counts:
            cpu_summary, cpu_time = time_run(
                f"CPU parallel, n_workers={n_workers}",
                basin_stability_cpu,
                **settings,
                n_workers=n_workers,
            )

            speedup = serial_time / cpu_time

            same_result = (
                serial_summary["basin_stability"] == cpu_summary["basin_stability"]
                and serial_summary["successes"] == cpu_summary["successes"]
                and serial_summary["sync_failures"] == cpu_summary["sync_failures"]
                and serial_summary["integration_failures"] == cpu_summary["integration_failures"]
            )

            print("Speedup:", speedup)
            print("Matches serial:", same_result)

    print()
    print("=" * 70)
    print("Timing test complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()