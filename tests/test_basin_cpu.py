"""
test_basin_cpu.py

Test that CPU-parallel basin stability matches the serial reference.

"""

import numpy as np
import networkx as nx
import matplotlib as plt

from network_dynamics.basin import basin_stability_serial, basin_stability_cpu


def compare_trial_results(serial_summary, cpu_summary):
    """
    Compare per-seed serial and CPU-parallel trial results.

    Returns True if all trial-level results match.
    """

    serial_results = serial_summary["results"]
    cpu_results = cpu_summary["results"]

    if len(serial_results) != len(cpu_results):
        print("Different number of trial results.")
        print("Serial:", len(serial_results))
        print("CPU:", len(cpu_results))
        return False

    all_match = True

    for serial_trial, cpu_trial in zip(serial_results, cpu_results):
        seed_match = serial_trial["trial_seed"] == cpu_trial["trial_seed"]
        success_match = serial_trial["success"] == cpu_trial["success"]
        final_success_match = serial_trial.get("final_success") == cpu_trial.get("final_success")
        integration_match = serial_trial["integration_failed"] == cpu_trial["integration_failed"]

        serial_sync_time = serial_trial.get("sync_time", np.inf)
        cpu_sync_time = cpu_trial.get("sync_time", np.inf)

        if np.isinf(serial_sync_time) and np.isinf(cpu_sync_time):
            sync_time_match = True
        else:
            sync_time_match = np.isclose(serial_sync_time, cpu_sync_time)

        serial_final_distance = serial_trial.get("final_distance", np.inf)
        cpu_final_distance = cpu_trial.get("final_distance", np.inf)

        if np.isinf(serial_final_distance) and np.isinf(cpu_final_distance):
            final_distance_match = True
        else:
            final_distance_match = np.isclose(serial_final_distance, cpu_final_distance)

        trial_match = (
            seed_match
            and success_match
            and final_success_match
            and integration_match
            and sync_time_match
            and final_distance_match
        )

        if not trial_match:
            all_match = False

            print()
            print("Mismatch found")
            print("--------------")
            print("Seed match:", seed_match)
            print("Serial seed:", serial_trial["trial_seed"])
            print("CPU seed:", cpu_trial["trial_seed"])
            print("Serial success:", serial_trial["success"])
            print("CPU success:", cpu_trial["success"])
            print("Serial final_success:", serial_trial.get("final_success"))
            print("CPU final_success:", cpu_trial.get("final_success"))
            print("Serial integration_failed:", serial_trial["integration_failed"])
            print("CPU integration_failed:", cpu_trial["integration_failed"])
            print("Serial sync_time:", serial_sync_time)
            print("CPU sync_time:", cpu_sync_time)
            print("Serial final_distance:", serial_final_distance)
            print("CPU final_distance:", cpu_final_distance)

    return all_match


def compare_summary_results(serial_summary, cpu_summary):
    """
    Compare aggregate serial and CPU-parallel basin-stability summaries.

    Returns True if aggregate results match.
    """

    checks = {
        "basin_stability": np.isclose(
            serial_summary["basin_stability"],
            cpu_summary["basin_stability"],
        ),
        "successes": serial_summary["successes"] == cpu_summary["successes"],
        "sync_failures": serial_summary["sync_failures"] == cpu_summary["sync_failures"],
        "integration_failures": serial_summary["integration_failures"] == cpu_summary["integration_failures"],
        "trial_seeds": serial_summary["trial_seeds"] == cpu_summary["trial_seeds"],
        "sync_time_mean": np.isclose(
            serial_summary["sync_time_mean"],
            cpu_summary["sync_time_mean"],
        ),
    }

    print()
    print("Aggregate comparison")
    print("--------------------")

    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    return all(checks.values())


def print_compact_summary(label, summary):
    """
    Print a compact basin-stability summary.
    """

    print()
    print(label)
    print("-" * len(label))
    print("Backend:", summary.get("backend"))
    print("Workers:", summary.get("n_workers"))
    print("Basin stability:", summary["basin_stability"])
    print("Number of trials:", summary["n_trials"])
    print("Successes:", summary["successes"])
    print("Sync failures:", summary["sync_failures"])
    print("Integration failures:", summary["integration_failures"])
    print("Mean sync time:", summary["sync_time_mean"])


def print_trial_comparison(serial_summary, cpu_summary):
    """
    Print per-trial serial vs CPU comparison.
    """

    print()
    print("Trial-by-trial comparison")
    print("-------------------------")

    for serial_trial, cpu_trial in zip(serial_summary["results"], cpu_summary["results"]):
        print(
            "Seed:",
            serial_trial["trial_seed"],
            "| Serial success:",
            serial_trial["success"],
            "| CPU success:",
            cpu_trial["success"],
            "| Serial sync_time:",
            serial_trial.get("sync_time"),
            "| CPU sync_time:",
            cpu_trial.get("sync_time"),
            "| Serial final_distance:",
            serial_trial.get("final_distance"),
            "| CPU final_distance:",
            cpu_trial.get("final_distance"),
        )


def main():
    print("=" * 70)
    print("CPU PARALLEL BASIN STABILITY TEST")
    print("=" * 70)

    G = nx.path_graph(5)

    settings = {
        "G": G,
        "n_trials": 10,
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

    print()
    print("Running serial reference...")

    serial_summary = basin_stability_serial(**settings)

    print_compact_summary("Serial summary", serial_summary)

    print()
    print("Running CPU-parallel version...")

    cpu_summary = basin_stability_cpu(
        **settings,
        n_workers=2,
    )

    print_compact_summary("CPU summary", cpu_summary)

    summary_match = compare_summary_results(serial_summary, cpu_summary)
    trials_match = compare_trial_results(serial_summary, cpu_summary)

    print()
    print("=" * 70)
    print("Final comparison")
    print("=" * 70)
    print("Aggregate summary match:", summary_match)
    print("Per-trial results match:", trials_match)

    if summary_match and trials_match:
        print("PASS: CPU-parallel results match serial results.")
    else:
        print("FAIL: CPU-parallel results do not match serial results.")

    print_trial_comparison(serial_summary, cpu_summary)


if __name__ == "__main__":
    main()
