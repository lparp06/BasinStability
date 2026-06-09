

from network_dynamics.cpu.basin import basin_stability_serial, run_single_trial
import networkx as nx
import numpy as np

def print_basin_summary(summary):
    """
    Print a compact basin-stability summary.
    """
    print("Basin stability:", summary["basin_stability"])
    print("Number of trials:", summary["n_trials"])
    print("Successes:", summary["successes"])
    print("Sync failures:", summary["sync_failures"])
    print("Integration failures:", summary["integration_failures"])
    print("Trial seeds:", summary["trial_seeds"])
    print("Sync time mean:", summary["sync_time_mean"])


def main():
    G = nx.path_graph(5)

    # Shared settings for all tests
    base_settings = {
        "G": G,
        "base_seed": 42,
        "parameters": [0.2, 0.2, 7],
        "coupling_strength": 1.0,
        "H": None,
        "tmax": 150,
        "tstep": 0.05,
        "dimension": 3,
        "tol_max": 1e6,
        "window_fraction": 0.1,
        "sampler": "uniform",
        "store_initial_conditions": False,
        "max_abs_threshold": 1e6,
    }

    print()
    print("=" * 70)
    print("BASIN STABILITY TEST SUITE")
    print("=" * 70)

    # ============================================================
    # Test 0: single-trial smoke test
    # ============================================================

    print()
    print("=" * 70)
    print("Test 0: Single-trial smoke test")
    print("=" * 70)

    single_result = run_single_trial(
        G=G,
        trial_seed=42,
        parameters=[0.2, 0.2, 7],
        coupling_strength=1.0,
        H=None,
        tmax=150,
        tstep=0.05,
        dimension=3,
        sync_tol=1e-2,
        tol_max=1e6,
        window_fraction=0.1,
        sampler="uniform",
        sampling_bounds=[-2, 2],
        store_initial_condition=False,
    )

    print("Success:", single_result["success"])
    print("Final success:", single_result["final_success"])
    print("Integration failed:", single_result["integration_failed"])
    print("Trial seed:", single_result["trial_seed"])
    print("Sampler:", single_result["sampler"])

    if not single_result["integration_failed"]:
        print("Initial condition shape:", single_result["initial_condition_shape"])
        print("State dimension:", single_result["state_dimension"])
        print("Final distance:", single_result["final_distance"])
        print("Sync time:", single_result["sync_time"])
        print("Sync tolerance:", single_result["sync_tol"])
        print("Window fraction:", single_result["window_fraction"])
        print("Solution health:", single_result["solution_health"])
    else:
        print("Error:", single_result.get("error"))

    # ============================================================
    # Test 1: convergence check
    # ============================================================

    print()
    print("=" * 70)
    print("Test 1: Convergence check over n_trials")
    print("=" * 70)

    trial_counts = [10, 25, 50, 100]

    convergence_results = []

    for n_trials in trial_counts:
        summary = basin_stability_serial(
            **base_settings,
            n_trials=n_trials,
            sync_tol=1e-2,
            sampling_bounds=[-2, 2],
        )

        convergence_results.append(summary)

        print()
        print(f"n_trials = {n_trials}")
        print("-" * 30)
        print_basin_summary(summary)

    # ============================================================
    # Test 2: sync tolerance sensitivity
    # ============================================================

    print()
    print("=" * 70)
    print("Test 2: Sync tolerance sensitivity")
    print("=" * 70)

    sync_tolerances = [1e-2, 1e-3]

    tolerance_results = []

    for sync_tol in sync_tolerances:
        summary = basin_stability_serial(
            **base_settings,
            n_trials=50,
            sync_tol=sync_tol,
            sampling_bounds=[-2, 2],
        )

        tolerance_results.append(summary)

        print()
        print(f"sync_tol = {sync_tol}")
        print("-" * 30)
        print_basin_summary(summary)

    # ============================================================
    # Test 3: sampling bounds sensitivity
    # ============================================================

    print()
    print("=" * 70)
    print("Test 3: Sampling bounds sensitivity")
    print("=" * 70)

    bounds_list = [
        [-2, 2],
        [-5, 5],
        [-10, 10],
    ]

    bounds_results = []

    for bounds in bounds_list:
        summary = basin_stability_serial(
            **base_settings,
            n_trials=50,
            sync_tol=1e-2,
            sampling_bounds=bounds,
        )

        bounds_results.append(summary)

        print()
        print(f"sampling_bounds = {bounds}")
        print("-" * 30)
        print_basin_summary(summary)

    # ============================================================
    # Test 4: reproducibility check
    # ============================================================

    print()
    print("=" * 70)
    print("Test 4: Reproducibility check")
    print("=" * 70)

    summary_a = basin_stability_serial(
        **base_settings,
        n_trials=25,
        sync_tol=1e-2,
        sampling_bounds=[-2, 2],
    )

    summary_b = basin_stability_serial(
        **base_settings,
        n_trials=25,
        sync_tol=1e-2,
        sampling_bounds=[-2, 2],
    )

    same_counts = (
        summary_a["basin_stability"] == summary_b["basin_stability"]
        and summary_a["successes"] == summary_b["successes"]
        and summary_a["sync_failures"] == summary_b["sync_failures"]
        and summary_a["integration_failures"] == summary_b["integration_failures"]
    )

    same_trial_results = [
        a["success"] == b["success"]
        and a.get("sync_time") == b.get("sync_time")
        and np.isclose(a.get("final_distance", np.nan), b.get("final_distance", np.nan))
        for a, b in zip(summary_a["results"], summary_b["results"])
    ]

    print("Same aggregate counts:", same_counts)
    print("Same per-trial results:", all(same_trial_results))

    # ============================================================
    # Final summary table
    # ============================================================

    print()
    print("=" * 70)
    print("Compact Summary")
    print("=" * 70)

    print()
    print("Convergence check:")
    for summary in convergence_results:
        print(
            "n_trials:",
            summary["n_trials"],
            "| BS:",
            summary["basin_stability"],
            "| successes:",
            summary["successes"],
            "| sync failures:",
            summary["sync_failures"],
            "| integration failures:",
            summary["integration_failures"],
        )

    print()
    print("Tolerance sensitivity:")
    for summary in tolerance_results:
        print(
            "sync_tol:",
            summary["sync_tol"],
            "| BS:",
            summary["basin_stability"],
            "| successes:",
            summary["successes"],
            "| sync failures:",
            summary["sync_failures"],
            "| integration failures:",
            summary["integration_failures"],
        )

    print()
    print("Sampling bounds sensitivity:")
    for summary in bounds_results:
        print(
            "bounds:",
            summary["sampling_bounds"],
            "| BS:",
            summary["basin_stability"],
            "| successes:",
            summary["successes"],
            "| sync failures:",
            summary["sync_failures"],
            "| integration failures:",
            summary["integration_failures"],
        )

    print()
    print("=" * 70)
    print("Finished basin stability tests.")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""
======================================================================
BASIN STABILITY TEST SUITE
======================================================================

======================================================================
Test 0: Single-trial smoke test
======================================================================
Success: True
Final success: True
Integration failed: False
Trial seed: 42
Sampler: uniform
Initial condition shape: (15,)
State dimension: 15
Final distance: 0.000624544188182968
Sync time: 100.85000000000001
Sync tolerance: 0.01
Window fraction: 0.1
Solution health: {'contains_nan': False, 'contains_inf': False, 'max_abs_value': 33.33518085353503, 'exceeds_max_abs_threshold': False, 'max_abs_threshold': 1000000.0}

======================================================================
Test 1: Convergence check over n_trials
======================================================================

n_trials = 10
------------------------------
Basin stability: 0.8
Number of trials: 10
Successes: 8
Sync failures: 2
Integration failures: 0
Trial seeds: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
Sync time mean: 72.15

n_trials = 25
------------------------------
Basin stability: 0.84
Number of trials: 25
Successes: 21
Sync failures: 4
Integration failures: 0
Trial seeds: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66]
Sync time mean: 77.36904761904762

n_trials = 50
------------------------------
Basin stability: 0.92
Number of trials: 50
Successes: 46
Sync failures: 4
Integration failures: 0
Trial seeds: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91]
Sync time mean: 75.84239130434784

n_trials = 100
------------------------------
Basin stability: 0.95
Number of trials: 100
Successes: 95
Sync failures: 5
Integration failures: 0
Trial seeds: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141]
Sync time mean: 77.52315789473685

======================================================================
Test 2: Sync tolerance sensitivity
======================================================================

sync_tol = 0.01
------------------------------
Basin stability: 0.92
Number of trials: 50
Successes: 46
Sync failures: 4
Integration failures: 0
Trial seeds: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91]
Sync time mean: 75.84239130434784

sync_tol = 0.001
------------------------------
Basin stability: 0.34
Number of trials: 50
Successes: 17
Sync failures: 33
Integration failures: 0
Trial seeds: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91]
Sync time mean: 93.53823529411765

======================================================================
Test 3: Sampling bounds sensitivity
======================================================================

sampling_bounds = [-2, 2]
------------------------------
Basin stability: 0.92
Number of trials: 50
Successes: 46
Sync failures: 4
Integration failures: 0
Trial seeds: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91]
Sync time mean: 75.84239130434784

sampling_bounds = [-5, 5]
------------------------------
Basin stability: 0.62
Number of trials: 50
Successes: 31
Sync failures: 19
Integration failures: 0
Trial seeds: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91]
Sync time mean: 89.71451612903226

sampling_bounds = [-10, 10]
------------------------------
Basin stability: 0.26
Number of trials: 50
Successes: 13
Sync failures: 37
Integration failures: 0
Trial seeds: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91]
Sync time mean: 97.3923076923077

======================================================================
Test 4: Reproducibility check
======================================================================
Same aggregate counts: True
Same per-trial results: True

======================================================================
Compact Summary
======================================================================

Convergence check:
n_trials: 10 | BS: 0.8 | successes: 8 | sync failures: 2 | integration failures: 0
n_trials: 25 | BS: 0.84 | successes: 21 | sync failures: 4 | integration failures: 0
n_trials: 50 | BS: 0.92 | successes: 46 | sync failures: 4 | integration failures: 0
n_trials: 100 | BS: 0.95 | successes: 95 | sync failures: 5 | integration failures: 0

Tolerance sensitivity:
sync_tol: 0.01 | BS: 0.92 | successes: 46 | sync failures: 4 | integration failures: 0
sync_tol: 0.001 | BS: 0.34 | successes: 17 | sync failures: 33 | integration failures: 0

Sampling bounds sensitivity:
bounds: [-2, 2] | BS: 0.92 | successes: 46 | sync failures: 4 | integration failures: 0
bounds: [-5, 5] | BS: 0.62 | successes: 31 | sync failures: 19 | integration failures: 0
bounds: [-10, 10] | BS: 0.26 | successes: 13 | sync failures: 37 | integration failures: 0

======================================================================
Finished basin stability tests.
======================================================================"""