

from network_dynamics.basin import basin_stability_serial, run_single_trial
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
