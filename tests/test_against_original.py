"""
test_against_original.py

Compare the new modular network_dynamics package against the original
GenerateDynamics.py behavior.

Run from project root:

    python3 -m tests.test_against_original
"""

import numpy as np
import networkx as nx

from GenerateDynamics import laplacian_dynamics

from network_dynamics.graphs import graph_laplacian
from network_dynamics.integration import integrate
from network_dynamics.sync import (
    final_max_pwd,
    time_to_sync,
)
from network_dynamics.basin import (
    basin_stability_serial,
    basin_stability_cpu,
)


def print_check(name, passed):
    status = "PASS" if passed else "FAIL"
    print(f"{name}: {status}")


def compare_laplacian():
    print()
    print("=" * 70)
    print("Test 1: Laplacian comparison")
    print("=" * 70)

    G = nx.path_graph(5)

    original = laplacian_dynamics()
    original_L = np.array(original.convert_graph_to_laplacian(G), dtype=float)

    new_L = np.array(graph_laplacian(G), dtype=float)

    print("Original L:")
    print(original_L)

    print("New L:")
    print(new_L)

    same_shape = original_L.shape == new_L.shape
    same_values = np.allclose(original_L, new_L)

    print_check("Same Laplacian shape", same_shape)
    print_check("Same Laplacian values", same_values)

    return same_shape and same_values


def compare_single_rossler_run():
    print()
    print("=" * 70)
    print("Test 2: Single Rössler trajectory comparison")
    print("=" * 70)

    G = nx.path_graph(5)

    parameters = [0.2, 0.2, 7]
    coupling_strength = 1.0
    tmax = 150
    timestep = 0.05
    dimension = 3

    np.random.seed(42)
    initial_condition = np.random.normal(0, 1, G.number_of_nodes() * dimension)

    original = laplacian_dynamics()

    original_sol, original_t = original.continuous_time_nonlinear_dynamics(
        G=G,
        tmax=tmax,
        timestep=timestep,
        init_cond=initial_condition.copy(),
        dynamics_type="Rossler",
        dynamics_params=parameters,
        coupling_strength=coupling_strength,
    )

    new_sol, new_t = integrate(
        G=G,
        initial_conditions=initial_condition.copy(),
        parameters=parameters,
        coupling_strength=coupling_strength,
        H=None,
        tmax=tmax,
        timestep=timestep,
    )

    print("Original sol.shape:", original_sol.shape)
    print("New sol.shape:", new_sol.shape)
    print("Original t.shape:", original_t.shape)
    print("New t.shape:", new_t.shape)

    same_sol_shape = original_sol.shape == new_sol.shape
    same_t_shape = original_t.shape == new_t.shape
    same_t_values = np.allclose(original_t, new_t)
    same_initial_condition = np.allclose(new_sol[0], initial_condition)

    print_check("Same solution shape", same_sol_shape)
    print_check("Same time shape", same_t_shape)
    print_check("Same time values", same_t_values)
    print_check("New solution preserves initial condition", same_initial_condition)

    original_final_distance = final_max_pwd(original_sol, dimension=dimension)
    new_final_distance = final_max_pwd(new_sol, dimension=dimension)

    print("Original final max pairwise distance:", original_final_distance)
    print("New final max pairwise distance:", new_final_distance)

    final_distances_reasonably_close = np.isclose(
        original_final_distance,
        new_final_distance,
        rtol=1e-1,
        atol=1e-3,
    )

    print_check(
        "Final synchronization distances reasonably close",
        final_distances_reasonably_close,
    )

    original_sync_time = original.nonlinear_find_time_to_sync(
        x=original_sol,
        t=original_t,
        d=dimension,
        criterion="maxpdist",
        Tol=1e-3,
        TolMax=1e6,
    )

    new_sync_time = time_to_sync(
        sol=new_sol,
        t=new_t,
        dimension=dimension,
        tol=1e-3,
        tol_max=1e6,
    )

    print("Original sync time at 1e-3:", original_sync_time)
    print("New sync time at 1e-3:", new_sync_time)

    both_finite = np.isfinite(original_sync_time) and np.isfinite(new_sync_time)
    both_inf = np.isinf(original_sync_time) and np.isinf(new_sync_time)

    if both_finite:
        sync_times_reasonably_close = np.isclose(
            original_sync_time,
            new_sync_time,
            rtol=0.25,
            atol=10,
        )
    else:
        sync_times_reasonably_close = both_inf

    print_check("Sync times reasonably close", sync_times_reasonably_close)

    return (
        same_sol_shape
        and same_t_shape
        and same_t_values
        and same_initial_condition
    )


def compare_saved_baseline_sync():
    print()
    print("=" * 70)
    print("Test 3: New sync.py against saved original baseline")
    print("=" * 70)

    try:
        sol = np.load("baseline_outputs/rossler_sol.npy")
        t = np.load("baseline_outputs/rossler_t.npy")
    except FileNotFoundError:
        print("Could not find baseline_outputs/rossler_sol.npy or rossler_t.npy")
        print("Run your original baseline script first.")
        return False

    final_distance = final_max_pwd(sol, dimension=3)

    sync_1e1 = time_to_sync(sol, t, dimension=3, tol=1e-1, tol_max=1e6)
    sync_1e2 = time_to_sync(sol, t, dimension=3, tol=1e-2, tol_max=1e6)
    sync_1e3 = time_to_sync(sol, t, dimension=3, tol=1e-3, tol_max=1e6)
    sync_1e4 = time_to_sync(sol, t, dimension=3, tol=1e-4, tol_max=1e6)

    print("Final max pairwise distance:", final_distance)
    print("Expected approximately: 0.000361")

    print("Sync time tol=1e-1:", sync_1e1)
    print("Expected approximately: 50.35")

    print("Sync time tol=1e-2:", sync_1e2)
    print("Expected approximately: 84.95")

    print("Sync time tol=1e-3:", sync_1e3)
    print("Expected approximately: 124.4")

    print("Sync time tol=1e-4:", sync_1e4)
    print("Expected: inf")

    checks = [
        np.isclose(final_distance, 0.00036113659678454086, rtol=1e-2, atol=1e-6),
        np.isclose(sync_1e1, 50.35),
        np.isclose(sync_1e2, 84.95),
        np.isclose(sync_1e3, 124.4),
        np.isinf(sync_1e4),
    ]

    print_check("Baseline sync metrics match expected values", all(checks))

    return all(checks)


def compare_serial_and_cpu_basin():
    print()
    print("=" * 70)
    print("Test 4: Serial basin stability vs CPU basin stability")
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

    serial = basin_stability_serial(**settings)

    cpu = basin_stability_cpu(
        **settings,
        n_workers=2,
    )

    print("Serial BS:", serial["basin_stability"])
    print("CPU BS:", cpu["basin_stability"])

    same_aggregate = (
        np.isclose(serial["basin_stability"], cpu["basin_stability"])
        and serial["successes"] == cpu["successes"]
        and serial["sync_failures"] == cpu["sync_failures"]
        and serial["integration_failures"] == cpu["integration_failures"]
    )

    same_trials = True

    for s_trial, c_trial in zip(serial["results"], cpu["results"]):
        if s_trial["trial_seed"] != c_trial["trial_seed"]:
            same_trials = False

        if s_trial["success"] != c_trial["success"]:
            same_trials = False

        if s_trial["integration_failed"] != c_trial["integration_failed"]:
            same_trials = False

        s_dist = s_trial.get("final_distance", np.inf)
        c_dist = c_trial.get("final_distance", np.inf)

        if not np.isclose(s_dist, c_dist):
            same_trials = False

    print_check("Same aggregate basin result", same_aggregate)
    print_check("Same per-trial results", same_trials)

    return same_aggregate and same_trials


def main():
    print("=" * 70)
    print("NEW MODULAR CODE VS ORIGINAL PROGRAM TESTS")
    print("=" * 70)

    results = []

    results.append(compare_laplacian())
    results.append(compare_single_rossler_run())
    results.append(compare_saved_baseline_sync())
    results.append(compare_serial_and_cpu_basin())

    print()
    print("=" * 70)
    print("Final result")
    print("=" * 70)

    if all(results):
        print("PASS: New modular code matches the original behavior for tested functionality.")
    else:
        print("CHECK: At least one comparison failed or needs review.")


if __name__ == "__main__":
    main()