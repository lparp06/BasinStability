"""
Checks synchronization

BS = M / T
    M is the number of initial conditions
    that successfully arrive at the desired stable state
   
    T is the total number of initial conditions drawn from
    a designated bounded subset of the phase space
"""

import numpy as np
import networkx as nx
from sampling import sample_uniform_initial_condition, trial_seeds
from integration import integrate
from sync import is_synchronized_over_win, is_synchronized_final, final_max_pwd, time_to_sync 

def run_single_trial(
    G=nx.path_graph(5),
    trial_seed=42,
    parameters=[0.2, 0.2, 7],
    coupling_strength=1.0,
    H=None,
    tmax=150,
    tstep=0.05,
    dimension=3,
    sync_tol=1e-3,
    tol_max=1e6,
    window_fraction=0.1,
    sampler="uniform",
    sampling_bounds=[-10, 10],
):
    """ 
    For one initial condition, determine whether the system synchronizes.

    Returns a dictionary containing:
    - success
    - whether integration failed
    - seed used
    - initial condition
    - final max pairwise distance
    - sync time
    """

    rng = np.random.default_rng(trial_seed)

    n_nodes = G.number_of_nodes()
    state_dimension = n_nodes * dimension

    if sampler == "uniform":
        initial_condition = sample_uniform_initial_condition(
            rng,
            n_nodes,
            dimension,
            sampling_bounds[0],
            sampling_bounds[1],
        )
    else:
        raise ValueError(f"Unknown sampler: {sampler}")

    try:
        sol, t = integrate(
            G=G,
            initial_conditions=initial_condition,
            parameters=parameters,
            coupling_strength=coupling_strength,
            H=H,
            tmax=tmax,
            timestep=tstep,
        )

        final_distance = final_max_pwd(sol, dimension)

        sync_time = time_to_sync(
            sol=sol,
            t=t,
            dimension=dimension,
            tol=sync_tol,
            tol_max=tol_max,
        )

        success = is_synchronized_over_win(
            sol=sol,
            dimension=dimension,
            tol=sync_tol,
            win_frac=window_fraction,
        )

        result = {
            "success": success,
            "integration_failed": False,
            "trial_seed": trial_seed,
            "sampler": sampler,
            "initial_condition": initial_condition,
            "initial_condition_shape": initial_condition.shape,
            "state_dimension": state_dimension,
            "final_distance": final_distance,
            "sync_time": sync_time,
            "sync_tol": sync_tol,
            "tol_max": tol_max,
            "window_fraction": window_fraction,
            "coupling_strength": coupling_strength,
            "tmax": tmax,
            "tstep": tstep,
        }

    except Exception as error:
        result = {
            "success": False,
            "integration_failed": True,
            "trial_seed": trial_seed,
            "sampler": sampler,
            "error": str(error),
            "sync_tol": sync_tol,
            "tol_max": tol_max,
            "window_fraction": window_fraction,
            "coupling_strength": coupling_strength,
            "tmax": tmax,
            "tstep": tstep,
        }

    return result

def basin_stability_serial(
    G,
    n_trials=25,
    base_seed=42,
    parameters=[0.2, 0.2, 7],
    coupling_strength=1,
    H=None,
    tmax=150,
    tstep=0.05,
    dimension=3,
    sync_tol=1e-2,
    tol_max=1e6,
    window_fraction=0.1,
    sampler="uniform",
    sampling_bounds=[-10, 10],
):
    
    seeds = trial_seeds(base_seed=base_seed, n_trials=n_trials)

    results = []
    successes = 0
    int_failures = 0

    for seed in seeds:
        trial = run_single_trial(
            G=G,
            trial_seed=seed,
            parameters=parameters,
            coupling_strength=coupling_strength,
            H=H,
            tmax=tmax,
            tstep=tstep,
            dimension=dimension,
            sync_tol=sync_tol,
            tol_max=tol_max,
            window_fraction=window_fraction,
            sampler=sampler,
            sampling_bounds=sampling_bounds,
        )

        results.append(trial)

    sync_time_mean = 0

    for result in results:
        if result["success"] is True:
            successes += 1
            sync_time_mean += result["sync_time"]

        if result["integration_failed"] is True:
            int_failures += 1

    sync_failures = n_trials - successes - int_failures

    bs_value = successes / n_trials

    sync_time_mean = sync_time_mean / successes



    basin_stability_results = {
        "basin_stability": bs_value,
        "n_trials": n_trials,
        "successes": successes,
        "sync_failures": sync_failures,
        "integration_failures": int_failures,
        "base_seed": base_seed,
        "trial_seeds": seeds,
        "sync_tol": sync_tol,
        "tol_max": tol_max,
        "window_fraction": window_fraction,
        "sampling_bounds": sampling_bounds,
        "coupling_strength": coupling_strength,
        "parameters": parameters,
        "sync_time_mean": sync_time_mean,
        "results": results,
    }

    return basin_stability_results

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
    }

    print()
    print("=" * 70)
    print("BASIN STABILITY TEST SUITE")
    print("=" * 70)

    # ============================================================
    # Test 1: Convergence check
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
        print("Basin stability:", summary["basin_stability"])
        print("Successes:", summary["successes"])
        print("Sync failures:", summary["sync_failures"])
        print("Integration failures:", summary["integration_failures"])
        print("Mean sync time:", summary["sync_time_mean"])

    # ============================================================
    # Test 2: Sync tolerance sensitivity
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
        print("Basin stability:", summary["basin_stability"])
        print("Successes:", summary["successes"])
        print("Sync failures:", summary["sync_failures"])
        print("Integration failures:", summary["integration_failures"])
        print("Mean sync time:", summary["sync_time_mean"])

    # ============================================================
    # Test 3: Sampling bounds sensitivity
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
        print("Basin stability:", summary["basin_stability"])
        print("Successes:", summary["successes"])
        print("Sync failures:", summary["sync_failures"])
        print("Integration failures:", summary["integration_failures"])
        print("Mean sync time:", summary["sync_time_mean"])

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