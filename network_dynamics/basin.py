"""
basin.py

Estimate basin stability for coupled oscillator networks.

Basin stability:
    BS = M / T

where:
    M = number of sampled initial conditions that successfully arrive
        at the desired stable/synchronized state

    T = total number of initial conditions drawn from a designated
        bounded subset of phase space
"""

import numpy as np
import networkx as nx

from .sampling import sample_uniform_initial_condition, trial_seeds
from .integration import integrate
from .sync import (
    is_synchronized_over_win,
    is_synchronized_final,
    final_max_pwd,
    time_to_sync,
)


def solution_health(sol, max_abs_threshold=1e6):
    """
    Check whether an integrated solution looks numerically healthy.

    Returns a dictionary with basic diagnostics.
    """
    contains_nan = bool(np.isnan(sol).any())
    contains_inf = bool(np.isinf(sol).any())

    if contains_nan or contains_inf:
        max_abs_value = np.inf
    else:
        max_abs_value = float(np.max(np.abs(sol)))

    return {
        "contains_nan": contains_nan,
        "contains_inf": contains_inf,
        "max_abs_value": max_abs_value,
        "exceeds_max_abs_threshold": bool(max_abs_value > max_abs_threshold),
        "max_abs_threshold": max_abs_threshold,
    }


def run_single_trial(
    G=None,
    trial_seed=42,
    parameters=None,
    coupling_strength=1.0,
    H=None,
    tmax=150,
    tstep=0.05,
    dimension=3,
    sync_tol=1e-3,
    tol_max=1e6,
    window_fraction=0.1,
    sampler="uniform",
    sampling_bounds=None,
    store_initial_condition=True,
    max_abs_threshold=1e6,
):
    """ 
    For one initial condition, determine whether the system synchronizes.

    Returns a dictionary containing:
    - success
    - whether integration failed
    - seed used
    - initial condition information
    - final max pairwise distance
    - sync time
    - solution health diagnostics
    """

    if G is None:
        G = nx.path_graph(5)

    if parameters is None:
        parameters = [0.2, 0.2, 7]

    if sampling_bounds is None:
        sampling_bounds = [-10, 10]

    rng = np.random.default_rng(trial_seed)

    n_nodes = G.number_of_nodes()
    state_dimension = n_nodes * dimension

    try:
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

        sol, t = integrate(
            G=G,
            initial_conditions=initial_condition,
            parameters=parameters,
            coupling_strength=coupling_strength,
            H=H,
            tmax=tmax,
            timestep=tstep,
        )

        health = solution_health(sol, max_abs_threshold=max_abs_threshold)

        if (
            health["contains_nan"]
            or health["contains_inf"]
            or health["exceeds_max_abs_threshold"]
        ):
            result = {
                "success": False,
                "integration_failed": True,
                "solution_invalid": True,
                "trial_seed": trial_seed,
                "sampler": sampler,
                "initial_condition_shape": initial_condition.shape,
                "state_dimension": state_dimension,
                "solution_health": health,
                "error": "Solution failed health check.",
                "sync_tol": sync_tol,
                "tol_max": tol_max,
                "window_fraction": window_fraction,
                "coupling_strength": coupling_strength,
                "tmax": tmax,
                "tstep": tstep,
            }

            if store_initial_condition:
                result["initial_condition"] = initial_condition

            return result

        final_distance = final_max_pwd(sol, dimension)

        sync_time = time_to_sync(
            sol=sol,
            t=t,
            dimension=dimension,
            tol=sync_tol,
            tol_max=tol_max,
        )

        final_success = is_synchronized_final(
            sol=sol,
            dimension=dimension,
            tol=sync_tol,
        )

        success = is_synchronized_over_win(
            sol=sol,
            dimension=dimension,
            tol=sync_tol,
            win_frac=window_fraction,
        )

        result = {
            "success": success,
            "final_success": final_success,
            "integration_failed": False,
            "solution_invalid": False,
            "trial_seed": trial_seed,
            "sampler": sampler,
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
            "solution_health": health,
        }

        if store_initial_condition:
            result["initial_condition"] = initial_condition

    except Exception as error:
        result = {
            "success": False,
            "final_success": False,
            "integration_failed": True,
            "solution_invalid": False,
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
    G=None,
    n_trials=25,
    base_seed=42,
    parameters=None,
    coupling_strength=1,
    H=None,
    tmax=150,
    tstep=0.05,
    dimension=3,
    sync_tol=1e-2,
    tol_max=1e6,
    window_fraction=0.1,
    sampler="uniform",
    sampling_bounds=None,
    store_initial_conditions=False,
    max_abs_threshold=1e6,
):
    """
    Estimate basin stability serially.

    Runs run_single_trial once per seed and computes:

        basin_stability = successes / n_trials

    Integration failures and invalid solutions are counted separately, but
    they are still included in the total trial count.
    """

    if G is None:
        G = nx.path_graph(5)

    if parameters is None:
        parameters = [0.2, 0.2, 7]

    if sampling_bounds is None:
        sampling_bounds = [-10, 10]

    seeds = trial_seeds(base_seed=base_seed, n_trials=n_trials)

    results = []
    successes = 0
    int_failures = 0
    successful_sync_times = []

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
            store_initial_condition=store_initial_conditions,
            max_abs_threshold=max_abs_threshold,
        )

        results.append(trial)

    for result in results:
        if result["success"] is True:
            successes += 1
            successful_sync_times.append(result["sync_time"])

        if result["integration_failed"] is True:
            int_failures += 1

    sync_failures = n_trials - successes - int_failures

    bs_value = successes / n_trials

    if successful_sync_times:
        sync_time_mean = float(np.mean(successful_sync_times))
    else:
        sync_time_mean = np.inf

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
        "tmax": tmax,
        "tstep": tstep,
        "dimension": dimension,
        "sampler": sampler,
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "is_directed": G.is_directed(),
        "sync_time_mean": sync_time_mean,
        "successful_sync_times": successful_sync_times,
        "store_initial_conditions": store_initial_conditions,
        "max_abs_threshold": max_abs_threshold,
        "results": results,
    }

    return basin_stability_results


