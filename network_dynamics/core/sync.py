"""
sync.py

Synchronization diagnostics for oscillator trajectories.

State layout:
    [x0, y0, z0, x1, y1, z1, ...]

Synchronization is measured using the maximum pairwise distance between
node states at each time point.
"""

import numpy as np
from scipy.spatial.distance import pdist


def reshape_state_by_node(state_vector, dimension=3):
    """
    Reshape one flat state vector into one row per oscillator.
    """

    state_vector = np.asarray(state_vector)

    if state_vector.ndim != 1:
        raise ValueError(
            "state_vector must be one-dimensional. " f"Got shape {state_vector.shape}."
        )

    if state_vector.size % dimension != 0:
        raise ValueError(
            "state_vector length must be divisible by dimension. "
            f"Got length {state_vector.size} and dimension {dimension}."
        )

    return state_vector.reshape(-1, dimension)


def max_pairwise_distance(state_vector, dimension=3):
    """
    Compute the maximum pairwise distance between oscillator states.
    """

    state_by_node = reshape_state_by_node(
        state_vector=state_vector,
        dimension=dimension,
    )

    if state_by_node.shape[0] <= 1:
        return 0.0

    distances = pdist(state_by_node)

    return float(distances.max())


def distance_time_series(sol, dimension=3):
    """
    Compute the maximum pairwise distance at every time point.
    """

    sol = np.asarray(sol)

    if sol.ndim != 2:
        raise ValueError(
            "sol must be two-dimensional with shape "
            "(n_time_points, state_dimension). "
            f"Got shape {sol.shape}."
        )

    distances = np.array(
        [
            max_pairwise_distance(
                state_vector=state,
                dimension=dimension,
            )
            for state in sol
        ],
        dtype=float,
    )

    return distances


def final_max_pwd(sol, dimension=3):
    """
    Maximum pairwise distance at the final time point.
    """

    distances = distance_time_series(
        sol=sol,
        dimension=dimension,
    )

    return float(distances[-1])


def max_distance_over_final_window(sol, dimension=3, win_frac=0.2):
    """
    Maximum pairwise distance over the final fraction of the trajectory.
    """

    if not 0 < win_frac <= 1:
        raise ValueError("win_frac must be in the interval (0, 1]. " f"Got {win_frac}.")

    distances = distance_time_series(
        sol=sol,
        dimension=dimension,
    )

    window_start = int((1.0 - win_frac) * len(distances))
    window_distances = distances[window_start:]

    return float(np.max(window_distances))


def time_to_sync(sol, t, dimension=3, tol=1e-3, tol_max=1e6):
    """
    Return the first time at which the oscillators become synchronized.
    """

    sol = np.asarray(sol)
    t = np.asarray(t)

    if len(sol) != len(t):
        raise ValueError(
            "sol and t must have the same length along the time axis. "
            f"Got len(sol)={len(sol)} and len(t)={len(t)}."
        )

    distances = distance_time_series(
        sol=sol,
        dimension=dimension,
    )

    for i, distance in enumerate(distances):
        if distance < tol:
            return float(t[i])

        if distance > tol_max:
            return np.inf

    return np.inf


def is_synchronized_final(sol, dimension=3, tol=1e-3):
    """
    Check synchronization only at the final time point.
    """

    return (
        final_max_pwd(
            sol=sol,
            dimension=dimension,
        )
        < tol
    )


def is_synchronized_over_win(sol, dimension=3, tol=1e-3, win_frac=0.2):
    """
    Check whether the oscillators are synchronized over the final window.
    """

    return (
        max_distance_over_final_window(
            sol=sol,
            dimension=dimension,
            win_frac=win_frac,
        )
        < tol
    )


def analyze_synchronization(sol, t, dimension=3, tol=1e-3, tol_max=1e6, win_frac=0.2):
    """
    Compute synchronization metrics for one trajectory.
    """

    sol = np.asarray(sol)
    t = np.asarray(t)

    if len(sol) != len(t):
        raise ValueError(
            "sol and t must have the same length along the time axis. "
            f"Got len(sol)={len(sol)} and len(t)={len(t)}."
        )

    distances = distance_time_series(
        sol=sol,
        dimension=dimension,
    )

    final_distance = float(distances[-1])

    window_start = int((1.0 - win_frac) * len(distances))
    window_distances = distances[window_start:]
    window_max_distance = float(np.max(window_distances))

    final_success = final_distance < tol
    window_success = window_max_distance < tol

    sync_time = np.inf

    for i, distance in enumerate(distances):
        if distance < tol:
            sync_time = float(t[i])
            break

        if distance > tol_max:
            sync_time = np.inf
            break

    return {
        "final_distance": final_distance,
        "window_max_distance": window_max_distance,
        "sync_time": sync_time,
        "final_success": final_success,
        "window_success": window_success,
    }
