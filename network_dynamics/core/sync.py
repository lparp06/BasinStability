"""
Given a trajectory sol and time vector t,
how synchronized are the oscillators?

For a network of oscillators, synchronization means
all nodes have approximately the same state

For Rössler oscillators, each node has
(x_i, y_i, z_i)

If node 0 and node 1 are synchronized, then
    x_0 ≈ x_1
    y_0 ≈ y_1
    z_0 ≈ z_1

- A common test is the maximum pairwise distance
    Look at every pair of nodes
    Measure how far apart their states are
    Take the maximum distance
    If it is below sync_tol, call the system synchronized
"""

import numpy as np
from scipy.spatial.distance import pdist

def reshape_state_by_node(state_vector, dimension=3):
    """
    Take one flat state vector and reshape it into one row per oscillator.

    Example:
        [x0, y0, z0, x1, y1, z1]
    becomes:
        [[x0, y0, z0],
         [x1, y1, z1]]
    """

    state_vector = np.asarray(state_vector)

    if state_vector.ndim != 1:
        raise ValueError(
            "state_vector must be one-dimensional. "
            f"Got shape {state_vector.shape}."
        )

    if state_vector.size % dimension != 0:
        raise ValueError(
            "state_vector length must be divisible by dimension. "
            f"Got length {state_vector.size} and dimension {dimension}."
        )

    return state_vector.reshape(-1, dimension)

def max_pairwise_distance(state_vector, dimension=3):
    state_by_node = reshape_state_by_node(state_vector, dimension)

    if state_by_node.shape[0] <= 1:
        return 0.0

    distances = pdist(state_by_node)
    return float(distances.max())

def final_max_pwd(sol, dimension=3):
    final_state = sol[-1]
    max_pwd_final = max_pairwise_distance(final_state, dimension)
    return max_pwd_final

def time_to_sync(sol, t, dimension=3, tol=1e-3, tol_max=1e6):
    """ 
    Return the first time at which the oscillators become synchronized.

    Returns np.inf if synchronization never occurs or if the pairwise
    distance exceeds tol_max.
    """

    if len(sol) != len(t):
        raise ValueError(
            "sol and t must have the same length along the time axis. "
            f"Got len(sol)={len(sol)} and len(t)={len(t)}."
        )

    for i in range(len(t)):
        state = sol[i]
        max_pwd = max_pairwise_distance(state, dimension)

        if max_pwd < tol:
            return float(t[i])

        if max_pwd > tol_max:
            return np.inf

    return np.inf

# Final state synchronization check

def is_synchronized_final(sol, dimension=3, tol=1e-3):
    return final_max_pwd(sol, dimension) < tol

def is_synchronized_over_win(sol, dimension=3, tol=1e-3, win_frac=0.2):
    """
    Check whether the oscillators are synchronized over the final window
    of the trajectory.
    """

    if not 0 < win_frac <= 1:
        raise ValueError(
            "win_frac must be in the interval (0, 1]. "
            f"Got {win_frac}."
        )

    window_start = int((1 - win_frac) * len(sol))
    final_window = sol[window_start:]

    for state in final_window:
        distance = max_pairwise_distance(state, dimension)

        if distance >= tol:
            return False

    return True

def analyze_synchronization(
        sol,
        t,
        dimension=3,
        tol=1e-3,
        tol_max=1e6,
        win_frac=0.2
    ):
    """
    Compute all synchronization metrics for one trajectory
    """
    final_distance = final_max_pwd(sol, dimension)

    sync_time = time_to_sync(
        sol=sol,
        t=t,
        dimension=dimension,
        tol=tol,
        tol_max=tol_max
    )

    final_success = is_synchronized_final(
        sol=sol,
        dimension=dimension,
        tol=tol
    )

    window_success = is_synchronized_over_win(
        sol=sol,
        dimension=dimension,
        tol=tol,
        win_frac=win_frac
    )

    return {
        "final_distance": final_distance,
        "sync_time": sync_time,
        "final_success": final_success,
        "window_success": window_success,
    }