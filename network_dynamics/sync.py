"""
Given a trajectory sol and time vector t,
how synchronized are the oscillators?
"""

import numpy as np
from scipy.spatial.distance import pdist

def reshape_state_by_node(state_vector, dimension = 3):
    """
    Take one flat state vector and reshape it into one row per oscillator
    """
    state = state_vector.reshape(-1, dimension)
    return state

def max_pairwise_distance(state_vector, dimension = 3):
    state_by_node = reshape_state_by_node(state_vector, dimension)
    distances = pdist(state_by_node)
    max_distance = distances.max()

    return max_distance

def final_max_pwd(sol, dimension=3):
    final_state = sol[-1]
    max_pwd_final = max_pairwise_distance(final_state, dimension)
    return max_pwd_final

def time_to_sync(sol, t, dimension, tol, tol_max):
    """ 
    At what time did the oscillators first become synchronized?
    sol: full trajectory from integration.py
    t: time vector
    dimension: dimension of 1 oscillator
    tol: synchronization tolerance
    tol_max: blowup threshold

    returns:
        sync_time - return first time at which synchronization happens
                    returns np.inf if we pass tol_max
    """

    for i in range (len(t)):
        state = sol[i]
        max_pwd = max_pairwise_distance(state)
        if (max_pwd < tol):
            return t[i]
        if max_pwd > tol_max:
            return np.inf 
    
    return np.inf



def main():

    print("Synchronizes at final time")
    t = np.array([0, 1, 2])
    sol = np.array([
        [0, 0, 0, 10, 0, 0],
        [0, 0, 0, 0.5, 0, 0],
        [1, 1, 1, 1, 1, 1]
    ])

    sync_time = time_to_sync(
        sol = sol,
        t = t,
        dimension = 3,
        tol = 0.1,
        tol_max = 1e6
    )
    
    print("Sync time:", sync_time)
    print("Expected: 2")
    print()

    print("Synchronizes earlier")
    t = np.array([0, 1, 2])
    sol = np.array([
        [0, 0, 0, 10, 0, 0],
        [0, 0, 0, 0.05, 0, 0],
        [1, 1, 1, 1, 1, 1]
    ])

    sync_time = time_to_sync(
        sol = sol,
        t = t,
        dimension = 3,
        tol = 0.1,
        tol_max = 1e6
    )
    
    print("Sync time:", sync_time)
    print("Expected: 1")
    print()

    print("Already synchronized")
    t = np.array([0, 1, 2])
    sol = np.array([
        [1, 1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3, 3]
    ])

    sync_time = time_to_sync(
        sol = sol,
        t = t,
        dimension = 3,
        tol = 0.1,
        tol_max = 1e6
    )
    
    print("Sync time:", sync_time)
    print("Expected: 0")
    print()

    print("Blows up")
    t = np.array([0, 1, 2])
    sol = np.array([
        [0, 0, 0, 10, 0, 0],
        [0, 0, 0, 0.5, 0, 0],
        [0, 0, 0, 1e9, 0, 0]
    ])

    sync_time = time_to_sync(
        sol = sol,
        t = t,
        dimension = 3,
        tol = 0.1,
        tol_max = 1e6
    )
    
    print("Sync time:", sync_time)
    print("Expected: inf")
    print()

if __name__ == "__main__":
    main()