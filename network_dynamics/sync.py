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
        max_pwd = max_pairwise_distance(state, dimension)
        if (max_pwd < tol):
            return t[i]
        if max_pwd > tol_max:
            return np.inf 
    
    return np.inf

# Simpler vers of is_synchronized_over_win

def is_synchronized_final(sol, dimension, tol):
    max_pwd = final_max_pwd(sol, dimension)
    if (max_pwd < tol):
        return True
    return False

def is_synchronized_over_win(sol, dimension, tol, win_frac = 0.2):
    """
    Check whether the oscillators are synchronized over the final window
    of the trajectory.

    sol: full trajectory, shape = time points x state variables
    dimension: dimension of one oscillator, 3 for Rössler
    tol: synchronization tolerance
    win_frac: fraction of the trajectory to check at the end

    returns:
        True if every state in the final window has max pairwise distance < tol
        False otherwise
    """
     
    window_start = int((1 - win_frac) * len(sol))

    final_window = sol[window_start:]

    for state in final_window:
        distance = max_pairwise_distance(state, dimension)

        if distance >= tol:
            return False
    
    return True
