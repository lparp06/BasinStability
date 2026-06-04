"""
Generates initial conditions for basin-stability trials
"""

import numpy as np


def sample_uniform_initial_condition(rng, n_nodes, dimension=3, low=10, high=10):
    """
    Generate one initial condition where 
    each state variable is sampled uniformly
    from a box
    """
    # rng is used so that randomness is controlled outside
    # the sampler; makes basin-stability trials reproducible
    state_dimension = n_nodes * dimension
    initial_condition = rng.uniform(low, high, state_dimension)
    return initial_condition

def sample_normal_initial_condition(rng, n_nodes, dimension=3, mean=0, std=1):
    """
    Sampler that draws from a normal distribution
    """
    state_dimension = n_nodes * dimension
    initial_condition = rng.normal(loc = mean, scale = std, size = state_dimension)
    return initial_condition

def trial_seeds(base_seed, n_trials):
    trials = list(range(base_seed, base_seed + n_trials))

    return trials

def main():
    print(trial_seeds(100, 5))

if __name__ == "__main__":
    main()