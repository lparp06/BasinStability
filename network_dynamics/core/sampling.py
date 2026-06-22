"""Initial-condition samplers for basin-stability trials."""


def sample_uniform_initial_condition(rng, n_nodes, dimension=3, low=-5.0, high=5.0):
    """
    Generate one flat initial-condition vector from a uniform box.
    """
    state_dimension = n_nodes * dimension
    return rng.uniform(low, high, state_dimension)


def sample_normal_initial_condition(rng, n_nodes, dimension=3, mean=0, std=1):
    """
    Generate one flat initial-condition vector from a normal distribution.
    """
    state_dimension = n_nodes * dimension
    return rng.normal(loc=mean, scale=std, size=state_dimension)


def trial_seeds(base_seed, n_trials):
    """Return deterministic per-trial seeds."""

    return list(range(base_seed, base_seed + n_trials))
