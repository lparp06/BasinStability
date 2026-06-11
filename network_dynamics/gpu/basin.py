"""
Calculate basin stability of a network using the GPU 
"""

import numpy as np
from network_dynamics.core.results import TrialResult, BasinSummary
from network_dynamics.core.sampling import sample_uniform_initial_condition, trial_seeds
from network_dynamics.core.diagnostics import (
    solution_health,
    is_solution_valid,
    format_health_message
)
from network_dynamics.core.sync import analyze_synchronization
from network_dynamics.gpu.integration import integrate_rk4_batch_jax

def sample_initial_conditions_batch(config, seeds):
    """
    Generate all initial conditions for a GPU basin run
    """
    if config.sampler != "uniform":
        raise ValueError(f"Unknown sampler: {config.sampler}")
    
    low, high = config.sampling_bounds

    initial_conditions = []

    for seed in seeds:
        rng = np.random.default_rng(seed)

        initial_condition = sample_uniform_initial_condition(
            rng = rng,
            n_nodes=config.n_nodes,
            dimension = config.dimension,
            low = low,
            high = high,
        )

        initial_conditions.append(initial_condition)

    initial_conditions_batch = np.asarray(
        initial_conditions,
        dtype=np.float32,
    )