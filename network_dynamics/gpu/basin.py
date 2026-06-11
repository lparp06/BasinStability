"""
Chunked GPU/JAX basin-stability experiment.

This backend integrates batches on the GPU, returns full trajectories to NumPy,
and then reuses the shared CPU-style classification logic. Use it for
debugging and CPU/GPU trajectory validation. For cluster-scale production runs,
prefer ``network_dynamics.gpu.basin_fast``.
"""

from network_dynamics.core.basin_common import (
    choose_success,
    classify_solution,
    sample_initial_conditions_batch,
)
from network_dynamics.core.results import BasinSummary
from network_dynamics.core.sampling import trial_seeds
from network_dynamics.gpu.integration import integrate_rk4_batch_from_config


def classify_single_trajectory(config, trial_seed, sol, t):
    """
    Classify one already-integrated trajectory.
    """

    return classify_solution(
        config=config,
        trial_seed=trial_seed,
        sol=sol,
        t=t,
    )


def basin_stability_gpu(config, batch_size=25, verbose=True):
    """
    Run the chunked GPU/JAX basin-stability experiment.
    """

    config.validate()

    if config.integrator != "RK4":
        raise ValueError(
            "GPU basin currently supports only integrator='RK4'. "
            "Use CPU for LSODA."
        )

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    seeds = trial_seeds(
        base_seed=config.base_seed,
        n_trials=config.n_trials,
    )
    results = []

    for start in range(0, len(seeds), batch_size):
        end = min(start + batch_size, len(seeds))
        seed_chunk = seeds[start:end]

        if verbose:
            print(f"GPU chunk: trials {start} to {end - 1}")

        initial_conditions_batch = sample_initial_conditions_batch(
            config=config,
            seeds=seed_chunk,
        )

        sol_batch, t = integrate_rk4_batch_from_config(
            config=config,
            initial_conditions_batch=initial_conditions_batch,
            return_numpy=True,
        )

        for trial_index, seed in enumerate(seed_chunk):
            result = classify_single_trajectory(
                config=config,
                trial_seed=seed,
                sol=sol_batch[trial_index],
                t=t,
            )
            results.append(result)

        del initial_conditions_batch
        del sol_batch

    return BasinSummary.from_results(
        config=config,
        seeds=seeds,
        results=results,
    )
