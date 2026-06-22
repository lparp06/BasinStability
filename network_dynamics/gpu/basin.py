"""
gpu/basin.py

GPU/JAX basin-stability experiment.

For now:
- samples initial conditions on CPU using NumPy
- integrates trajectories on GPU using JAX RK4
- processes trials in chunks to avoid GPU memory pressure
- brings trajectories back to NumPy
- reuses CPU/core diagnostics and synchronization code
- returns the same BasinSummary class as CPU basin.py
"""

import numpy as np

from network_dynamics.core.jax_config import enable_x64

enable_x64()

import jax.numpy as jnp

from network_dynamics.core.basin_common import (
    classify_solution,
    sample_initial_conditions_batch,
)
from network_dynamics.core.coupling import build_coupling_matrix
from network_dynamics.core.graphs import graph_laplacian
from network_dynamics.core.results import BasinSummary
from network_dynamics.core.sampling import trial_seeds
from network_dynamics.gpu.dynamics import dynamics_code, rk4_step_batch_jax
from network_dynamics.gpu.integration import integrate_rk4_batch_scan_jax


def basin_stability_gpu(config, batch_size=25, verbose=True):
    """
    Run a GPU/JAX basin-stability experiment.

    Parameters
    ----------
    config : BasinConfig
        Experiment configuration.

    batch_size : int
        Number of trials to integrate on the GPU at once.

        Smaller batch_size uses less GPU memory.
        Larger batch_size may be faster, but can overwhelm Apple MPS memory.

    verbose : bool
        If True, print chunk progress.

    Returns
    -------
    BasinSummary
        Summary object containing counts and trial results.
    """

    config.validate()

    if config.integrator != "RK4":
        raise ValueError(
            "GPU basin currently supports only integrator='RK4'. " "Use CPU for LSODA."
        )

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    # Build coupling matrix once and keep it on device for all chunks.
    L = graph_laplacian(config.G)
    coupling_matrix = jnp.asarray(
        build_coupling_matrix(L=L, H=config.H, strength=config.coupling_strength, dimension=config.dimension),
        dtype=jnp.float64,
    )
    parameters = jnp.asarray(config.parameters, dtype=jnp.float64)
    dt_jax = jnp.asarray(config.dt, dtype=jnp.float64)
    n_steps = int(round(config.tmax / config.dt))
    t = np.arange(n_steps, dtype=np.float64) * config.dt
    dyn_code = dynamics_code(config.dynamics)

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
            dtype=np.float64,
        )

        initial_states = jnp.asarray(initial_conditions_batch, dtype=jnp.float64)

        sol_batch_jax = integrate_rk4_batch_scan_jax(
            initial_states=initial_states,
            coupling_matrix=coupling_matrix,
            parameters=parameters,
            dt=dt_jax,
            n_steps=n_steps,
            dynamics_code_value=dyn_code,
        )
        sol_batch = np.asarray(sol_batch_jax)
        del sol_batch_jax

        for trial_index, seed in enumerate(seed_chunk):
            sol = sol_batch[trial_index]

            result = classify_solution(
                config=config,
                trial_seed=seed,
                sol=sol,
                t=t,
            )

            results.append(result)

        del initial_conditions_batch, sol_batch

    summary = BasinSummary.from_results(
        config=config,
        seeds=seeds,
        results=results,
    )

    return summary
