"""
compare_cpu_gpu_batch_rk4.py

Compare CPU RK4 loop vs GPU/JAX batched RK4.

Run from project root:

    python -m network_dynamics.experiments.compare_cpu_gpu_batch_rk4
"""

import numpy as np
import networkx as nx

from network_dynamics.core.sampling import (
    sample_uniform_initial_condition,
    trial_seeds,
)
from network_dynamics.cpu.integration import integrate_rk4
from network_dynamics.gpu.integration import integrate_rk4_batch_jax


def make_initial_conditions_batch(
    seeds,
    n_nodes,
    dimension,
    low,
    high,
):
    """
    Generate a batch of NumPy initial conditions.

    Shape:
        (n_trials, n_nodes * dimension)
    """

    initial_conditions = []

    for seed in seeds:
        rng = np.random.default_rng(seed)

        initial_condition = sample_uniform_initial_condition(
            rng=rng,
            n_nodes=n_nodes,
            dimension=dimension,
            low=low,
            high=high,
        )

        initial_conditions.append(initial_condition)

    return np.asarray(initial_conditions)


def integrate_cpu_batch_loop(
    G,
    initial_conditions_batch,
    parameters,
    coupling_strength,
    H,
    tmax,
    dt,
    dimension,
):
    """
    CPU reference: integrate each trajectory one at a time with CPU RK4.
    """

    solutions = []

    for initial_condition in initial_conditions_batch:
        sol, t = integrate_rk4(
            G=G,
            initial_conditions=initial_condition,
            parameters=parameters,
            coupling_strength=coupling_strength,
            H=H,
            tmax=tmax,
            dt=dt,
            dimension=dimension,
        )

        solutions.append(sol)

    sol_batch = np.asarray(solutions)

    return sol_batch, t


def main():
    print("=" * 70)
    print("Compare CPU RK4 loop vs GPU/JAX batched RK4")
    print("=" * 70)

    G = nx.path_graph(5)
    n_nodes = G.number_of_nodes()
    dimension = 3

    n_trials = 5
    base_seed = 42
    seeds = trial_seeds(
        base_seed=base_seed,
        n_trials=n_trials,
    )

    parameters = (0.2, 0.2, 7.0)
    coupling_strength = 1.0
    H = None

    sampling_bounds = (-5.0, 5.0)
    low, high = sampling_bounds

    tmax = 20.0
    dt = 0.005

    initial_conditions_batch = make_initial_conditions_batch(
        seeds=seeds,
        n_nodes=n_nodes,
        dimension=dimension,
        low=low,
        high=high,
    )

    print("Initial condition batch shape:", initial_conditions_batch.shape)

    cpu_sol_batch, cpu_t = integrate_cpu_batch_loop(
        G=G,
        initial_conditions_batch=initial_conditions_batch,
        parameters=parameters,
        coupling_strength=coupling_strength,
        H=H,
        tmax=tmax,
        dt=dt,
        dimension=dimension,
    )

    gpu_sol_batch, gpu_t = integrate_rk4_batch_jax(
        G=G,
        initial_conditions_batch=initial_conditions_batch,
        parameters=parameters,
        coupling_strength=coupling_strength,
        H=H,
        tmax=tmax,
        dt=dt,
        dimension=dimension,
        return_numpy=True,
    )

    max_difference = np.max(np.abs(cpu_sol_batch - gpu_sol_batch))

    final_state_differences = np.max(
        np.abs(cpu_sol_batch[:, -1, :] - gpu_sol_batch[:, -1, :]),
        axis=1,
    )

    print()
    print("CPU batch solution shape:", cpu_sol_batch.shape)
    print("GPU batch solution shape:", gpu_sol_batch.shape)
    print("CPU time shape:", cpu_t.shape)
    print("GPU time shape:", gpu_t.shape)

    print()
    print("CPU contains NaN:", np.isnan(cpu_sol_batch).any())
    print("GPU contains NaN:", np.isnan(gpu_sol_batch).any())
    print("CPU contains Inf:", np.isinf(cpu_sol_batch).any())
    print("GPU contains Inf:", np.isinf(gpu_sol_batch).any())

    print()
    print("Max absolute CPU-GPU difference over full batch:")
    print(max_difference)

    print()
    print("Max final-state difference per trial:")
    for seed, difference in zip(seeds, final_state_differences):
        print(f"seed={seed}: {difference}")

    print()
    if max_difference < 1e-3:
        print("Batch GPU RK4 comparison passed.")
    else:
        print("Batch GPU RK4 ran, but CPU-GPU difference is larger than expected.")


if __name__ == "__main__":
    main()
