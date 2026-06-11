import numpy as np
import networkx as nx

from network_dynamics.core.sampling import sample_uniform_initial_condition
from network_dynamics.gpu.integration import integrate_rk4_jax, integrate_rk4_batch_jax

G = nx.path_graph(5)
seed = 82
rng = np.random.default_rng(seed)

initial_condition = sample_uniform_initial_condition(
    rng=rng,
    n_nodes=5,
    dimension=3,
    low=-5.0,
    high=5.0,
)

initial_batch = initial_condition[None, :]

single_sol, t = integrate_rk4_jax(
    G=G,
    initial_conditions=initial_condition,
    parameters=(0.2, 0.2, 7.0),
    coupling_strength=1.0,
    H=None,
    tmax=150.0,
    dt=0.005,
    dimension=3,
    return_numpy=True,
)

batch_sol, batch_t = integrate_rk4_batch_jax(
    G=G,
    initial_conditions_batch=initial_batch,
    parameters=(0.2, 0.2, 7.0),
    coupling_strength=1.0,
    H=None,
    tmax=150.0,
    dt=0.005,
    dimension=3,
    return_numpy=True,
)

batch_sol = batch_sol[0]

print("single shape:", single_sol.shape)
print("batch shape:", batch_sol.shape)
print("max single-batch difference:", np.max(np.abs(single_sol - batch_sol)))
print("single final:", single_sol[-1])
print("batch final:", batch_sol[-1])
