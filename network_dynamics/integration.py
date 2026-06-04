"""
Integrates one trajectory
"""
import numpy as np
import networkx as nx
from scipy.integrate import solve_ivp
from graphs import graph_laplacian
from coupling import build_coupling_matrix
from oscillators import rossler


def integrate(G, initial_conditions, parameters, coupling_strength, H, tmax, timestep):
    
    L = graph_laplacian(G)

    coupling_matrix = build_coupling_matrix(L = L, H = H, strength = coupling_strength)

    t = np.arange(0, tmax, timestep)

    def rhs (time, state):
        return rossler(time, state, coupling_matrix, parameters)
    
    result = solve_ivp(rhs, t_span = (t[0], t[-1]), y0 = initial_conditions, t_eval = t)
    
    sol = result.y.T

    sol_t = result.t

    return sol, sol_t


def main():
    print("Testing integration.py...")

    G = nx.path_graph(5)

    initial_conditions = np.arange(15, dtype = "float")

    parameters = [0.2, 0.2, 7]

    sol, t = integrate(
        G=G,
        initial_conditions=initial_conditions,
        parameters=parameters,
        coupling_strength=1,
        H=None,
        tmax=1,
        timestep=0.05,
    )

    print("sol.shape:", sol.shape)
    print("t.shape:", t.shape)
    print("first state:", sol[0])
    print("final state:", sol[-1])



if __name__ == "__main__":
    main()