"""
Integrates one trajectory
"""
import numpy as np
from scipy.integrate import solve_ivp
from .graphs import graph_laplacian
from .coupling import build_coupling_matrix
from .oscillators import rossler


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


