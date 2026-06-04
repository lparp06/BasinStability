"""
oscillators.py
Builds derivative functions
For now, only handles Rössler derivative
"""

import numpy as np 
import networkx as nx


def rossler(time, state_vector, coupling_matrix, parameters):
    a, b, c = parameters
    state_length = len(state_vector)
    
    X = state_vector[0:state_length:3]
    Y = state_vector[1:state_length:3]
    Z = state_vector[2:state_length:3]

    dx = np.zeros_like(state_vector, dtype=float)

    dx[0:state_length:3] = - Y - Z
    dx[1:state_length:3] = X + a * Y
    dx[2:state_length:3] = b + Z * (X - c)

    derivative = np.array(dx - np.dot(coupling_matrix, state_vector)).flatten()
    
    return derivative

