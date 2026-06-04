"""
Module that builds coupling matrices
Coupling matrix governs the strength of influence between two nodes
Eigenvalues of the graph Laplacian are often used in synchronization analysis.
"""
import numpy as np

def build_coupling_matrix(L, H = None, strength = 1):
    if (H is None):
        H = np.eye(3)
        H[1, 1] = 0
        H[2, 2] = 0

    coupling_matrix = strength * np.kron(L, H)
    coupling_matrix[np.isclose(coupling_matrix, 0)] = 0
    return coupling_matrix
