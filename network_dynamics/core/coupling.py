"""
Coupling matrix construction.

The coupling matrix determines how oscillator states influence one another
through the network.

The graph Laplacian L describes coupling between nodes.
The inner coupling matrix H describes which state variables are coupled
inside each oscillator.

For a network with:
    n_nodes nodes
    dimension variables per oscillator

L has shape:
    (n_nodes, n_nodes)

H has shape:
    (dimension, dimension)

The full coupling matrix has shape:
    (n_nodes * dimension, n_nodes * dimension)
"""

import numpy as np


def default_x_coupling_matrix(dimension=3):
    """
    Build a default inner coupling matrix that couples only the first variable.

    For Rössler systems with state (x, y, z), this couples x only:

        [[1, 0, 0],
         [0, 0, 0],
         [0, 0, 0]]
    """

    H = np.zeros((dimension, dimension))
    H[0, 0] = 1.0

    return H


def rank_one_inner_coupling_matrix(target, source, dimension=3):
    """
    Build an inner coupling matrix with one nonzero entry.

    ``target`` is the row receiving the coupled variable, and ``source`` is
    the column of the variable being coupled.
    """

    if not (0 <= target < dimension and 0 <= source < dimension):
        raise ValueError(
            "target and source must be valid variable indices for "
            f"dimension={dimension}; got target={target}, source={source}."
        )

    H = np.zeros((dimension, dimension))
    H[target, source] = 1.0

    return H


def validate_inner_coupling_matrix(H, dimension):
    """
    Check that H has shape (dimension, dimension).
    """

    H = np.asarray(H, dtype=float)

    expected_shape = (dimension, dimension)

    if H.shape != expected_shape:
        raise ValueError(
            "Inner coupling matrix H has the wrong shape. "
            f"Expected {expected_shape}, got {H.shape}."
        )

    return H


def build_coupling_matrix(L, H=None, strength=1.0, dimension=3):
    """
    Build the full coupling matrix for the network system.

    Parameters
    ----------
    L : np.ndarray
        Graph Laplacian with shape (n_nodes, n_nodes).

    H : np.ndarray or None
        Inner coupling matrix with shape (dimension, dimension).
        If None, defaults to x-only coupling.

    strength : float
        Coupling strength multiplier.

    dimension : int
        Number of variables per oscillator.

    Returns
    -------
    np.ndarray
        Full coupling matrix with shape:
            (n_nodes * dimension, n_nodes * dimension)
    """

    L = np.asarray(L, dtype=float)

    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError("L must be a square matrix. " f"Got shape {L.shape}.")

    if H is None:
        H = default_x_coupling_matrix(dimension=dimension)
    else:
        H = validate_inner_coupling_matrix(H, dimension=dimension)

    coupling_matrix = strength * np.kron(L, H)

    # Remove tiny floating-point roundoff values that should be zero.
    coupling_matrix[np.isclose(coupling_matrix, 0.0)] = 0.0

    return coupling_matrix
