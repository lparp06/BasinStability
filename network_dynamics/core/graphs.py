"""
Graph and Laplacian utilities
"""

import numpy as np
import networkx as nx


def graph_laplacian(G):
    """
    Returns the graph Laplacian for both directed and undirected graphs

    Inputs:
        G : a Networkx graph, either directed or undirected

    """
    if G.is_directed():
        A = nx.to_numpy_array(G)
        D = np.diag(A.sum(axis=1))
        laplacian_directed = D - A
        return laplacian_directed

    laplacian_undirected = nx.laplacian_matrix(G).toarray().astype(float)
    return laplacian_undirected
