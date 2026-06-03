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
        D = np.diag(A.sum(axis = 1))
        laplacian_directed = D - A
        return laplacian_directed
    
    laplacian_undirected = nx.laplacian_matrix(G).toarray().astype(float)
    return laplacian_undirected

# Code I used to test
"""
def main():
    G = nx.Graph()
    G.add_edges_from([
        (0, 1),
        (1, 2),
        (1, 3)
    ])

    L = graph_laplacian(G)

    print(L)

if __name__ == "__main__":
    main()

"""