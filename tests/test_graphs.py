"""
graphs.py tester
"""

import networkx as nx
from network_dynamics.graphs import graph_laplacian


def main():
    print("=" * 70)
    print("Undirected Graph")
    print("=" * 70)

    G = nx.Graph([
        (1, 2),
        (2, 3),
        (4, 5),
    ])

    L = graph_laplacian(G)

    print("Graph:", G)
    print("Nodes:", list(G.nodes()))
    print("Edges:", list(G.edges()))
    print("Laplacian:")
    print(L)

    print("=" * 70)
    print("Directed Graph")
    print("=" * 70)

    edges = [
        (1, 2),
        (2, 1),
        (2, 4),
        (4, 3),
        (3, 4),
    ]

    G_directed = nx.DiGraph(edges)

    L_directed = graph_laplacian(G_directed)

    print("Graph:", G_directed)
    print("Nodes:", list(G_directed.nodes()))
    print("Edges:", list(G_directed.edges()))
    print("Laplacian:")
    print(L_directed)


if __name__ == "__main__":
    main()