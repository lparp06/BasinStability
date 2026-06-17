import numpy as np
import networkx as nx

from network_dynamics.core.graphs import graph_laplacian


def main():
    print("=" * 70)
    print("TEST graphs.py")
    print("=" * 70)

    G = nx.path_graph(5)

    L = graph_laplacian(G)

    print("L shape:", L.shape)
    print("L:")
    print(L)
    print("Row sums:", L.sum(axis=1))

    assert L.shape == (5, 5)
    assert np.allclose(L.sum(axis=1), 0)

    print("graphs.py passed.")


if __name__ == "__main__":
    main()
