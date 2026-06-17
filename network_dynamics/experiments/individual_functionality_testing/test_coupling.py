import networkx as nx

from network_dynamics.core.graphs import graph_laplacian
from network_dynamics.core.coupling import build_coupling_matrix


def main():
    print("=" * 70)
    print("TEST coupling.py")
    print("=" * 70)

    G = nx.path_graph(5)
    L = graph_laplacian(G)

    C = build_coupling_matrix(
        L=L,
        H=None,
        strength=1.0,
    )

    print("Coupling matrix shape:", C.shape)
    print("Coupling matrix:")
    print(C)

    assert C.shape == (15, 15)

    # Check default x-only coupling:
    # y and z diagonal coupling entries should be zero for a node block.
    first_block = C[0:3, 0:3]
    print("First 3x3 block:")
    print(first_block)

    print("coupling.py passed.")


if __name__ == "__main__":
    main()
