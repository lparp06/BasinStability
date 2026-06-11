import numpy as np
import networkx as nx

from network_dynamics.core.graphs import graph_laplacian
from network_dynamics.core.coupling import build_coupling_matrix
from network_dynamics.core.oscillators import rossler


def main():
    print("=" * 70)
    print("TEST oscillators.py")
    print("=" * 70)

    G = nx.path_graph(5)
    L = graph_laplacian(G)

    C = build_coupling_matrix(
        L=L,
        H=None,
        strength=1.0,
    )

    parameters = [0.2, 0.2, 7.0]

    # Deterministic test state: x0,y0,z0,x1,y1,z1,...
    state = np.arange(15, dtype=float) / 10.0

    dx = rossler(
        0.0,
        state,
        C,
        parameters,
    )

    print("state:")
    print(state)
    print("derivative:")
    print(dx)
    print("derivative shape:", dx.shape)
    print("finite:", np.all(np.isfinite(dx)))

    assert dx.shape == state.shape
    assert np.all(np.isfinite(dx))

    print("oscillators.py passed.")


if __name__ == "__main__":
    main()
