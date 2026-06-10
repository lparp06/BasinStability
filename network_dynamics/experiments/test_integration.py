import numpy as np
import networkx as nx

from network_dynamics.cpu.integration import integrate_lsoda


def main():
    print("=" * 70)
    print("CHECK integration.py")
    print("=" * 70)

    G = nx.path_graph(5)

    initial_condition = np.array([
        1.0, 0.0, 0.0,
        1.1, 0.0, 0.0,
        0.9, 0.0, 0.0,
        1.2, 0.0, 0.0,
        0.8, 0.0, 0.0,
    ])

    sol, t = integrate_lsoda(
        G=G,
        initial_conditions=initial_condition,
        parameters=[0.2, 0.2, 7.0],
        coupling_strength=1.0,
        H=None,
        tmax=20.0,
        dt=0.05,
        dimension=3,
    )

    print("t[0], t[-1], len(t):", t[0], t[-1], len(t))
    print("sol shape:", sol.shape)
    print("contains NaN:", np.isnan(sol).any())
    print("contains Inf:", np.isinf(sol).any())
    print("max abs:", np.max(np.abs(sol)))
    print("final state:")
    print(sol[-1])

    if np.isnan(sol).any() or np.isinf(sol).any():
        raise RuntimeError("Integration produced NaN or Inf.")

    print("integration.py sanity check passed.")


if __name__ == "__main__":
    main()