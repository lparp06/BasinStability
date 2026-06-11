import networkx as nx

from network_dynamics.core.config import BasinConfig
from network_dynamics.cpu.basin import basin_stability_serial, print_basin_summary


def main():
    bounds_list = [
        (-2.0, 2.0),
        (-5.0, 5.0),
        (-7.5, 7.5),
        (-10.0, 10.0),
    ]

    for bounds in bounds_list:
        print()
        print("=" * 70)
        print(f"Sampling bounds: {bounds}")
        print("=" * 70)

        config = BasinConfig(
            G=nx.path_graph(5),
            n_trials=100,
            base_seed=42,
            parameters=(0.2, 0.2, 7.0),
            coupling_strength=1.0,
            H=None,
            tmax=150.0,
            dt=0.05,
            dimension=3,
            sampling_bounds=bounds,
            sync_tol=1e-2,
            tol_max=1e6,
            window_fraction=0.1,
            max_abs_threshold=1e6,
            backend="serial",
        ).validate()

        summary = basin_stability_serial(config)
        print_basin_summary(summary)


if __name__ == "__main__":
    main()
