"""
Local check that MSF source/target choices reach the basin coupling matrix.

Run from the project root:

    python3 -m network_dynamics.experiments.individual_functionality_testing.check_msf_basin_coupling

This does not run a basin simulation. It builds the same BasinConfig used by
coupling_basin_scan for several source/target pairs, then checks the inner
coupling matrix H and a representative full network-coupling entry.
"""

import numpy as np

from network_dynamics.core.coupling import build_coupling_matrix
from network_dynamics.core.graphs import graph_laplacian
from network_dynamics.experiments.coupling_basin_scan import (
    ScanRequest,
    make_basin_config,
)


def _request(source, target):
    return ScanRequest(
        graph_type="path_graph",
        n_nodes=3,
        edge_probability=0.15,
        graph_seed=42,
        base_seed=42,
        n_trials=2,
        dynamics="lorenz",
        tmax=1.0,
        dt=0.01,
        n_strengths=1,
        coupling_low=None,
        coupling_high=None,
        interval_index=0,
        progress_interval=5,
        backend="cpu",
        n_workers=1,
        integrator="RK4",
        sync_tol=1e-3,
        tol_max=1e6,
        window_fraction=0.1,
        success_definition="first_crossing",
        sampling_low=-5.0,
        sampling_high=5.0,
        max_abs_threshold=1e9,
        a=10.0,
        b=2.0,
        c=28.0,
        msf_source=source,
        msf_target=target,
        K_min=0.0,
        K_max=50.0,
        n_K=101,
        msf_cache="outputs/msf_zero_cache.csv",
        msf_transient_time=100.0,
        msf_measurement_time=3000.0,
        msf_dt=0.001,
        msf_chunk_size=None,
    )


def _state_index(node, variable, dimension=3):
    return node * dimension + variable


def check_case(graph, source, target, strength):
    config = make_basin_config(
        request=_request(source=source, target=target),
        graph=graph,
        coupling_strength=strength,
    )
    L = graph_laplacian(graph)
    full_coupling = build_coupling_matrix(
        L=L,
        H=config.H,
        strength=config.coupling_strength,
        dimension=config.dimension,
    )

    expected_H = np.zeros((3, 3))
    expected_H[target, source] = 1.0

    row = _state_index(node=0, variable=target)
    diag_col = _state_index(node=0, variable=source)
    neighbor_col = _state_index(node=1, variable=source)

    h_ok = np.array_equal(config.H, expected_H)
    diag_ok = np.isclose(full_coupling[row, diag_col], strength)
    neighbor_ok = np.isclose(full_coupling[row, neighbor_col], -strength)
    zero_count_ok = np.count_nonzero(config.H) == 1

    return {
        "source": source,
        "target": target,
        "H": config.H.astype(int),
        "diag_entry": full_coupling[row, diag_col],
        "neighbor_entry": full_coupling[row, neighbor_col],
        "passed": h_ok and diag_ok and neighbor_ok and zero_count_ok,
    }


def main():
    graph = __import__("networkx").path_graph(3)
    strength = 0.7
    cases = (
        (0, 0),  # x -> x default-style coupling
        (1, 0),  # y -> x, Lorenz s1t0
        (0, 1),  # x -> y
        (2, 2),  # z -> z
    )

    print("MSF source/target -> basin H check")
    print("=" * 44)
    print("Graph: path_graph(3)")
    print(f"Coupling strength: {strength}")
    print()

    all_passed = True
    for source, target in cases:
        result = check_case(
            graph=graph,
            source=source,
            target=target,
            strength=strength,
        )
        all_passed = all_passed and result["passed"]

        print(f"case s{source}t{target}: {'PASS' if result['passed'] else 'FAIL'}")
        print(result["H"])
        print(
            "  full matrix entries: "
            f"self={result['diag_entry']:.3g}, "
            f"neighbor={result['neighbor_entry']:.3g}"
        )
        print()

    if not all_passed:
        raise SystemExit(1)

    print("All checked source/target pairs reached BasinConfig.H correctly.")


if __name__ == "__main__":
    main()
