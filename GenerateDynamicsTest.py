"""
Purpose
-------
Run the original GenerateDynamics.py functions and save baseline outputs
for comparison against future implementations.

"""

import traceback
import warnings
from pathlib import Path
from datetime import datetime as dt

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

from GenerateDynamics import laplacian_dynamics


OUTPUT_DIR = Path("baseline_outputs")
OUTPUT_FILE = OUTPUT_DIR / "output_summary.txt"

ROSSLER_PARAMS = [0.2, 0.2, 7]
TOLERANCES = [1e-1, 1e-2, 1e-3, 1e-4, 1e-6]


def make_output():
    OUTPUT_DIR.mkdir(exist_ok=True)


def start_summary_file():
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        file.write(f"Summary for testing GenerateDynamics.py - {dt.now()}\n")


def format_value(value):
    if isinstance(value, np.ndarray):
        return np.array2string(value, precision = 6, suppress_small = False)

    if isinstance(value, np.generic):
        return str(value.item())

    if isinstance(value, Path):
        return str(value)

    return str(value)


def add_to_summary(label = "", value = None, indent = 0):
    prefix = " " * indent

    if value is None:
        line = prefix + str(label)
    else:
        line = prefix + f"{label}: {format_value(value)}"

    print(line)

    with open(OUTPUT_FILE, "a") as file:
        file.write(line + "\n")


def section(title):
    add_to_summary("")
    add_to_summary("=" * 70)
    add_to_summary(title)
    add_to_summary("=" * 70)


def subsection(title):
    add_to_summary("")
    add_to_summary(f"--- {title} ---")


def test_graph():
    return nx.path_graph(5)


def graph_info(G):
    subsection("Graph Information")
    add_to_summary("Number of nodes", G.number_of_nodes(), indent = 2)
    add_to_summary("List of nodes", list(G.nodes()), indent = 2)
    add_to_summary("Number of edges", G.number_of_edges(), indent = 2)
    add_to_summary("List of edges", list(G.edges()), indent = 2)
    add_to_summary("Directed", G.is_directed(), indent = 2)


def laplacian_info(dynamics, G, filename="laplacian.npy"):
    laplacian = dynamics.convert_graph_to_laplacian(G)
    matrix = np.array(laplacian, dtype=float)

    shape = np.shape(matrix)
    row_sums = np.sum(matrix, axis=1)
    eigenvalues = np.linalg.eigvals(matrix)

    subsection("Laplacian Information")
    add_to_summary("Shape", shape, indent=2)
    add_to_summary("Matrix", matrix, indent=2)
    add_to_summary("Row sums", row_sums, indent=2)
    add_to_summary("Eigenvalues", eigenvalues, indent=2)

    np.save(OUTPUT_DIR / filename, matrix)
    add_to_summary("Saved Laplacian", OUTPUT_DIR / filename, indent=2)

    return matrix



def final_max_pairwise_distance(sol, n_nodes, dimension):
    final_state = sol[-1]
    node_states = final_state.reshape(n_nodes, dimension)
    distances = pdist(node_states)
    return np.max(distances)


def record_solution_info(sol, t, expected_state_variables):
    sol_shape = np.shape(sol)
    t_shape = np.shape(t)
    state_dimension_pass = sol_shape[1] == expected_state_variables

    subsection("Solution Information")
    add_to_summary("Solution shape", sol_shape, indent=2)
    add_to_summary("Solution rows", sol_shape[0], indent=2)
    add_to_summary("Solution columns", sol_shape[1], indent=2)
    add_to_summary("Time vector shape", t_shape, indent=2)
    add_to_summary("Time vector length", len(t), indent=2)
    add_to_summary("Expected state variables", expected_state_variables, indent=2)
    add_to_summary("State dimension check", "PASS" if state_dimension_pass else "FAIL", indent=2)

    subsection("Time Vector Information")
    add_to_summary("First time value", t[0], indent=2)
    add_to_summary("Final time value", t[-1], indent=2)

    subsection("State Information")
    add_to_summary("First state vector", sol[0], indent=2)
    add_to_summary("Final state vector", sol[-1], indent=2)




def record_sync_times(dynamics, sol, t, oscillator_dimension, tolerances=TOLERANCES):
    subsection("Synchronization Times")

    sync_times = {}

    for tol in tolerances:
        sync_time = dynamics.nonlinear_find_time_to_sync(
            x = sol,
            t = t,
            d = oscillator_dimension,
            criterion = "maxpdist",
            Tol = tol,
            TolMax = 1e6,
        )

        sync_times[tol] = sync_time
        add_to_summary(f"Tol={tol}", sync_time, indent=2)

    return sync_times


def record_sync_consistency(final_distance, sync_times):
    subsection("Synchronization Consistency Check")

    for tol, sync_time in sync_times.items():
        final_distance_below_tol = final_distance < tol
        sync_time_is_finite = np.isfinite(sync_time)

        if final_distance_below_tol and not sync_time_is_finite:
            result = "CHECK: final distance is below tolerance but sync_time is inf"
        else:
            result = "OK"

        add_to_summary(f"Tol={tol} final_distance < tol", final_distance_below_tol, indent=2)
        add_to_summary(f"Tol={tol} sync_time finite", sync_time_is_finite, indent=2)
        add_to_summary(f"Tol={tol} consistency", result, indent=2)


def save_rossler_x_plot(sol, t, n_nodes):
    plt.figure(figsize=(10, 6))

    for node in range(n_nodes):
        x_index = 3 * node
        plt.plot(t, sol[:, x_index], label=f"node {node} x")

    plt.xlabel("time")
    plt.ylabel("Rössler x variable")
    plt.title("Rössler x variables by node")
    plt.legend()
    plt.tight_layout()

    path = OUTPUT_DIR / "rossler_x_variables.png"
    plt.savefig(path, dpi=200)
    plt.close()

    add_to_summary("Saved plot", path, indent=2)


def save_rossler_node0_3d_plot(sol):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(sol[:, 0], sol[:, 1], sol[:, 2])
    ax.set_xlabel("node 0 x")
    ax.set_ylabel("node 0 y")
    ax.set_zlabel("node 0 z")
    ax.set_title("Node 0 Rössler trajectory")

    path = OUTPUT_DIR / "rossler_node0_3d.png"
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

    add_to_summary("Saved plot", path, indent=2)


def save_linear_plot(sol, t, n_nodes):
    plt.figure(figsize=(10, 6))

    for node in range(n_nodes):
        plt.plot(t, sol[:, node], label=f"node {node}")

    plt.xlabel("time")
    plt.ylabel("state value")
    plt.title("Linear Laplacian dynamics by node")
    plt.legend()
    plt.tight_layout()

    path = OUTPUT_DIR / "linear_dynamics.png"
    plt.savefig(path, dpi=200)
    plt.close()

    add_to_summary("Saved plot", path, indent=2)


def rossler():
    section("Rössler Baseline")

    np.random.seed(42)

    dynamics = laplacian_dynamics()
    G = test_graph()

    graph_info(G)
    laplacian_info(dynamics, G, filename="rossler_laplacian.npy")

    tmax = 150
    timestep = 0.05
    coupling_strength = 1

    subsection("Rössler Parameters")
    add_to_summary("Random seed", 42, indent=2)
    add_to_summary("Graph nodes", G.number_of_nodes(), indent=2)
    add_to_summary("Graph edges", G.number_of_edges(), indent=2)
    add_to_summary("tmax", tmax, indent=2)
    add_to_summary("timestep", timestep, indent=2)
    add_to_summary("method", "random", indent=2)
    add_to_summary("init_cond_type", "normal", indent=2)
    add_to_summary("init_cond_params", [0, 1], indent=2)
    add_to_summary("init_cond_offset", 0, indent=2)
    add_to_summary("dynamics_type", "Rossler", indent=2)
    add_to_summary("dynamics_params", ROSSLER_PARAMS, indent=2)
    add_to_summary("coupling_strength", coupling_strength, indent=2)

    def run_simulation():
        return dynamics.continuous_time_nonlinear_dynamics(
            G=G,
            tmax=tmax,
            timestep=timestep,
            method="random",
            init_cond_type="normal",
            init_cond_params=[0, 1],
            init_cond_offset=0,
            dynamics_type="Rossler",
            dynamics_params=ROSSLER_PARAMS,
            coupling_strength=coupling_strength,
        )

    sol, t = run_simulation()

    
    record_solution_info(sol, t, expected_state_variables=G.number_of_nodes() * 3)

    final_distance = final_max_pairwise_distance(
        sol=sol,
        n_nodes=G.number_of_nodes(),
        dimension=3,
    )

    subsection("Final Synchronization Distance")
    add_to_summary("Final max pairwise distance", final_distance, indent=2)

    initial_condition = dynamics.return_init_cond()

    subsection("Initial Condition")
    add_to_summary("Initial condition length", len(initial_condition), indent=2)
    add_to_summary("Initial condition", initial_condition, indent=2)

    np.save(OUTPUT_DIR / "rossler_sol.npy", sol)
    np.save(OUTPUT_DIR / "rossler_t.npy", t)
    np.save(OUTPUT_DIR / "rossler_initial_condition.npy", initial_condition)

    subsection("Saved Rössler Arrays")
    add_to_summary("Saved", OUTPUT_DIR / "rossler_sol.npy", indent=2)
    add_to_summary("Saved", OUTPUT_DIR / "rossler_t.npy", indent=2)
    add_to_summary("Saved", OUTPUT_DIR / "rossler_initial_condition.npy", indent=2)

    sync_times = record_sync_times(
        dynamics=dynamics,
        sol=sol,
        t=t,
        oscillator_dimension=3,
    )

    record_sync_consistency(final_distance, sync_times)

    subsection("Saved Rössler Plots")
    save_rossler_x_plot(sol, t, G.number_of_nodes())
    save_rossler_node0_3d_plot(sol)

    add_to_summary("")
    add_to_summary("Rössler baseline complete.")

    return sol, t


def coupling_sweep():
    section("Coupling Strength Sweep")

    np.random.seed(123)

    G = test_graph()
    n_nodes = G.number_of_nodes()
    oscillator_dimension = 3
    state_dimension = n_nodes * oscillator_dimension

    fixed_init_cond = np.random.normal(0, 1, state_dimension)

    tmax = 150
    timestep = 0.05
    coupling_values = [0.01, 0.1, 0.5, 1.0, 1.25, 1.5] # anything over 1 causes an error!

    subsection("Sweep Setup")
    add_to_summary("Random seed", 123, indent=2)
    add_to_summary("Graph nodes", n_nodes, indent=2)
    add_to_summary("Graph edges", G.number_of_edges(), indent=2)
    add_to_summary("State dimension", state_dimension, indent=2)
    add_to_summary("Fixed initial condition", fixed_init_cond, indent=2)
    add_to_summary("tmax", tmax, indent=2)
    add_to_summary("timestep", timestep, indent=2)
    add_to_summary("Coupling values", coupling_values, indent=2)

    for coupling_strength in coupling_values:
        section(f"Coupling strength = {coupling_strength}")

        dynamics = laplacian_dynamics()

        add_to_summary("Requested coupling strength", coupling_strength, indent=2)
        add_to_summary("Passing coupling strength", coupling_strength, indent=2)

        def run_simulation():
            return dynamics.continuous_time_nonlinear_dynamics(
                G=G,
                tmax=tmax,
                timestep=timestep,
                init_cond=fixed_init_cond.copy(),
                dynamics_type="Rossler",
                dynamics_params=ROSSLER_PARAMS,
                coupling_strength=coupling_strength,
            )

        sol, t = run_simulation()

        record_solution_info(sol, t, expected_state_variables=state_dimension)

        final_distance = final_max_pairwise_distance(
            sol=sol,
            n_nodes=n_nodes,
            dimension=oscillator_dimension,
        )

        subsection("Final Synchronization Distance")
        add_to_summary("Final max pairwise distance", final_distance, indent=2)

        sync_times = record_sync_times(
            dynamics=dynamics,
            sol=sol,
            t=t,
            oscillator_dimension=oscillator_dimension,
        )

        record_sync_consistency(final_distance, sync_times)

    add_to_summary("")
    add_to_summary("Coupling strength sweep complete.")


def linear_baseline():
    section("Linear Laplacian Baseline")

    np.random.seed(7)

    dynamics = laplacian_dynamics()
    G = test_graph()

    graph_info(G)
    laplacian_info(dynamics, G, filename="linear_laplacian.npy")

    tmax = 10
    timestep = 0.05

    subsection("Linear Parameters")
    add_to_summary("Random seed", 7, indent=2)
    add_to_summary("Graph nodes", G.number_of_nodes(), indent=2)
    add_to_summary("Graph edges", G.number_of_edges(), indent=2)
    add_to_summary("tmax", tmax, indent=2)
    add_to_summary("timestep", timestep, indent=2)
    add_to_summary("init_cond_type", "normal", indent=2)
    add_to_summary("init_cond_params", [0, 1], indent=2)
    add_to_summary("init_cond_offset", 0, indent=2)

    def run_simulation():
        return dynamics.continuous_time_linear_dynamics(
            G=G,
            tmax=tmax,
            timestep=timestep,
            init_cond_type="normal",
            init_cond_params=[0, 1],
            init_cond_offset=0,
        )

    sol, t = run_simulation()

    record_solution_info(sol, t, expected_state_variables=G.number_of_nodes())

    initial_average = np.mean(sol[0])
    final_average = np.mean(sol[-1])
    final_spread = np.max(sol[-1]) - np.min(sol[-1])

    subsection("Linear Consensus Information")
    add_to_summary("Initial average", initial_average, indent=2)
    add_to_summary("Final average", final_average, indent=2)
    add_to_summary("Final spread max-min", final_spread, indent=2)

    np.save(OUTPUT_DIR / "linear_sol.npy", sol)
    np.save(OUTPUT_DIR / "linear_t.npy", t)

    subsection("Saved Linear Arrays")
    add_to_summary("Saved", OUTPUT_DIR / "linear_sol.npy", indent=2)
    add_to_summary("Saved", OUTPUT_DIR / "linear_t.npy", indent=2)

    subsection("Saved Linear Plot")
    save_linear_plot(sol, t, G.number_of_nodes())

    add_to_summary("")
    add_to_summary("Linear baseline complete.")

    return sol, t


def main():
    make_output()
    start_summary_file()


    rossler()
    coupling_sweep()
    linear_baseline()

    section("Baseline Script Complete")
    add_to_summary("All baseline runs completed successfully.")
    add_to_summary("Output directory", OUTPUT_DIR)

    

if __name__ == "__main__":
    main()
