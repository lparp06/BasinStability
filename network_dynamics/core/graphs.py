"""
Graph and Laplacian utilities
"""

import numpy as np
import networkx as nx


GRAPH_TYPES = (
    "path_graph",
    "erdos_renyi",
    "barabasi_albert",
    "watts_strogatz",
)


def normalize_graph_type(graph_type):
    """
    Normalize command-line graph names to the canonical helper names.
    """

    aliases = {
        "path": "path_graph",
        "path-graph": "path_graph",
        "path_graph": "path_graph",
        "er": "erdos_renyi",
        "erdos-renyi": "erdos_renyi",
        "erdos_renyi": "erdos_renyi",
        "ba": "barabasi_albert",
        "barabasi_albert": "barabasi_albert",
        "barabasi-albert": "barabasi_albert",
        "ws": "watts_strogatz",
        "watts_strogatz": "watts_strogatz",
        "watts-strogatz": "watts_strogatz",
    }

    normalized = aliases.get(graph_type.lower())

    if normalized is None:
        raise ValueError(
            "Unknown graph_type. Supported graph types are: "
            + ", ".join(GRAPH_TYPES)
        )

    return normalized


def make_graph(
    graph_type,
    n_nodes=5,
    seed=42,
    edge_probability=0.16,
    ba_m=8,
    ws_k=6,
):
    """
    Build a NetworkX graph from a compact graph-type name and parameters.
    """

    graph_type = normalize_graph_type(graph_type)

    if graph_type == "path_graph":
        return nx.path_graph(n_nodes)

    if graph_type == "erdos_renyi":
        return nx.erdos_renyi_graph(n=n_nodes, p=edge_probability, seed=seed)

    if graph_type == "barabasi_albert":
        return nx.barabasi_albert_graph(n=n_nodes, m=ba_m, seed=seed)

    if graph_type == "watts_strogatz":
        return nx.watts_strogatz_graph(n=n_nodes, k=ws_k, p=edge_probability, seed=seed)

    raise ValueError(f"Unsupported graph_type: {graph_type}")


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
