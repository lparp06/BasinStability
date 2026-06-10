"""
config.py

Configuration object for basin-stability experiments.
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np
import networkx as nx


@dataclass
class BasinConfig:
    """
    Stores all settings for one basin-stability experiment.

    The goal is to keep experiment settings in one place instead of
    passing many separate arguments through the code.
    """

    # Network and oscillator settings
    G: nx.Graph = field(default_factory=lambda: nx.path_graph(5))
    dimension: int = 3
    parameters: Sequence[float] = (0.2, 0.2, 7.0)

    # Coupling settings
    coupling_strength: float = 1.0
    H: Optional[np.ndarray] = None

    # Integration settings
    tmax: float = 150.0
    dt: float = 0.05
    integrator: str = "LSODA"

    # Basin sampling settings
    n_trials: int = 25
    base_seed: int = 42
    sampler: str = "uniform"
    sampling_bounds: Tuple[float, float] = (-5.0, 5.0)

    # Synchronization settings
    sync_tol: float = 1e-2
    tol_max: float = 1e6
    window_fraction: float = 0.1
    success_definition: str = "window_success"

    # Numerical health settings
    max_abs_threshold: float = 1e6

    # Storage/debug settings
    store_initial_conditions: bool = False

    # Backend settings
    backend: str = "serial"
    n_workers: Optional[int] = None

    def validate(self):
        """
        Validate the configuration.

        Returns
        -------
        self
            Returning self allows convenient usage:

                config = BasinConfig(...).validate()
        """

        if self.G is None:
            raise ValueError("G must be a NetworkX graph.")

        if self.G.number_of_nodes() <= 0:
            raise ValueError("G must contain at least one node.")

        if self.dimension <= 0:
            raise ValueError("dimension must be positive.")

        if len(self.parameters) != 3:
            raise ValueError(
                "parameters must contain exactly three values: (a, b, c)."
            )

        if self.coupling_strength < 0:
            raise ValueError("coupling_strength must be nonnegative.")

        if self.H is not None:
            H_array = np.asarray(self.H)

            expected_shape = (self.dimension, self.dimension)

            if H_array.shape != expected_shape:
                raise ValueError(
                    f"H must have shape {expected_shape}, "
                    f"but got {H_array.shape}."
                )

        if self.tmax <= 0:
            raise ValueError("tmax must be positive.")

        if self.dt <= 0:
            raise ValueError("dt must be positive.")

        if self.integrator not in ("LSODA", "RK4", ):
            raise ValueError(
                "Only integrator='LSODA' or integrator='RK4' is currently supported."
            )

        if self.n_trials <= 0:
            raise ValueError("n_trials must be positive.")

        if not isinstance(self.base_seed, int):
            raise ValueError("base_seed must be an integer.")

        if self.sampler != "uniform":
            raise ValueError(
                "Only sampler='uniform' is currently supported."
            )

        if len(self.sampling_bounds) != 2:
            raise ValueError(
                "sampling_bounds must be a tuple like (low, high)."
            )

        low, high = self.sampling_bounds

        if low >= high:
            raise ValueError(
                "sampling_bounds must satisfy low < high."
            )

        if self.sync_tol <= 0:
            raise ValueError("sync_tol must be positive.")

        if self.tol_max <= 0:
            raise ValueError("tol_max must be positive.")

        if not (0 < self.window_fraction <= 1):
            raise ValueError(
                "window_fraction must satisfy 0 < window_fraction <= 1."
            )

        if self.success_definition not in (
            "final_success",
            "window_success",
        ):
            raise ValueError(
                "success_definition must be either "
                "'final_success' or 'window_success'."
            )

        if self.max_abs_threshold <= 0:
            raise ValueError("max_abs_threshold must be positive.")

        if self.backend not in ("serial", "cpu", "gpu"):
            raise ValueError(
                "backend must be one of: 'serial', 'cpu', or 'gpu'."
            )

        if self.n_workers is not None and self.n_workers <= 0:
            raise ValueError(
                "n_workers must be positive or None."
            )

        return self

    @property
    def n_nodes(self):
        """
        Number of nodes in the graph.
        """

        return self.G.number_of_nodes()

    @property
    def n_edges(self):
        """
        Number of edges in the graph.
        """

        return self.G.number_of_edges()

    @property
    def state_dimension(self):
        """
        Total state-vector size.

        For Rössler:
            state_dimension = n_nodes * 3
        """

        return self.n_nodes * self.dimension

    @property
    def n_time_points(self):
        """
        Number of time points used by the current integration convention.

        Your integration.py currently uses:

            np.arange(0, tmax, dt)

        so we match that here.
        """

        return len(np.arange(0.0, self.tmax, self.dt))

    def to_dict(self):
        """
        Convert config into a simple dictionary for saving or printing.
        """

        return {
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "dimension": self.dimension,
            "state_dimension": self.state_dimension,
            "parameters": tuple(self.parameters),
            "coupling_strength": self.coupling_strength,
            "H": self.H,
            "tmax": self.tmax,
            "dt": self.dt,
            "integrator": self.integrator,
            "n_time_points": self.n_time_points,
            "n_trials": self.n_trials,
            "base_seed": self.base_seed,
            "sampler": self.sampler,
            "sampling_bounds": self.sampling_bounds,
            "sync_tol": self.sync_tol,
            "tol_max": self.tol_max,
            "window_fraction": self.window_fraction,
            "success_definition": self.success_definition,
            "max_abs_threshold": self.max_abs_threshold,
            "store_initial_conditions": self.store_initial_conditions,
            "backend": self.backend,
            "n_workers": self.n_workers,
        }