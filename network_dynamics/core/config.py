"""Configuration for basin-stability experiments."""

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np
import networkx as nx

from network_dynamics.core.basin_common import SUCCESS_DEFINITIONS
from network_dynamics.core.oscillators import normalize_dynamics_type


@dataclass
class BasinConfig:
    """Settings for one basin-stability experiment."""

    # Network and oscillator settings
    G: nx.Graph = field(default_factory=lambda: nx.path_graph(5))
    dynamics: str = "rossler"
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
    max_abs_threshold: float = 1e9

    # Backend settings
    backend: str = "serial"
    n_workers: Optional[int] = None

    def validate(self):
        """Validate settings and return ``self`` for fluent construction."""

        if self.G is None or self.G.number_of_nodes() <= 0:
            raise ValueError("G must be a nonempty NetworkX graph.")

        if self.dimension <= 0:
            raise ValueError("dimension must be positive.")

        self.dynamics = normalize_dynamics_type(self.dynamics)

        if len(self.parameters) != 3:
            raise ValueError(
                "parameters must contain three values for the selected dynamics."
            )

        if self.coupling_strength < 0:
            raise ValueError("coupling_strength must be nonnegative.")

        if self.H is not None:
            H_array = np.asarray(self.H)
            expected_shape = (self.dimension, self.dimension)
            if H_array.shape != expected_shape:
                raise ValueError(f"H must have shape {expected_shape}.")

        for name in ("tmax", "dt", "sync_tol", "tol_max", "max_abs_threshold"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")

        if self.integrator not in ("LSODA", "RK4"):
            raise ValueError("integrator must be 'LSODA' or 'RK4'.")

        if self.n_trials <= 0:
            raise ValueError("n_trials must be positive.")

        if not isinstance(self.base_seed, int):
            raise ValueError("base_seed must be an integer.")

        if self.sampler != "uniform":
            raise ValueError("Only sampler='uniform' is currently supported.")

        if len(self.sampling_bounds) != 2:
            raise ValueError("sampling_bounds must be (low, high).")

        low, high = self.sampling_bounds

        if low >= high:
            raise ValueError("sampling_bounds must satisfy low < high.")

        if not (0 < self.window_fraction <= 1):
            raise ValueError("window_fraction must satisfy 0 < window_fraction <= 1.")

        if self.success_definition not in SUCCESS_DEFINITIONS:
            raise ValueError(
                "success_definition must be one of "
                f"{', '.join(SUCCESS_DEFINITIONS)}."
            )

        if self.backend not in ("serial", "cpu", "gpu"):
            raise ValueError("backend must be 'serial', 'cpu', or 'gpu'.")

        if self.n_workers is not None and self.n_workers <= 0:
            raise ValueError("n_workers must be positive or None.")

        return self

    @property
    def n_nodes(self):
        return self.G.number_of_nodes()

    @property
    def n_edges(self):
        return self.G.number_of_edges()

    @property
    def state_dimension(self):
        return self.n_nodes * self.dimension

    @property
    def n_time_points(self):
        return int(round(self.tmax / self.dt))

    def to_dict(self):
        return {
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "dynamics": self.dynamics,
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
            "backend": self.backend,
            "n_workers": self.n_workers,
        }
