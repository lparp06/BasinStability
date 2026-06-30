"""
Generate coupling-strength intervals from MSF zeros and graph Laplacian spectra.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import networkx as nx

from network_dynamics.core.graphs import graph_laplacian
from network_dynamics.core.msf import MSFParams, find_msf_zeros


@dataclass(frozen=True)
class CouplingStrengthInterval:
    """
    Coupling-strength interval induced by one stable MSF interval.
    """

    lower: float
    upper: float
    msf_zero_low: float
    msf_zero_high: float
    laplacian_first_nonzero: float
    laplacian_largest: float


def laplacian_nonzero_eigenvalue_bounds(G, tolerance: float = 1e-10) -> tuple[float, float]:
    """
    Return the first nonzero and largest graph-Laplacian eigenvalues.

    The first value is the smallest eigenvalue whose absolute value is larger
    than ``tolerance``. Numerical zero eigenvalues are ignored.
    """

    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative.")

    laplacian = np.asarray(graph_laplacian(G), dtype=float)

    if np.allclose(laplacian, laplacian.T, atol=tolerance, rtol=0.0):
        eigenvalues = np.linalg.eigvalsh(laplacian)
    else:
        eigenvalues_complex = np.linalg.eigvals(laplacian)
        if np.any(np.abs(eigenvalues_complex.imag) > tolerance):
            raise ValueError(
                "Laplacian has complex eigenvalues; scalar coupling-strength "
                "intervals require a real Laplacian spectrum."
            )
        eigenvalues = eigenvalues_complex.real

    eigenvalues = np.sort(np.asarray(eigenvalues, dtype=float))
    nonzero_eigenvalues = eigenvalues[np.abs(eigenvalues) > tolerance]

    if nonzero_eigenvalues.size == 0:
        raise ValueError("Laplacian has no nonzero eigenvalues.")

    first_nonzero = float(nonzero_eigenvalues[0])
    largest = float(eigenvalues[-1])

    if first_nonzero <= 0 or largest <= 0:
        raise ValueError("Coupling-strength intervals require positive Laplacian eigenvalues.")

    return first_nonzero, largest


def coupling_strength_intervals_from_zeros(
    G,
    msf_zeros: Iterable[float],
    tolerance: float = 1e-10,
) -> list[CouplingStrengthInterval]:
    """
    Convert MSF zeros into graph-valid scalar coupling-strength intervals.

    For each stable MSF interval ``(K_low, K_high)``, the scalar coupling
    strength ``sigma`` must satisfy:

        K_low / lambda_first_nonzero < sigma < K_high / lambda_largest

    where the lambdas are graph-Laplacian eigenvalues.
    """

    zeros = sorted(float(zero) for zero in msf_zeros)

    if len(zeros) < 2:
        return []

    first_nonzero, largest = laplacian_nonzero_eigenvalue_bounds(
        G,
        tolerance=tolerance,
    )

    intervals = []
    for index in range(0, len(zeros) - 1, 2):
        zero_low = zeros[index]
        zero_high = zeros[index + 1]
        lower = zero_low / first_nonzero
        upper = zero_high / largest

        if lower < upper:
            intervals.append(
                CouplingStrengthInterval(
                    lower=lower,
                    upper=upper,
                    msf_zero_low=zero_low,
                    msf_zero_high=zero_high,
                    laplacian_first_nonzero=first_nonzero,
                    laplacian_largest=largest,
                )
            )

    return intervals


def coupling_strength_intervals_from_stable(
    G,
    stable_intervals: list[tuple[float, float]],
    tolerance: float = 1e-10,
) -> list[CouplingStrengthInterval]:
    """
    Convert stable MSF intervals directly to graph-valid coupling-strength intervals.

    Handles type I (single zero, stable forever) and type III (unbounded tail)
    as well as type II, by working from pre-computed stable intervals rather than
    re-pairing zeros. Each stable interval ``(K_lo, K_hi)`` maps to:

        lower = K_lo / lambda_first_nonzero
        upper = K_hi / lambda_largest
    """
    if not stable_intervals:
        return []

    first_nonzero, largest = laplacian_nonzero_eigenvalue_bounds(G, tolerance)

    intervals = []
    for k_lo, k_hi in stable_intervals:
        lower = k_lo / first_nonzero
        upper = k_hi / largest
        if lower < upper:
            intervals.append(
                CouplingStrengthInterval(
                    lower=lower,
                    upper=upper,
                    msf_zero_low=k_lo,
                    msf_zero_high=k_hi,
                    laplacian_first_nonzero=first_nonzero,
                    laplacian_largest=largest,
                )
            )

    return intervals


def find_coupling_strength_intervals(
    G,
    params: MSFParams,
    K_min: float = 0.0,
    K_max: float = 10.0,
    n_K: int = 101,
    eigenvalue_tolerance: float = 1e-10,
    verbose: bool = False,
) -> list[CouplingStrengthInterval]:
    """Find MSF zeros (Numba CPU) and convert them to coupling-strength intervals."""

    _, stable_intervals = find_msf_zeros(
        params_obj=params,
        K_min=K_min,
        K_max=K_max,
        n_K=n_K,
        verbose=verbose,
    )

    return coupling_strength_intervals_from_stable(
        G=G,
        stable_intervals=stable_intervals,
        tolerance=eigenvalue_tolerance,
    )


def interval_coupling_strengths(
    interval: CouplingStrengthInterval,
    n_strengths: int,
    endpoint: bool = True,
) -> np.ndarray:
    """
    Return evenly spaced coupling strengths inside one interval.
    """

    if n_strengths <= 0:
        raise ValueError("n_strengths must be positive.")

    return np.linspace(
        interval.lower,
        interval.upper,
        n_strengths,
        endpoint=endpoint,
        dtype=float,
    )


def coupling_strengths_from_intervals(
    intervals: Iterable[CouplingStrengthInterval],
    n_strengths_per_interval: int,
    endpoint: bool = True,
) -> list[np.ndarray]:
    """
    Return one coupling-strength array for each interval.
    """

    return [
        interval_coupling_strengths(
            interval=interval,
            n_strengths=n_strengths_per_interval,
            endpoint=endpoint,
        )
        for interval in intervals
    ]

