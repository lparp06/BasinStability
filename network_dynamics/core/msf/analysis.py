"""CPU-side analysis helpers for MSF scans."""

from __future__ import annotations

from typing import Iterable

from network_dynamics.core.jax_config import enable_x64

enable_x64()

import jax
import jax.numpy as jnp
import numpy as np

from network_dynamics.core.msf.config import MSFConfig, config_to_jax_arrays
from network_dynamics.core.msf.lyapunov import scan_msf_jax


def find_zero_brackets(
    K_values: Iterable[float],
    psi_values: Iterable[float],
) -> list[tuple[float, float]]:
    """Find adjacent K intervals where Psi(K) changes sign."""

    K_values = np.asarray(K_values, dtype=float)
    psi_values = np.asarray(psi_values, dtype=float)

    left_psi = psi_values[:-1]
    right_psi = psi_values[1:]
    sign_change = (
        np.isfinite(left_psi)
        & np.isfinite(right_psi)
        & (left_psi * right_psi < 0.0)
    )

    return list(
        zip(
            K_values[:-1][sign_change].tolist(),
            K_values[1:][sign_change].tolist(),
        )
    )


def merge_close_brackets(
    brackets: list[tuple[float, float]],
    min_separation: float = 1.0,
) -> list[tuple[float, float]]:
    """Collapse clusters of nearby sign-change brackets into single brackets.

    Numerical noise can produce several rapid sign flips where only one true
    zero crossing exists.  Brackets whose left edge is within *min_separation*
    of the previous bracket's right edge are grouped into a cluster.  A cluster
    with an odd count represents a net sign change and is kept as one bracket
    spanning the full cluster; a cluster with an even count is discarded.
    """
    if not brackets:
        return []

    merged: list[tuple[float, float]] = []
    cluster_start = brackets[0][0]
    cluster_end = brackets[0][1]
    cluster_count = 1

    for left, right in brackets[1:]:
        if left - cluster_end < min_separation:
            cluster_end = right
            cluster_count += 1
        else:
            if cluster_count % 2 == 1:
                merged.append((cluster_start, cluster_end))
            cluster_start = left
            cluster_end = right
            cluster_count = 1

    if cluster_count % 2 == 1:
        merged.append((cluster_start, cluster_end))

    return merged


def stable_intervals_from_brackets(
    brackets: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """For the standard +,-,+ case, each pair of brackets bounds one stable interval."""

    if len(brackets) < 2:
        return []
    return [(brackets[i][0], brackets[i + 1][1]) for i in range(0, len(brackets) - 1, 2)]


def find_msf_zeros_jax(
    config: MSFConfig,
    K_min: float = 0.0,
    K_max: float = 10.0,
    n_K: int = 101,
    chunk_size: int | None = None,
    verbose: bool = False,
) -> list[float]:
    """Compute all zeros of the master-stability function Psi(K)."""

    config.validate()

    if K_max <= K_min:
        raise ValueError("K_max must be greater than K_min.")
    if n_K < 2:
        raise ValueError("n_K must be at least 2.")

    params, initial_state, H = config_to_jax_arrays(config)
    K_values_np = np.linspace(K_min, K_max, n_K)
    psi_chunks = []

    if chunk_size is None:
        if verbose:
            print(f"Scanning {n_K} K values from {K_min} to {K_max}...")

        K_values = jnp.asarray(K_values_np, dtype=jnp.float64)
        psi_values, _exponent_values = scan_msf_jax(
            K_values,
            initial_state,
            H,
            params,
            config.transient_steps,
            config.measurement_steps,
            config.dt,
            config.qr_interval_steps,
            config.dynamics,
        )

        psi_values.block_until_ready()
        psi_values_np = np.asarray(jax.device_get(psi_values))

    else:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive or None.")

        for start in range(0, n_K, chunk_size):
            stop = min(start + chunk_size, n_K)

            if verbose:
                print(f"Scanning K chunk {start}:{stop} of {n_K}")

            K_chunk = jnp.asarray(K_values_np[start:stop], dtype=jnp.float64)
            psi_chunk, _exponent_chunk = scan_msf_jax(
                K_chunk,
                initial_state,
                H,
                params,
                config.transient_steps,
                config.measurement_steps,
                config.dt,
                config.qr_interval_steps,
                config.dynamics,
            )

            psi_chunk.block_until_ready()
            psi_chunks.append(np.asarray(jax.device_get(psi_chunk)))

        psi_values_np = np.concatenate(psi_chunks)

    brackets = find_zero_brackets(K_values_np, psi_values_np)

    if verbose:
        print("Zero brackets:", brackets)

    if not brackets:
        return []

    return [0.5 * (left + right) for left, right in brackets]
