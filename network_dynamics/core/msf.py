"""Clean JAX/MPS-compatible Rössler MSF smoke-test code.

This script computes the master-stability function Psi(K) for the
Rössler oscillator with one-component 1->1 coupling, scans K values,
finds sign-change brackets, and optionally refines zeros by bisection.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterable

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax


# ============================================================
# Configuration
# ============================================================

@dataclass(frozen=True)
class RosslerMSFConfig:
    """Run settings for the Rössler MSF calculation."""

    a: float = 0.2
    b: float = 0.2
    c: float = 9.0
    initial_state: tuple[float, float, float] = (1.0, 1.0, 1.0)
    target: int = 0
    source: int = 0
    dt: float = 0.05
    transient_time: float = 100.0
    measurement_time: float = 300.0
    qr_interval_steps: int = 10

    @property
    def transient_steps(self) -> int:
        return int(self.transient_time / self.dt)

    @property
    def measurement_steps(self) -> int:
        return int(self.measurement_time / self.dt)

    def validate(self) -> None:
        if self.dt <= 0:
            raise ValueError("dt must be positive.")
        if self.transient_steps <= 0:
            raise ValueError("transient_steps must be positive.")
        if self.measurement_steps <= 0:
            raise ValueError("measurement_steps must be positive.")
        if self.qr_interval_steps <= 0:
            raise ValueError("qr_interval_steps must be positive.")
        if self.measurement_steps % self.qr_interval_steps != 0:
            raise ValueError(
                "measurement_steps must be divisible by qr_interval_steps "
                "so all tangent growth is included in log_stretch."
            )
        if not (0 <= self.target < 3 and 0 <= self.source < 3):
            raise ValueError("target and source must be in {0, 1, 2}.")


def config_to_jax_arrays(config: RosslerMSFConfig):
    """Convert a Python config into JAX arrays used by the compiled code."""
    params = jnp.array([config.a, config.b, config.c], dtype=jnp.float32)
    initial_state = jnp.array(config.initial_state, dtype=jnp.float32)
    H = inner_coupling_matrix_jax(dimension=3, target=config.target, source=config.source)
    return params, initial_state, H


# ============================================================
# Rössler oscillator and variational equation
# ============================================================

def rossler_rhs_jax(state: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    """Rössler vector field F(s)."""
    x, y, z = state
    a, b, c = params
    return jnp.array([
        -y - z,
        x + a * y,
        b + z * (x - c),
    ], dtype=state.dtype)


def rossler_jacobian_jax(state: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    """Analytical Jacobian DF(s) for the Rössler vector field."""
    x, _y, z = state
    a, _b, c = params
    return jnp.array([
        [0.0, -1.0, -1.0],
        [1.0,    a,  0.0],
        [z,    0.0, x - c],
    ], dtype=state.dtype)


def inner_coupling_matrix_jax(dimension: int = 3, target: int = 0, source: int = 0) -> jnp.ndarray:
    """Build one-component inner coupling matrix H.

    Paper notation source+1 -> target+1.
    Example: target=0, source=0 means 1->1 coupling.
    """
    H = jnp.zeros((dimension, dimension), dtype=jnp.float32)
    return H.at[target, source].set(1.0)


def msf_rhs_jax(
    state: jnp.ndarray,
    tangent_matrix: jnp.ndarray,
    K: float | jnp.ndarray,
    H: jnp.ndarray,
    params: jnp.ndarray,
):
    """Combined RHS: ds/dt = F(s), dY/dt = [DF(s) - K H]Y."""
    dsdt = rossler_rhs_jax(state, params)
    A = rossler_jacobian_jax(state, params) - K * H
    dYdt = A @ tangent_matrix
    return dsdt, dYdt


# ============================================================
# RK4 integration
# ============================================================

def rk4_step_state_jax(state: jnp.ndarray, dt: float, params: jnp.ndarray) -> jnp.ndarray:
    """One RK4 step for the isolated synchronized oscillator."""
    k1 = rossler_rhs_jax(state, params)
    k2 = rossler_rhs_jax(state + 0.5 * dt * k1, params)
    k3 = rossler_rhs_jax(state + 0.5 * dt * k2, params)
    k4 = rossler_rhs_jax(state + dt * k3, params)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk4_step_msf_jax(
    state: jnp.ndarray,
    tangent_matrix: jnp.ndarray,
    dt: float,
    K: float | jnp.ndarray,
    H: jnp.ndarray,
    params: jnp.ndarray,
):
    """One RK4 step for the synchronized state and tangent matrix."""
    k1_s, k1_Y = msf_rhs_jax(state, tangent_matrix, K, H, params)
    k2_s, k2_Y = msf_rhs_jax(
        state + 0.5 * dt * k1_s,
        tangent_matrix + 0.5 * dt * k1_Y,
        K,
        H,
        params,
    )
    k3_s, k3_Y = msf_rhs_jax(
        state + 0.5 * dt * k2_s,
        tangent_matrix + 0.5 * dt * k2_Y,
        K,
        H,
        params,
    )
    k4_s, k4_Y = msf_rhs_jax(
        state + dt * k3_s,
        tangent_matrix + dt * k3_Y,
        K,
        H,
        params,
    )

    next_state = state + (dt / 6.0) * (k1_s + 2.0 * k2_s + 2.0 * k3_s + k4_s)
    next_tangent = tangent_matrix + (dt / 6.0) * (k1_Y + 2.0 * k2_Y + 2.0 * k3_Y + k4_Y)
    return next_state, next_tangent


def run_transient_jax(
    initial_state: jnp.ndarray,
    transient_steps: int,
    dt: float,
    params: jnp.ndarray,
) -> jnp.ndarray:
    """Integrate the isolated oscillator before measuring exponents."""
    def body_fun(_step, state):
        return rk4_step_state_jax(state, dt, params)

    return lax.fori_loop(0, transient_steps, body_fun, initial_state)


# ============================================================
# Lyapunov exponent / MSF calculation
# ============================================================

def qr_update_jax(tangent_matrix: jnp.ndarray, log_stretch: jnp.ndarray):
    """QR-renormalize tangent matrix and accumulate log stretching."""
    Q, R = jnp.linalg.qr(tangent_matrix)
    diagonal = jnp.diag(R)
    log_stretch = log_stretch + jnp.log(jnp.abs(diagonal) + 1e-30)
    return Q, log_stretch


def msf_value_jax_impl(
    K: float | jnp.ndarray,
    initial_state: jnp.ndarray,
    H: jnp.ndarray,
    params: jnp.ndarray,
    transient_steps: int,
    measurement_steps: int,
    dt: float,
    qr_interval_steps: int,
):
    """Compute Psi(K) and full Lyapunov exponent vector."""
    dimension = initial_state.shape[0]
    state = run_transient_jax(initial_state, transient_steps, dt, params)
    tangent_matrix = jnp.eye(dimension, dtype=initial_state.dtype)
    log_stretch = jnp.zeros(dimension, dtype=initial_state.dtype)

    def body_fun(step, carry):
        state, tangent_matrix, log_stretch = carry
        state, tangent_matrix = rk4_step_msf_jax(state, tangent_matrix, dt, K, H, params)

        do_qr = ((step + 1) % qr_interval_steps) == 0
        tangent_matrix, log_stretch = lax.cond(
            do_qr,
            lambda args: qr_update_jax(*args),
            lambda args: args,
            operand=(tangent_matrix, log_stretch),
        )
        return state, tangent_matrix, log_stretch

    state, tangent_matrix, log_stretch = lax.fori_loop(
        0,
        measurement_steps,
        body_fun,
        (state, tangent_matrix, log_stretch),
    )

    lyapunov_exponents = log_stretch / (measurement_steps * dt)
    psi = jnp.max(lyapunov_exponents)
    return psi, lyapunov_exponents


msf_value_jax = jax.jit(
    msf_value_jax_impl,
    static_argnames=("transient_steps", "measurement_steps", "qr_interval_steps"),
)


def scan_msf_jax_impl(
    K_values: jnp.ndarray,
    initial_state: jnp.ndarray,
    H: jnp.ndarray,
    params: jnp.ndarray,
    transient_steps: int,
    measurement_steps: int,
    dt: float,
    qr_interval_steps: int,
):
    """Vectorized MSF scan over a batch of K values."""
    def one_K(K):
        return msf_value_jax(
            K,
            initial_state,
            H,
            params,
            transient_steps,
            measurement_steps,
            dt,
            qr_interval_steps,
        )

    return jax.vmap(one_K)(K_values)


scan_msf_jax = jax.jit(
    scan_msf_jax_impl,
    static_argnames=("transient_steps", "measurement_steps", "qr_interval_steps"),
)


# ============================================================
# CPU-side analysis helpers
# ============================================================

def find_zero_brackets(K_values: Iterable[float], psi_values: Iterable[float]) -> list[tuple[float, float]]:
    """Find adjacent K intervals where Psi(K) changes sign."""
    K_values = np.asarray(K_values, dtype=float)
    psi_values = np.asarray(psi_values, dtype=float)

    left_psi = psi_values[:-1]
    right_psi = psi_values[1:]
    sign_change = np.isfinite(left_psi) & np.isfinite(right_psi) & (left_psi * right_psi < 0.0)

    return list(zip(K_values[:-1][sign_change].tolist(), K_values[1:][sign_change].tolist()))


def stable_intervals_from_brackets(brackets: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """For the standard +,-,+ case, each pair of brackets bounds one stable interval."""
    if len(brackets) < 2:
        return []
    return [(brackets[i][0], brackets[i + 1][1]) for i in range(0, len(brackets) - 1, 2)]


def psi_scalar_jax(K: float, config: RosslerMSFConfig) -> float:
    """Evaluate one Psi(K) value and return it as a Python float."""
    params, initial_state, H = config_to_jax_arrays(config)
    psi, _exponents = msf_value_jax(
        jnp.asarray(K, dtype=jnp.float32),
        initial_state,
        H,
        params,
        config.transient_steps,
        config.measurement_steps,
        config.dt,
        config.qr_interval_steps,
    )
    psi.block_until_ready()
    return float(jax.device_get(psi))


def bisect_zero_jax(
    left: float,
    right: float,
    config: RosslerMSFConfig,
    tolerance: float = 1e-3,
    max_iterations: int = 30,
) -> tuple[float, float]:
    """Refine one zero bracket using Python bisection with JAX Psi evaluations."""
    psi_left = psi_scalar_jax(left, config)
    psi_right = psi_scalar_jax(right, config)

    if psi_left == 0.0:
        return left, psi_left
    if psi_right == 0.0:
        return right, psi_right
    if psi_left * psi_right > 0:
        raise ValueError("Bisection requires opposite signs at the bracket endpoints.")

    for _ in range(max_iterations):
        midpoint = 0.5 * (left + right)
        psi_mid = psi_scalar_jax(midpoint, config)

        if abs(psi_mid) < tolerance or 0.5 * (right - left) < tolerance:
            return midpoint, psi_mid

        if psi_left * psi_mid < 0:
            right = midpoint
            psi_right = psi_mid
        else:
            left = midpoint
            psi_left = psi_mid

    zero = 0.5 * (left + right)
    return zero, psi_scalar_jax(zero, config)


def refine_brackets_jax(
    brackets: list[tuple[float, float]],
    config: RosslerMSFConfig,
    tolerance: float = 1e-3,
) -> list[tuple[float, float]]:
    """Refine all zero brackets using bisection."""
    return [bisect_zero_jax(left, right, config, tolerance=tolerance) for left, right in brackets]

def find_msf_zeros_jax(
    config: RosslerMSFConfig,
    K_min: float = 0.0,
    K_max: float = 10.0,
    n_K: int = 101,
    refine: bool = True,
    tolerance: float = 1e-3,
    chunk_size: int | None = None,
    verbose: bool = False,
) -> list[float]:
    """
    Compute all zeros of the Rössler master-stability function Psi(K).

    This function is meant to be called by other code.

    Parameters
    ----------
    config : RosslerMSFConfig
        MSF configuration containing oscillator parameters, timestep,
        transient time, measurement time, and coupling component.

    K_min : float
        Left edge of K scan.

    K_max : float
        Right edge of K scan.

    n_K : int
        Number of K values in the coarse scan.

    refine : bool
        If True, refine each zero bracket using bisection.
        If False, return midpoint estimates from the coarse brackets.

    tolerance : float
        Bisection tolerance used only when refine=True.

    chunk_size : int | None
        If None, scan all K values in one JAX call.
        If an integer, scan K values in chunks. Useful for progress printing
        and avoiding overly large compiled batches.

    verbose : bool
        If True, print progress updates.

    Returns
    -------
    zeros : list[float]
        Estimated K values where Psi(K) = 0.
    """
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

        K_values = jnp.asarray(K_values_np, dtype=jnp.float32)

        psi_values, _exponent_values = scan_msf_jax(
            K_values,
            initial_state,
            H,
            params,
            config.transient_steps,
            config.measurement_steps,
            config.dt,
            config.qr_interval_steps,
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

            K_chunk = jnp.asarray(K_values_np[start:stop], dtype=jnp.float32)

            psi_chunk, _exponent_chunk = scan_msf_jax(
                K_chunk,
                initial_state,
                H,
                params,
                config.transient_steps,
                config.measurement_steps,
                config.dt,
                config.qr_interval_steps,
            )

            psi_chunk.block_until_ready()
            psi_chunks.append(np.asarray(jax.device_get(psi_chunk)))

        psi_values_np = np.concatenate(psi_chunks)

    brackets = find_zero_brackets(K_values_np, psi_values_np)

    if verbose:
        print("Zero brackets:", brackets)

    if not brackets:
        return []

    if refine:
        refined = refine_brackets_jax(
            brackets=brackets,
            config=config,
            tolerance=tolerance,
        )

        zeros = [zero for zero, _psi_at_zero in refined]

    else:
        zeros = [
            0.5 * (left + right)
            for left, right in brackets
        ]

    return zeros
