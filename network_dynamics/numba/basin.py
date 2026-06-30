"""Numba-parallelized basin stability for networks of coupled oscillators.

Mirrors the MSF scan_msf pattern: one @njit(parallel=True) kernel with
prange over initial conditions instead of K values. All temporary arrays
are pre-allocated at _run_trial entry — no heap allocation in the hot loop.

Coupling is computed using the *sparse* Laplacian in CSR format + the 3×3
inner coupling matrix H, rather than the dense (N*3)×(N*3) Kronecker product.
For a typical ER graph at p=0.15 with N=100 this reduces the coupling
computation from 90,000 multiply-adds to ~1,560 — ~57× fewer FLOPs — and
the CSR arrays (~20 KB) fit entirely in L1 cache.

Primary entry point:
    run_basin_numba(initial_conditions,
                    sigma_L_data, sigma_L_indices, sigma_L_indptr,
                    H_matrix, params, dyn_id,
                    dt, n_steps, sync_tol, max_abs_threshold,
                    window_start, success_code)
        -> (ever_synced, sync_time, final_distance, window_max_distance,
            integration_failed)  each shape (n_trials,)

success_code: 0=final_success, 1=window_success, 2=first_crossing (early exit)

Helper:
    build_sparse_coupling(L, sigma, H) -> (data, indices, indptr, H_matrix)
"""

from __future__ import annotations

import numpy as np
import numba
from numba import prange
import scipy.sparse as sp

from network_dynamics.core.msf.dynamics import ROSSLER, LORENZ, CHEN, CHUA, HR

FINAL_SUCCESS  = 0
WINDOW_SUCCESS = 1
FIRST_CROSSING = 2


# ─── helpers (Python-level) ───────────────────────────────────────────────────

def build_sparse_coupling(L: np.ndarray, sigma: float, H: np.ndarray):
    """Convert dense Laplacian + coupling strength into CSR arrays for Numba.

    Returns (sigma_L_data, sigma_L_indices, sigma_L_indptr, H_c, h_tgt, h_src)

    h_tgt, h_src encode the coupling scheme:
      - If H has exactly one non-zero H[tgt, src], returns (tgt, src) and
        absorbs H[tgt,src] into sigma_L_data. The kernel skips the 3×3 multiply
        and directly does ds[i*3+tgt] -= w * state[j*3+src]  (~9× fewer FLOPs).
      - If H is not rank-1, returns (-1, -1) and the full 3×3 H_c is used.
    """
    sigma_L = sp.csr_matrix(L * sigma)
    sigma_L.eliminate_zeros()
    H_arr = np.asarray(H, dtype=np.float64)

    nnz = np.count_nonzero(H_arr)
    if nnz == 1:
        h_tgt, h_src = [int(x) for x in np.argwhere(H_arr != 0)[0]]
        h_val = float(H_arr[h_tgt, h_src])
        # Absorb H value into the CSR data so the kernel avoids the 3×3 loop
        data = np.ascontiguousarray(sigma_L.data * h_val, dtype=np.float64)
    else:
        h_tgt, h_src = -1, -1
        data = np.ascontiguousarray(sigma_L.data, dtype=np.float64)

    return (
        data,
        np.ascontiguousarray(sigma_L.indices, dtype=np.int32),
        np.ascontiguousarray(sigma_L.indptr,  dtype=np.int32),
        np.ascontiguousarray(H_arr,           dtype=np.float64),
        h_tgt,
        h_src,
    )


# ─── per-node uncoupled RHS ───────────────────────────────────────────────────

@numba.njit(cache=True, inline="always")
def _node_rhs_into(x, y, z, params, dyn_id, ds, base):
    """Write uncoupled single-node RHS into ds[base:base+3]."""
    if dyn_id == ROSSLER:
        a = params[0]; b = params[1]; c = params[2]
        ds[base]   = -y - z
        ds[base+1] = x + a * y
        ds[base+2] = b + z * (x - c)

    elif dyn_id == LORENZ:
        sig = params[0]; beta = params[1]; rho = params[2]
        ds[base]   = sig * (y - x)
        ds[base+1] = x * (rho - z) - y
        ds[base+2] = x * y - beta * z

    elif dyn_id == CHEN:
        a = params[0]; b = params[1]; c = params[2]
        ds[base]   = a * (y - x)
        ds[base+1] = (c - a - z) * x + c * y
        ds[base+2] = x * y - b * z

    elif dyn_id == CHUA:
        al = params[0]; be = params[1]; ga = params[2]
        a_nl = params[3]; b_nl = params[4]
        if abs(x) <= 1.0:
            f = -a_nl * x
        elif x > 1.0:
            f = -b_nl * x - a_nl + b_nl
        else:
            f = -b_nl * x + a_nl - b_nl
        ds[base]   = al * (y - x + f)
        ds[base+1] = x - y + z
        ds[base+2] = -be * y - ga * z

    else:  # HR
        I   = params[0]; r = params[1]; s_p = params[2]
        ds[base]   = y + 3.0 * x * x - x * x * x - z + I
        ds[base+1] = 1.0 - 5.0 * x * x - y
        ds[base+2] = -r * z + r * s_p * (x + 1.6)


# ─── sparse coupling term ────────────────────────────────────────────────────

@numba.njit(cache=True, inline="always")
def _apply_coupling_sparse(state, ds,
                            sigma_L_data, sigma_L_indices, sigma_L_indptr,
                            H_matrix, h_tgt, h_src):
    """Subtract sparse coupling from ds: ds -= (sigma*L ⊗ H) @ state.

    Iterates only over non-zero Laplacian entries — O(edges) instead of O(N²).

    h_tgt >= 0 signals a rank-1 H (sigma*H value already absorbed into
    sigma_L_data): only 1 multiply-add per non-zero instead of 9.
    h_tgt < 0 falls back to the full 3×3 H_matrix multiply.
    """
    n_nodes = len(sigma_L_indptr) - 1
    if h_tgt >= 0:
        # Rank-1 fast path: ds[i*3+h_tgt] -= w * state[j*3+h_src]
        for i in range(n_nodes):
            bi = i * 3 + h_tgt
            for ptr in range(sigma_L_indptr[i], sigma_L_indptr[i + 1]):
                ds[bi] -= sigma_L_data[ptr] * state[sigma_L_indices[ptr] * 3 + h_src]
    else:
        # General H path
        for i in range(n_nodes):
            bi = i * 3
            for ptr in range(sigma_L_indptr[i], sigma_L_indptr[i + 1]):
                j   = sigma_L_indices[ptr]
                w   = sigma_L_data[ptr]
                bj  = j * 3
                sj0 = state[bj];  sj1 = state[bj + 1];  sj2 = state[bj + 2]
                ds[bi]     -= w * (H_matrix[0, 0] * sj0 + H_matrix[0, 1] * sj1 + H_matrix[0, 2] * sj2)
                ds[bi + 1] -= w * (H_matrix[1, 0] * sj0 + H_matrix[1, 1] * sj1 + H_matrix[1, 2] * sj2)
                ds[bi + 2] -= w * (H_matrix[2, 0] * sj0 + H_matrix[2, 1] * sj1 + H_matrix[2, 2] * sj2)


# ─── full network RHS ────────────────────────────────────────────────────────

@numba.njit(cache=True, inline="always")
def _rhs_network_into(state, ds,
                      sigma_L_data, sigma_L_indices, sigma_L_indptr,
                      H_matrix, h_tgt, h_src, params, dyn_id):
    """Network RHS: uncoupled per-node dynamics then subtract sparse coupling."""
    n_nodes = (len(sigma_L_indptr) - 1)
    for i in range(n_nodes):
        base = i * 3
        _node_rhs_into(state[base], state[base + 1], state[base + 2],
                       params, dyn_id, ds, base)
    _apply_coupling_sparse(state, ds,
                           sigma_L_data, sigma_L_indices, sigma_L_indptr,
                           H_matrix, h_tgt, h_src)


# ─── RK4 step ────────────────────────────────────────────────────────────────

@numba.njit(cache=True, inline="always")
def _rk4_step(state, state_out, dt,
              sigma_L_data, sigma_L_indices, sigma_L_indptr, H_matrix,
              h_tgt, h_src, params, dyn_id, k1, k2, k3, k4, tmp):
    """Fixed-step RK4. Writes next state into state_out.

    k1..k4, tmp are pre-allocated (N*3,) scratch arrays.
    """
    dt2 = 0.5 * dt
    dt6 = dt / 6.0
    m = len(state)

    _rhs_network_into(state, k1,
                      sigma_L_data, sigma_L_indices, sigma_L_indptr,
                      H_matrix, h_tgt, h_src, params, dyn_id)

    for i in range(m):
        tmp[i] = state[i] + dt2 * k1[i]
    _rhs_network_into(tmp, k2,
                      sigma_L_data, sigma_L_indices, sigma_L_indptr,
                      H_matrix, h_tgt, h_src, params, dyn_id)

    for i in range(m):
        tmp[i] = state[i] + dt2 * k2[i]
    _rhs_network_into(tmp, k3,
                      sigma_L_data, sigma_L_indices, sigma_L_indptr,
                      H_matrix, h_tgt, h_src, params, dyn_id)

    for i in range(m):
        tmp[i] = state[i] + dt * k3[i]
    _rhs_network_into(tmp, k4,
                      sigma_L_data, sigma_L_indices, sigma_L_indptr,
                      H_matrix, h_tgt, h_src, params, dyn_id)

    for i in range(m):
        state_out[i] = state[i] + dt6 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])


# ─── synchronization distance ────────────────────────────────────────────────

@numba.njit(cache=True, inline="always")
def _sync_distance(state, n_nodes):
    """O(N) synchronization metric: bounding-box diagonal of node states.

    Returns sqrt(range_x² + range_y² + range_z²) which satisfies:
        max_pairwise_distance ≤ result ≤ sqrt(3) × max_pairwise_distance

    Using this as the sync criterion is conservative (requires slightly
    tighter synchronization than max-pairwise) but 50× faster for N=100.
    State layout: [x0,y0,z0, x1,y1,z1, ...].
    """
    xmin = state[0]; xmax = state[0]
    ymin = state[1]; ymax = state[1]
    zmin = state[2]; zmax = state[2]
    for i in range(1, n_nodes):
        b = i * 3
        v = state[b]
        if v < xmin:
            xmin = v
        elif v > xmax:
            xmax = v
        v = state[b + 1]
        if v < ymin:
            ymin = v
        elif v > ymax:
            ymax = v
        v = state[b + 2]
        if v < zmin:
            zmin = v
        elif v > zmax:
            zmax = v
    rx = xmax - xmin;  ry = ymax - ymin;  rz = zmax - zmin
    return (rx * rx + ry * ry + rz * rz) ** 0.5


# ─── single trial ────────────────────────────────────────────────────────────

@numba.njit(cache=True)
def _run_trial(initial_state,
               sigma_L_data, sigma_L_indices, sigma_L_indptr, H_matrix,
               h_tgt, h_src,
               params, dyn_id, dt, n_steps,
               sync_tol, max_abs_threshold, window_start, success_code):
    """Integrate one trial and return basin metrics.

    Returns (ever_synced, sync_time, final_distance, window_max_distance,
             integration_failed).

    success_code == FIRST_CROSSING enables early exit once the trial resolves.
    """
    m = len(initial_state)
    n_nodes = m // 3

    # Pre-allocate all working buffers once at function entry
    state      = initial_state.copy()
    state_next = np.empty(m)
    k1 = np.empty(m); k2 = np.empty(m)
    k3 = np.empty(m); k4 = np.empty(m)
    tmp = np.empty(m)

    ever_synced    = False
    sync_time      = np.inf
    health_ok      = True
    window_max     = 0.0
    window_max_set = False

    # Check initial synchronization
    dist = _sync_distance(state, n_nodes)
    if dist < sync_tol:
        ever_synced = True
        sync_time   = 0.0

    for step in range(1, n_steps + 1):
        _rk4_step(state, state_next, dt,
                  sigma_L_data, sigma_L_indices, sigma_L_indptr, H_matrix,
                  h_tgt, h_src, params, dyn_id, k1, k2, k3, k4, tmp)

        # Copy next state
        for i in range(m):
            state[i] = state_next[i]

        # Health check
        all_finite = True
        ma = 0.0
        for i in range(m):
            v = state[i]
            if v != v:           # NaN
                all_finite = False
                break
            av = v if v >= 0.0 else -v
            if av > ma:
                ma = av
        step_ok = all_finite and ma <= max_abs_threshold
        if not step_ok:
            health_ok = False
            if success_code == FIRST_CROSSING:
                break
            continue

        dist = _sync_distance(state, n_nodes)

        if not ever_synced and dist < sync_tol:
            ever_synced = True
            sync_time   = step * dt

        if step >= window_start:
            if not window_max_set or dist > window_max:
                window_max     = dist
                window_max_set = True

        if success_code == FIRST_CROSSING and ever_synced:
            break

    final_distance     = _sync_distance(state, n_nodes)
    integration_failed = not health_ok and not ever_synced

    if not window_max_set:
        window_max = final_distance

    return ever_synced, sync_time, final_distance, window_max, integration_failed


# ─── parallel sweep over trials ──────────────────────────────────────────────

@numba.njit(parallel=True, cache=True)
def run_basin_numba(initial_conditions,
                    sigma_L_data, sigma_L_indices, sigma_L_indptr, H_matrix,
                    h_tgt, h_src,
                    params, dyn_id,
                    dt, n_steps, sync_tol, max_abs_threshold,
                    window_start, success_code):
    """Parallel basin stability scan over all initial conditions.

    Parameters
    ----------
    initial_conditions  : float64 array, shape (n_trials, N*3)
    sigma_L_data        : float64 array — CSR values of sigma * L
    sigma_L_indices     : int32 array   — CSR column indices
    sigma_L_indptr      : int32 array   — CSR row pointers (length N+1)
    H_matrix            : float64 array, shape (3, 3) — inner coupling matrix
    params              : float64 array — oscillator parameters
    dyn_id              : int           — oscillator type (ROSSLER=0 … HR=4)
    dt                  : float
    n_steps             : int
    sync_tol            : float
    max_abs_threshold   : float
    window_start        : int           — step index where window tracking begins
    success_code        : int           — 0/1/2 for final/window/first_crossing

    Returns
    -------
    Five bool/float64 arrays of shape (n_trials,):
        ever_synced, sync_time, final_distance, window_max_distance,
        integration_failed
    """
    n_trials = initial_conditions.shape[0]

    ever_synced_out        = np.empty(n_trials, dtype=np.bool_)
    sync_time_out          = np.empty(n_trials, dtype=np.float64)
    final_distance_out     = np.empty(n_trials, dtype=np.float64)
    window_max_out         = np.empty(n_trials, dtype=np.float64)
    integration_failed_out = np.empty(n_trials, dtype=np.bool_)

    for trial in prange(n_trials):
        es, st, fd, wm, fi = _run_trial(
            initial_conditions[trial],
            sigma_L_data, sigma_L_indices, sigma_L_indptr, H_matrix,
            h_tgt, h_src,
            params, dyn_id, dt, n_steps,
            sync_tol, max_abs_threshold, window_start, success_code,
        )
        ever_synced_out[trial]        = es
        sync_time_out[trial]          = st
        final_distance_out[trial]     = fd
        window_max_out[trial]         = wm
        integration_failed_out[trial] = fi

    return (ever_synced_out, sync_time_out, final_distance_out,
            window_max_out, integration_failed_out)


# ─── helpers ─────────────────────────────────────────────────────────────────

def choose_success_code(success_definition: str) -> int:
    """Map success_definition string to integer code for the Numba kernel."""
    codes = {
        "final_success":  FINAL_SUCCESS,
        "window_success": WINDOW_SUCCESS,
        "first_crossing": FIRST_CROSSING,
    }
    if success_definition not in codes:
        raise ValueError(
            f"success_definition must be one of "
            f"{list(codes)}; got {success_definition!r}."
        )
    return codes[success_definition]
