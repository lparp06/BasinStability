"""Pre-compile the Numba basin kernel so the first real call is not cold.

Call warmup() once at program start (e.g. before the scan loop).
"""

from __future__ import annotations

import numpy as np

from network_dynamics.core.msf.dynamics import ROSSLER


def warmup() -> None:
    """Trigger JIT compilation on a tiny N=2, 10-trial, 50-step problem."""
    from network_dynamics.numba.basin import (
        run_basin_numba, build_sparse_coupling, FIRST_CROSSING,
    )

    n_nodes   = 2
    state_dim = n_nodes * 3
    n_trials  = 10
    n_steps   = 50

    rng = np.random.default_rng(0)
    ics = rng.uniform(-1.0, 1.0, (n_trials, state_dim)).astype(np.float64)

    L = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float64)
    H = np.zeros((3, 3), dtype=np.float64); H[0, 0] = 1.0
    sigma = 0.5

    sigma_L_data, sigma_L_indices, sigma_L_indptr, H_c, h_tgt, h_src = build_sparse_coupling(L, sigma, H)
    params = np.array([0.2, 0.2, 9.0], dtype=np.float64)

    run_basin_numba(
        initial_conditions=ics,
        sigma_L_data=sigma_L_data,
        sigma_L_indices=sigma_L_indices,
        sigma_L_indptr=sigma_L_indptr,
        H_matrix=H_c,
        h_tgt=h_tgt,
        h_src=h_src,
        params=params,
        dyn_id=ROSSLER,
        dt=0.05,
        n_steps=n_steps,
        sync_tol=1e-3,
        max_abs_threshold=1e9,
        window_start=45,
        success_code=FIRST_CROSSING,
    )
