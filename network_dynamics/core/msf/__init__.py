"""Master Stability Function (MSF) package — Numba-accelerated CPU implementation.

Primary entry points:
    run_transient(state0, params, dyn_id, dt, steps) -> np.ndarray
    scan_msf(K_arr, state0, H_tgt, H_src, params, dyn_id, dt, msteps, qr_steps) -> np.ndarray
    find_zeros(K, psi, min_sep, max_zeros) -> (zeros, brackets, stable_intervals)
    warmup() — trigger Numba JIT compilation before timed runs

Oscillator IDs (pass as dyn_id):
    ROSSLER=0  LORENZ=1  CHEN=2  CHUA=3  HR=4
"""

from network_dynamics.core.msf.dynamics import (
    ROSSLER, LORENZ, CHEN, CHUA, HR,
    DYN_IDS, PARAM_LENGTHS, INITIAL_STATES,
    rhs_jac,
)
from network_dynamics.core.msf.compute import (
    run_transient,
    scan_msf,
    warmup,
)
from network_dynamics.core.msf.zeros import find_zeros
from network_dynamics.core.msf.params import MSFParams
from network_dynamics.core.msf.compute import find_msf_zeros

__all__ = [
    "ROSSLER", "LORENZ", "CHEN", "CHUA", "HR",
    "DYN_IDS", "PARAM_LENGTHS", "INITIAL_STATES",
    "rhs_jac",
    "run_transient",
    "scan_msf",
    "warmup",
    "find_zeros",
    "find_msf_zeros",
    "MSFParams",
]
