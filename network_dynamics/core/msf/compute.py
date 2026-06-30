"""Numba-parallelized MSF computation.

Key functions:
    run_transient(state0, params, dyn_id, dt, steps) -> settled state
    scan_msf(K_arr, state0, H_tgt, H_src, params, dyn_id, dt, msteps, qr_steps) -> psi array

"""

from math import sqrt, log
import numpy as np
import numba
from numba import prange

from network_dynamics.core.msf.dynamics import (
    ROSSLER, LORENZ, CHEN, CHUA, HR,
)


# ─── inlined 3×3 matrix multiply ─────────────────────────────────────────────

@numba.njit(cache=True, inline="always")
def _matmul3(A, B, C):
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
    C[2,2] = A[2,0]*B[0,2] + A[2,1]*B[1,2] + A[2,2]*B[2,2]


# ─── inline 3×3 Modified Gram-Schmidt ────────────────────────────────────────

@numba.njit(cache=True, inline="always")
def _qr3(Y):
    """In-place Modified Gram-Schmidt on columns of 3×3 Y.

    Returns (log|r00|, log|r11|, log|r22|) — log-stretching factors for
    Lyapunov exponent accumulation. ~30 FLOP vs ~2 µs for LAPACK dispatch.
    """
    # Column 0
    n0 = sqrt(Y[0,0]*Y[0,0] + Y[1,0]*Y[1,0] + Y[2,0]*Y[2,0])
    n0 = max(n0, 1e-15)
    Y[0,0] /= n0;  Y[1,0] /= n0;  Y[2,0] /= n0

    # Column 1: subtract projection onto col 0
    d01 = Y[0,0]*Y[0,1] + Y[1,0]*Y[1,1] + Y[2,0]*Y[2,1]
    Y[0,1] -= d01*Y[0,0];  Y[1,1] -= d01*Y[1,0];  Y[2,1] -= d01*Y[2,0]
    n1 = sqrt(Y[0,1]*Y[0,1] + Y[1,1]*Y[1,1] + Y[2,1]*Y[2,1])
    n1 = max(n1, 1e-15)
    Y[0,1] /= n1;  Y[1,1] /= n1;  Y[2,1] /= n1

    # Column 2: subtract projections onto col 0 and col 1
    d02 = Y[0,0]*Y[0,2] + Y[1,0]*Y[1,2] + Y[2,0]*Y[2,2]
    Y[0,2] -= d02*Y[0,0];  Y[1,2] -= d02*Y[1,0];  Y[2,2] -= d02*Y[2,0]
    d12 = Y[0,1]*Y[0,2] + Y[1,1]*Y[1,2] + Y[2,1]*Y[2,2]
    Y[0,2] -= d12*Y[0,1];  Y[1,2] -= d12*Y[1,1];  Y[2,2] -= d12*Y[2,1]
    n2 = sqrt(Y[0,2]*Y[0,2] + Y[1,2]*Y[1,2] + Y[2,2]*Y[2,2])
    n2 = max(n2, 1e-15)
    Y[0,2] /= n2;  Y[1,2] /= n2;  Y[2,2] /= n2

    return log(n0), log(n1), log(n2)


# ─── RHS and Jacobian written into pre-allocated arrays ──────────────────────

@numba.njit(cache=True, inline="always")
def _rhs_into(state, params, dyn_id, ds):
    """Compute RHS vector, writing into pre-allocated ds (shape 3,)."""
    x = state[0]; y = state[1]; z = state[2]

    if dyn_id == ROSSLER:
        a = params[0]; b = params[1]; c = params[2]
        ds[0] = -y - z
        ds[1] = x + a*y
        ds[2] = b + z*(x - c)

    elif dyn_id == LORENZ:
        sig = params[0]; beta = params[1]; rho = params[2]
        ds[0] = sig*(y - x)
        ds[1] = x*(rho - z) - y
        ds[2] = x*y - beta*z

    elif dyn_id == CHEN:
        a = params[0]; b = params[1]; c = params[2]
        ds[0] = a*(y - x)
        ds[1] = (c - a - z)*x + c*y
        ds[2] = x*y - b*z

    elif dyn_id == CHUA:
        al = params[0]; be = params[1]; ga = params[2]
        a_nl = params[3]; b_nl = params[4]
        if abs(x) <= 1.0:
            f = -a_nl * x
        elif x > 1.0:
            f = -b_nl*x - a_nl + b_nl
        else:
            f = -b_nl*x + a_nl - b_nl
        ds[0] = al*(y - x + f)
        ds[1] = x - y + z
        ds[2] = -be*y - ga*z

    else:  # HR
        I = params[0]; r = params[1]; s_p = params[2]
        ds[0] = y + 3.*x*x - x*x*x - z + I
        ds[1] = 1. - 5.*x*x - y
        ds[2] = -r*z + r*s_p*(x + 1.6)


@numba.njit(cache=True, inline="always")
def _jac_into(state, params, dyn_id, J):
    """Compute Jacobian matrix, writing into pre-allocated J (shape 3,3)."""
    x = state[0]; y = state[1]; z = state[2]

    if dyn_id == ROSSLER:
        a = params[0]; c = params[2]
        J[0,0] = 0.;  J[0,1] = -1.; J[0,2] = -1.
        J[1,0] = 1.;  J[1,1] =  a;  J[1,2] =  0.
        J[2,0] = z;   J[2,1] =  0.; J[2,2] =  x - c

    elif dyn_id == LORENZ:
        sig = params[0]; beta = params[1]; rho = params[2]
        J[0,0] = -sig;   J[0,1] =  sig; J[0,2] =  0.
        J[1,0] = rho-z;  J[1,1] = -1.;  J[1,2] = -x
        J[2,0] = y;      J[2,1] =  x;   J[2,2] = -beta

    elif dyn_id == CHEN:
        a = params[0]; b = params[1]; c = params[2]
        J[0,0] = -a;      J[0,1] = a;  J[0,2] = 0.
        J[1,0] = c-a-z;   J[1,1] = c;  J[1,2] = -x
        J[2,0] = y;       J[2,1] = x;  J[2,2] = -b

    elif dyn_id == CHUA:
        al = params[0]; be = params[1]; ga = params[2]
        a_nl = params[3]; b_nl = params[4]
        fc = a_nl if abs(x) <= 1.0 else b_nl
        J[0,0] = -al*(1. + fc); J[0,1] =  al; J[0,2] =  0.
        J[1,0] =  1.;           J[1,1] = -1.; J[1,2] =  1.
        J[2,0] =  0.;           J[2,1] = -be; J[2,2] = -ga

    else:  # HR
        r = params[1]; s_p = params[2]
        J[0,0] = 6.*x - 3.*x*x; J[0,1] =  1.; J[0,2] = -1.
        J[1,0] = -10.*x;         J[1,1] = -1.; J[1,2] =  0.
        J[2,0] =  r*s_p;         J[2,1] =  0.; J[2,2] = -r


# ─── transient ───────────────────────────────────────────────────────────────

@numba.njit(cache=True)
def run_transient(state0, params, dyn_id, dt, steps):
    """RK4-integrate the free oscillator to settle onto the chaotic attractor."""
    s = state0.copy()
    ds1 = np.empty(3); ds2 = np.empty(3); ds3 = np.empty(3); ds4 = np.empty(3)
    s2  = np.empty(3); s3  = np.empty(3); s4  = np.empty(3)
    dt2 = 0.5 * dt;    dt6 = dt / 6.

    for _ in range(steps):
        _rhs_into(s, params, dyn_id, ds1)
        for i in range(3): s2[i] = s[i] + dt2 * ds1[i]
        _rhs_into(s2, params, dyn_id, ds2)
        for i in range(3): s3[i] = s[i] + dt2 * ds2[i]
        _rhs_into(s3, params, dyn_id, ds3)
        for i in range(3): s4[i] = s[i] + dt  * ds3[i]
        _rhs_into(s4, params, dyn_id, ds4)
        for i in range(3):
            s[i] += dt6 * (ds1[i] + 2.*ds2[i] + 2.*ds3[i] + ds4[i])
    return s


# ─── single-K MSF value — zero heap allocations in hot loop ──────────────────

@numba.njit(cache=True)
def _one_k(K, state0, H_tgt, H_src, params, dyn_id, dt, msteps, qr_steps):
    """Largest Lyapunov exponent Psi(K) for one coupling strength.

    All working arrays are pre-allocated here (once per K-value call).
    The inner RK4 loop never calls np.empty/np.zeros, avoiding malloc
    contention when many threads run simultaneously via prange.
    """
    # State and tangent matrix
    s = state0.copy()
    Y = np.eye(3)

    # Pre-allocate RK4 working arrays (reused every step)
    ds1 = np.empty(3); ds2 = np.empty(3); ds3 = np.empty(3); ds4 = np.empty(3)
    s2  = np.empty(3); s3  = np.empty(3); s4  = np.empty(3)
    J   = np.empty((3, 3))
    dY1 = np.empty((3, 3)); dY2 = np.empty((3, 3))
    dY3 = np.empty((3, 3)); dY4 = np.empty((3, 3))
    Ytmp = np.empty((3, 3))   # Y + 0.5*dt*dY intermediate

    acc0 = 0.; acc1 = 0.; acc2 = 0.
    n_qr = msteps // qr_steps
    dt2 = 0.5 * dt;  dt6 = dt / 6.

    for _ in range(n_qr):
        for _ in range(qr_steps):

            # ── RK4 stage 1 ─────────────────────────────────────────
            _rhs_into(s, params, dyn_id, ds1)
            _jac_into(s, params, dyn_id, J)
            _matmul3(J, Y, dY1)                    # dY1 = J @ Y
            dY1[H_tgt,0] -= K*Y[H_src,0]           # subtract K*(H@Y) row H_tgt
            dY1[H_tgt,1] -= K*Y[H_src,1]
            dY1[H_tgt,2] -= K*Y[H_src,2]

            # ── RK4 stage 2 ─────────────────────────────────────────
            s2[0]=s[0]+dt2*ds1[0]; s2[1]=s[1]+dt2*ds1[1]; s2[2]=s[2]+dt2*ds1[2]
            for i in range(3):
                for j in range(3):
                    Ytmp[i,j] = Y[i,j] + dt2*dY1[i,j]
            _rhs_into(s2, params, dyn_id, ds2)
            _jac_into(s2, params, dyn_id, J)
            _matmul3(J, Ytmp, dY2)
            dY2[H_tgt,0] -= K*Ytmp[H_src,0]
            dY2[H_tgt,1] -= K*Ytmp[H_src,1]
            dY2[H_tgt,2] -= K*Ytmp[H_src,2]

            # ── RK4 stage 3 ─────────────────────────────────────────
            s3[0]=s[0]+dt2*ds2[0]; s3[1]=s[1]+dt2*ds2[1]; s3[2]=s[2]+dt2*ds2[2]
            for i in range(3):
                for j in range(3):
                    Ytmp[i,j] = Y[i,j] + dt2*dY2[i,j]
            _rhs_into(s3, params, dyn_id, ds3)
            _jac_into(s3, params, dyn_id, J)
            _matmul3(J, Ytmp, dY3)
            dY3[H_tgt,0] -= K*Ytmp[H_src,0]
            dY3[H_tgt,1] -= K*Ytmp[H_src,1]
            dY3[H_tgt,2] -= K*Ytmp[H_src,2]

            # ── RK4 stage 4 ─────────────────────────────────────────
            s4[0]=s[0]+dt*ds3[0]; s4[1]=s[1]+dt*ds3[1]; s4[2]=s[2]+dt*ds3[2]
            for i in range(3):
                for j in range(3):
                    Ytmp[i,j] = Y[i,j] + dt*dY3[i,j]
            _rhs_into(s4, params, dyn_id, ds4)
            _jac_into(s4, params, dyn_id, J)
            _matmul3(J, Ytmp, dY4)
            dY4[H_tgt,0] -= K*Ytmp[H_src,0]
            dY4[H_tgt,1] -= K*Ytmp[H_src,1]
            dY4[H_tgt,2] -= K*Ytmp[H_src,2]

            # ── update state and tangent ─────────────────────────────
            s[0] += dt6*(ds1[0]+2.*ds2[0]+2.*ds3[0]+ds4[0])
            s[1] += dt6*(ds1[1]+2.*ds2[1]+2.*ds3[1]+ds4[1])
            s[2] += dt6*(ds1[2]+2.*ds2[2]+2.*ds3[2]+ds4[2])
            for i in range(3):
                for j in range(3):
                    Y[i,j] += dt6*(dY1[i,j]+2.*dY2[i,j]+2.*dY3[i,j]+dY4[i,j])

        # QR reorthonormalize (in-place Modified Gram-Schmidt)
        l0, l1, l2 = _qr3(Y)
        acc0 += l0;  acc1 += l1;  acc2 += l2

    total_time = msteps * dt
    le0 = acc0 / total_time
    le1 = acc1 / total_time
    le2 = acc2 / total_time
    if le0 >= le1:
        if le0 >= le2: return le0
        return le2
    if le1 >= le2: return le1
    return le2


# ─── parallel K sweep ────────────────────────────────────────────────────────

@numba.njit(parallel=True, cache=True)
def scan_msf(K_arr, state0, H_tgt, H_src, params, dyn_id, dt, msteps, qr_steps):
    """Compute Psi(K) for every K in K_arr in parallel across CPU cores.

    Args:
        K_arr:     1D array of coupling strengths.
        state0:    Settled attractor state from ``run_transient``.
        H_tgt:     Row index of the 1 in the inner coupling matrix H.
        H_src:     Column index of the 1 in H.
        params:    Parameter array for the chosen oscillator.
        dyn_id:    Integer oscillator ID (ROSSLER=0 … HR=4).
        dt:        Integration timestep.
        msteps:    Total measurement steps (must be divisible by qr_steps).
        qr_steps:  Steps between QR reorthonormalisations.

    Returns:
        psi: 1D array of max Lyapunov exponent for each K.
    """
    n = len(K_arr)
    psi = np.empty(n)
    for i in prange(n):
        psi[i] = _one_k(K_arr[i], state0, H_tgt, H_src,
                        params, dyn_id, dt, msteps, qr_steps)
    return psi


# ─── high-level zero-finding ─────────────────────────────────────────────────

def find_msf_zeros(params_obj, K_min, K_max, n_K,
                   min_sep=1.0, max_zeros=4, verbose=False):
    """Run transient + MSF scan and return zero crossings.

    Args:
        params_obj: Object with attributes matching ``MSFParams``.
        K_min, K_max: Coupling-strength scan range.
        n_K:          Number of K values to evaluate.
        min_sep:      Minimum bracket separation for noise filtering.
        max_zeros:    Maximum number of zeros to return.
        verbose:      Print progress if True.

    Returns:
        zeros: list of float K values where Psi crosses zero.
    """
    from network_dynamics.core.msf.dynamics import PARAM_LENGTHS, INITIAL_STATES
    from network_dynamics.core.msf.zeros import find_zeros as _find_zeros

    dyn_id   = params_obj.dyn_id
    n_p      = PARAM_LENGTHS[dyn_id]
    params   = np.array([params_obj.a, params_obj.b, params_obj.c,
                         params_obj.d, params_obj.e][:n_p])
    s0       = INITIAL_STATES[dyn_id].copy()
    tr_steps = int(round(params_obj.transient_time / params_obj.dt))
    m_steps  = int(round(params_obj.measurement_time / params_obj.dt))

    if verbose:
        print(f"  transient: {tr_steps:,} steps ...", end="", flush=True)
    settled = run_transient(s0, params, dyn_id, params_obj.dt, tr_steps)
    if verbose:
        print(" done")

    K_arr = np.linspace(K_min, K_max, n_K)
    if verbose:
        print(f"  scanning {n_K} K values ({m_steps:,} steps each) ...",
              end="", flush=True)
    psi = scan_msf(K_arr, settled, params_obj.target, params_obj.source,
                   params, dyn_id, params_obj.dt, m_steps,
                   params_obj.qr_interval_steps)
    if verbose:
        print(" done")

    zeros, _, stable_intervals = _find_zeros(K_arr, psi, min_sep, max_zeros)
    return zeros, stable_intervals


# ─── warmup ──────────────────────────────────────────────────────────────────

def warmup():
    """Trigger Numba JIT compilation. Call once before timing any runs."""
    s0 = np.array([1., 1., 1.])
    p  = np.array([0.2, 0.2, 9.])
    settled = run_transient(s0, p, ROSSLER, 0.05, 2)
    K_test  = np.array([0.5])
    scan_msf(K_test, settled, 0, 0, p, ROSSLER, 0.05, 2, 1)
