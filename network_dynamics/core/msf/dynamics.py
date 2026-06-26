"""Numba-JIT RHS and Jacobian for all 5 supported oscillators.

Oscillator ID constants:
    ROSSLER = 0
    LORENZ  = 1
    CHEN    = 2
    CHUA    = 3
    HR      = 4

Each oscillator is a 3D autonomous system.  ``rhs_jac(state, params, dyn_id)``
returns the derivative vector and Jacobian matrix as pre-allocated NumPy arrays.
"""

import numpy as np
import numba

ROSSLER = 0
LORENZ  = 1
CHEN    = 2
CHUA    = 3
HR      = 4

DYN_IDS = {"rossler": ROSSLER, "lorenz": LORENZ, "chen": CHEN, "chua": CHUA, "hr": HR}

# Number of parameters per oscillator
PARAM_LENGTHS = {ROSSLER: 3, LORENZ: 3, CHEN: 3, CHUA: 5, HR: 3}

# Safe initial states that quickly reach the chaotic attractor
INITIAL_STATES = {
    ROSSLER: np.array([1.0,  1.0,  1.0]),
    LORENZ:  np.array([1.0,  1.0,  1.0]),
    CHEN:    np.array([1.0,  1.0,  1.0]),
    CHUA:    np.array([0.1,  0.0,  0.0]),
    HR:      np.array([-1.3, -8.0, 1.0]),
}


@numba.njit(cache=True, inline="always")
def rhs_jac(state, params, dyn_id):
    """RHS vector and Jacobian for one 3D oscillator state.

    Returns (ds, J) where ds is shape (3,) and J is shape (3,3).
    """
    x = state[0]; y = state[1]; z = state[2]

    if dyn_id == ROSSLER:
        a = params[0]; b = params[1]; c = params[2]
        ds = np.empty(3)
        ds[0] = -y - z
        ds[1] =  x + a*y
        ds[2] =  b + z*(x - c)
        J = np.empty((3, 3))
        J[0,0] = 0.;  J[0,1] = -1.; J[0,2] = -1.
        J[1,0] = 1.;  J[1,1] =  a;  J[1,2] =  0.
        J[2,0] = z;   J[2,1] =  0.; J[2,2] =  x - c
        return ds, J

    elif dyn_id == LORENZ:
        sig = params[0]; beta = params[1]; rho = params[2]
        ds = np.empty(3)
        ds[0] = sig*(y - x)
        ds[1] = x*(rho - z) - y
        ds[2] = x*y - beta*z
        J = np.empty((3, 3))
        J[0,0] = -sig;  J[0,1] =  sig; J[0,2] =  0.
        J[1,0] = rho-z; J[1,1] = -1.;  J[1,2] = -x
        J[2,0] = y;     J[2,1] =  x;   J[2,2] = -beta
        return ds, J

    elif dyn_id == CHEN:
        a = params[0]; b = params[1]; c = params[2]
        ds = np.empty(3)
        ds[0] = a*(y - x)
        ds[1] = (c - a - z)*x + c*y
        ds[2] = x*y - b*z
        J = np.empty((3, 3))
        J[0,0] = -a;     J[0,1] = a;  J[0,2] = 0.
        J[1,0] = c-a-z;  J[1,1] = c;  J[1,2] = -x
        J[2,0] = y;      J[2,1] = x;  J[2,2] = -b
        return ds, J

    elif dyn_id == CHUA:
        al = params[0]; be = params[1]; ga = params[2]
        a_nl = params[3]; b_nl = params[4]
        if abs(x) <= 1.0:
            f  = -a_nl * x
            fc =  a_nl
        elif x > 1.0:
            f  = -b_nl*x - a_nl + b_nl
            fc =  b_nl
        else:
            f  = -b_nl*x + a_nl - b_nl
            fc =  b_nl
        ds = np.empty(3)
        ds[0] = al*(y - x + f)
        ds[1] = x - y + z
        ds[2] = -be*y - ga*z
        J = np.empty((3, 3))
        J[0,0] = -al*(1. + fc); J[0,1] =  al; J[0,2] =  0.
        J[1,0] =  1.;           J[1,1] = -1.; J[1,2] =  1.
        J[2,0] =  0.;           J[2,1] = -be; J[2,2] = -ga
        return ds, J

    else:  # HR
        I = params[0]; r = params[1]; s = params[2]
        ds = np.empty(3)
        ds[0] = y + 3.*x*x - x*x*x - z + I
        ds[1] = 1. - 5.*x*x - y
        ds[2] = -r*z + r*s*(x + 1.6)
        J = np.empty((3, 3))
        J[0,0] = 6.*x - 3.*x*x; J[0,1] =  1.; J[0,2] = -1.
        J[1,0] = -10.*x;         J[1,1] = -1.; J[1,2] =  0.
        J[2,0] =  r*s;           J[2,1] =  0.; J[2,2] = -r
        return ds, J
