"""
integration.py

Integrates trajectories on the CPU.
"""

import numpy as np
from scipy.integrate import solve_ivp

from network_dynamics.core.graphs import graph_laplacian
from network_dynamics.core.coupling import build_coupling_matrix
from network_dynamics.core.oscillators import rossler


def make_time_grid(tmax, dt):
    """
    Match the original project convention:
    include 0, exclude tmax.
    """

    return np.arange(0.0, tmax, dt)


def build_rhs(G, parameters, coupling_strength, H):
    """
    Build the right-hand side function for the coupled Rössler system.
    """

    L = graph_laplacian(G)

    coupling_matrix = build_coupling_matrix(
        L=L,
        H=H,
        strength=coupling_strength,
    )

    def rhs(time, state):
        return rossler(
            time,
            state,
            coupling_matrix,
            parameters,
        )

    return rhs


def integrate_lsoda(
    G,
    initial_conditions,
    parameters,
    coupling_strength,
    H,
    tmax,
    dt,
    dimension=3,
):
    """
    Integrate one trajectory using SciPy LSODA.
    """

    t = make_time_grid(tmax=tmax, dt=dt)

    rhs = build_rhs(
        G=G,
        parameters=parameters,
        coupling_strength=coupling_strength,
        H=H,
    )

    result = solve_ivp(
        rhs,
        t_span=(t[0], t[-1]),
        y0=initial_conditions,
        t_eval=t,
        method="LSODA",
    )

    sol = result.y.T
    sol_t = result.t

    return sol, sol_t


def rk4_step(rhs, time, state, dt):
    """
    Take one fixed-step RK4 step.

    RK4 means Runge-Kutta 4th order.

    It estimates the slope four times:
    - k1 at the beginning
    - k2 halfway using k1
    - k3 halfway using k2
    - k4 at the end using k3

    Then it combines them into one better update.
    """

    k1 = rhs(time, state)
    k2 = rhs(time + 0.5 * dt, state + 0.5 * dt * k1)
    k3 = rhs(time + 0.5 * dt, state + 0.5 * dt * k2)
    k4 = rhs(time + dt, state + dt * k3)

    next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return next_state


def integrate_rk4(
    G,
    initial_conditions,
    parameters,
    coupling_strength,
    H,
    tmax,
    dt,
    dimension=3,
):
    """
    Integrate one trajectory using fixed-step RK4.

    Returns
    -------
    sol : np.ndarray
        Array with shape (n_time_points, state_dimension).

    t : np.ndarray
        Time array with shape (n_time_points,).
    """

    t = make_time_grid(tmax=tmax, dt=dt)

    rhs = build_rhs(
        G=G,
        parameters=parameters,
        coupling_strength=coupling_strength,
        H=H,
    )

    state = np.asarray(initial_conditions, dtype=np.float32)

    sol = np.zeros(
        (len(t), len(state)),
        dtype=np.float32,
    )

    sol[0] = state

    for i in range(1, len(t)):
        state = rk4_step(
            rhs=rhs,
            time=t[i - 1],
            state=state,
            dt=dt,
        )

        sol[i] = state

    return sol, t


def integrate(
    G,
    initial_conditions,
    parameters,
    coupling_strength,
    H,
    tmax,
    dt,
    dimension=3,
    integrator="LSODA",
):
    """
    General integration wrapper.

    integrator can be:
    - "LSODA"
    - "RK4"
    """

    if integrator == "LSODA":
        return integrate_lsoda(
            G=G,
            initial_conditions=initial_conditions,
            parameters=parameters,
            coupling_strength=coupling_strength,
            H=H,
            tmax=tmax,
            dt=dt,
            dimension=dimension,
        )

    if integrator == "RK4":
        return integrate_rk4(
            G=G,
            initial_conditions=initial_conditions,
            parameters=parameters,
            coupling_strength=coupling_strength,
            H=H,
            tmax=tmax,
            dt=dt,
            dimension=dimension,
        )

    raise ValueError("Unknown integrator. Use integrator='LSODA' or integrator='RK4'.")
