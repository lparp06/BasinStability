"""
integration.py

Integrates trajectories on the CPU.
"""

import numpy as np
from scipy.integrate import solve_ivp

from network_dynamics.core.graphs import graph_laplacian
from network_dynamics.core.coupling import build_coupling_matrix
from network_dynamics.core.oscillators import get_oscillator_rhs


def make_n_steps(tmax, dt):
    """Compute number of time steps robustly, avoiding float-arange drift."""
    return int(round(tmax / dt))


def make_time_grid(tmax, dt):
    """
    Build time grid [0, dt, 2*dt, ...) with exactly round(tmax/dt) points.

    Using integer indexing avoids float-accumulation errors that can cause
    np.arange(0, tmax, dt) to produce n-1 or n+1 points for small dt.
    """
    n = make_n_steps(tmax, dt)
    return np.arange(n, dtype=np.float64) * dt


def build_rhs(G, parameters, coupling_strength, H, dimension=3, dynamics="rossler"):
    """
    Build the right-hand side function for the coupled oscillator system.
    """

    L = graph_laplacian(G)
    oscillator_rhs = get_oscillator_rhs(dynamics)

    coupling_matrix = build_coupling_matrix(
        L=L,
        H=H,
        strength=coupling_strength,
        dimension=dimension,
    )

    def rhs(time, state):
        return oscillator_rhs(
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
    dynamics="rossler",
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
        dimension=dimension,
        dynamics=dynamics,
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
    dynamics="rossler",
    divergence_threshold=1e9,
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
        dimension=dimension,
        dynamics=dynamics,
    )

    state = np.asarray(initial_conditions, dtype=np.float64)
    n_steps = len(t)

    sol = np.zeros((n_steps, len(state)), dtype=np.float64)
    sol[0] = state

    for i in range(1, n_steps):
        state = rk4_step(rhs=rhs, time=t[i - 1], state=state, dt=dt)

        if not np.all(np.isfinite(state)) or np.max(np.abs(state)) > divergence_threshold:
            sol[i:] = np.inf
            break

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
    dynamics="rossler",
    divergence_threshold=1e9,
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
            dynamics=dynamics,
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
            dynamics=dynamics,
            divergence_threshold=divergence_threshold,
        )

    raise ValueError("Unknown integrator. Use integrator='LSODA' or integrator='RK4'.")


def integrate_from_config(config, initial_conditions):
    """
    Integrate one trajectory using a BasinConfig.
    """

    return integrate(
        G=config.G,
        initial_conditions=initial_conditions,
        parameters=config.parameters,
        coupling_strength=config.coupling_strength,
        H=config.H,
        tmax=config.tmax,
        dt=config.dt,
        dimension=config.dimension,
        integrator=config.integrator,
        dynamics=config.dynamics,
        divergence_threshold=getattr(config, "max_abs_threshold", 1e9),
    )
