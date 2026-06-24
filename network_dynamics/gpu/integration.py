"""
JAX/GPU RK4 integrator for one coupled Rössler trajectory.
"""

from functools import partial

import numpy as np

from network_dynamics.core.jax_config import enable_x64

enable_x64()

import jax
import jax.numpy as jnp
from jax import lax

from network_dynamics.core.graphs import graph_laplacian
from network_dynamics.core.coupling import build_coupling_matrix
from network_dynamics.gpu.dynamics import DYNAMICS_CODES, dynamics_code, rk4_step_batch_jax


def rossler_jax(state_vector, coupling_matrix, parameters):
    """JAX coupled Rössler derivative for a single state vector."""

    a, b, c = parameters

    X = state_vector[0::3]
    Y = state_vector[1::3]
    Z = state_vector[2::3]

    # Stack per-node derivatives then flatten — avoids scatter (at[].set) ops
    dx = jnp.stack([-Y - Z, X + a * Y, b + Z * (X - c)], axis=-1).ravel()

    return dx - coupling_matrix @ state_vector


def lorenz_jax(state_vector, coupling_matrix, parameters):
    """JAX coupled Lorenz derivative for a single state vector."""

    sigma, beta, rho = parameters

    X = state_vector[0::3]
    Y = state_vector[1::3]
    Z = state_vector[2::3]

    dx = jnp.stack([sigma * (Y - X), X * (rho - Z) - Y, X * Y - beta * Z], axis=-1).ravel()

    return dx - coupling_matrix @ state_vector


def chen_jax(state_vector, coupling_matrix, parameters):
    """JAX coupled Chen derivative for a single state vector."""

    a, beta, c = parameters

    X = state_vector[0::3]
    Y = state_vector[1::3]
    Z = state_vector[2::3]

    dx = jnp.stack(
        [a * (Y - X), (c - a - Z) * X + c * Y, X * Y - beta * Z],
        axis=-1,
    ).ravel()

    return dx - coupling_matrix @ state_vector


def chua_jax(state_vector, coupling_matrix, parameters):
    """JAX coupled Chua's circuit derivative for a single state vector."""

    alpha, beta, gamma, a_nl, b_nl = parameters

    X = state_vector[0::3]
    Y = state_vector[1::3]
    Z = state_vector[2::3]

    f = jnp.where(
        jnp.abs(X) <= 1.0,
        -a_nl * X,
        jnp.where(X > 1.0, -b_nl * X - a_nl + b_nl, -b_nl * X + a_nl - b_nl),
    )

    dx = jnp.stack(
        [alpha * (Y - X + f), X - Y + Z, -beta * Y - gamma * Z],
        axis=-1,
    ).ravel()

    return dx - coupling_matrix @ state_vector


def oscillator_jax(state_vector, coupling_matrix, parameters, dynamics_code_value):
    if dynamics_code_value == DYNAMICS_CODES["rossler"]:
        return rossler_jax(state_vector, coupling_matrix, parameters)

    if dynamics_code_value == DYNAMICS_CODES["lorenz"]:
        return lorenz_jax(state_vector, coupling_matrix, parameters)

    if dynamics_code_value == DYNAMICS_CODES["chen"]:
        return chen_jax(state_vector, coupling_matrix, parameters)

    if dynamics_code_value == DYNAMICS_CODES["chua"]:
        return chua_jax(state_vector, coupling_matrix, parameters)

    raise ValueError(f"Unsupported dynamics code: {dynamics_code_value}")


def rk4_step_jax(
    state,
    dt,
    coupling_matrix,
    parameters,
    dynamics_code_value=DYNAMICS_CODES["rossler"],
):
    """
    One RK4 step using JAX.
    """

    k1 = oscillator_jax(
        state,
        coupling_matrix,
        parameters,
        dynamics_code_value,
    )

    k2 = oscillator_jax(
        state + 0.5 * dt * k1,
        coupling_matrix,
        parameters,
        dynamics_code_value,
    )

    k3 = oscillator_jax(
        state + 0.5 * dt * k2,
        coupling_matrix,
        parameters,
        dynamics_code_value,
    )

    k4 = oscillator_jax(
        state + dt * k3,
        coupling_matrix,
        parameters,
        dynamics_code_value,
    )

    next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return next_state


@partial(jax.jit, static_argnames=("n_steps", "dynamics_code_value"))
def integrate_rk4_scan_jax(
    initial_state,
    coupling_matrix,
    parameters,
    dt,
    n_steps,
    dynamics_code_value,
):
    """
    JIT-compiled RK4 trajectory integrator.

    Uses lax.scan instead of a Python for-loop.
    """

    def step_function(state, _):
        next_state = rk4_step_jax(
            state=state,
            dt=dt,
            coupling_matrix=coupling_matrix,
            parameters=parameters,
            dynamics_code_value=dynamics_code_value,
        )

        return next_state, next_state

    final_state, states_after_initial = lax.scan(
        step_function,
        initial_state,
        xs=None,
        length=n_steps - 1,
    )

    sol = jnp.concatenate(
        [
            initial_state[None, :],
            states_after_initial,
        ],
        axis=0,
    )

    return sol


def integrate_rk4_jax(
    G,
    initial_conditions,
    parameters,
    coupling_strength,
    H,
    tmax,
    dt,
    dimension=3,
    dynamics="rossler",
    return_numpy=True,
):
    """
    Integrate one trajectory using JAX RK4.

    Parameters are intentionally similar to CPU integrate_rk4.

    Returns
    -------
    sol, t
        sol has shape (n_time_points, state_dimension)
        t has shape (n_time_points,)
    """

    n_steps = int(round(tmax / dt))
    t = np.arange(n_steps, dtype=np.float64) * dt

    L = graph_laplacian(G)

    coupling_matrix_np = build_coupling_matrix(
        L=L,
        H=H,
        strength=coupling_strength,
        dimension=dimension,
    )

    initial_state = jnp.asarray(
        initial_conditions,
        dtype=jnp.float64,
    )

    coupling_matrix = jnp.asarray(
        coupling_matrix_np,
        dtype=jnp.float64,
    )

    parameters = jnp.asarray(
        parameters,
        dtype=jnp.float64,
    )

    dt_jax = jnp.asarray(
        dt,
        dtype=jnp.float64,
    )

    sol = integrate_rk4_scan_jax(
        initial_state=initial_state,
        coupling_matrix=coupling_matrix,
        parameters=parameters,
        dt=dt_jax,
        n_steps=n_steps,
        dynamics_code_value=dynamics_code(dynamics),
    )

    if return_numpy:
        return np.asarray(sol), t

    return sol, jnp.asarray(t)


@partial(jax.jit, static_argnames=("n_steps", "dynamics_code_value"))
def integrate_rk4_batch_scan_jax(
    initial_states,
    coupling_matrix,
    parameters,
    dt,
    n_steps,
    dynamics_code_value,
):
    """
    JIT-compiled batched RK4 trajectory integrator

    Parameters
    initial_states: jax array
        Shape: (n_trials, state_dimension)
    coupling_matrix: jax array
        Shape: (state_dimension, state_dimension)
    parameters : jax array
        Shape: (3, )

    dt : float

    n_steps : int

    Returns
    sol : jax array
        Shape : (n_trials, n_time_points, state_dimension)

    """

    def step_function(state_batch, _):
        next_state_batch = rk4_step_batch_jax(
            state_batch=state_batch,
            dt=dt,
            coupling_matrix=coupling_matrix,
            parameters=parameters,
            dynamics_code_value=dynamics_code_value,
        )
        return next_state_batch, next_state_batch

    final_state_batch, states_after_initial = lax.scan(
        step_function,
        initial_states,
        xs=None,
        length=n_steps - 1,
    )

    # states_after_initial has shape:
    #     (n_time_points - 1, n_trials, state_dimension)
    #
    # Add the initial states at the beginning:
    #     (n_time_points, n_trials, state_dimension)
    sol_time_first = jnp.concatenate(
        [
            initial_states[None, :, :],
            states_after_initial,
        ],
        axis=0,
    )
    # Reorder to:
    #     (n_trials, n_time_points, state_dimension)
    sol_batch = jnp.swapaxes(sol_time_first, 0, 1)

    return sol_batch


def integrate_rk4_batch_jax(
    G,
    initial_conditions_batch,
    parameters,
    coupling_strength,
    H,
    tmax,
    dt,
    dimension=3,
    dynamics="rossler",
    return_numpy=True,
):
    """
    Integrate a batch of trajectories using JAX RK4.

    Parameters
    ----------
    initial_conditions_batch : array
        Shape: (n_trials, state_dimension)

    Returns
    -------
    sol_batch, t

    sol_batch shape:
        (n_trials, n_time_points, state_dimension)

    t shape:
        (n_time_points,)
    """

    initial_conditions_batch = np.asarray(
        initial_conditions_batch,
        dtype=np.float64,
    )

    if initial_conditions_batch.ndim != 2:
        raise ValueError(
            "initial_conditions_batch must have shape " "(n_trials, state_dimension)."
        )

    n_steps = int(round(tmax / dt))
    t = np.arange(n_steps, dtype=np.float64) * dt

    L = graph_laplacian(G)

    coupling_matrix_np = build_coupling_matrix(
        L=L,
        H=H,
        strength=coupling_strength,
        dimension=dimension,
    )

    initial_states = jnp.asarray(
        initial_conditions_batch,
        dtype=jnp.float64,
    )

    coupling_matrix = jnp.asarray(
        coupling_matrix_np,
        dtype=jnp.float64,
    )

    parameters = jnp.asarray(
        parameters,
        dtype=jnp.float64,
    )

    dt_jax = jnp.asarray(
        dt,
        dtype=jnp.float64,
    )

    sol_batch = integrate_rk4_batch_scan_jax(
        initial_states=initial_states,
        coupling_matrix=coupling_matrix,
        parameters=parameters,
        dt=dt_jax,
        n_steps=n_steps,
        dynamics_code_value=dynamics_code(dynamics),
    )

    if return_numpy:
        return np.asarray(sol_batch), t

    return sol_batch, jnp.asarray(t)


def integrate_rk4_batch_from_config(
    config,
    initial_conditions_batch,
    return_numpy=True,
):
    """
    Integrate a batch of trajectories using the values in a BasinConfig.
    """

    return integrate_rk4_batch_jax(
        G=config.G,
        initial_conditions_batch=initial_conditions_batch,
        parameters=config.parameters,
        coupling_strength=config.coupling_strength,
        H=config.H,
        tmax=config.tmax,
        dt=config.dt,
        dimension=config.dimension,
        dynamics=config.dynamics,
        return_numpy=return_numpy,
    )
