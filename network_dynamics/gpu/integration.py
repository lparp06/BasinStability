"""
JAX/GPU RK4 integrator for one coupled Rössler trajectory.
"""

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from network_dynamics.core.graphs import graph_laplacian
from network_dynamics.core.coupling import build_coupling_matrix


def rossler_jax(state_vector, coupling_matrix, parameters):
    """
    JAX vers of the coupled Rössler derivative
    """

    a, b, c = parameters

    X = state_vector[0::3]
    Y = state_vector[1::3]
    Z = state_vector[2::3]

    dx = jnp.zeros_like(state_vector)

    dx = dx.at[0::3].set(-Y - Z)
    dx = dx.at[1::3].set(X + a * Y)
    dx = dx.at[2::3].set(b + Z * (X - c))

    derivative = dx - coupling_matrix @ state_vector

    return derivative


def rk4_step_jax(state, dt, coupling_matrix, parameters):
    """
    One RK4 step using JAX.
    """

    k1 = rossler_jax(
        state,
        coupling_matrix,
        parameters,
    )

    k2 = rossler_jax(
        state + 0.5 * dt * k1,
        coupling_matrix,
        parameters,
    )

    k3 = rossler_jax(
        state + 0.5 * dt * k2,
        coupling_matrix,
        parameters,
    )

    k4 = rossler_jax(
        state + dt * k3,
        coupling_matrix,
        parameters,
    )

    next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return next_state


@partial(jax.jit, static_argnames=("n_steps",))
def integrate_rk4_scan_jax(
    initial_state,
    coupling_matrix,
    parameters,
    dt,
    n_steps,
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

    # Match CPU time convention exactly.
    t = np.arange(0.0, tmax, dt)
    n_steps = len(t)

    L = graph_laplacian(G)

    coupling_matrix_np = build_coupling_matrix(
        L=L,
        H=H,
        strength=coupling_strength,
    )

    initial_state = jnp.asarray(
        initial_conditions,
        dtype=jnp.float32,
    )

    coupling_matrix = jnp.asarray(
        coupling_matrix_np,
        dtype=jnp.float32,
    )

    parameters = jnp.asarray(
        parameters,
        dtype=jnp.float32,
    )

    dt_jax = jnp.asarray(
        dt,
        dtype=jnp.float32,
    )

    sol = integrate_rk4_scan_jax(
        initial_state=initial_state,
        coupling_matrix=coupling_matrix,
        parameters=parameters,
        dt=dt_jax,
        n_steps=n_steps,
    )

    if return_numpy:
        return np.asarray(sol), t

    return sol, jnp.asarray(t)


@partial(jax.jit, static_argnames=("n_steps",))
def integrate_rk4_batch_scan_jax(
    initial_states,
    coupling_matrix,
    parameters,
    dt,
    n_steps,
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
        next_state_batch = jax.vmap(
            rk4_step_jax,
            in_axes=(0, None, None, None),
        )(
            state_batch,
            dt,
            coupling_matrix,
            parameters,
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
        dtype=np.float32,
    )

    if initial_conditions_batch.ndim != 2:
        raise ValueError(
            "initial_conditions_batch must have shape " "(n_trials, state_dimension)."
        )

    t = np.arange(0.0, tmax, dt)
    n_steps = len(t)

    L = graph_laplacian(G)

    coupling_matrix_np = build_coupling_matrix(
        L=L,
        H=H,
        strength=coupling_strength,
    )

    initial_states = jnp.asarray(
        initial_conditions_batch,
        dtype=jnp.float32,
    )

    coupling_matrix = jnp.asarray(
        coupling_matrix_np,
        dtype=jnp.float32,
    )

    parameters = jnp.asarray(
        parameters,
        dtype=jnp.float32,
    )

    dt_jax = jnp.asarray(
        dt,
        dtype=jnp.float32,
    )

    sol_batch = integrate_rk4_batch_scan_jax(
        initial_states=initial_states,
        coupling_matrix=coupling_matrix,
        parameters=parameters,
        dt=dt_jax,
        n_steps=n_steps,
    )

    if return_numpy:
        return np.asarray(sol_batch), t

    return sol_batch, jnp.asarray(t)
