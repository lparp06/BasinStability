"""CPU oscillator derivative functions."""

import numpy as np


DYNAMICS_TYPES = (
    "rossler",
    "lorenz",
)


def normalize_dynamics_type(dynamics):
    aliases = {
        "rossler": "rossler",
        "rössler": "rossler",
        "roessler": "rossler",
        "lorenz": "lorenz",
    }

    normalized = aliases.get(dynamics.lower())

    if normalized is None:
        raise ValueError(
            "Unknown dynamics. Supported dynamics are: "
            + ", ".join(DYNAMICS_TYPES)
        )

    return normalized


def get_oscillator_rhs(dynamics):
    dynamics = normalize_dynamics_type(dynamics)

    if dynamics == "rossler":
        return rossler

    if dynamics == "lorenz":
        return lorenz

    raise ValueError(f"Unsupported dynamics: {dynamics}")


def rossler(time, state_vector, coupling_matrix, parameters):
    a, b, c = parameters
    state_length = len(state_vector)

    X = state_vector[0:state_length:3]
    Y = state_vector[1:state_length:3]
    Z = state_vector[2:state_length:3]

    dx = np.zeros_like(state_vector, dtype=float)

    dx[0:state_length:3] = -Y - Z
    dx[1:state_length:3] = X + a * Y

    dx[2:state_length:3] = b + Z * (X - c)

    derivative = dx - np.dot(coupling_matrix, state_vector)

    return derivative


def lorenz(time, state_vector, coupling_matrix, parameters):
    sigma, beta, rho = parameters

    state_length = len(state_vector)

    X = state_vector[0:state_length:3]
    Y = state_vector[1:state_length:3]
    Z = state_vector[2:state_length:3]

    dx = np.zeros_like(state_vector, dtype=float)

    dx[0:state_length:3] = sigma * (Y - X)
    dx[1:state_length:3] = X * (rho - Z) - Y
    dx[2:state_length:3] = X * Y - beta * Z

    derivative = dx - np.dot(coupling_matrix, state_vector)

    return derivative
