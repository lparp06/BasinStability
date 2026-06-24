"""CPU oscillator derivative functions."""

import numpy as np


DYNAMICS_TYPES = (
    "rossler",
    "lorenz",
    "chen",
    "chua",
    "hr",
)

# Number of parameters each dynamics type expects.
DYNAMICS_PARAMETER_COUNTS = {
    "rossler": 3,
    "lorenz": 3,
    "chen": 3,
    "chua": 5,  # alpha, beta, gamma, a_nl, b_nl
    "hr": 3,    # I, r, s
}


def normalize_dynamics_type(dynamics):
    aliases = {
        "rossler": "rossler",
        "rössler": "rossler",
        "roessler": "rossler",
        "lorenz": "lorenz",
        "chen": "chen",
        "chua": "chua",
        "hr": "hr",
        "hindmarsh_rose": "hr",
        "hindmarsh-rose": "hr",
        "hindmarshrose": "hr",
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

    if dynamics == "chen":
        return chen

    if dynamics == "chua":
        return chua

    if dynamics == "hr":
        return hr

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


def chen(time, state_vector, coupling_matrix, parameters):
    a, beta, c = parameters  # a=35, beta=8/3, c=25

    state_length = len(state_vector)

    X = state_vector[0:state_length:3]
    Y = state_vector[1:state_length:3]
    Z = state_vector[2:state_length:3]

    dx = np.zeros_like(state_vector, dtype=float)

    dx[0:state_length:3] = a * (Y - X)
    dx[1:state_length:3] = (c - a - Z) * X + c * Y
    dx[2:state_length:3] = X * Y - beta * Z

    derivative = dx - np.dot(coupling_matrix, state_vector)

    return derivative


def _chua_f(X, a_nl, b_nl):
    """Piecewise nonlinearity f(x) for Chua's circuit (vectorised)."""
    return np.where(
        np.abs(X) <= 1.0,
        -a_nl * X,
        np.where(X > 1.0, -b_nl * X - a_nl + b_nl, -b_nl * X + a_nl - b_nl),
    )


def chua(time, state_vector, coupling_matrix, parameters):
    # parameters = (alpha, beta, gamma, a_nl, b_nl)
    # Defaults from literature: alpha=10, beta=14.87, gamma=0, a=-1.27, b=-0.68
    alpha, beta, gamma, a_nl, b_nl = parameters

    state_length = len(state_vector)

    X = state_vector[0:state_length:3]
    Y = state_vector[1:state_length:3]
    Z = state_vector[2:state_length:3]

    f = _chua_f(X, a_nl, b_nl)

    dx = np.zeros_like(state_vector, dtype=float)
    dx[0:state_length:3] = alpha * (Y - X + f)
    dx[1:state_length:3] = X - Y + Z
    dx[2:state_length:3] = -beta * Y - gamma * Z

    return dx - np.dot(coupling_matrix, state_vector)


def hr(time, state_vector, coupling_matrix, parameters):
    # parameters = (I, r, s)
    # Default from paper (PhysRevE.80.036204 Eq. 16): I=3.2, r=0.006, s=4
    I, r, s = parameters

    state_length = len(state_vector)

    X = state_vector[0:state_length:3]
    Y = state_vector[1:state_length:3]
    Z = state_vector[2:state_length:3]

    dx = np.zeros_like(state_vector, dtype=float)
    dx[0:state_length:3] = Y + 3.0 * X**2 - X**3 - Z + I
    dx[1:state_length:3] = 1.0 - 5.0 * X**2 - Y
    dx[2:state_length:3] = -r * Z + r * s * (X + 1.6)

    return dx - np.dot(coupling_matrix, state_vector)
