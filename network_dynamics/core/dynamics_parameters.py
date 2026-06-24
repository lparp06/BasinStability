"""Default oscillator parameters used by the experiment CLIs."""

from __future__ import annotations

from dataclasses import dataclass

from network_dynamics.core.oscillators import normalize_dynamics_type


@dataclass(frozen=True)
class DynamicsParameterDefaults:
    names: tuple[str, ...]
    values: tuple[float, ...]
    source: str


# Per-dynamics safe starting points for MSF transient integration.
# Must lie inside the basin of the target attractor; (1,1,1) diverges for Chua.
DYNAMICS_MSF_INITIAL_STATES: dict[str, tuple[float, float, float]] = {
    "rossler": (1.0, 1.0, 1.0),
    "lorenz":  (1.0, 1.0, 1.0),
    "chen":    (1.0, 1.0, 1.0),
    "chua":    (0.1, 0.0, 0.0),  # (1,1,1) escapes to ~1e100; use inner-region point
    "hr":      (-1.3, -8.0, 1.0),  # Inside bursting attractor; use --transient-time 1000 (slow z-timescale 1/r≈167)
}


DYNAMICS_PARAMETER_DEFAULTS = {
    "rossler": DynamicsParameterDefaults(
        names=("a", "b", "c"),
        values=(0.2, 0.2, 9.0),
        source="paper",
    ),
    "lorenz": DynamicsParameterDefaults(
        names=("sigma", "beta", "rho"),
        values=(10.0, 2.0, 28.0),
        source="paper",
    ),
    "chen": DynamicsParameterDefaults(
        names=("a", "beta", "c"),
        values=(35.0, 8.0 / 3.0, 28.0),
        source="paper",
    ),
    "chua": DynamicsParameterDefaults(
        # alpha, beta, gamma are the circuit params; a_nl/b_nl are the
        # slopes of the piecewise nonlinearity f(x).
        names=("alpha", "beta", "gamma", "a_nl", "b_nl"),
        values=(10.0, 14.87, 0.0, -1.27, -0.68),
        source="paper",
    ),
    "hr": DynamicsParameterDefaults(
        # Hindmarsh-Rose neuron (PhysRevE.80.036204 Eq. 16).
        # I=external current, r=slow adaptation rate, s=slope parameter.
        names=("I", "r", "s"),
        values=(3.2, 0.006, 4.0),
        source="paper",
    ),
}


def parameter_defaults_for_dynamics(dynamics: str) -> DynamicsParameterDefaults:
    dynamics = normalize_dynamics_type(dynamics)

    try:
        return DYNAMICS_PARAMETER_DEFAULTS[dynamics]
    except KeyError as error:
        raise ValueError(
            f"No parameter defaults are registered for dynamics={dynamics!r}."
        ) from error


def resolve_dynamics_parameters(
    dynamics: str,
    a: float | None = None,
    b: float | None = None,
    c: float | None = None,
    d: float | None = None,
    e: float | None = None,
) -> tuple[float, ...]:
    """
    Resolve CLI overrides against per-dynamics defaults.

    Positional slots map as: a→[0], b→[1], c→[2], d→[3], e→[4].
    Slots beyond the dynamics' parameter count are ignored.
    """
    defaults = parameter_defaults_for_dynamics(dynamics).values
    overrides = (a, b, c, d, e)
    return tuple(
        float(overrides[i]) if i < len(overrides) and overrides[i] is not None
        else defaults[i]
        for i in range(len(defaults))
    )


def format_parameter_defaults() -> str:
    parts = []
    for dynamics, defaults in DYNAMICS_PARAMETER_DEFAULTS.items():
        assignments = ", ".join(
            f"{name}={value:g}"
            for name, value in zip(defaults.names, defaults.values)
        )
        parts.append(f"{dynamics}: {assignments}")
    return "; ".join(parts)
