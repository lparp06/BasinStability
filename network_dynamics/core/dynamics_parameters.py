"""Default oscillator parameters used by the experiment CLIs."""

from __future__ import annotations

from dataclasses import dataclass

from network_dynamics.core.oscillators import normalize_dynamics_type


@dataclass(frozen=True)
class DynamicsParameterDefaults:
    names: tuple[str, str, str]
    values: tuple[float, float, float]
    source: str


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
) -> tuple[float, float, float]:
    defaults = parameter_defaults_for_dynamics(dynamics).values
    values = (a, b, c)
    return tuple(
        float(value) if value is not None else defaults[index]
        for index, value in enumerate(values)
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
