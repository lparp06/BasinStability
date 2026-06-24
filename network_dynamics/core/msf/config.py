"""Configuration objects for MSF calculations."""

from __future__ import annotations

from dataclasses import dataclass

from network_dynamics.core.jax_config import enable_x64

enable_x64()

import jax.numpy as jnp

from network_dynamics.core.msf.coupling import inner_coupling_matrix_jax
from network_dynamics.core.msf.dynamics import normalize_msf_dynamics


@dataclass(frozen=True)
class MSFConfig:
    """Run settings for one 3D oscillator MSF calculation.

    Parameters ``a``–``e`` map to the first five positional parameters of the
    selected dynamics.  For most systems only ``a``, ``b``, ``c`` are used.
    Chua's circuit additionally uses ``d`` (a_nl) and ``e`` (b_nl).

    Slot mapping by dynamics:
      rossler : a, b, c
      lorenz  : a=sigma, b=beta, c=rho
      chen    : a, b=beta, c
      chua    : a=alpha, b=beta, c=gamma, d=a_nl, e=b_nl
      hr      : a=I, b=r, c=s
    """

    dynamics: str = "rossler"
    a: float = 0.2
    b: float = 0.2
    c: float = 9.0
    d: float = 0.0   # extra param slot (used by chua: a_nl)
    e: float = 0.0   # extra param slot (used by chua: b_nl)
    initial_state: tuple[float, float, float] = (1.0, 1.0, 1.0)
    target: int = 0
    source: int = 0
    dt: float = 0.05
    transient_time: float = 100.0
    measurement_time: float = 300.0
    qr_interval_steps: int = 10

    @property
    def transient_steps(self) -> int:
        return int(round(self.transient_time / self.dt))

    @property
    def measurement_steps(self) -> int:
        return int(round(self.measurement_time / self.dt))

    def validate(self) -> None:
        normalize_msf_dynamics(self.dynamics)

        if self.dt <= 0:
            raise ValueError("dt must be positive.")
        if self.transient_steps <= 0:
            raise ValueError("transient_steps must be positive.")
        if self.measurement_steps <= 0:
            raise ValueError("measurement_steps must be positive.")
        if self.qr_interval_steps <= 0:
            raise ValueError("qr_interval_steps must be positive.")
        if self.measurement_steps % self.qr_interval_steps != 0:
            n_qr = self.measurement_steps // self.qr_interval_steps
            covered = n_qr * self.qr_interval_steps
            raise ValueError(
                f"measurement_steps={self.measurement_steps} is not divisible by "
                f"qr_interval_steps={self.qr_interval_steps}. "
                f"The last {self.measurement_steps - covered} integration steps "
                "would not be included in log_stretch, biasing the Lyapunov estimate. "
                f"Use measurement_steps={covered} or {covered + self.qr_interval_steps}."
            )
        if not (0 <= self.target < 3 and 0 <= self.source < 3):
            raise ValueError("target and source must be in {0, 1, 2}.")


_DYNAMICS_PARAM_SLOTS = {
    "rossler": ("a", "b", "c"),
    "lorenz":  ("a", "b", "c"),
    "chen":    ("a", "b", "c"),
    "chua":    ("a", "b", "c", "d", "e"),
    "hr":      ("a", "b", "c"),
}


def config_to_jax_arrays(config: MSFConfig):
    """Convert a Python config into JAX arrays used by the compiled code."""

    dynamics = normalize_msf_dynamics(config.dynamics)
    slots = _DYNAMICS_PARAM_SLOTS[dynamics]
    params = jnp.array([getattr(config, s) for s in slots], dtype=jnp.float64)

    initial_state = jnp.array(config.initial_state, dtype=jnp.float64)
    H = inner_coupling_matrix_jax(
        dimension=3,
        target=config.target,
        source=config.source,
    )
    return params, initial_state, H
