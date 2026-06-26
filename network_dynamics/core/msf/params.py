"""MSFParams: lightweight configuration for one MSF run ."""

from dataclasses import dataclass, field
from network_dynamics.core.msf.dynamics import DYN_IDS


@dataclass
class MSFParams:
    """
    All settings needed to compute Psi(K) for one oscillator + coupling scheme.
    Slot mapping by dynamics:
        rossler : a, b, c
        lorenz  : a=sigma, b=beta, c=rho
        chen    : a, b=beta, c
        chua    : a=alpha, b=beta, c=gamma, d=a_nl, e=b_nl
        hr      : a=I, b=r, c=s
    """
    dynamics:           str   = "rossler"
    a:                  float = 0.2
    b:                  float = 0.2
    c:                  float = 9.0
    d:                  float = 0.0   # extra param (Chua: a_nl)
    e:                  float = 0.0   # extra param (Chua: b_nl)
    target:             int   = 0     # row index in inner coupling matrix H
    source:             int   = 0     # column index in H
    dt:                 float = 0.001
    transient_time:     float = 100.0
    measurement_time:   float = 3000.0
    qr_interval_steps:  int   = 10

    def __post_init__(self):
        if self.dynamics not in DYN_IDS:
            raise ValueError(f"Unknown dynamics {self.dynamics!r}. "
                             f"Available: {sorted(DYN_IDS)}")
        if not (0 <= self.target <= 2 and 0 <= self.source <= 2):
            raise ValueError("target and source must be in {0, 1, 2}.")
        m = int(round(self.measurement_time / self.dt))
        if m % self.qr_interval_steps != 0:
            raise ValueError(
                f"measurement_steps={m} must be divisible by "
                f"qr_interval_steps={self.qr_interval_steps}."
            )

    @property
    def dyn_id(self) -> int:
        return DYN_IDS[self.dynamics]
