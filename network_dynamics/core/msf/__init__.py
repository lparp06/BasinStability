"""Master-stability-function tools.

To add a new oscillator's MSF implementation, start in ``dynamics.py``.
"""

from network_dynamics.core.msf.analysis import (
    find_msf_zeros_jax,
    find_zero_brackets,
    merge_close_brackets,
    stable_intervals_from_brackets,
)
from network_dynamics.core.msf.config import (
    MSFConfig,
    config_to_jax_arrays,
)
from network_dynamics.core.msf.coupling import inner_coupling_matrix_jax
from network_dynamics.core.msf.dynamics import (
    MSF_DYNAMICS,
    get_msf_dynamics,
    lorenz_jacobian_jax,
    lorenz_rhs_jax,
    normalize_msf_dynamics,
    rossler_jacobian_jax,
    rossler_rhs_jax,
)
from network_dynamics.core.msf.integration import (
    msf_rhs_jax,
    rk4_step_msf_jax,
    rk4_step_state_jax,
    run_transient_jax,
)
from network_dynamics.core.msf.lyapunov import (
    msf_value_jax,
    msf_value_from_state_jax,
    msf_value_from_state_jax_impl,
    msf_value_jax_impl,
    qr_update_jax,
    scan_msf_jax,
    scan_msf_jax_impl,
)

__all__ = [
    "MSFConfig",
    "MSF_DYNAMICS",
    "config_to_jax_arrays",
    "find_msf_zeros_jax",
    "find_zero_brackets",
    "get_msf_dynamics",
    "merge_close_brackets",
    "inner_coupling_matrix_jax",
    "lorenz_jacobian_jax",
    "lorenz_rhs_jax",
    "msf_rhs_jax",
    "msf_value_jax",
    "msf_value_from_state_jax",
    "msf_value_from_state_jax_impl",
    "msf_value_jax_impl",
    "normalize_msf_dynamics",
    "qr_update_jax",
    "rk4_step_msf_jax",
    "rk4_step_state_jax",
    "rossler_jacobian_jax",
    "rossler_rhs_jax",
    "run_transient_jax",
    "scan_msf_jax",
    "scan_msf_jax_impl",
    "stable_intervals_from_brackets",
]
