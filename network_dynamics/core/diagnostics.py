"""Numerical health checks for integrated trajectories."""

import numpy as np


def solution_health(sol, max_abs_threshold=1e6):
    """Return finite/threshold health metrics for a solution array."""

    contains_nan = bool(np.isnan(sol).any())
    contains_inf = bool(np.isinf(sol).any())
    max_abs_value = (
        np.inf
        if contains_nan or contains_inf
        else float(np.max(np.abs(sol)))
    )

    return {
        "contains_nan": contains_nan,
        "contains_inf": contains_inf,
        "max_abs_value": max_abs_value,
        "exceeds_max_abs_threshold": bool(max_abs_value > max_abs_threshold),
        "max_abs_threshold": max_abs_threshold,
    }


def is_solution_valid(health):
    return (
        not health["contains_nan"]
        and not health["contains_inf"]
        and not health["exceeds_max_abs_threshold"]
    )


def format_health_message(health):
    if health["contains_nan"]:
        return "Solution contains NaN values."
    if health["contains_inf"]:
        return "Solution contains infinite values."
    if health["exceeds_max_abs_threshold"]:
        return (
            "Solution exceeded max absolute value threshold: "
            f"{health['max_abs_value']} > {health['max_abs_threshold']}"
        )
    return "Solution passed health checks."
