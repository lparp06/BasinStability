"""
diagnostics.py
Functions that answer:
Did the numerical computation produce a trustworthy result?
"""

import numpy as np

def solution_health(sol, max_abs_threshold=1e6):
    '''
    Inspect a solution array and report numerical problems
    Checks:
        Does the solution contain NaN or infinity?
        Did the values explode beyond a threshold?
        What is the largest absolute value?

    Returns a dictionary
    '''
    contains_nan = bool(np.isnan(sol).any())
    contains_inf = bool(np.isinf(sol).any())

    if contains_nan or contains_inf:
        max_abs_value = np.inf
    else:
        max_abs_value = float(np.max(np.abs(sol)))

    health = {
        "contains_nan": contains_nan,
        "contains_inf": contains_inf,
        "max_abs_value": max_abs_value,
        "exceeds_max_abs_threshold": bool(max_abs_value > max_abs_threshold),
        "max_abs_threshold": max_abs_threshold,
    }
    return health 

def is_solution_valid(health):
    '''
    Turns health dictionary into a simple True/False answer
    '''
    return (
        not health["contains_nan"]
        and not health["contains_inf"]
        and not health["exceeds_max_abs_threshold"]
    )
    

def format_health_message(health):
    '''
    Create a readable explanation when something goes wrong
    '''
    if health["contains_nan"]: 
        return "Solution contains NaN values."
    if health["contains_inf"]:
        return "Solution contains infinite values."
    if health["exceeds_max_abs_threshold"]:
        return(
            "Solution exceeded max absolute value threshold:"
            f"{health['max_abs_value']} > {health['max_abs_threshold']}" 
        )
    return "Solution passed health checks."