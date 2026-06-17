"""
Run one local fast-JAX/GPU basin-stability test.

Reusable computation belongs in ``network_dynamics.core`` or
``network_dynamics.gpu``. This script is only the local command-line harness:
parse inputs, build a config, call the GPU basin backend, and print results.
"""

import argparse
import math
import time
from dataclasses import dataclass
from typing import Tuple

from network_dynamics.core.basin_common import sample_initial_conditions_batch
from network_dynamics.core.config import BasinConfig
from network_dynamics.core.graphs import make_graph
from network_dynamics.core.sampling import trial_seeds
from network_dynamics.gpu.basin_fast import (
    basin_stability_gpu_fast_from_initial_conditions,
)


@dataclass(frozen=True)
class LocalGpuRequest:
    graph_type: str
    n_nodes: int
    edge_probability: float
    graph_seed: int
    base_seed: int
    n_trials: int
    tmax: float
    dt: float
    sync_tol: float
    tol_max: float
    window_fraction: float
    max_abs_threshold: float
    success_definition: str
    sampling_bounds: Tuple[float, float]
    network: str
    coupling_strength: float
    show_trials: int


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one local fast-JAX/GPU basin-stability test."
    )
    parser.add_argument(
        "--graph",
        "--graph-type",
        "--graph_type",
        dest="graph_type",
        default="erdos-renyi",
        help="Graph family: path or erdos-renyi.",
    )
    parser.add_argument("--n-nodes", type=int, default=5)
    parser.add_argument("--edge-probability", type=float, default=0.5)
    parser.add_argument("--graph-seed", type=int, default=123)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--n-trials", type=int, default=8)
    parser.add_argument("--tmax", type=float, default=150.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--sync-tol", type=float, default=1e-3)
    parser.add_argument("--tol-max", type=float, default=1e6)
    parser.add_argument("--window-fraction", type=float, default=0.2)
    parser.add_argument("--max-abs-threshold", type=float, default=1e6)
    parser.add_argument(
        "--success-definition",
        choices=("final_success", "window_success", "first_crossing"),
        default="first_crossing",
    )
    parser.add_argument("--sampling-low", type=float, default=-5.0)
    parser.add_argument("--sampling-high", type=float, default=5.0)
    parser.add_argument(
        "--network",
        "--dynamics",
        dest="network",
        default="rossler",
        help="Currently supported dynamics preset.",
    )
    parser.add_argument("--coupling-strength", type=float, default=1.0)
    parser.add_argument(
        "--show-trials",
        type=int,
        default=8,
        help="Show up to this many non-successful trial rows.",
    )
    return parser.parse_args()


def request_from_args(args):
    return LocalGpuRequest(
        graph_type=args.graph_type,
        n_nodes=args.n_nodes,
        edge_probability=args.edge_probability,
        graph_seed=args.graph_seed,
        base_seed=args.base_seed,
        n_trials=args.n_trials,
        tmax=args.tmax,
        dt=args.dt,
        sync_tol=args.sync_tol,
        tol_max=args.tol_max,
        window_fraction=args.window_fraction,
        max_abs_threshold=args.max_abs_threshold,
        success_definition=args.success_definition,
        sampling_bounds=(args.sampling_low, args.sampling_high),
        network=args.network,
        coupling_strength=args.coupling_strength,
        show_trials=args.show_trials,
    )


def make_config(request):
    if request.network != "rossler":
        raise ValueError("Only network='rossler' is currently supported.")

    graph = make_graph(
        graph_type=request.graph_type,
        n_nodes=request.n_nodes,
        seed=request.graph_seed,
        edge_probability=request.edge_probability,
    )

    return BasinConfig(
        G=graph,
        n_trials=request.n_trials,
        base_seed=request.base_seed,
        coupling_strength=request.coupling_strength,
        H=None,
        tmax=request.tmax,
        dt=request.dt,
        dimension=3,
        sampling_bounds=request.sampling_bounds,
        sync_tol=request.sync_tol,
        tol_max=request.tol_max,
        window_fraction=request.window_fraction,
        max_abs_threshold=request.max_abs_threshold,
        success_definition=request.success_definition,
        integrator="RK4",
        backend="gpu",
        n_workers=None,
    ).validate()


def time_call(function, *args, **kwargs):
    start = time.perf_counter()
    value = function(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return value, elapsed


def format_float(value):
    if value is None:
        return "None"
    if not math.isfinite(value):
        return str(value)
    return f"{value:.6g}"


def print_header(request, config):
    import jax

    print("Local fast-JAX basin test")
    print("=" * 40)
    print("JAX backend:", jax.default_backend())
    print("JAX devices:", jax.devices())
    print(
        "Graph:",
        f"type={request.graph_type}, nodes={config.n_nodes}, "
        f"edges={config.n_edges}, seed={request.graph_seed}",
    )
    print("Dynamics:", request.network)
    print(
        "Numerics:",
        f"trials={config.n_trials}, tmax={config.tmax}, dt={config.dt}, "
        f"sync_tol={config.sync_tol}, max_abs_threshold={config.max_abs_threshold}, "
        f"success_definition={config.success_definition}",
    )
    print("Coupling:", f"strength={config.coupling_strength}")
    print()


def print_summary(summary, elapsed_seconds):
    print("Results")
    print("-" * 40)
    print("Fast-JAX basin stability:", summary.basin_stability)
    print("Successes:", f"{summary.successes}/{summary.n_trials}")
    print("Sync failures:", summary.sync_failures)
    print("Integration failures:", summary.integration_failures)
    print("Mean sync time:", summary.sync_time_mean)
    print("Fast-JAX seconds:", f"{elapsed_seconds:.3f}")


def print_trial_diagnostics(summary, limit):
    if limit <= 0:
        return

    interesting_results = [
        result
        for result in summary.results
        if result.integration_failed or not result.success
    ]

    if not interesting_results:
        return

    print()
    print(f"First {min(limit, len(interesting_results))} non-successful trials")
    print("-" * 40)

    for result in interesting_results[:limit]:
        status = "integration_failed" if result.integration_failed else "sync_failed"
        print(
            "seed={seed} status={status} min_distance={min_distance} "
            "sync_time={sync_time} final_distance={final_distance} "
            "window_max={window_max} error={error}".format(
                seed=result.trial_seed,
                status=status,
                min_distance=format_float(result.min_distance),
                sync_time=format_float(result.sync_time),
                final_distance=format_float(result.final_distance),
                window_max=format_float(result.window_max_distance),
                error=result.error or "",
            )
        )


def run(request, config):
    seeds = trial_seeds(
        base_seed=config.base_seed,
        n_trials=config.n_trials,
    )
    initial_conditions = sample_initial_conditions_batch(
        config=config,
        seeds=seeds,
    )

    print_header(
        request=request,
        config=config,
    )

    summary, elapsed_seconds = time_call(
        basin_stability_gpu_fast_from_initial_conditions,
        config,
        initial_conditions,
        seeds,
    )

    print_summary(
        summary=summary,
        elapsed_seconds=elapsed_seconds,
    )
    print_trial_diagnostics(
        summary=summary,
        limit=request.show_trials,
    )

    return summary


def main():
    args = parse_args()
    request = request_from_args(args)
    config = make_config(request)
    run(request, config)


if __name__ == "__main__":
    main()
