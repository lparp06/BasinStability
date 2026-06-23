"""
Scan coupling strengths from MSF intervals and basin stability.

Example
-------
python3 -m network_dynamics.experiments.coupling_basin_scan \
    --n-trials 25 \
    --tmax 150 \
    --n-strengths 5
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass

import numpy as np

from network_dynamics.core.basin_common import sample_initial_conditions_batch
from network_dynamics.core.config import BasinConfig
from network_dynamics.core.coupling_strengths import (
    CouplingStrengthInterval,
    coupling_strength_intervals_from_zeros,
    interval_coupling_strengths,
)
from network_dynamics.core.graphs import make_graph
from network_dynamics.core.msf import MSFConfig, find_msf_zeros_jax
from network_dynamics.core.msf_cache import (
    append_msf_cache_result,
    find_cached_msf_result,
    make_msf_cache_key,
)
from network_dynamics.core.dynamics_parameters import (
    format_parameter_defaults,
    resolve_dynamics_parameters,
)
from network_dynamics.core.oscillators import normalize_dynamics_type
from network_dynamics.core.sampling import trial_seeds
from network_dynamics.cpu.basin import basin_stability_cpu_from_initial_conditions
from network_dynamics.gpu.basin_fast import basin_stability_gpu_fast_from_initial_conditions


@dataclass(frozen=True)
class ScanRequest:
    graph_type: str
    n_nodes: int
    edge_probability: float
    graph_seed: int
    base_seed: int
    n_trials: int
    dynamics: str
    tmax: float
    dt: float
    n_strengths: int
    coupling_low: float | None
    coupling_high: float | None
    interval_index: int
    progress_interval: int
    backend: str
    n_workers: int | None
    integrator: str
    sync_tol: float
    tol_max: float
    window_fraction: float
    success_definition: str
    sampling_low: float
    sampling_high: float
    max_abs_threshold: float
    a: float
    b: float
    c: float
    K_min: float
    K_max: float
    n_K: int
    msf_cache: str
    msf_transient_time: float
    msf_measurement_time: float
    msf_dt: float
    msf_chunk_size: int | None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute MSF zeros, convert them to graph-valid coupling-strength "
            "intervals, and run basin stability across one selected interval."
        )
    )
    parser.add_argument("--graph", "--graph-type", dest="graph_type", default="erdos-renyi")
    parser.add_argument("--n-nodes", type=int, default=100)
    parser.add_argument("--edge-probability", type=float, default=0.15)
    parser.add_argument("--graph-seed", type=int, default=42)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--n-trials", type=int, default=1000)
    parser.add_argument(
        "--dynamics",
        default="rossler",
        help="Oscillator dynamics used for basin integration.",
    )
    parser.add_argument("--tmax", type=float, default=5000.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--n-strengths", type=int, default=10)
    parser.add_argument(
        "--coupling-low",
        type=float,
        default=None,
        help=(
            "Manual lower coupling-strength bound. If used with "
            "--coupling-high, skip MSF zero calculation."
        ),
    )
    parser.add_argument(
        "--coupling-high",
        type=float,
        default=None,
        help=(
            "Manual upper coupling-strength bound. If used with "
            "--coupling-low, skip MSF zero calculation."
        ),
    )
    parser.add_argument("--interval-index", type=int, default=0)
    parser.add_argument("--progress-interval", type=int, default=5)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument(
        "--n-workers",
        type=int,
        default=0,
        help=(
            "CPU worker processes. Use 0 to auto-detect from "
            "SLURM_CPUS_PER_TASK or local CPU count. Ignored for GPU."
        ),
    )
    parser.add_argument("--integrator", choices=("LSODA", "RK4"), default="RK4")
    parser.add_argument("--sync-tol", type=float, default=1e-3)
    parser.add_argument("--tol-max", type=float, default=1e6)
    parser.add_argument("--window-fraction", type=float, default=0.1)
    parser.add_argument(
        "--success-definition",
        choices=("final_success", "window_success", "first_crossing"),
        default="window_success",
    )
    parser.add_argument("--sampling-low", type=float, default=-5.0)
    parser.add_argument("--sampling-high", type=float, default=5.0)
    parser.add_argument("--max-abs-threshold", type=float, default=1e9)
    parser.add_argument(
        "--a",
        type=float,
        default=None,
        help="First dynamics parameter. Omit to use the selected dynamics default.",
    )
    parser.add_argument(
        "--b",
        type=float,
        default=None,
        help="Second dynamics parameter. Omit to use the selected dynamics default.",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=None,
        help="Third dynamics parameter. Omit to use the selected dynamics default.",
    )
    parser.add_argument("--K-min", type=float, default=0.0)
    parser.add_argument("--K-max", type=float, default=10.0)
    parser.add_argument("--n-K", type=int, default=101)
    parser.add_argument(
        "--msf-cache",
        default="outputs/msf_zero_cache.csv",
        help="CSV cache path for MSF zeros and settings.",
    )
    parser.add_argument("--msf-transient-time", type=float, default=1000.0)
    parser.add_argument("--msf-measurement-time", type=float, default=3000.0)
    parser.add_argument("--msf-dt", type=float, default=0.05)
    parser.add_argument(
        "--msf-chunk-size",
        type=int,
        default=None,
        help="Optional K-scan chunk size for MSF progress updates.",
    )
    return parser.parse_args()


def auto_cpu_worker_count(n_trials):
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")

    if slurm_cpus is not None:
        try:
            available_cpus = int(slurm_cpus)
        except ValueError:
            available_cpus = os.cpu_count() or 1
    else:
        available_cpus = os.cpu_count() or 1

    return max(1, min(available_cpus, n_trials))


def resolve_n_workers(args):
    if args.backend == "gpu":
        return None

    if args.n_workers < 0:
        raise ValueError("n_workers must be nonnegative.")

    if args.n_workers == 0:
        return auto_cpu_worker_count(args.n_trials)

    return args.n_workers


def request_from_args(args):
    n_workers = resolve_n_workers(args)
    dynamics = normalize_dynamics_type(args.dynamics)
    a, b, c = resolve_dynamics_parameters(
        dynamics=dynamics,
        a=args.a,
        b=args.b,
        c=args.c,
    )

    return ScanRequest(
        graph_type=args.graph_type,
        n_nodes=args.n_nodes,
        edge_probability=args.edge_probability,
        graph_seed=args.graph_seed,
        base_seed=args.base_seed,
        n_trials=args.n_trials,
        dynamics=dynamics,
        tmax=args.tmax,
        dt=args.dt,
        n_strengths=args.n_strengths,
        coupling_low=args.coupling_low,
        coupling_high=args.coupling_high,
        interval_index=args.interval_index,
        progress_interval=args.progress_interval,
        backend=args.backend,
        n_workers=n_workers,
        integrator=args.integrator,
        sync_tol=args.sync_tol,
        tol_max=args.tol_max,
        window_fraction=args.window_fraction,
        success_definition=args.success_definition,
        sampling_low=args.sampling_low,
        sampling_high=args.sampling_high,
        max_abs_threshold=args.max_abs_threshold,
        a=a,
        b=b,
        c=c,
        K_min=args.K_min,
        K_max=args.K_max,
        n_K=args.n_K,
        msf_cache=args.msf_cache,
        msf_transient_time=args.msf_transient_time,
        msf_measurement_time=args.msf_measurement_time,
        msf_dt=args.msf_dt,
        msf_chunk_size=args.msf_chunk_size,
    )


def make_msf_config(request):
    return MSFConfig(
        dynamics=request.dynamics,
        a=request.a,
        b=request.b,
        c=request.c,
        target=0,
        source=0,
        dt=request.msf_dt,
        transient_time=request.msf_transient_time,
        measurement_time=request.msf_measurement_time,
    )


def make_manual_coupling_interval(request):
    if request.coupling_low is None and request.coupling_high is None:
        return None

    if request.coupling_low is None or request.coupling_high is None:
        raise ValueError(
            "Use both --coupling-low and --coupling-high, or neither."
        )

    if request.coupling_low >= request.coupling_high:
        raise ValueError("--coupling-low must be less than --coupling-high.")

    return CouplingStrengthInterval(
        lower=float(request.coupling_low),
        upper=float(request.coupling_high),
        msf_zero_low=float("nan"),
        msf_zero_high=float("nan"),
        laplacian_first_nonzero=float("nan"),
        laplacian_largest=float("nan"),
    )


def make_basin_config(request, graph, coupling_strength):
    if request.backend == "gpu" and request.integrator != "RK4":
        raise ValueError("GPU backend requires integrator='RK4'.")

    return BasinConfig(
        G=graph,
        dynamics=request.dynamics,
        dimension=3,
        parameters=(request.a, request.b, request.c),
        coupling_strength=float(coupling_strength),
        H=None,
        tmax=request.tmax,
        dt=request.dt,
        integrator=request.integrator,
        n_trials=request.n_trials,
        base_seed=request.base_seed,
        sampling_bounds=(request.sampling_low, request.sampling_high),
        sync_tol=request.sync_tol,
        tol_max=request.tol_max,
        window_fraction=request.window_fraction,
        success_definition=request.success_definition,
        max_abs_threshold=request.max_abs_threshold,
        backend=request.backend,
        n_workers=request.n_workers,
    ).validate()


def format_float(value):
    if value is None:
        return "None"
    if not math.isfinite(value):
        return str(value)
    return f"{value:.6g}"


def print_run_header(request, graph):
    print("Coupling basin scan")
    print("=" * 44)
    if request.backend == "gpu":
        import jax

        jax_backend = jax.default_backend()
        print("JAX backend:", jax_backend)
        print("JAX devices:", jax.devices())
        if jax_backend != "gpu":
            print("WARNING: backend=gpu requested, but JAX is not using a GPU.")
    print(
        "Graph:",
        f"type={request.graph_type}, nodes={graph.number_of_nodes()}, "
        f"edges={graph.number_of_edges()}, seed={request.graph_seed}",
    )
    print(
        "Basin settings:",
        f"trials={request.n_trials}, tmax={request.tmax}, dt={request.dt}, "
        f"strengths={request.n_strengths}, integrator={request.integrator}, "
        f"backend={request.backend}, workers={request.n_workers}",
    )
    print(
        "Dynamics:",
        f"type={request.dynamics}, parameters=({request.a}, {request.b}, {request.c})",
    )
    print("Registered defaults:", format_parameter_defaults())
    print(
        "Coupling interval:",
        (
            f"manual=[{request.coupling_low}, {request.coupling_high}]"
            if request.coupling_low is not None or request.coupling_high is not None
            else (
                f"from MSF K=[{request.K_min}, {request.K_max}], "
                f"n_K={request.n_K}"
            )
        ),
    )
    print()


def warn_if_msf_step_is_large(dt, K_max):
    stiffness_scale = dt * K_max
    if stiffness_scale <= 2.5:
        return

    print(
        "WARNING: msf_dt*K_max is large for explicit RK4 "
        f"({dt}*{K_max}={stiffness_scale:.3g}). "
        "High-K MSF values may show artificial positive growth; "
        "try a smaller --msf-dt such as 0.01 or 0.005.",
        flush=True,
    )
    print()


def print_zero_summary(zeros):
    print("MSF zeros")
    print("-" * 44)

    if not zeros:
        print("No zeros found.")
        return

    for index, zero in enumerate(zeros):
        print(f"zero[{index}] = {format_float(zero)}")
    print()


def print_interval_summary(intervals, selected_index):
    print("Coupling-strength intervals")
    print("-" * 44)

    if not intervals:
        print("No graph-compatible coupling-strength intervals found.")
        return

    for index, interval in enumerate(intervals):
        marker = "*" if index == selected_index else " "
        if math.isnan(interval.msf_zero_low):
            print(
                f"{marker} interval[{index}]: "
                f"[{format_float(interval.lower)}, {format_float(interval.upper)}] "
                "from manual coupling bounds"
            )
        else:
            print(
                f"{marker} interval[{index}]: "
                f"[{format_float(interval.lower)}, {format_float(interval.upper)}] "
                f"from K=[{format_float(interval.msf_zero_low)}, "
                f"{format_float(interval.msf_zero_high)}], "
                f"lambda=[{format_float(interval.laplacian_first_nonzero)}, "
                f"{format_float(interval.laplacian_largest)}]"
            )
    print()


def run_basin_scan(request, graph, interval):
    strengths = interval_coupling_strengths(
        interval=interval,
        n_strengths=request.n_strengths,
        endpoint=True,
    )
    first_config = make_basin_config(
        request=request,
        graph=graph,
        coupling_strength=strengths[0],
    )
    seeds = trial_seeds(
        base_seed=first_config.base_seed,
        n_trials=first_config.n_trials,
    )
    initial_conditions = sample_initial_conditions_batch(
        config=first_config,
        seeds=seeds,
    )

    rows = []
    total_strengths = len(strengths)
    scan_start = time.perf_counter()

    for index, strength in enumerate(strengths, start=1):
        config = make_basin_config(
            request=request,
            graph=graph,
            coupling_strength=strength,
        )

        elapsed_total = time.perf_counter() - scan_start
        remaining = total_strengths - index
        eta_str = (
            f"  ETA ~{elapsed_total / (index - 1) * remaining:.0f}s"
            if index > 1 and remaining > 0
            else ""
        )
        print(
            f"[{index}/{total_strengths}] "
            f"coupling_strength={format_float(strength)}  "
            f"elapsed {elapsed_total:.0f}s{eta_str}",
            flush=True,
        )
        run_start = time.perf_counter()
        if request.backend == "gpu":
            summary = basin_stability_gpu_fast_from_initial_conditions(
                config=config,
                initial_conditions_batch=initial_conditions,
                seeds=seeds,
            )
        else:
            summary = basin_stability_cpu_from_initial_conditions(
                config=config,
                initial_conditions_batch=initial_conditions,
                seeds=seeds,
                progress_label=f"coupling {index}/{total_strengths}",
                progress_interval=request.progress_interval,
                progress_stream=sys.__stdout__,
            )
        elapsed = time.perf_counter() - run_start
        elapsed_total = time.perf_counter() - scan_start
        remaining = total_strengths - index
        eta_str = (
            f"  ETA ~{elapsed_total / index * remaining:.0f}s remaining"
            if remaining > 0
            else ""
        )
        print(
            f"  -> basin={format_float(summary.basin_stability)}  "
            f"this={elapsed:.1f}s  total={elapsed_total:.0f}s{eta_str}",
            flush=True,
        )
        print()

        rows.append(
            {
                "index": index - 1,
                "coupling_strength": float(strength),
                "basin_stability": summary.basin_stability,
                "successes": summary.successes,
                "sync_failures": summary.sync_failures,
                "integration_failures": summary.integration_failures,
                "sync_time_mean": summary.sync_time_mean,
                "seconds": elapsed,
            }
        )

    return rows


def print_results_table(rows):
    print("Results table")
    print("-" * 108)
    header = (
        f"{'idx':>3} "
        f"{'coupling':>14} "
        f"{'basin':>10} "
        f"{'successes':>10} "
        f"{'sync_fail':>10} "
        f"{'int_fail':>9} "
        f"{'mean_sync_t':>13} "
        f"{'seconds':>10}"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        print(
            f"{row['index']:>3d} "
            f"{format_float(row['coupling_strength']):>14} "
            f"{format_float(row['basin_stability']):>10} "
            f"{row['successes']:>10d} "
            f"{row['sync_failures']:>10d} "
            f"{row['integration_failures']:>9d} "
            f"{format_float(row['sync_time_mean']):>13} "
            f"{row['seconds']:>10.3f}"
        )


def main():
    args = parse_args()
    request = request_from_args(args)
    graph = make_graph(
        graph_type=request.graph_type,
        n_nodes=request.n_nodes,
        seed=request.graph_seed,
        edge_probability=request.edge_probability,
    )
    manual_interval = make_manual_coupling_interval(request)

    print_run_header(request, graph)

    if manual_interval is None:
        msf_config = make_msf_config(request)
        warn_if_msf_step_is_large(msf_config.dt, request.K_max)
        cache_key = make_msf_cache_key(
            config=msf_config,
            K_min=request.K_min,
            K_max=request.K_max,
            n_K=request.n_K,
        )
        cached = find_cached_msf_result(
            cache_path=request.msf_cache,
            key=cache_key,
        )

        if cached is None:
            print("No matching MSF cache entry found. Running MSF scan.")
            zeros = find_msf_zeros_jax(
                config=msf_config,
                K_min=request.K_min,
                K_max=request.K_max,
                n_K=request.n_K,
                chunk_size=request.msf_chunk_size,
                verbose=True,
            )
            append_msf_cache_result(
                cache_path=request.msf_cache,
                key=cache_key,
                zeros=zeros,
            )
            print("Saved MSF zeros to cache:", request.msf_cache)
        else:
            zeros = cached["zeros"]
            print("Using cached MSF zeros from:", request.msf_cache)

        print_zero_summary(zeros)

        intervals = coupling_strength_intervals_from_zeros(
            G=graph,
            msf_zeros=zeros,
        )
    else:
        print("Skipping MSF scan because manual coupling bounds were provided.")
        print()
        intervals = [manual_interval]

    if not intervals:
        print_interval_summary(intervals, selected_index=request.interval_index)
        return 1

    if not (0 <= request.interval_index < len(intervals)):
        raise ValueError(
            f"interval_index must be between 0 and {len(intervals) - 1}."
        )

    print_interval_summary(intervals, selected_index=request.interval_index)
    rows = run_basin_scan(
        request=request,
        graph=graph,
        interval=intervals[request.interval_index],
    )
    print_results_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
