"""
Create CSV benchmark tables for serial CPU, parallel CPU, and fast GPU runs.

This script is intended for supercomputer/cluster runs where you want tabular
timing data that can be copied back locally and plotted later.

Example:

    python -m network_dynamics.experiments.hpc_benchmark \
        --trial-counts 100 1000 5000 \
        --workers 16 \
        --output timing_outputs/hpc_benchmark_results.csv
"""

import argparse
import csv
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path

import networkx as nx
import numpy as np

from network_dynamics.core.config import BasinConfig
from network_dynamics.core.sampling import sample_uniform_initial_condition, trial_seeds
from network_dynamics.cpu.basin import basin_stability_cpu_from_initial_conditions
BACKENDS = (""
"_cpu", "parallel_cpu", "fast_gpu")
JAX_BACKEND = "not_run"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark basin-stability backends and write a CSV table."
    )
    parser.add_argument(
        "--trial-counts",
        nargs="+",
        type=int,
        default=[100, 1000],
        help="Trial counts to benchmark.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Worker count for the parallel CPU backend.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of repeats per backend/trial-count combination.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("timing_outputs/hpc_benchmark_results.csv"),
        help="CSV path for benchmark rows.",
    )
    parser.add_argument(
        "--skip-serial",
        action="store_true",
        help="Skip serial CPU runs for large cluster benchmarks.",
    )
    parser.add_argument(
        "--skip-gpu",
        action="store_true",
        help="Skip fast GPU/JAX runs when JAX is unavailable or not needed.",
    )
    parser.add_argument("--tmax", type=float, default=150.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--sync-tol", type=float, default=1e-3)
    parser.add_argument("--sampling-low", type=float, default=-5.0)
    parser.add_argument("--sampling-high", type=float, default=5.0)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--n-nodes", type=int, default=5)
    return parser.parse_args()


def make_config(args, backend, n_trials):
    n_workers = args.workers if backend == "cpu" else None

    return BasinConfig(
        G=nx.path_graph(args.n_nodes),
        n_trials=n_trials,
        base_seed=args.base_seed,
        parameters=(0.2, 0.2, 7.0),
        coupling_strength=1.0,
        H=None,
        tmax=args.tmax,
        dt=args.dt,
        dimension=3,
        sampling_bounds=(args.sampling_low, args.sampling_high),
        sync_tol=args.sync_tol,
        tol_max=1e6,
        window_fraction=0.2,
        max_abs_threshold=1e6,
        success_definition="first_crossing",
        integrator="RK4",
        backend=backend,
        n_workers=n_workers,
    ).validate()


def make_initial_conditions(config, seeds):
    low, high = config.sampling_bounds
    initial_conditions = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        initial_conditions.append(
            sample_uniform_initial_condition(
                rng=rng,
                n_nodes=config.n_nodes,
                dimension=config.dimension,
                low=low,
                high=high,
            )
        )

    return np.asarray(initial_conditions, dtype=np.float32)


def time_call(function, *args, **kwargs):
    start = time.perf_counter()
    value = function(*args, **kwargs)
    elapsed = time.perf_counter() - start

    return value, elapsed


def load_gpu_backend():
    import jax
    from network_dynamics.gpu.basin_fast import (
        basin_stability_gpu_fast_from_initial_conditions,
    )

    return jax, basin_stability_gpu_fast_from_initial_conditions


def run_backend(backend_name, config, initial_conditions, seeds, progress_interval):
    global JAX_BACKEND

    if backend_name == "serial_cpu":
        run_config = replace(config, backend="serial", n_workers=1)
        return time_call(
            basin_stability_cpu_from_initial_conditions,
            run_config,
            initial_conditions,
            seeds,
            progress_label=f"{backend_name} ({config.n_trials})",
            progress_interval=progress_interval,
            progress_stream=sys.stdout,
        )

    if backend_name == "parallel_cpu":
        run_config = replace(config, backend="cpu")
        return time_call(
            basin_stability_cpu_from_initial_conditions,
            run_config,
            initial_conditions,
            seeds,
            progress_label=f"{backend_name} ({config.n_trials})",
            progress_interval=progress_interval,
            progress_stream=sys.stdout,
        )

    if backend_name == "fast_gpu":
        jax, basin_stability_gpu_fast_from_initial_conditions = load_gpu_backend()
        JAX_BACKEND = jax.default_backend()
        run_config = replace(config, backend="gpu", n_workers=None)
        return time_call(
            basin_stability_gpu_fast_from_initial_conditions,
            run_config,
            initial_conditions,
            seeds,
        )

    raise ValueError(f"Unknown backend: {backend_name}")


def make_row(args, backend_name, repeat_index, summary, runtime_seconds):
    trials_per_second = summary.n_trials / runtime_seconds

    return {
        "backend": backend_name,
        "repeat": repeat_index,
        "n_trials": summary.n_trials,
        "runtime_seconds": runtime_seconds,
        "trials_per_second": trials_per_second,
        "basin_stability": summary.basin_stability,
        "successes": summary.successes,
        "sync_failures": summary.sync_failures,
        "integration_failures": summary.integration_failures,
        "sync_time_mean": summary.sync_time_mean,
        "n_workers": args.workers if backend_name == "parallel_cpu" else 1,
        "n_nodes": args.n_nodes,
        "tmax": args.tmax,
        "dt": args.dt,
        "sync_tol": args.sync_tol,
        "sampling_low": args.sampling_low,
        "sampling_high": args.sampling_high,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "jax_backend": JAX_BACKEND,
    }


def add_speedups(rows):
    grouped = {}

    for row in rows:
        key = (row["n_trials"], row["repeat"])
        grouped.setdefault(key, {})[row["backend"]] = row

    for group in grouped.values():
        serial_time = group.get("serial_cpu", {}).get("runtime_seconds")
        parallel_time = group.get("parallel_cpu", {}).get("runtime_seconds")

        for row in group.values():
            runtime = row["runtime_seconds"]
            row["speedup_vs_serial"] = (
                serial_time / runtime if serial_time is not None else ""
            )
            row["speedup_vs_parallel_cpu"] = (
                parallel_time / runtime if parallel_time is not None else ""
            )


def write_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "backend",
        "repeat",
        "n_trials",
        "runtime_seconds",
        "trials_per_second",
        "speedup_vs_serial",
        "speedup_vs_parallel_cpu",
        "basin_stability",
        "successes",
        "sync_failures",
        "integration_failures",
        "sync_time_mean",
        "n_workers",
        "n_nodes",
        "tmax",
        "dt",
        "sync_tol",
        "sampling_low",
        "sampling_high",
        "platform",
        "python",
        "jax_backend",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def backend_order(skip_serial, skip_gpu):
    backends = list(BACKENDS)

    if skip_serial:
        backends.remove("serial_cpu")

    if skip_gpu:
        backends.remove("fast_gpu")

    return tuple(backends)


def main():
    args = parse_args()
    rows = []

    for n_trials in args.trial_counts:
        base_config = make_config(
            args=args,
            backend="cpu",
            n_trials=n_trials,
        )
        seeds = trial_seeds(
            base_seed=base_config.base_seed,
            n_trials=base_config.n_trials,
        )
        initial_conditions = make_initial_conditions(
            config=base_config,
            seeds=seeds,
        )
        progress_interval = max(1, n_trials // 10)

        for repeat_index in range(1, args.repeats + 1):
            for backend_name in backend_order(args.skip_serial, args.skip_gpu):
                print(
                    f"Running {backend_name}: "
                    f"n_trials={n_trials}, repeat={repeat_index}",
                    flush=True,
                )

                summary, runtime_seconds = run_backend(
                    backend_name=backend_name,
                    config=base_config,
                    initial_conditions=initial_conditions,
                    seeds=seeds,
                    progress_interval=progress_interval,
                )

                rows.append(
                    make_row(
                        args=args,
                        backend_name=backend_name,
                        repeat_index=repeat_index,
                        summary=summary,
                        runtime_seconds=runtime_seconds,
                    )
                )

                print(
                    f"Finished {backend_name}: "
                    f"{runtime_seconds:.3f} s, "
                    f"{summary.n_trials / runtime_seconds:.3f} trials/s",
                    flush=True,
                )

    add_speedups(rows)
    write_csv(rows, args.output)
    print(f"Wrote benchmark table to {args.output}")


if __name__ == "__main__":
    main()
