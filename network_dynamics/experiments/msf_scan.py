"""
Run an MSF-only scan for one oscillator.

Example
-------
python3 -m network_dynamics.experiments.msf_scan \
    --measurement-time 5000 \
    --transient-time 1000 \
    --K-min 0 \
    --K-max 10 \
    --n-K 201
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from network_dynamics.core.jax_config import enable_x64

enable_x64()

import jax
import jax.numpy as jnp
import numpy as np

from network_dynamics.core.msf import (
    MSFConfig,
    config_to_jax_arrays,
    find_zero_brackets,
    normalize_msf_dynamics,
    scan_msf_jax,
    stable_intervals_from_brackets,
)
from network_dynamics.core.msf_cache import (
    append_msf_cache_result,
    find_cached_msf_result,
    make_msf_cache_key,
)
from network_dynamics.core.dynamics_parameters import (
    format_parameter_defaults,
    resolve_dynamics_parameters,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a master-stability-function scan only."
    )
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
    parser.add_argument(
        "--dynamics",
        default="rossler",
        help=(
            "Oscillator dynamics for the synchronized trajectory and "
            "variational equation."
        ),
    )
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--transient-time", type=float, default=100.0)
    parser.add_argument("--measurement-time", "--tmax", type=float, default=300.0)
    parser.add_argument("--qr-interval-steps", type=int, default=10)
    parser.add_argument("--K-min", type=float, default=0.0)
    parser.add_argument("--K-max", type=float, default=10.0)
    parser.add_argument("--n-K", type=int, default=101)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Number of CPU worker processes for parallel K chunks.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional path to write K, psi, and Lyapunov exponents.",
    )
    parser.add_argument(
        "--msf-cache",
        default="outputs/msf_zero_cache.csv",
        help="CSV cache path for MSF zeros and settings.",
    )
    return parser.parse_args()


def make_config(args):
    dynamics = normalize_msf_dynamics(args.dynamics)
    a, b, c = resolve_dynamics_parameters(
        dynamics=dynamics,
        a=args.a,
        b=args.b,
        c=args.c,
    )
    return MSFConfig(
        a=a,
        b=b,
        c=c,
        dynamics=dynamics,
        target=args.target,
        source=args.source,
        dt=args.dt,
        transient_time=args.transient_time,
        measurement_time=args.measurement_time,
        qr_interval_steps=args.qr_interval_steps,
    )


def scan_msf_batch(config, K_values_np):
    params, initial_state, H = config_to_jax_arrays(config)

    K_values = jnp.asarray(K_values_np, dtype=jnp.float64)
    psi_values, exponent_values = scan_msf_jax(
        K_values,
        initial_state,
        H,
        params,
        config.transient_steps,
        config.measurement_steps,
        config.dt,
        config.qr_interval_steps,
        config.dynamics,
    )
    psi_values.block_until_ready()
    return (
        np.asarray(jax.device_get(psi_values)),
        np.asarray(jax.device_get(exponent_values)),
    )


def make_chunk_tasks(config, K_values_np, chunk_size, n_workers):
    if chunk_size is None:
        index_chunks = np.array_split(
            np.arange(len(K_values_np)),
            min(n_workers, len(K_values_np)),
        )
        return [
            (chunk_index, config, int(indices[0]), int(indices[-1]) + 1, K_values_np[indices])
            for chunk_index, indices in enumerate(index_chunks)
            if len(indices) > 0
        ]

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive or None.")

    tasks = []
    for chunk_index, start in enumerate(range(0, len(K_values_np), chunk_size)):
        stop = min(start + chunk_size, len(K_values_np))
        tasks.append((chunk_index, config, start, stop, K_values_np[start:stop]))

    return tasks


def scan_msf_chunk_worker(task):
    chunk_index, config, start, stop, K_chunk = task
    psi_values, exponent_values = scan_msf_batch(config, K_chunk)
    return chunk_index, start, stop, psi_values, exponent_values


def scan_msf_serial(config, tasks, total_K):
    psi_chunks = []
    exponent_chunks = []

    for _chunk_index, _config, start, stop, K_chunk in tasks:
        print(f"Scanning K chunk {start}:{stop} of {total_K}", flush=True)
        psi_chunk, exponent_chunk = scan_msf_batch(config, K_chunk)
        psi_chunks.append(psi_chunk)
        exponent_chunks.append(exponent_chunk)

    return np.concatenate(psi_chunks), np.concatenate(exponent_chunks)


def scan_msf_parallel(tasks, total_K, n_workers):
    psi_chunks = [None] * len(tasks)
    exponent_chunks = [None] * len(tasks)
    context = multiprocessing.get_context("spawn")

    print(
        f"Scanning {total_K} K values across {len(tasks)} chunks "
        f"with {n_workers} workers...",
        flush=True,
    )

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=context) as executor:
        future_to_task = {
            executor.submit(scan_msf_chunk_worker, task): task
            for task in tasks
        }

        for completed, future in enumerate(as_completed(future_to_task), start=1):
            _task = future_to_task[future]
            chunk_index, start, stop, psi_chunk, exponent_chunk = future.result()
            psi_chunks[chunk_index] = psi_chunk
            exponent_chunks[chunk_index] = exponent_chunk
            print(
                f"Finished K chunk {start}:{stop} "
                f"({completed}/{len(tasks)})",
                flush=True,
            )

    return np.concatenate(psi_chunks), np.concatenate(exponent_chunks)


def scan_msf(config, K_values_np, chunk_size, n_workers):
    tasks = make_chunk_tasks(
        config=config,
        K_values_np=K_values_np,
        chunk_size=chunk_size,
        n_workers=n_workers,
    )

    if n_workers <= 1:
        return scan_msf_serial(
            config=config,
            tasks=tasks,
            total_K=len(K_values_np),
        )

    return scan_msf_parallel(
        tasks=tasks,
        total_K=len(K_values_np),
        n_workers=n_workers,
    )


def write_csv(path, K_values, psi_values, exponent_values):
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["K", "psi", "lyapunov_0", "lyapunov_1", "lyapunov_2"])
        for K, psi, exponents in zip(K_values, psi_values, exponent_values):
            writer.writerow([K, psi, *exponents.tolist()])


def print_summary(args, config):
    print("MSF scan")
    print("=" * 40)
    print("JAX backend:", jax.default_backend())
    print("JAX devices:", jax.devices())
    print(
        "Parameters:",
        f"dynamics={config.dynamics}, a={config.a}, b={config.b}, c={config.c}, "
        f"coupling={config.source + 1}->{config.target + 1}",
    )
    print("Registered defaults:", format_parameter_defaults())
    print(
        "Time:",
        f"dt={config.dt}, transient_time={config.transient_time}, "
        f"measurement_time={config.measurement_time}, "
        f"transient_steps={config.transient_steps}, "
        f"measurement_steps={config.measurement_steps}",
    )
    print(
        "K scan:",
        f"K_min={args.K_min}, K_max={args.K_max}, n_K={args.n_K}, "
        f"chunk_size={args.chunk_size}, workers={args.n_workers}",
    )
    print()


def warn_if_msf_step_is_large(dt, K_max):
    stiffness_scale = dt * K_max
    if stiffness_scale <= 2.5:
        return

    print(
        "WARNING: dt*K_max is large for explicit RK4 "
        f"({dt}*{K_max}={stiffness_scale:.3g}). "
        "High-K MSF values may show artificial positive growth; "
        "try a smaller --dt such as 0.01 or 0.005.",
        flush=True,
    )
    print()


def main():
    args = parse_args()
    config = make_config(args)
    config.validate()

    if args.K_max <= args.K_min:
        raise ValueError("K_max must be greater than K_min.")
    if args.n_K < 2:
        raise ValueError("n_K must be at least 2.")
    if args.n_workers <= 0:
        raise ValueError("n_workers must be positive.")

    print_summary(args, config)
    warn_if_msf_step_is_large(config.dt, args.K_max)
    K_values = np.linspace(args.K_min, args.K_max, args.n_K)

    start = time.perf_counter()
    psi_values, exponent_values = scan_msf(
        config=config,
        K_values_np=K_values,
        chunk_size=args.chunk_size,
        n_workers=args.n_workers,
    )
    elapsed = time.perf_counter() - start

    brackets = find_zero_brackets(K_values, psi_values)
    stable_intervals = stable_intervals_from_brackets(brackets)
    cache_key = make_msf_cache_key(
        config=config,
        K_min=args.K_min,
        K_max=args.K_max,
        n_K=args.n_K,
    )

    print()
    print("MSF scan complete")
    print("-" * 40)
    print("Seconds:", f"{elapsed:.3f}")
    print("Psi min:", float(np.nanmin(psi_values)))
    print("Psi max:", float(np.nanmax(psi_values)))
    print("Zero brackets:", brackets)
    print("Stable K intervals from brackets:", stable_intervals)

    zeros = [0.5 * (left + right) for left, right in brackets]
    print("Midpoint zeros:", zeros)

    cached = find_cached_msf_result(
        cache_path=args.msf_cache,
        key=cache_key,
    )
    if cached is None:
        cache_path = append_msf_cache_result(
            cache_path=args.msf_cache,
            key=cache_key,
            zeros=zeros,
            zero_brackets=brackets,
            stable_intervals=stable_intervals,
        )
        print("Saved MSF zeros to cache:", cache_path)
    else:
        print("MSF zeros already present in cache:", args.msf_cache)

    if args.csv is not None:
        write_csv(
            path=args.csv,
            K_values=K_values,
            psi_values=psi_values,
            exponent_values=exponent_values,
        )
        print("Wrote CSV:", args.csv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
