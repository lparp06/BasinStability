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
import math
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from network_dynamics.core.jax_config import enable_x64

enable_x64()

import jax
import jax.numpy as jnp
import numpy as np


def _setup_jax_compilation_cache() -> str | None:
    """Enable JAX persistent compilation cache if JAX_COMPILATION_CACHE_DIR is set."""
    cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    if not cache_dir:
        return None
    os.makedirs(cache_dir, exist_ok=True)
    try:
        jax.config.update("jax_compilation_cache_dir", cache_dir)
        return cache_dir
    except Exception:
        pass
    try:
        from jax.experimental.compilation_cache import compilation_cache as cc
        cc.set_cache_dir(cache_dir)
        return cache_dir
    except Exception:
        return None

from network_dynamics.core.msf import (
    MSFConfig,
    config_to_jax_arrays,
    find_zero_brackets,
    interpolate_zeros,
    merge_close_brackets,
    normalize_msf_dynamics,
    scan_msf_from_transient_state_jax,
    stable_intervals_from_brackets,
)
from network_dynamics.core.msf.integration import run_transient_jax
from network_dynamics.core.msf_cache import (
    append_msf_cache_result,
    find_cached_msf_result,
    make_msf_cache_key,
)
from network_dynamics.core.dynamics_parameters import (
    DYNAMICS_MSF_INITIAL_STATES,
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
        "--d",
        type=float,
        default=None,
        help="Fourth dynamics parameter (Chua: a_nl, inner-region slope). Omit to use default.",
    )
    parser.add_argument(
        "--e",
        type=float,
        default=None,
        help="Fifth dynamics parameter (Chua: b_nl, outer-region slope). Omit to use default.",
    )
    parser.add_argument(
        "--dynamics",
        default="rossler",
        help=(
            "Oscillator dynamics for the synchronized trajectory and "
            "variational equation."
        ),
    )
    parser.add_argument(
        "--initial-state",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help=(
            "Initial state for the synchronized oscillator transient. "
            "Must be inside the basin of the attractor. "
            "Defaults are dynamics-specific (e.g. Chua uses 0.1 0 0, "
            "not 1 1 1 which escapes to infinity)."
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
    parser.add_argument(
        "--min-zero-separation",
        type=float,
        default=1.0,
        help=(
            "Merge sign-change brackets closer than this K distance into one zero. "
            "Eliminates spurious crossings from numerical noise near the true zero."
        ),
    )
    parser.add_argument(
        "--progress-chunks",
        type=int,
        default=10,
        help=(
            "Number of sub-batches to split serial work into for progress reporting. "
            "Ignored when n_workers > 1 (parallel mode reports per completed chunk)."
        ),
    )
    return parser.parse_args()


def make_config(args):
    dynamics = normalize_msf_dynamics(args.dynamics)
    params = resolve_dynamics_parameters(
        dynamics=dynamics,
        a=args.a,
        b=args.b,
        c=args.c,
        d=getattr(args, "d", None),
        e=getattr(args, "e", None),
    )
    # Unpack into positional slots; MSFConfig accepts up to 5 (a–e).
    a, b, c = params[0], params[1], params[2]
    d = params[3] if len(params) > 3 else 0.0
    e = params[4] if len(params) > 4 else 0.0

    # Use the per-dynamics safe default, or the user-supplied override.
    if args.initial_state is not None:
        initial_state = tuple(args.initial_state)
    else:
        initial_state = DYNAMICS_MSF_INITIAL_STATES.get(dynamics, (1.0, 1.0, 1.0))

    return MSFConfig(
        a=a,
        b=b,
        c=c,
        d=d,
        e=e,
        dynamics=dynamics,
        initial_state=initial_state,
        target=args.target,
        source=args.source,
        dt=args.dt,
        transient_time=args.transient_time,
        measurement_time=args.measurement_time,
        qr_interval_steps=args.qr_interval_steps,
    )


def scan_msf_batch(config, K_values_np):
    params, initial_state, H = config_to_jax_arrays(config)

    # Run and validate the transient before committing to the expensive
    # measurement phase. initial_state must be inside the attractor basin;
    # e.g. (1,1,1) escapes to ~1e100 for Chua, corrupting all Psi values.
    transient_state = run_transient_jax(
        initial_state, config.transient_steps, config.dt, params, config.dynamics
    )
    transient_state.block_until_ready()
    ts_np = np.asarray(jax.device_get(transient_state))

    if not np.all(np.isfinite(ts_np)) or np.max(np.abs(ts_np)) > 1e6:
        raise ValueError(
            f"Transient integration diverged: state={ts_np}. "
            "The initial_state is outside the attractor basin. "
            "For Chua try --initial-state 0.1 0 0; "
            "for Rössler/Lorenz (1 1 1) is usually safe."
        )

    K_values = jnp.asarray(K_values_np, dtype=jnp.float64)
    psi_values, exponent_values = scan_msf_from_transient_state_jax(
        K_values,
        transient_state,
        H,
        params,
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
    # MPS (Apple Metal) does not support float64. Even if MPS was selected as
    # the default backend when this module was imported, the CPU backend is
    # always present alongside it. Routing computation through the CPU device
    # avoids the float64 restriction without requiring any environment variable
    # tricks that race against JAX's module-level backend initialization.
    cpu_devices = jax.devices("cpu")
    with jax.default_device(cpu_devices[0]):
        chunk_index, config, start, stop, K_chunk = task
        psi_values, exponent_values = scan_msf_batch(config, K_chunk)
    return chunk_index, start, stop, psi_values, exponent_values


def _fmt_seconds(s):
    """Format a duration in seconds as a human-readable string."""
    s = int(s)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def scan_msf_serial(config, tasks, total_K, progress_chunks=10):
    psi_chunks = []
    exponent_chunks = []

    sub_size = max(1, math.ceil(total_K / max(1, progress_chunks * len(tasks))))
    total_sub = sum(math.ceil(len(task[4]) / sub_size) for task in tasks)
    completed_sub = 0
    scan_start = time.perf_counter()
    batch_times: list[float] = []

    print(
        f"Serial scan: {total_K} K values in {total_sub} batches of ~{sub_size} each.",
        flush=True,
    )
    print(
        "  Batch 1 includes JAX JIT compilation — it will be slower than the rest.",
        flush=True,
    )

    for _chunk_index, _config, chunk_start, _chunk_stop, K_chunk in tasks:
        sub_psi: list[np.ndarray] = []
        sub_exp: list[np.ndarray] = []

        # Pad K_chunk so every sub-batch is exactly sub_size elements.
        # A mismatched last batch would cause JAX to recompile the entire
        # GPU kernel (another ~20 min), so we avoid it by padding with the
        # last K value and trimming the results afterward.
        real_len = len(K_chunk)
        n_subs = math.ceil(real_len / sub_size)
        pad_len = n_subs * sub_size - real_len
        if pad_len > 0:
            K_chunk_padded = np.concatenate(
                [K_chunk, np.full(pad_len, K_chunk[-1])]
            )
        else:
            K_chunk_padded = K_chunk

        for s in range(0, len(K_chunk_padded), sub_size):
            e = s + sub_size  # always exactly sub_size after padding
            k_start_val = chunk_start + s
            k_end_val = chunk_start + min(e, real_len)
            batch_label = f"{completed_sub + 1}/{total_sub}"
            is_first = completed_sub == 0

            print(
                f"  [{batch_label}] K[{k_start_val}:{k_end_val}]  starting"
                + ("  (JIT compile on first batch)" if is_first else ""),
                flush=True,
            )

            batch_start = time.perf_counter()
            psi_sub, exp_sub = scan_msf_batch(config, K_chunk_padded[s:e])
            batch_elapsed = time.perf_counter() - batch_start
            batch_times.append(batch_elapsed)

            # Trim padding from the last sub-batch before storing.
            keep = min(sub_size, real_len - s)
            sub_psi.append(psi_sub[:keep])
            sub_exp.append(exp_sub[:keep])
            completed_sub += 1

            total_elapsed = time.perf_counter() - scan_start
            k_done = chunk_start + e
            pct = 100.0 * completed_sub / total_sub
            k_per_s = k_done / total_elapsed if total_elapsed > 0 else 0.0

            if completed_sub >= 2:
                avg_batch = sum(batch_times[1:]) / len(batch_times[1:])
                remaining_batches = total_sub - completed_sub
                eta = avg_batch * remaining_batches
                eta_str = f"  ETA ~{_fmt_seconds(eta)}"
            elif completed_sub == 1:
                eta_str = "  (ETA available after batch 2)"
            else:
                eta_str = ""

            compile_note = (
                f"  [compile+run: {_fmt_seconds(batch_times[0])}]"
                if is_first
                else ""
            )
            print(
                f"  [{batch_label}] done  "
                f"this={_fmt_seconds(batch_elapsed)}  "
                f"total={_fmt_seconds(total_elapsed)}  "
                f"{pct:.0f}%  {k_per_s:.1f} K/s"
                f"{compile_note}{eta_str}",
                flush=True,
            )

        psi_chunks.append(np.concatenate(sub_psi))
        exponent_chunks.append(np.concatenate(sub_exp))

    return np.concatenate(psi_chunks), np.concatenate(exponent_chunks)


def scan_msf_parallel(tasks, total_K, n_workers):
    psi_chunks = [None] * len(tasks)
    exponent_chunks = [None] * len(tasks)
    context = multiprocessing.get_context("spawn")

    print(
        f"Scanning {total_K} K values across {len(tasks)} chunks "
        f"with {n_workers} workers (CPU/XLA per worker)...",
        flush=True,
    )

    scan_start = time.perf_counter()
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
            elapsed = time.perf_counter() - scan_start
            remaining = len(tasks) - completed
            eta_str = (
                f"  ETA ~{_fmt_seconds(elapsed / completed * remaining)}"
                if remaining > 0
                else ""
            )
            print(
                f"Finished K chunk {start}:{stop} "
                f"({completed}/{len(tasks)})  "
                f"elapsed {_fmt_seconds(elapsed)}{eta_str}",
                flush=True,
            )

    return np.concatenate(psi_chunks), np.concatenate(exponent_chunks)


def scan_msf(config, K_values_np, chunk_size, n_workers, progress_chunks=10):
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
            progress_chunks=progress_chunks,
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


def print_summary(args, config, cache_dir=None):
    print("MSF scan")
    print("=" * 40)
    print("JAX backend:", jax.default_backend())
    print("JAX devices:", jax.devices())
    if cache_dir:
        print(f"JAX compile cache: {cache_dir}  (first run compiles, subsequent runs reuse)")
    else:
        print(
            "JAX compile cache: OFF  "
            "(set JAX_COMPILATION_CACHE_DIR to cache compiled kernels across runs)"
        )
    extra_params = (
        f", d={config.d}, e={config.e}" if config.dynamics == "chua" else ""
    )
    print(
        "Parameters:",
        f"dynamics={config.dynamics}, a={config.a}, b={config.b}, c={config.c}"
        f"{extra_params}, "
        f"initial_state={config.initial_state}, "
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
        f"chunk_size={args.chunk_size}, workers={args.n_workers}, "
        f"progress_chunks={args.progress_chunks}",
    )
    steps_per_K = config.transient_steps + config.measurement_steps
    print(f"Steps per K value: {steps_per_K:,}  ({config.transient_steps:,} transient + {config.measurement_steps:,} measurement)")
    print(f"Total RK4 steps across all K: {args.n_K * steps_per_K:,}")
    if args.n_workers <= 1:
        print(
            f"GPU/serial note: batch 1/{args.progress_chunks} will be slower than the rest "
            f"due to JAX JIT compilation. Use --progress-chunks to control update frequency."
        )
    print(
        "Paper-accurate settings (PhysRevE.80.036204, dt=0.001):\n"
        "  Rossler/Lorenz/Chen/Chua: --dt 0.001 --transient-time 100 --measurement-time 300\n"
        "  HR neuron: --dt 0.001 --transient-time 1000 --measurement-time 300\n"
        "    (HR slow-adaptation timescale 1/r≈167 time units; short transients give wrong λ)\n"
        "  Lorenz β=2 (Fig. 2): default --b 2.0 | Lorenz β=8/3 (Fig. 3): --b 2.6667\n"
        "  Coupling i→j: --source <i-1> --target <j-1>  (e.g. 2→1: --source 1 --target 0)"
    )
    print()


def warn_if_msf_step_is_large(dt, K_max):
    stiffness_scale = dt * K_max

    if stiffness_scale > 2.5:
        print(
            f"WARNING: dt*K_max={stiffness_scale:.3g} exceeds RK4 stability limit (~2.785). "
            "MSF values at high K will diverge. Reduce --dt.",
            flush=True,
        )
        print()
    elif stiffness_scale > 0.1:
        print(
            f"ACCURACY WARNING: dt*K_max={stiffness_scale:.3g} ({dt}*{K_max}). "
            "For diagonal coupling the variational equation stiffness scales with K, "
            "so RK4 accuracy degrades at high K. "
            "Zero crossings above K~{:.0f} may be shifted by O(dt^4*K^5) error. "
            "The paper uses dt=0.001; for K_max={} that gives dt*K_max={:.3g}.".format(
                K_max / 10, K_max, 0.001 * K_max
            ),
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

    cache_dir = _setup_jax_compilation_cache()
    print_summary(args, config, cache_dir=cache_dir)
    warn_if_msf_step_is_large(config.dt, args.K_max)
    K_values = np.linspace(args.K_min, args.K_max, args.n_K)

    start = time.perf_counter()
    psi_values, exponent_values = scan_msf(
        config=config,
        K_values_np=K_values,
        chunk_size=args.chunk_size,
        n_workers=args.n_workers,
        progress_chunks=args.progress_chunks,
    )
    elapsed = time.perf_counter() - start

    raw_brackets = find_zero_brackets(K_values, psi_values)
    brackets = merge_close_brackets(raw_brackets, min_separation=args.min_zero_separation)
    stable_intervals = stable_intervals_from_brackets(brackets)
    cache_key = make_msf_cache_key(
        config=config,
        K_min=args.K_min,
        K_max=args.K_max,
        n_K=args.n_K,
    )

    zeros = interpolate_zeros(K_values, psi_values, brackets)
    midpoints = [0.5 * (left + right) for left, right in brackets]

    print()
    print("MSF scan complete")
    print("-" * 40)
    print("Seconds:", f"{elapsed:.3f}")
    print("Psi min:", float(np.nanmin(psi_values)))
    print("Psi max:", float(np.nanmax(psi_values)))
    print("MSF zeros (K, interpolated):", [round(z, 4) for z in zeros])
    print("MSF zeros (K, midpoints):   ", [round(z, 4) for z in midpoints])
    print("Stable K intervals:", stable_intervals)

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
