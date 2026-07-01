"""
Basin stability scan across N randomly-seeded graphs of a given type.

For each graph seed the script:
  1. Builds the graph and computes its coupling-strength interval from MSF zeros.
  2. Writes an MSF+eigenvalue overlay plot (three sigma values at 25/50/75% of the interval).
  3. Runs a full basin-stability scan across n_strengths evenly-spaced coupling strengths.
  4. Saves results to outputs/network_scan/<graph_tag>/seed_<NN>/.

MSF zeros are looked up in the shared cache (outputs/msf_zero_cache.csv) and computed
on first use; subsequent seeds reuse the same zeros because MSF is graph-independent.

Usage
-----
python -m network_dynamics.experiments.network_scan \\
    --graph-type erdos_renyi --n-nodes 100 --edge-probability 0.15 \\
    --n-seeds 10 --dynamics rossler --source 1 --target 0 \\
    --n-trials 1000 --n-strengths 10 --tmax 5000 --backend gpu

python -m network_dynamics.experiments.network_scan \\
    --graph-type watts_strogatz --ws-k 6 --n-seeds 10 --backend cpu
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

from network_dynamics.core.basin_common import sample_initial_conditions_batch
from network_dynamics.core.config import BasinConfig
from network_dynamics.core.coupling import rank_one_inner_coupling_matrix
from network_dynamics.core.coupling_strengths import (
    coupling_strength_intervals_from_stable,
    interval_coupling_strengths,
    laplacian_nonzero_eigenvalue_bounds,
)
from network_dynamics.core.dynamics_parameters import resolve_dynamics_parameters
from network_dynamics.core.graphs import graph_laplacian, make_graph
from network_dynamics.core.msf import MSFParams, find_msf_zeros, default_k_range
from network_dynamics.core.msf_cache import (
    append_msf_cache_result,
    find_cached_msf_result,
    make_msf_cache_key,
)
from network_dynamics.core.oscillators import normalize_dynamics_type
from network_dynamics.core.sampling import trial_seeds
from network_dynamics.cpu.basin import (
    basin_stability_cpu_from_initial_conditions,
    basin_stability_numba_from_initial_conditions,
)
from network_dynamics.gpu.basin_fast import basin_stability_gpu_fast_from_initial_conditions
from network_dynamics.experiments.plot_msf_with_eigenvalues import make_eigenvalue_plot
from network_dynamics.experiments.plot_stability_curves import plot_basin_stability_vs_k

_MSF_CACHE_DEFAULT = "outputs/msf_zero_cache.csv"

logger = logging.getLogger("network_scan")


class _FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


def _setup_logging(log_path: Path) -> None:
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(message)s")

    file_handler = _FlushFileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Basin stability scan across N randomly-seeded graphs."
    )

    # Graph
    p.add_argument("--graph-type", default="erdos_renyi",
                   help="Graph type: erdos_renyi, barabasi_albert, watts_strogatz, path_graph")
    p.add_argument("--n-nodes", type=int, default=100)
    p.add_argument("--edge-probability", type=float, default=0.15,
                   help="ER edge probability or WS rewiring probability")
    p.add_argument("--ba-m", type=int, default=8,
                   help="BA: number of edges to attach per new node")
    p.add_argument("--ws-k", type=int, default=6,
                   help="WS: number of nearest-ring neighbors")

    # Seeds
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--seed-start", type=int, default=42)

    # Dynamics
    p.add_argument("--dynamics", default="rossler")
    p.add_argument("--source", "--msf-source", dest="source", type=int, default=1,
                   help="Column index of 1 in coupling matrix H (MSF source variable)")
    p.add_argument("--target", "--msf-target", dest="target", type=int, default=0,
                   help="Row index of 1 in coupling matrix H (MSF target variable)")
    p.add_argument("--a", type=float, default=None, help="First dynamics parameter (override default)")
    p.add_argument("--b", type=float, default=None, help="Second dynamics parameter")
    p.add_argument("--c", type=float, default=None, help="Third dynamics parameter")

    # Basin
    p.add_argument("--n-trials", type=int, default=1000)
    p.add_argument("--n-strengths", type=int, default=10,
                   help="Number of coupling strengths to scan per graph")
    p.add_argument("--tmax", type=float, default=5000.0)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--backend", choices=("cpu", "gpu", "numba"), default="gpu")
    p.add_argument("--n-workers", type=int, default=8,
                   help="CPU worker processes (0=auto-detect, ignored for gpu)")
    p.add_argument("--integrator", choices=("LSODA", "RK4"), default="RK4")
    p.add_argument("--sync-tol", type=float, default=1e-3)
    p.add_argument(
        "--success-definition",
        choices=("final_success", "window_success", "first_crossing"),
        default="first_crossing",
    )
    p.add_argument("--interval-index", type=int, default=0,
                   help="Which MSF stable interval to use (0 = first)")

    # MSF
    p.add_argument("--K-min", type=float, default=None)
    p.add_argument("--K-max", type=float, default=None)
    p.add_argument("--n-K", type=int, default=1001)
    p.add_argument("--msf-cache", default=_MSF_CACHE_DEFAULT)

    # Output
    p.add_argument("--output-dir", default="outputs/network_scan")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _graph_subdir_name(graph_type, n_nodes, edge_prob, ba_m, ws_k):
    if graph_type == "barabasi_albert":
        return f"{graph_type}_n{n_nodes}_m{ba_m}"
    if graph_type == "watts_strogatz":
        return f"{graph_type}_n{n_nodes}_k{ws_k}_p{edge_prob}"
    return f"{graph_type}_n{n_nodes}_p{edge_prob}"


def _run_subdir_name(graph_type, n_nodes, edge_prob, ba_m, ws_k, dynamics, source, target):
    graph_part = _graph_subdir_name(graph_type, n_nodes, edge_prob, ba_m, ws_k)
    return f"{graph_part}__{dynamics}_s{source}t{target}"


def _auto_n_workers(args):
    if args.backend == "gpu":
        return None
    if args.n_workers > 0:
        return args.n_workers
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    return int(slurm) if slurm else (os.cpu_count() or 1)


def _fmt_seconds(s):
    s = int(s)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def _log_seed_spectral_intervals(G, stable_intervals):
    """
    Log Laplacian spectral bounds and candidate sigma intervals for a seed.
    """
    try:
        lambda_2, lambda_n = laplacian_nonzero_eigenvalue_bounds(G)
    except ValueError as exc:
        logger.info(f"  Laplacian eigenvalues: unavailable ({exc})")
        return

    eigenratio = lambda_n / lambda_2
    logger.info(
        "  Laplacian eigenvalues: "
        f"lambda_2={lambda_2:.6g}, lambda_N={lambda_n:.6g}, "
        f"lambda_N/lambda_2={eigenratio:.6g}"
    )

    for interval_index, (k_lo, k_hi) in enumerate(stable_intervals):
        sigma_lo = k_lo / lambda_2
        sigma_hi = k_hi / lambda_n
        status = "valid" if sigma_lo < sigma_hi else "invalid"
        logger.info(
            f"  interval {interval_index}: "
            f"K=[{k_lo:.6g}, {k_hi:.6g}] -> "
            f"sigma=[{sigma_lo:.6g}, {sigma_hi:.6g}] ({status})"
        )


def _laplacian_eigenvalues(G):
    laplacian = np.asarray(graph_laplacian(G), dtype=float)
    return np.sort(np.linalg.eigvalsh(laplacian))


def _write_eigenvalues_file(path, eigenvalues):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for value in eigenvalues:
            f.write(f"{value:.10g}\n")


# ---------------------------------------------------------------------------
# MSF zeros (graph-independent — compute once, reuse for all seeds)
# ---------------------------------------------------------------------------

def _get_msf_zeros(msf_config, K_min, K_max, n_K, cache_path):
    key = make_msf_cache_key(config=msf_config, K_min=K_min, K_max=K_max, n_K=n_K)
    cached = find_cached_msf_result(cache_path=cache_path, key=key)

    if cached is not None:
        zeros = cached["zeros"]
        stable_intervals = cached["stable_intervals"]
        expected = (len(zeros) + 1) // 2 if zeros else 0
        stale = zeros and (not stable_intervals or len(stable_intervals) < expected)
        if not stale:
            logger.info(f"Using cached MSF zeros from {cache_path}")
            return zeros, stable_intervals
        logger.info("Stale cache entry (stable_intervals incomplete) — recomputing MSF scan.")

    logger.info("Running MSF scan...")
    zeros, stable_intervals = find_msf_zeros(
        params_obj=msf_config, K_min=K_min, K_max=K_max, n_K=n_K, verbose=True,
    )
    append_msf_cache_result(
        cache_path=cache_path, key=key, zeros=zeros, stable_intervals=stable_intervals,
    )
    logger.info(f"Saved MSF zeros to {cache_path}")
    return zeros, stable_intervals


# ---------------------------------------------------------------------------
# Basin scan
# ---------------------------------------------------------------------------

def _make_basin_config(G, dynamics, params, source, target, coupling_strength,
                       tmax, dt, integrator, n_trials, base_seed, sync_tol,
                       backend, n_workers, success_definition):
    H = rank_one_inner_coupling_matrix(target=target, source=source, dimension=3)
    return BasinConfig(
        G=G,
        dynamics=dynamics,
        dimension=3,
        parameters=params,
        coupling_strength=float(coupling_strength),
        H=H,
        tmax=tmax,
        dt=dt,
        integrator=integrator,
        n_trials=n_trials,
        base_seed=base_seed,
        sampling_bounds=(-5.0, 5.0),
        sync_tol=sync_tol,
        success_definition=success_definition,
        backend=backend,
        n_workers=n_workers,
    ).validate()


def _run_basin_scan(G, args, dynamics, params, source, target, interval, n_workers):
    strengths = interval_coupling_strengths(interval=interval, n_strengths=args.n_strengths)

    first_config = _make_basin_config(
        G=G, dynamics=dynamics, params=params, source=source, target=target,
        coupling_strength=strengths[0], tmax=args.tmax, dt=args.dt,
        integrator=args.integrator, n_trials=args.n_trials, base_seed=args.base_seed,
        sync_tol=args.sync_tol, backend=args.backend, n_workers=n_workers,
        success_definition=args.success_definition,
    )
    seeds = trial_seeds(base_seed=first_config.base_seed, n_trials=first_config.n_trials)
    initial_conditions = sample_initial_conditions_batch(config=first_config, seeds=seeds)

    rows = []
    n = len(strengths)
    scan_start = time.perf_counter()

    for i, strength in enumerate(strengths, 1):
        config = _make_basin_config(
            G=G, dynamics=dynamics, params=params, source=source, target=target,
            coupling_strength=strength, tmax=args.tmax, dt=args.dt,
            integrator=args.integrator, n_trials=args.n_trials, base_seed=args.base_seed,
            sync_tol=args.sync_tol, backend=args.backend, n_workers=n_workers,
            success_definition=args.success_definition,
        )

        elapsed = time.perf_counter() - scan_start
        eta = (
            f"  ETA ~{_fmt_seconds(elapsed / (i - 1) * (n - i))}"
            if i > 1 and i < n else ""
        )
        logger.info(
            f"  [{i}/{n}] sigma={strength:.6g}  elapsed={_fmt_seconds(elapsed)}{eta}",
        )

        t0 = time.perf_counter()
        if args.backend == "gpu":
            summary = basin_stability_gpu_fast_from_initial_conditions(
                config=config,
                initial_conditions_batch=initial_conditions,
                seeds=seeds,
            )
        elif args.backend == "numba":
            summary = basin_stability_numba_from_initial_conditions(
                config=config,
                initial_conditions_batch=initial_conditions,
                seeds=seeds,
            )
        else:
            summary = basin_stability_cpu_from_initial_conditions(
                config=config,
                initial_conditions_batch=initial_conditions,
                seeds=seeds,
                progress_label=f"strength {i}/{n}",
                progress_interval=5,
                progress_stream=sys.__stdout__,
            )

        elapsed_this = time.perf_counter() - t0
        logger.info(
            f"  -> basin={summary.basin_stability:.4f}  ({_fmt_seconds(elapsed_this)})",
        )

        rows.append({
            "index": i - 1,
            "coupling_strength": float(strength),
            "basin_stability": summary.basin_stability,
            "successes": summary.successes,
            "sync_failures": summary.sync_failures,
            "integration_failures": summary.integration_failures,
            "sync_time_mean": summary.sync_time_mean,
            "seconds": elapsed_this,
        })

    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _write_csv(path, rows, n_trials, dynamics):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "dynamics", "coupling_strength", "basin_stability", "n_trials",
        "successes", "sync_failures", "integration_failures", "sync_time_mean", "seconds",
    )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "dynamics": dynamics,
                "coupling_strength": row["coupling_strength"],
                "basin_stability": row["basin_stability"],
                "n_trials": n_trials,
                "successes": row["successes"],
                "sync_failures": row["sync_failures"],
                "integration_failures": row["integration_failures"],
                "sync_time_mean": row["sync_time_mean"],
                "seconds": row["seconds"],
            })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    dynamics = normalize_dynamics_type(args.dynamics)
    params = resolve_dynamics_parameters(dynamics, a=args.a, b=args.b, c=args.c)
    source, target = args.source, args.target
    n_workers = _auto_n_workers(args)

    # Output root
    run_subdir = _run_subdir_name(
        args.graph_type, args.n_nodes, args.edge_probability, args.ba_m, args.ws_k,
        dynamics, source, target,
    )
    output_root = Path(args.output_dir) / run_subdir
    output_root.mkdir(parents=True, exist_ok=True)
    _setup_logging(output_root / "output.txt")

    logger.info("Run configuration:")
    for key, value in sorted(vars(args).items()):
        logger.info(f"  {key} = {value}")
    logger.info("")

    # MSF zeros (same for all graph seeds — depends only on dynamics/coupling scheme)
    msf_config = MSFParams(
        dynamics=dynamics,
        a=params[0], b=params[1], c=params[2],
        target=target, source=source,
    )
    k_defaults = default_k_range(dynamics, source, target)
    K_min = args.K_min if args.K_min is not None else k_defaults[0]
    K_max = args.K_max if args.K_max is not None else k_defaults[1]

    zeros, stable_intervals = _get_msf_zeros(
        msf_config, K_min, K_max, args.n_K, args.msf_cache,
    )

    if not stable_intervals:
        logger.error(
            f"ERROR: No stable MSF intervals found for "
            f"{dynamics} source={source} target={target}. "
            "Try a different coupling scheme or increase K_max."
        )
        return 1

    logger.info(f"Stable K intervals: {stable_intervals}")

    # MSF CSV for eigenvalue plots (optional — skip plot if missing)
    msf_csv = Path(f"outputs/{dynamics}/csv/msf_{dynamics}_s{source}t{target}.csv")
    if not msf_csv.exists():
        logger.warning(f"WARNING: MSF CSV not found at {msf_csv}; eigenvalue plots will be skipped.")
        msf_csv = None

    logger.info(f"\nNetwork scan")
    logger.info(f"  graph={args.graph_type}  n={args.n_nodes}  seeds={args.seed_start}–"
          f"{args.seed_start + args.n_seeds - 1}")
    logger.info(f"  dynamics={dynamics}  source={source}  target={target}")
    logger.info(f"  trials={args.n_trials}  strengths={args.n_strengths}  backend={args.backend}")
    logger.info(f"  output: {output_root}\n")

    for seed in range(args.seed_start, args.seed_start + args.n_seeds):
        seed_num = seed - args.seed_start + 1
        logger.info(f"{'=' * 54}")
        logger.info(f"Seed {seed}  ({seed_num}/{args.n_seeds})")
        logger.info(f"{'=' * 54}")

        G = make_graph(
            args.graph_type,
            n_nodes=args.n_nodes,
            seed=seed,
            edge_probability=args.edge_probability,
            ba_m=args.ba_m,
            ws_k=args.ws_k,
        )
        logger.info(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        _log_seed_spectral_intervals(G, stable_intervals)

        sigma_intervals = coupling_strength_intervals_from_stable(G, stable_intervals)

        if not sigma_intervals:
            logger.info(
                f"  No valid coupling interval for seed {seed} — "
                "graph spectrum does not fit inside the stable MSF interval. Skipping."
            )
            continue

        if args.interval_index >= len(sigma_intervals):
            logger.info(
                f"  interval_index={args.interval_index} out of range "
                f"(only {len(sigma_intervals)} interval(s)). Skipping."
            )
            continue

        interval = sigma_intervals[args.interval_index]
        logger.info(f"  sigma interval [{interval.lower:.4f}, {interval.upper:.4f}]")

        seed_dir = output_root / f"seed_{seed:02d}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = seed_dir / "plots"
        csv_dir = seed_dir / "csv"

        # Laplacian eigenvalues (drop the trivial first eigenvalue)
        eigenvalues = _laplacian_eigenvalues(G)
        _write_eigenvalues_file(seed_dir / "eigenvalues.txt", eigenvalues[1:])

        # Eigenvalue overlay plot
        if msf_csv is not None:
            w = interval.upper - interval.lower
            plot_sigmas = [interval.lower + f * w for f in [0.25, 0.5, 0.75]]
            try:
                make_eigenvalue_plot(
                    G=G,
                    msf_csv_path=msf_csv,
                    coupling_strengths=plot_sigmas,
                    output_path=plots_dir / "msf_eigenvalues.png",
                    title=f"{args.graph_type} n={args.n_nodes} seed={seed}",
                )
            except Exception as exc:
                logger.warning(f"  WARNING: eigenvalue plot failed: {exc}")

        # Basin stability scan
        rows = _run_basin_scan(
            G=G,
            args=args,
            dynamics=dynamics,
            params=params,
            source=source,
            target=target,
            interval=interval,
            n_workers=n_workers,
        )

        csv_path = csv_dir / f"basin_{dynamics}_s{source}t{target}.csv"
        _write_csv(csv_path, rows, n_trials=args.n_trials, dynamics=dynamics)
        logger.info(f"  Wrote: {csv_path}")

        basin_plot_path = plots_dir / f"basin_{dynamics}_s{source}t{target}_vs_k.png"
        plot_basin_stability_vs_k(
            K_values=[row["coupling_strength"] for row in rows],
            basin_stabilities=[row["basin_stability"] for row in rows],
            output_path=basin_plot_path,
            n_trials=args.n_trials,
            title=f"{args.graph_type} n={args.n_nodes} seed={seed} — {dynamics} s{source}t{target}",
        )
        logger.info(f"  Wrote: {basin_plot_path}\n")

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
