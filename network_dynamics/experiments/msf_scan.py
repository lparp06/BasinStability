"""Compute Psi(K) over a range of coupling strengths and write msf_scan.csv.

The output CSV has two columns (K, psi) and is consumed directly by
plot_stability_curves.py.

Examples
--------
# Rössler x→x coupling, K in [0, 10]:
python -m network_dynamics.experiments.msf_scan \
    --dynamics rossler --target 0 --source 0 \
    

# Lorenz σ→ρ coupling, K in [0, 50], fine grid:
python -m network_dynamics.experiments.msf_scan \
    --dynamics lorenz --a 10 --b 2 --c 28 \
    --target 1 --source 0 --K-max 50 --n-K 2001 \
    --output outputs/msf_scan_lorenz.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from multiprocessing import cpu_count
from pathlib import Path


def _parse_workers_early():
    for i, arg in enumerate(sys.argv):
        if arg == "--workers" and i + 1 < len(sys.argv):
            try:
                return max(1, int(sys.argv[i + 1]))
            except ValueError:
                pass
    return max(1, cpu_count() - 1)


_N_CORES = _parse_workers_early()
os.environ["NUMBA_NUM_THREADS"]    = str(_N_CORES)
os.environ["OMP_NUM_THREADS"]      = str(_N_CORES)
os.environ["OPENBLAS_NUM_THREADS"] = str(_N_CORES)

import numba
numba.set_num_threads(_N_CORES)

import numpy as np

from network_dynamics.core.msf import (
    PARAM_LENGTHS, INITIAL_STATES,
    run_transient, scan_msf, warmup,
    MSFParams, default_k_range,
)


def _fmt(seconds: float) -> str:
    s = int(seconds)
    if s < 60:   return f"{s}s"
    if s < 3600: return f"{s//60}m{s%60:02d}s"
    return f"{s//3600}h{(s%3600)//60:02d}m"


def run_scan(p: MSFParams, K_min: float, K_max: float, n_K: int) -> tuple[np.ndarray, np.ndarray]:
    """Run transient + MSF scan, return (K_arr, psi_arr)."""
    dyn_id   = p.dyn_id
    n_params = PARAM_LENGTHS[dyn_id]
    params   = np.array([p.a, p.b, p.c, p.d, p.e][:n_params])
    s0       = INITIAL_STATES[dyn_id].copy()

    tr_steps = int(round(p.transient_time / p.dt))
    m_steps  = int(round(p.measurement_time / p.dt))

    print(f"  transient: {tr_steps:,} steps ... ", end="", flush=True)
    t0 = time.perf_counter()
    settled = run_transient(s0, params, dyn_id, p.dt, tr_steps)
    print(f"done ({_fmt(time.perf_counter() - t0)})  state={np.round(settled, 3)}")

    if not np.isfinite(settled).all() or np.abs(settled).max() > 1e6:
        raise RuntimeError("Transient diverged — check oscillator parameters.")

    K_arr = np.linspace(K_min, K_max, n_K)
    print(f"  scanning {n_K} K values in [{K_min}, {K_max}] ({m_steps:,} steps each) ... ",
          end="", flush=True)
    t0 = time.perf_counter()
    psi = scan_msf(K_arr, settled, p.target, p.source,
                   params, dyn_id, p.dt, m_steps, p.qr_interval_steps)
    print(f"done ({_fmt(time.perf_counter() - t0)})")

    return K_arr, psi


def write_csv(output_path: Path, K_arr: np.ndarray, psi_arr: np.ndarray,
              dynamics: str, target: int, source: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dynamics", "target", "source", "K", "psi"])
        for k, psi in zip(K_arr.tolist(), psi_arr.tolist()):
            w.writerow([dynamics, target, source, k, psi])


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dynamics",  default="rossler",
                    choices=["rossler", "lorenz", "chen", "chua", "hr"],
                    help="Oscillator type (default: rossler)")
    ap.add_argument("--a",         type=float, default=None, help="Parameter a")
    ap.add_argument("--b",         type=float, default=None, help="Parameter b")
    ap.add_argument("--c",         type=float, default=None, help="Parameter c")
    ap.add_argument("--d",         type=float, default=0.0,  help="Extra param d (Chua a_nl)")
    ap.add_argument("--e",         type=float, default=0.0,  help="Extra param e (Chua b_nl)")
    ap.add_argument("--target",    type=int,   default=0,    help="H row index (default: 0)")
    ap.add_argument("--source",    type=int,   default=0,    help="H column index (default: 0)")
    ap.add_argument("--dt",        type=float, default=0.001)
    ap.add_argument("--transient-time",   type=float, default=None,
                    help="Transient integration time (default: 100, HR: 1000)")
    ap.add_argument("--measurement-time", type=float, default=3000.0)
    ap.add_argument("--qr-interval-steps", type=int,  default=10)
    ap.add_argument("--K-min",     type=float, default=None,
                    help="Minimum K (default: from paper config for this oscillator/coupling)")
    ap.add_argument("--K-max",     type=float, default=None,
                    help="Maximum K (default: from paper config for this oscillator/coupling)")
    ap.add_argument("--n-K",       type=int,   default=1001)
    ap.add_argument("--output",    type=Path,  default=None,
                    help="Output CSV path (default: outputs/msf_<dynamics>_s<source>t<target>.csv)")
    ap.add_argument("--workers",   type=int,   default=_N_CORES,
                    help=f"CPU threads for Numba prange (default: {_N_CORES})")
    return ap.parse_args()



# Default parameter sets matching the paper configurations
_DEFAULTS: dict[str, dict] = {
    "rossler": dict(a=0.2,  b=0.2,  c=9.0),
    "lorenz":  dict(a=10.,  b=2.,   c=28.),
    "chen":    dict(a=35.,  b=8/3,  c=28.),
    "chua":    dict(a=10.,  b=14.87, c=0., d=-1.27, e=-0.68),
    "hr":      dict(a=3.2,  b=0.006, c=4.),
}

_TRANSIENT_DEFAULTS: dict[str, float] = {
    "hr": 1000.0,
}


def main():
    args = parse_args()

    defaults = _DEFAULTS[args.dynamics]
    a = args.a if args.a is not None else defaults["a"]
    b = args.b if args.b is not None else defaults["b"]
    c = args.c if args.c is not None else defaults["c"]
    d = args.d if args.d != 0.0   else defaults.get("d", 0.0)
    e = args.e if args.e != 0.0   else defaults.get("e", 0.0)

    t_tr = args.transient_time
    if t_tr is None:
        t_tr = _TRANSIENT_DEFAULTS.get(args.dynamics, 100.0)

    k_defaults = default_k_range(args.dynamics, args.source, args.target)
    K_min = args.K_min if args.K_min is not None else k_defaults[0]
    K_max = args.K_max if args.K_max is not None else k_defaults[1]

    p = MSFParams(
        dynamics=args.dynamics,
        a=a, b=b, c=c, d=d, e=e,
        target=args.target,
        source=args.source,
        dt=args.dt,
        transient_time=t_tr,
        measurement_time=args.measurement_time,
        qr_interval_steps=args.qr_interval_steps,
    )

    print(f"MSF scan  |  {args.dynamics}  target={args.target}  source={args.source}  "
          f"K=[{K_min}, {K_max}]  n_K={args.n_K}  {_N_CORES} threads")

    print("Compiling Numba kernels (~30 s on first run, cached after) ...", flush=True)
    t0 = time.perf_counter()
    warmup()
    print(f"Compiled in {_fmt(time.perf_counter() - t0)}\n")

    output = args.output or Path(
        f"outputs/msf_{args.dynamics}_s{args.source}t{args.target}.csv"
    )

    K_arr, psi = run_scan(p, K_min, K_max, args.n_K)

    write_csv(output, K_arr, psi, args.dynamics, args.target, args.source)
    print(f"\nWrote {args.n_K} rows -> {output}")
    print(f"  psi range: [{psi.min():.4f}, {psi.max():.4f}]")
    n_stable = int(np.sum(psi < 0))
    print(f"  stable (psi < 0): {n_stable}/{args.n_K} points "
          f"({100*n_stable/args.n_K:.1f}%)")


if __name__ == "__main__":
    main()
