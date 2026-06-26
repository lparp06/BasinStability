"""
run_msf_cache.py
================
Compute MSF zero crossings for all paper oscillators and write them to the cache.

Uses Numba JIT with parallel prange — compiles once (~30 s), then runs at near-C
speed across all available CPU cores.  All 22 configs finish in ~15–30 minutes at
measurement_time=3000; use --fast for a quick smoke-test (~2 min, t_ms=300).

Usage:
    python run_msf_cache.py                  # all 22 configs, t_ms=3000
    python run_msf_cache.py --only rossler   # one oscillator
    python run_msf_cache.py --fast           # t_ms=300, quick test
    python run_msf_cache.py --dry-run        # show table, no computation
    python run_msf_cache.py --force          # recompute even if cached
    python run_msf_cache.py --workers 4      # override core count
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np

# ── Numba thread count must be set before importing numba ────────────────────
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

from network_dynamics.core.msf import (
    DYN_IDS, PARAM_LENGTHS, INITIAL_STATES,
    run_transient, scan_msf, warmup, find_zeros,
)


# ─── timing helper ───────────────────────────────────────────────────────────

def _fmt(seconds):
    s = int(seconds)
    if s < 60:   return f"{s}s"
    if s < 3600: return f"{s//60}m{s%60:02d}s"
    return f"{s//3600}h{(s%3600)//60:02d}m"


# ─── cache I/O ───────────────────────────────────────────────────────────────

_FIELDS = (
    "created_at", "dynamics", "a", "b", "c", "target", "source",
    "dt", "transient_time", "measurement_time", "qr_interval_steps",
    "K_min", "K_max", "n_K",
    "zeros_json", "zero_brackets_json", "stable_intervals_json",
)

def _fk(v):
    return f"{float(v):.17g}"

def _cache_key(cfg):
    return {
        "dynamics":         cfg["dynamics"],
        "a":                _fk(cfg["a"]),
        "b":                _fk(cfg["b"]),
        "c":                _fk(cfg["c"]),
        "target":           str(cfg["target"]),
        "source":           str(cfg["source"]),
        "dt":               _fk(cfg["dt"]),
        "transient_time":   _fk(cfg["transient_time"]),
        "measurement_time": _fk(cfg["measurement_time"]),
        "qr_interval_steps": str(cfg["qr_interval_steps"]),
        "K_min":            _fk(cfg["K_min"]),
        "K_max":            _fk(cfg["K_max"]),
        "n_K":              str(cfg["n_K"]),
    }

def _is_cached(path, key):
    p = Path(path)
    if not p.exists():
        return False
    with p.open() as f:
        for row in csv.DictReader(f):
            if all(row.get(k) == v for k, v in key.items()):
                return True
    return False

def _write_cache(path, key, zeros, brackets, stable):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    is_new = not p.exists() or p.stat().st_size == 0
    with p.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow({
            "created_at":            datetime.now(timezone.utc).isoformat(),
            **key,
            "zeros_json":            json.dumps([float(z) for z in zeros]),
            "zero_brackets_json":    json.dumps([[float(a), float(b)] for a, b in brackets]),
            "stable_intervals_json": json.dumps([[float(a), float(b)] for a, b in stable]),
        })


# ─── configurations ──────────────────────────────────────────────────────────
# Source: Huang et al., Phys. Rev. E 80, 036204 (2009).
# Column order: dynamics, src, tgt, K_max, max_zeros, a, b, c
# Optional kwargs: d, e (Chua extra params), dt, t_tr, t_ms, qr, min_sep, nK, Km

def _cfg(dyn, src, tgt, K_max, mz, a, b, c, d=0., e=0.,
         dt=0.001, t_tr=100., t_ms=3000., qr=10, ms=1.0, nK=1001, Km=0.):
    return dict(
        dynamics=dyn, dyn_id=DYN_IDS[dyn],
        source=src, target=tgt,
        K_min=Km, K_max=K_max, n_K=nK,
        a=a, b=b, c=c, d=d, e=e,
        dt=dt,
        transient_time=t_tr,
        measurement_time=t_ms,
        qr_interval_steps=qr,
        min_sep=ms,
        max_zeros=mz,
    )

_R  = dict(a=0.2,  b=0.2,   c=9.)
_Lo = dict(a=10.,  b=2.,    c=28.)
_Ch = dict(a=35.,  b=8/3,   c=28.)
_Cu = dict(a=10.,  b=14.87, c=0.,  d=-1.27, e=-0.68)
_HR = dict(a=3.2,  b=0.006, c=4.,  t_tr=1000.)

CONFIGS = [
    # Rössler
    _cfg("rossler", 0, 0, 10.,  2, **_R),
    _cfg("rossler", 1, 1,  5.,  1, **_R),
    _cfg("rossler", 2, 0, 100., 1, **_R),
    # Lorenz
    _cfg("lorenz",  0, 0, 30.,  1, **_Lo),
    _cfg("lorenz",  0, 1, 30.,  1, **_Lo),
    _cfg("lorenz",  1, 0, 50.,  2, **_Lo),
    _cfg("lorenz",  1, 1, 20.,  1, **_Lo),
    _cfg("lorenz",  2, 2, 100., 3, **_Lo),
    # Chen
    _cfg("chen",    0, 1, 30.,  1, **_Ch),
    _cfg("chen",    1, 1, 20.,  1, **_Ch),
    _cfg("chen",    2, 2, 100., 2, **_Ch),
    # Chua
    _cfg("chua",    0, 0, 20.,  1, **_Cu),
    _cfg("chua",    0, 1,  5.,  1, **_Cu),
    _cfg("chua",    1, 0, 30.,  1, **_Cu),
    _cfg("chua",    1, 1, 10.,  1, **_Cu),
    _cfg("chua",    1, 2, 50.,  1, **_Cu),
    _cfg("chua",    2, 0, 10.,  2, **_Cu, ms=0.3),
    _cfg("chua",    2, 2, 10.,  2, **_Cu),
    # Hindmarsh–Rose
    _cfg("hr",      0, 0,  5.,  1, **_HR),
    _cfg("hr",      0, 1,  5.,  1, **_HR),
    _cfg("hr",      1, 0,  5.,  2, **_HR, ms=0.5),
    _cfg("hr",      1, 1,  3.,  1, **_HR),
]


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Populate MSF zero cache (Numba CPU)")
    ap.add_argument("--cache",    default="outputs/msf_zero_cache.csv")
    ap.add_argument("--only",     default=None, help="Run only this oscillator name")
    ap.add_argument("--force",    action="store_true", help="Recompute even if cached")
    ap.add_argument("--dry-run",  action="store_true", help="Print config table and exit")
    ap.add_argument("--fast",     action="store_true",
                    help="Use measurement_time=300 (~2 min total, for smoke testing)")
    ap.add_argument("--workers",  type=int, default=_N_CORES,
                    help=f"CPU threads for Numba prange (default: {_N_CORES})")
    args = ap.parse_args()

    configs = [c for c in CONFIGS if args.only is None or c["dynamics"] == args.only]
    if not configs:
        sys.exit(f"Unknown oscillator: {args.only!r}. "
                 f"Available: {sorted(set(c['dynamics'] for c in CONFIGS))}")

    if args.fast:
        configs = [{**c, "measurement_time": 300.} for c in configs]

    if args.dry_run:
        t_label = "300 (fast)" if args.fast else "3000"
        print(f"{'#':>2}  {'oscillator':10}  {'src→tgt':7}  {'K_max':>6}  "
              f"{'t_tr':>5}  {'t_ms':>5}")
        print("-" * 48)
        for i, c in enumerate(configs, 1):
            print(f"{i:>2}  {c['dynamics']:10}  "
                  f"{c['source']+1}→{c['target']+1}       "
                  f"{c['K_max']:>6.1f}  "
                  f"{c['transient_time']:>5.0f}  "
                  f"{c['measurement_time']:>5.0f}")
        print(f"\n{len(configs)} configs  |  t_ms={t_label}  |  cache={args.cache}")
        return

    print(f"MSF cache  |  {len(configs)} configs  |  {_N_CORES} threads  |  {args.cache}")
    if args.fast:
        print("FAST MODE: measurement_time=300 (smoke test)\n")

    # Compile Numba kernels once
    print("Compiling Numba kernels (~30 s on first run, cached after)...", flush=True)
    t0 = time.perf_counter()
    warmup()
    print(f"Compiled in {_fmt(time.perf_counter() - t0)}\n")

    # Group by oscillator so the transient is computed once per oscillator
    groups = defaultdict(list)
    for c in configs:
        key = (c["dynamics"], c["a"], c["b"], c["c"], c["d"], c["e"],
               c["dt"], c["transient_time"])
        groups[key].append(c)

    wall_start = time.perf_counter()
    n_ok = n_skip = n_err = 0

    for group in groups.values():
        rep = group[0]
        dyn_id = rep["dyn_id"]
        n_params = PARAM_LENGTHS[dyn_id]
        params = np.array([rep["a"], rep["b"], rep["c"], rep["d"], rep["e"]][:n_params])

        all_cached = not args.force and all(
            _is_cached(args.cache, _cache_key(c)) for c in group
        )
        print(f"{'═'*56}\n{rep['dynamics'].upper()}  "
              f"elapsed={_fmt(time.perf_counter() - wall_start)}")

        if all_cached:
            print("  all cached — skipping")
            n_skip += len(group)
            continue

        # Settle onto the attractor (shared across all coupling schemes)
        s0 = INITIAL_STATES[dyn_id].copy()
        tr_steps = int(round(rep["transient_time"] / rep["dt"]))
        t1 = time.perf_counter()
        print(f"  transient: {tr_steps:,} steps ... ", end="", flush=True)
        settled = run_transient(s0, params, dyn_id, rep["dt"], tr_steps)
        print(f"done ({_fmt(time.perf_counter() - t1)})  "
              f"state={np.round(settled, 3)}")

        if not np.isfinite(settled).all() or np.abs(settled).max() > 1e6:
            print("  ERROR: transient diverged — skipping group")
            n_err += len(group)
            continue

        for c in group:
            key = _cache_key(c)
            if not args.force and _is_cached(args.cache, key):
                print(f"  {c['source']+1}→{c['target']+1}  cached — skip")
                n_skip += 1
                continue

            K_arr   = np.linspace(c["K_min"], c["K_max"], c["n_K"])
            msteps  = int(round(c["measurement_time"] / c["dt"]))
            scheme  = f"{c['source']+1}→{c['target']+1}"
            print(f"  {scheme}  K=[{c['K_min']}, {c['K_max']}]  "
                  f"{msteps:,} steps × {c['n_K']} K values ... ",
                  end="", flush=True)

            t1  = time.perf_counter()
            psi = scan_msf(
                K_arr, settled,
                c["target"], c["source"],
                params, dyn_id,
                c["dt"], msteps, c["qr_interval_steps"],
            )
            elapsed = time.perf_counter() - t1

            zeros, brackets, stable = find_zeros(
                K_arr, psi, c["min_sep"], c["max_zeros"]
            )
            _write_cache(args.cache, key, zeros, brackets, stable)

            print(f"done ({_fmt(elapsed)})  zeros={[round(z, 4) for z in zeros]}")
            n_ok += 1

    total = time.perf_counter() - wall_start
    print(f"\n{'═'*56}")
    print(f"Done in {_fmt(total)}.  OK={n_ok}  SKIPPED={n_skip}  ERRORS={n_err}")


if __name__ == "__main__":
    main()
