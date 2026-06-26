"""
run_msf_all.py
==============
Compute Psi(K) for all 22 paper oscillator/coupling configs (or a subset),
write per-config K/psi CSVs for use with plot_stability_curves.py, and
update the MSF zero cache.

Usage
-----
    python run_msf_all.py                           # all 22 configs
    python run_msf_all.py --only rossler            # one oscillator
    python run_msf_all.py --only rossler,lorenz     # multiple oscillators
    python run_msf_all.py --fast                    # measurement_time=300 smoke-test
    python run_msf_all.py --dry-run                 # print config table and exit
    python run_msf_all.py --force                   # recompute even if cached
    python run_msf_all.py --workers 4               # override CPU thread count
    python run_msf_all.py --csv-dir outputs/scans   # custom CSV output directory
    python run_msf_all.py --cache outputs/my.csv    # custom zero-cache path
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


# ── Thread count must be set before importing numba ───────────────────────────

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
    DYN_IDS, PARAM_LENGTHS, INITIAL_STATES,
    run_transient, scan_msf, warmup, find_zeros,
)


# ── Timing helper ─────────────────────────────────────────────────────────────

def _fmt(seconds):
    s = int(seconds)
    if s < 60:   return f"{s}s"
    if s < 3600: return f"{s//60}m{s%60:02d}s"
    return f"{s//3600}h{(s%3600)//60:02d}m"


# ── Zero-cache I/O (matches run_msf_cache.py format) ─────────────────────────

_CACHE_FIELDS = (
    "created_at", "dynamics", "a", "b", "c", "target", "source",
    "dt", "transient_time", "measurement_time", "qr_interval_steps",
    "K_min", "K_max", "n_K",
    "zeros_json", "zero_brackets_json", "stable_intervals_json",
)


def _fk(v):
    return f"{float(v):.17g}"


def _cache_key(cfg):
    return {
        "dynamics":          cfg["dynamics"],
        "a":                 _fk(cfg["a"]),
        "b":                 _fk(cfg["b"]),
        "c":                 _fk(cfg["c"]),
        "target":            str(cfg["target"]),
        "source":            str(cfg["source"]),
        "dt":                _fk(cfg["dt"]),
        "transient_time":    _fk(cfg["transient_time"]),
        "measurement_time":  _fk(cfg["measurement_time"]),
        "qr_interval_steps": str(cfg["qr_interval_steps"]),
        "K_min":             _fk(cfg["K_min"]),
        "K_max":             _fk(cfg["K_max"]),
        "n_K":               str(cfg["n_K"]),
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
        w = csv.DictWriter(f, fieldnames=_CACHE_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow({
            "created_at":            datetime.now(timezone.utc).isoformat(),
            **key,
            "zeros_json":            json.dumps([float(z) for z in zeros]),
            "zero_brackets_json":    json.dumps([[float(a), float(b)] for a, b in brackets]),
            "stable_intervals_json": json.dumps([[float(a), float(b)] for a, b in stable]),
        })


# ── Scan CSV I/O ──────────────────────────────────────────────────────────────

def _write_scan_csv(path, dynamics, target, source, K_arr, psi):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dynamics", "target", "source", "K", "psi"])
        for k, p in zip(K_arr.tolist(), psi.tolist()):
            w.writerow([dynamics, target, source, k, p])


def _scan_csv_path(csv_dir, cfg):
    dyn = cfg["dynamics"]
    src = cfg["source"]
    tgt = cfg["target"]
    return Path(csv_dir) / f"msf_{dyn}_s{src}t{tgt}.csv"


# ── Paper configurations (Huang et al. 2009) ─────────────────────────────────

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
    _cfg("rossler", 0, 0,  10., 2, **_R),
    _cfg("rossler", 1, 1,   5., 1, **_R),
    _cfg("rossler", 2, 0, 100., 1, **_R),
    # Lorenz
    _cfg("lorenz",  0, 0,  30., 1, **_Lo),
    _cfg("lorenz",  0, 1,  30., 1, **_Lo),
    _cfg("lorenz",  1, 0,  50., 2, **_Lo),
    _cfg("lorenz",  1, 1,  20., 1, **_Lo),
    _cfg("lorenz",  2, 2, 100., 3, **_Lo),
    # Chen
    _cfg("chen",    0, 1,  30., 1, **_Ch),
    _cfg("chen",    1, 1,  20., 1, **_Ch),
    _cfg("chen",    2, 2, 100., 2, **_Ch),
    # Chua
    _cfg("chua",    0, 0,  20., 1, **_Cu),
    _cfg("chua",    0, 1,   5., 1, **_Cu),
    _cfg("chua",    1, 0,  30., 1, **_Cu),
    _cfg("chua",    1, 1,  10., 1, **_Cu),
    _cfg("chua",    1, 2,  50., 1, **_Cu),
    _cfg("chua",    2, 0,  10., 2, **_Cu, ms=0.3),
    _cfg("chua",    2, 2,  10., 2, **_Cu),
    # Hindmarsh–Rose
    _cfg("hr",      0, 0,   5., 1, **_HR),
    _cfg("hr",      0, 1,   5., 1, **_HR),
    _cfg("hr",      1, 0,   5., 2, **_HR, ms=0.5),
    _cfg("hr",      1, 1,   3., 1, **_HR),
]


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only",    default=None,
                    help="Comma-separated oscillator names to run, e.g. rossler,lorenz")
    ap.add_argument("--cache",   default="outputs/msf_zero_cache.csv",
                    help="Zero-cache CSV path (default: outputs/msf_zero_cache.csv)")
    ap.add_argument("--csv-dir", default="outputs",
                    help="Directory for per-config K/psi CSVs (default: outputs)")
    ap.add_argument("--force",   action="store_true",
                    help="Recompute even if already cached")
    ap.add_argument("--fast",    action="store_true",
                    help="Use measurement_time=300 for a quick smoke-test")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print config table and exit without computing")
    ap.add_argument("--workers", type=int, default=_N_CORES,
                    help=f"CPU threads for Numba prange (default: {_N_CORES})")
    return ap.parse_args()


def main():
    args = parse_args()

    # ── Filter configs ────────────────────────────────────────────────────────
    if args.only:
        requested = {s.strip().lower() for s in args.only.split(",")}
        available = {c["dynamics"] for c in CONFIGS}
        unknown   = requested - available
        if unknown:
            sys.exit(f"Unknown oscillator(s): {', '.join(sorted(unknown))}. "
                     f"Available: {', '.join(sorted(available))}")
        configs = [c for c in CONFIGS if c["dynamics"] in requested]
    else:
        configs = list(CONFIGS)

    if args.fast:
        configs = [{**c, "measurement_time": 300.} for c in configs]

    # ── Dry run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        t_label = "300 (fast)" if args.fast else "3000"
        print(f"{'#':>2}  {'oscillator':10}  {'src→tgt':7}  {'K_range':>14}  "
              f"{'t_tr':>5}  {'t_ms':>5}  {'output CSV'}")
        print("─" * 78)
        for i, c in enumerate(configs, 1):
            csv_path = _scan_csv_path(args.csv_dir, c)
            print(f"{i:>2}  {c['dynamics']:10}  "
                  f"{c['source']}→{c['target']}        "
                  f"[{c['K_min']:.0f}, {c['K_max']:.0f}]{' ':>6}  "
                  f"{c['transient_time']:>5.0f}  "
                  f"{c['measurement_time']:>5.0f}  "
                  f"{csv_path}")
        print(f"\n{len(configs)} configs  |  t_ms={t_label}  "
              f"|  cache={args.cache}  |  csv-dir={args.csv_dir}")
        return

    print(f"MSF full scan  |  {len(configs)} configs  |  {_N_CORES} threads")
    print(f"  zero cache : {args.cache}")
    print(f"  CSV output : {args.csv_dir}/")
    if args.fast:
        print("  FAST MODE  : measurement_time=300\n")
    else:
        print()

    # ── JIT warmup ────────────────────────────────────────────────────────────
    print("Compiling Numba kernels (~30 s on first run, cached after) ...",
          flush=True)
    t0 = time.perf_counter()
    warmup()
    print(f"Compiled in {_fmt(time.perf_counter() - t0)}\n")

    # ── Group by oscillator so the transient is shared ────────────────────────
    groups = defaultdict(list)
    for c in configs:
        key = (c["dynamics"], c["a"], c["b"], c["c"], c.get("d", 0.),
               c.get("e", 0.), c["dt"], c["transient_time"])
        groups[key].append(c)

    wall_start = time.perf_counter()
    n_ok = n_skip = n_err = 0

    for group in groups.values():
        rep    = group[0]
        dyn_id = rep["dyn_id"]
        n_p    = PARAM_LENGTHS[dyn_id]
        params = np.array([rep["a"], rep["b"], rep["c"],
                           rep.get("d", 0.), rep.get("e", 0.)][:n_p])

        all_cached = not args.force and all(
            _is_cached(args.cache, _cache_key(c)) and
            _scan_csv_path(args.csv_dir, c).exists()
            for c in group
        )

        print(f"{'═'*60}\n{rep['dynamics'].upper()}  "
              f"elapsed={_fmt(time.perf_counter() - wall_start)}")

        if all_cached:
            print("  all cached — skipping")
            n_skip += len(group)
            continue

        # ── Transient (shared across coupling schemes for this oscillator) ──
        s0       = INITIAL_STATES[dyn_id].copy()
        tr_steps = int(round(rep["transient_time"] / rep["dt"]))
        print(f"  transient: {tr_steps:,} steps ... ", end="", flush=True)
        t1 = time.perf_counter()
        settled = run_transient(s0, params, dyn_id, rep["dt"], tr_steps)
        print(f"done ({_fmt(time.perf_counter() - t1)})  "
              f"state={np.round(settled, 3)}")

        if not np.isfinite(settled).all() or np.abs(settled).max() > 1e6:
            print("  ERROR: transient diverged — skipping group")
            n_err += len(group)
            continue

        for c in group:
            ckey     = _cache_key(c)
            csv_path = _scan_csv_path(args.csv_dir, c)
            cached   = not args.force and _is_cached(args.cache, ckey)
            csv_done = not args.force and csv_path.exists()

            if cached and csv_done:
                print(f"  {c['source']}→{c['target']}  cached + CSV exists — skip")
                n_skip += 1
                continue

            K_arr  = np.linspace(c["K_min"], c["K_max"], c["n_K"])
            msteps = int(round(c["measurement_time"] / c["dt"]))
            scheme = f"{c['source']}→{c['target']}"
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
            print(f"done ({_fmt(elapsed)})  zeros={[round(z, 4) for z in zeros]}")

            if not csv_done:
                _write_scan_csv(csv_path, c["dynamics"], c["target"],
                                c["source"], K_arr, psi)
                print(f"    wrote {csv_path}")

            if not cached:
                _write_cache(args.cache, ckey, zeros, brackets, stable)

            n_ok += 1

    total = time.perf_counter() - wall_start
    print(f"\n{'═'*60}")
    print(f"Done in {_fmt(total)}.  OK={n_ok}  SKIPPED={n_skip}  ERRORS={n_err}")


if __name__ == "__main__":
    main()
