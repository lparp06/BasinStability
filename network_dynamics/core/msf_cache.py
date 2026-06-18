"""CSV cache helpers for Rössler MSF zero calculations."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


MSF_CACHE_FIELDS = (
    "created_at",
    "a",
    "b",
    "c",
    "target",
    "source",
    "dt",
    "transient_time",
    "measurement_time",
    "qr_interval_steps",
    "K_min",
    "K_max",
    "n_K",
    "refine",
    "tolerance",
    "zeros_json",
    "zero_brackets_json",
    "stable_intervals_json",
)


def _float_key(value):
    return f"{float(value):.17g}"


def _bool_key(value):
    return "1" if bool(value) else "0"


def make_msf_cache_key(
    config,
    K_min,
    K_max,
    n_K,
    refine,
    tolerance,
):
    return {
        "a": _float_key(config.a),
        "b": _float_key(config.b),
        "c": _float_key(config.c),
        "target": str(int(config.target)),
        "source": str(int(config.source)),
        "dt": _float_key(config.dt),
        "transient_time": _float_key(config.transient_time),
        "measurement_time": _float_key(config.measurement_time),
        "qr_interval_steps": str(int(config.qr_interval_steps)),
        "K_min": _float_key(K_min),
        "K_max": _float_key(K_max),
        "n_K": str(int(n_K)),
        "refine": _bool_key(refine),
        "tolerance": _float_key(tolerance),
    }


def row_matches_key(row, key):
    return all(row.get(field) == value for field, value in key.items())


def read_msf_cache(cache_path):
    path = Path(cache_path)

    if not path.exists():
        return []

    with path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def find_cached_msf_result(cache_path, key):
    rows = read_msf_cache(cache_path)

    for row in reversed(rows):
        if row_matches_key(row, key):
            return {
                "zeros": json.loads(row.get("zeros_json") or "[]"),
                "zero_brackets": json.loads(row.get("zero_brackets_json") or "[]"),
                "stable_intervals": json.loads(row.get("stable_intervals_json") or "[]"),
                "row": row,
            }

    return None


def append_msf_cache_result(
    cache_path,
    key,
    zeros,
    zero_brackets=None,
    stable_intervals=None,
):
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0

    row = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        **key,
        "zeros_json": json.dumps([float(zero) for zero in zeros]),
        "zero_brackets_json": json.dumps(zero_brackets or []),
        "stable_intervals_json": json.dumps(stable_intervals or []),
    }

    with path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=MSF_CACHE_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return path
