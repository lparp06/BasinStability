"""Plot basin stability vs K for network-scan basin CSV files.

By default this discovers every basin CSV under Erdos-Renyi network-scan
directories and writes one PNG next to each CSV.

Examples
--------
python -m network_dynamics.experiments.plot_network_scan_basins

python -m network_dynamics.experiments.plot_network_scan_basins \\
    --root outputs/network_scan --graph-prefix erdos-renyi --output-dir outputs/network_scan/plots
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "matplotlib-generate-dynamics"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from network_dynamics.core.coupling_strengths import laplacian_nonzero_eigenvalue_bounds
from network_dynamics.core.graphs import make_graph
from network_dynamics.experiments.plot_stability_curves import (
    plot_basin_stability_vs_k,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/network_scan"),
        help="Network-scan output root.",
    )
    parser.add_argument(
        "--graph-prefix",
        default="erdos",
        help=(
            "Only scan graph directories whose names start with this prefix. "
            "The default matches both erdos-renyi_* and erdos_renyi_* names."
        ),
    )
    parser.add_argument(
        "--csv-glob",
        default="seed_*/basin_*.csv",
        help="Glob, relative to each matching graph directory, for basin CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for all PNGs. By default each plot is written "
            "next to its source CSV."
        ),
    )
    parser.add_argument(
        "--suffix",
        default="_vs_k.png",
        help="Suffix appended to each CSV stem for output PNG names.",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--no-eigen-label",
        action="store_true",
        help="Do not annotate plots with the graph Laplacian eigenvalue range.",
    )
    return parser.parse_args()


def discover_csvs(root: Path, graph_prefix: str, csv_glob: str) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Network-scan root does not exist: {root}")

    csvs = []
    for graph_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        normalized = graph_dir.name.replace("_", "-")
        if not normalized.startswith(graph_prefix.replace("_", "-")):
            continue
        csvs.extend(sorted(graph_dir.glob(csv_glob)))

    return csvs


def read_basin_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, str]:
    with csv_path.open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))

    if not rows:
        raise ValueError(f"CSV contains no data rows: {csv_path}")

    fieldnames = set(rows[0].keys())
    k_column = "K" if "K" in fieldnames else "coupling_strength"
    for column in (k_column, "basin_stability"):
        if column not in fieldnames:
            raise ValueError(f"CSV is missing required column {column!r}: {csv_path}")

    K = np.asarray([float(row[k_column]) for row in rows], dtype=float)
    basin = np.asarray([float(row["basin_stability"]) for row in rows], dtype=float)
    n_trials = None
    if "n_trials" in fieldnames:
        n_trials = np.asarray([float(row["n_trials"]) for row in rows], dtype=float)

    dynamics = rows[0].get("dynamics", "dynamics")
    return K, basin, n_trials, dynamics


def output_path_for(csv_path: Path, root: Path, output_dir: Path | None, suffix: str) -> Path:
    filename = f"{csv_path.stem}{suffix}"
    if output_dir is None:
        return csv_path.with_name(filename)

    graph_dir = csv_path.parents[1].name
    seed_dir = csv_path.parent.name
    return output_dir / graph_dir / seed_dir / filename


def _parse_seed(seed_dir_name: str) -> int | None:
    prefix = "seed_"
    if not seed_dir_name.startswith(prefix):
        return None
    try:
        return int(seed_dir_name[len(prefix):])
    except ValueError:
        return None


def eigenvalue_range_for_scan_path(csv_path: Path) -> tuple[float, float] | None:
    graph_dir = csv_path.parents[1].name.replace("-", "_")
    seed = _parse_seed(csv_path.parent.name)
    if seed is None:
        return None

    parts = graph_dir.split("_")
    if parts[:2] != ["erdos", "renyi"]:
        return None

    values = {}
    for part in parts[2:]:
        if part.startswith("n"):
            values["n_nodes"] = int(part[1:])
        elif part.startswith("p"):
            values["edge_probability"] = float(part[1:])

    if {"n_nodes", "edge_probability"} - values.keys():
        return None

    G = make_graph(
        "erdos_renyi",
        n_nodes=values["n_nodes"],
        seed=seed,
        edge_probability=values["edge_probability"],
    )
    return laplacian_nonzero_eigenvalue_bounds(G)


def add_eigenvalue_label(ax, eigenvalue_range: tuple[float, float] | None) -> None:
    if eigenvalue_range is None:
        return

    lambda_2, lambda_n = eigenvalue_range
    ax.text(
        0.03,
        0.06,
        rf"Laplacian eigenvalues: $\lambda_2={lambda_2:.4g}$, $\lambda_N={lambda_n:.4g}$",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#d1d5db",
            "alpha": 0.88,
        },
    )


def plot_csv(csv_path: Path, output_path: Path, dpi: int, show_eigen_label: bool) -> Path:
    K, basin, n_trials, dynamics = read_basin_csv(csv_path)
    graph_dir = csv_path.parents[1].name
    seed_dir = csv_path.parent.name
    title = f"{dynamics.capitalize()} Basin Stability vs K ({graph_dir}, {seed_dir})"

    fig, ax = plot_basin_stability_vs_k(
        K,
        basin,
        output_path,
        n_trials=n_trials,
        title=title,
        dpi=dpi,
    )
    ax.set_xlabel(r"$K$ (coupling strength)")
    if show_eigen_label:
        add_eigenvalue_label(ax, eigenvalue_range_for_scan_path(csv_path))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("dpi must be positive.")
    if not args.suffix.endswith(".png"):
        raise ValueError("--suffix must end with .png")

    csvs = discover_csvs(args.root, args.graph_prefix, args.csv_glob)
    if not csvs:
        raise SystemExit(
            f"No CSV files matched {args.csv_glob!r} under {args.root}/{args.graph_prefix}*"
        )

    written = []
    for csv_path in csvs:
        output_path = output_path_for(csv_path, args.root, args.output_dir, args.suffix)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        written.append(plot_csv(csv_path, output_path, args.dpi, not args.no_eigen_label))

    print(f"Wrote {len(written)} plot(s):")
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
