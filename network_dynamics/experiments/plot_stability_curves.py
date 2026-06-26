"""Create MSF and basin-stability plots against coupling parameter K.

Examples
--------
python -m network_dynamics.experiments.plot_stability_curves msf \
    outputs/msf_scan.csv --output outputs/msf_vs_k.png

python -m network_dynamics.experiments.plot_stability_curves basin \
    outputs/basin_scan.csv --output outputs/basin_stability_vs_k.png
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "matplotlib-generate-dynamics"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from network_dynamics.core.msf import find_zeros as _find_msf_zeros

_VAR_NAMES = {0: "x", 1: "y", 2: "z"}


def _coupling_label(source: int, target: int) -> str:
    s = _VAR_NAMES.get(source, str(source))
    t = _VAR_NAMES.get(target, str(target))
    return f"{s}→{t}"


def _auto_msf_title(dynamics: str, source: int, target: int) -> str:
    return f"{dynamics.capitalize()} MSF  ({_coupling_label(source, target)} coupling)"


def _auto_plot_path(dynamics: str, source: int, target: int) -> Path:
    return Path("outputs") / dynamics / "plots" / f"msf_{dynamics}_s{source}t{target}.png"


def _curve_arrays(
    K_values: Iterable[float],
    y_values: Iterable[float],
    y_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    K = np.asarray(
        K_values if isinstance(K_values, np.ndarray) else list(K_values),
        dtype=float,
    )
    y = np.asarray(
        y_values if isinstance(y_values, np.ndarray) else list(y_values),
        dtype=float,
    )

    if K.ndim != 1 or y.ndim != 1:
        raise ValueError("K_values and plotted values must be one-dimensional.")
    if K.size == 0:
        raise ValueError("At least one K value is required.")
    if K.shape != y.shape:
        raise ValueError(f"K_values and {y_name} must have the same length.")
    if not np.all(np.isfinite(K)):
        raise ValueError("K_values must all be finite.")

    order = np.argsort(K)
    return K[order], y[order]


def _save_figure(fig, output_path, dpi: int) -> Path | None:
    if output_path is None:
        return None

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def plot_msf_vs_k(
    K_values: Iterable[float],
    psi_values: Iterable[float],
    output_path: str | Path | None = None,
    *,
    title: str = "Master Stability Function",
    dpi: int = 200,
    zeros: list[float] | None = None,
):
    """Plot Psi(K) against K and optionally save the figure.

    Negative (linearly stable) portions of the MSF are lightly shaded. If
    ``zeros`` is provided, each crossing is marked with a vertical line and
    annotated with its K value. The returned ``(figure, axes)`` can be further
    customized by callers.
    """
    K, psi = _curve_arrays(K_values, psi_values, "psi_values")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(K, psi, color="tab:blue", linewidth=1.8, label=r"$\Psi(K)$")
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    finite = np.isfinite(psi)
    ax.fill_between(
        K,
        psi,
        0.0,
        where=finite & (psi < 0.0),
        color="tab:green",
        alpha=0.18,
        interpolate=True,
        label=r"stable ($\Psi<0$)",
    )

    if zeros:
        xform = ax.get_xaxis_transform()  # x=data, y=axes [0,1]
        for i, z in enumerate(zeros):
            ax.axvline(z, color="tab:red", linewidth=1.0, linestyle=":",
                       alpha=0.8, label=r"$K^*$" if i == 0 else None)
            ax.text(z, 0.97, f"$K^*={z:.3f}$",
                    transform=xform, ha="center", va="top",
                    fontsize=7, color="tab:red", rotation=90)

    ax.set(title=title, xlabel=r"$K$", ylabel=r"$\Psi(K)$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, output_path, dpi)
    return fig, ax


def plot_basin_stability_vs_k(
    K_values: Iterable[float],
    basin_stabilities: Iterable[float],
    output_path: str | Path | None = None,
    *,
    n_trials: int | Iterable[int] | None = None,
    title: str = "Basin Stability vs Coupling",
    dpi: int = 200,
):
    """Plot basin stability against K and optionally save the figure.

    If ``n_trials`` is supplied, binomial standard-error bars are included.
    It may be one positive integer shared by all points or one per K value.
    """
    raw_K = np.asarray(
        K_values if isinstance(K_values, np.ndarray) else list(K_values),
        dtype=float,
    )
    K, basin = _curve_arrays(raw_K, basin_stabilities, "basin_stabilities")
    if np.any(~np.isfinite(basin)) or np.any((basin < 0.0) | (basin > 1.0)):
        raise ValueError("basin_stabilities must be finite values between 0 and 1.")

    yerr = None
    if n_trials is not None:
        trials = np.asarray(
            n_trials
            if np.isscalar(n_trials) or isinstance(n_trials, np.ndarray)
            else list(n_trials),
            dtype=float,
        )
        if trials.ndim == 0:
            trials = np.full(basin.shape, trials)
        if trials.shape != basin.shape or np.any(~np.isfinite(trials)) or np.any(trials <= 0):
            raise ValueError("n_trials must be positive and scalar or one value per K.")
        order = np.argsort(raw_K)
        trials = trials[order]
        yerr = np.sqrt(basin * (1.0 - basin) / trials)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        K,
        basin,
        yerr=yerr,
        color="tab:purple",
        marker="o",
        markersize=4,
        linewidth=1.8,
        capsize=3 if yerr is not None else 0,
        label="Basin stability",
    )
    ax.set(title=title, xlabel=r"$K$", ylabel="Basin stability", ylim=(-0.02, 1.02))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, output_path, dpi)
    return fig, ax


def _read_numeric_columns(csv_path, required_columns):
    with Path(csv_path).open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))

    if not rows:
        raise ValueError(f"CSV contains no data rows: {csv_path}")

    missing = [name for name in required_columns if name not in rows[0]]
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

    return {
        name: np.asarray([float(row[name]) for row in rows], dtype=float)
        for name in required_columns
    }


def plot_msf_csv(csv_path, output_path, *, title=None, dpi=200):
    """Create an MSF plot from an ``msf_scan`` CSV file.

    When the CSV contains ``dynamics``, ``target``, and ``source`` columns the
    title is built automatically and zero crossings are detected and labelled.
    Pass ``title`` explicitly to override the auto-generated title.
    """
    with Path(csv_path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"CSV contains no data rows: {csv_path}")
    for col in ("K", "psi"):
        if col not in rows[0]:
            raise ValueError(f"CSV is missing required column: {col}")

    K   = np.asarray([float(r["K"])   for r in rows])
    psi = np.asarray([float(r["psi"]) for r in rows])

    auto_title = "Master Stability Function"
    if {"dynamics", "target", "source"} <= rows[0].keys():
        dyn    = rows[0]["dynamics"]
        target = int(rows[0]["target"])
        source = int(rows[0]["source"])
        auto_title = _auto_msf_title(dyn, source, target)
        if output_path is None:
            output_path = _auto_plot_path(dyn, source, target)

    zeros, _, _ = _find_msf_zeros(K, psi)

    return plot_msf_vs_k(
        K, psi, output_path,
        title=title if title is not None else auto_title,
        dpi=dpi,
        zeros=zeros,
    )


def plot_basin_stability_csv(
    csv_path,
    output_path,
    *,
    title="Basin Stability vs Coupling",
    dpi=200,
):
    """Create a basin-stability plot from a coupling-basin scan CSV file."""
    with Path(csv_path).open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = reader.fieldnames or []

    K_column = "K" if "K" in fieldnames else "coupling_strength"
    required = [K_column, "basin_stability"]
    if "n_trials" in fieldnames:
        required.append("n_trials")
    columns = _read_numeric_columns(csv_path, required)

    return plot_basin_stability_vs_k(
        columns[K_column],
        columns["basin_stability"],
        output_path,
        n_trials=columns.get("n_trials"),
        title=title,
        dpi=dpi,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("msf", "basin"))
    parser.add_argument("csv", type=Path, help="Input scan CSV.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output image path (MSF default: outputs/<dynamics>/plots/msf_<dyn>_s<src>t<tgt>.png)")
    parser.add_argument("--title", default=None)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("dpi must be positive.")

    kwargs = {"title": args.title} if args.title else {}
    out = args.output

    if args.kind == "msf":
        if out is None:
            with args.csv.open(newline="", encoding="utf-8") as f:
                first = next(csv.DictReader(f), {})
            if {"dynamics", "target", "source"} <= first.keys():
                out = _auto_plot_path(
                    first["dynamics"], int(first["source"]), int(first["target"])
                )
        fig, _ = plot_msf_csv(args.csv, out, dpi=args.dpi, **kwargs)
    else:
        fig, _ = plot_basin_stability_csv(args.csv, out, dpi=args.dpi, **kwargs)

    plt.close(fig)
    print("Wrote plot:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
