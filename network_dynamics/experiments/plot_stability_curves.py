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
import re
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


def _auto_msf_plot_path(dynamics: str, source: int, target: int) -> Path:
    return Path("outputs") / dynamics / "plots" / f"msf_{dynamics}_s{source}t{target}.png"


def _source_target_from_path(path: str | Path) -> tuple[int, int] | None:
    match = re.search(r"_s(?P<source>\d+)_?t(?P<target>\d+)$", Path(path).stem)
    if match is None:
        return None

    return int(match.group("source")), int(match.group("target"))


def _auto_basin_plot_path(
    dynamics: str,
    source: int | None = None,
    target: int | None = None,
) -> Path:
    if source is None or target is None:
        filename = f"basin_stability_{dynamics}.png"
    else:
        filename = f"basin_stability_{dynamics}_s{source}_t{target}.png"

    return Path("outputs") / dynamics / "stability" / "plots" / filename


def _basin_source_target_from_metadata(
    fieldnames: Iterable[str],
    row: dict[str, str],
    csv_path: str | Path,
) -> tuple[int, int] | None:
    fieldnames = set(fieldnames)
    if {"source", "target"} <= fieldnames:
        return int(row["source"]), int(row["target"])
    if {"msf_source", "msf_target"} <= fieldnames:
        return int(row["msf_source"]), int(row["msf_target"])

    return _source_target_from_path(csv_path)


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
    ``zeros`` is provided, each crossing is marked with a subtle vertical line.
    The returned ``(figure, axes)`` can be further customized by callers.
    """
    K, psi = _curve_arrays(K_values, psi_values, "psi_values")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(K, psi, color="#2563eb", linewidth=2.0, label=r"$\Psi(K)$")
    ax.axhline(0.0, color="#111827", linewidth=1.0, linestyle="--", alpha=0.8)
    finite = np.isfinite(psi)
    ax.fill_between(
        K,
        psi,
        0.0,
        where=finite & (psi < 0.0),
        color="#16a34a",
        alpha=0.14,
        interpolate=True,
        label=r"stable ($\Psi<0$)",
    )

    if zeros:
        for i, z in enumerate(zeros):
            ax.axvline(
                z,
                color="#dc2626",
                linewidth=1.0,
                linestyle=":",
                alpha=0.75,
                label=r"$K^*$" if i == 0 else None,
            )

    ax.set(title=title, xlabel=r"$K$", ylabel=r"$\Psi(K)$")
    ax.grid(True, color="#d1d5db", linewidth=0.8, alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
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
    """Plot basin stability against coupling strength and optionally save the figure.

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

    order = np.argsort(raw_K)

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
        trials = trials[order]
        yerr = np.sqrt(basin * (1.0 - basin) / trials)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        K,
        basin,
        yerr=yerr,
        color="#7c3aed",
        marker="o",
        markersize=4,
        linewidth=2.0,
        capsize=3 if yerr is not None else 0,
        label="Basin stability",
    )

    ax.set(
        title=title,
        xlabel=r"$\sigma$ (coupling strength)",
        ylabel="Basin stability",
        ylim=(-0.02, 1.02),
    )
    ax.grid(True, color="#d1d5db", linewidth=0.8, alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_figure(fig, output_path, dpi)
    return fig, ax

def plot_msf_csv(csv_path, output_path, *, title=None, dpi=200):
    """Create an MSF plot from an ``msf_scan`` CSV file.

    When the CSV contains ``dynamics``, ``target``, and ``source`` columns the
    title is built automatically and zero crossings are detected and marked.
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
            output_path = _auto_msf_plot_path(dyn, source, target)

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
    title=None,
    dpi=200,
):
    """Create a basin-stability plot from a coupling-basin scan CSV file."""
    with Path(csv_path).open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))

    if not rows:
        raise ValueError(f"CSV contains no data rows: {csv_path}")

    fieldnames = list(rows[0].keys())
    K_column = "K" if "K" in fieldnames else "coupling_strength"
    for col in (K_column, "basin_stability"):
        if col not in fieldnames:
            raise ValueError(f"CSV is missing required column: {col}")

    K       = np.asarray([float(r[K_column])        for r in rows])
    basin   = np.asarray([float(r["basin_stability"]) for r in rows])
    n_trials = (
        np.asarray([float(r["n_trials"]) for r in rows])
        if "n_trials" in fieldnames else None
    )

    auto_title = "Basin Stability vs Coupling"
    if "dynamics" in fieldnames:
        dyn = rows[0]["dynamics"]
        source_target = _basin_source_target_from_metadata(
            fieldnames=fieldnames,
            row=rows[0],
            csv_path=csv_path,
        )

        if source_target is None:
            auto_title = f"{dyn.capitalize()} Basin Stability"
        else:
            source, target = source_target
            auto_title = (
                f"{dyn.capitalize()} Basin Stability "
                f"({_coupling_label(source, target)} coupling)"
            )

        if output_path is None:
            source_target = source_target or (None, None)
            output_path = _auto_basin_plot_path(dyn, *source_target)

    return plot_basin_stability_vs_k(
        K, basin, output_path,
        n_trials=n_trials,
        title=title if title is not None else auto_title,
        dpi=dpi,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("msf", "basin"))
    parser.add_argument("csv", type=Path, help="Input scan CSV.")
    parser.add_argument("--output", type=Path, default=None,
                        help=(
                            "Output image path. Defaults: MSF plots go to "
                            "outputs/<dynamics>/plots/; basin stability plots "
                            "go to outputs/<dynamics>/stability/plots/."
                        ))
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
                out = _auto_msf_plot_path(
                    first["dynamics"], int(first["source"]), int(first["target"])
                )
        fig, _ = plot_msf_csv(args.csv, out, dpi=args.dpi, **kwargs)
    else:
        if out is None:
            with args.csv.open(newline="", encoding="utf-8") as f:
                first = next(csv.DictReader(f), {})
            if "dynamics" in first:
                fieldnames = first.keys()
                source_target = _basin_source_target_from_metadata(
                    fieldnames=fieldnames,
                    row=first,
                    csv_path=args.csv,
                )
                source_target = source_target or (None, None)
                out = _auto_basin_plot_path(first["dynamics"], *source_target)
        fig, _ = plot_basin_stability_csv(args.csv, out, dpi=args.dpi, **kwargs)

    plt.close(fig)
    print("Wrote plot:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
