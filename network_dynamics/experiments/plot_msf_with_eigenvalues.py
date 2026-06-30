"""
Overlay Laplacian eigenvalues (scaled by coupling strength) on the Lorenz s1t0 MSF plot.

For coupling strength sigma and Laplacian eigenvalue lambda_i, the effective
K value is sigma * lambda_i. Plotting these on the MSF curve shows which
modes fall in the stable (Psi < 0) vs unstable (Psi > 0) region.

The bottom panel shows a KDE of the scaled eigenvalue distribution for each
sigma, making the spread and density of the eigenvalue cloud immediately clear.

Usage
-----
python -m network_dynamics.experiments.plot_msf_with_eigenvalues
"""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import gaussian_kde

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "matplotlib-generate-dynamics"),
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from network_dynamics.core.graphs import graph_laplacian, make_graph
from network_dynamics.core.msf import find_zeros as find_msf_zeros

_MSF_CSV = Path("outputs/lorenz/csv/msf_lorenz_s1t0.csv")
_OUTPUT = Path("outputs/lorenz/plots/msf_lorenz_s1t0_eigs_overlay.png")

_COUPLING_STRENGTHS = [0.68, 0.77, 0.83]

_GRAPH_TYPE = "erdos_renyi"
_N_NODES = 100
_EDGE_PROB = 0.15
_GRAPH_SEED = 42
_EIG_TOLERANCE = 1e-10


def load_msf_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    K = np.array([float(r["K"]) for r in rows])
    psi = np.array([float(r["psi"]) for r in rows])
    order = np.argsort(K)
    return K[order], psi[order]


def nonzero_laplacian_eigenvalues(G, tol: float = _EIG_TOLERANCE) -> np.ndarray:
    L = np.asarray(graph_laplacian(G), dtype=float)
    eigs = np.linalg.eigvalsh(L)
    eigs = np.sort(eigs)
    return eigs[np.abs(eigs) > tol]


_COLORS = ["#7c3aed", "#ea580c", "#0891b2"]  # purple, orange, teal (cycles if more sigmas needed)


def make_eigenvalue_plot(
    G,
    msf_csv_path: Path,
    coupling_strengths: list,
    output_path: Path,
    title: str | None = None,
) -> None:
    """
    Overlay scaled Laplacian eigenvalues on an MSF curve and write a PNG.

    Parameters
    ----------
    G:
        NetworkX graph whose Laplacian eigenvalues are plotted.
    msf_csv_path:
        Path to a CSV with columns ``K`` and ``psi``.
    coupling_strengths:
        List of sigma values to overlay (up to 3; extra values cycle through _COLORS).
    output_path:
        Destination PNG path (parent directories are created if needed).
    title:
        Optional plot title override.
    """
    K, psi = load_msf_csv(msf_csv_path)
    zeros, _, _ = find_msf_zeros(K, psi)

    eigenvalues = nonzero_laplacian_eigenvalues(G)
    print(f"Non-zero Laplacian eigenvalues: {len(eigenvalues)}")
    print(f"Eigenvalue range: [{eigenvalues[0]:.4f}, {eigenvalues[-1]:.4f}]")

    colors = [_COLORS[i % len(_COLORS)] for i in range(len(coupling_strengths))]

    fig, (ax_msf, ax_kde) = plt.subplots(
        2, 1,
        figsize=(8, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    # --- Top panel: MSF curve ---
    finite = np.isfinite(psi)
    ax_msf.plot(K, psi, color="#2563eb", linewidth=2.0, label=r"$\Psi(K)$")
    ax_msf.axhline(0.0, color="#111827", linewidth=1.0, linestyle="--", alpha=0.8)
    ax_msf.fill_between(
        K, psi, 0.0,
        where=finite & (psi < 0.0),
        color="#16a34a", alpha=0.14, interpolate=True,
        label=r"stable ($\Psi<0$)",
    )
    for i, z in enumerate(zeros):
        ax_msf.axvline(
            z, color="#dc2626", linewidth=1.0, linestyle=":", alpha=0.75,
            label=r"$K^*$" if i == 0 else None,
        )

    # --- Mean point + KDE per sigma ---
    x_eval = np.linspace(K.min(), K.max(), 500)

    for sigma, color in zip(coupling_strengths, colors):
        K_i = sigma * eigenvalues
        K_mean = float(np.mean(K_i))
        psi_mean = float(np.interp(K_mean, K, psi))

        ax_msf.scatter(
            K_mean, psi_mean,
            s=180, color=color, zorder=5, alpha=0.95,
            label=f"$\\sigma={sigma:.4g}$,  $\\bar{{K}}={K_mean:.2f}$",
        )

        kde = gaussian_kde(K_i)
        density = kde(x_eval)
        ax_kde.plot(x_eval, density, color=color, linewidth=1.5)
        ax_kde.fill_between(x_eval, density, alpha=0.2, color=color)
        ax_kde.axvline(K_mean, color=color, linewidth=1.0, linestyle="--", alpha=0.7)

    ax_msf.set(
        title=title or "MSF — mean scaled eigenvalue per σ",
        ylabel=r"$\Psi(K)$",
    )
    ax_msf.grid(True, color="#d1d5db", linewidth=0.8, alpha=0.65)
    ax_msf.spines["top"].set_visible(False)
    ax_msf.spines["right"].set_visible(False)
    ax_msf.legend(frameon=False, fontsize=9)

    ax_kde.set(
        xlabel=r"$K = \sigma \cdot \lambda_i$",
        ylabel="density",
    )
    ax_kde.grid(True, color="#d1d5db", linewidth=0.8, alpha=0.65)
    ax_kde.spines["top"].set_visible(False)
    ax_kde.spines["right"].set_visible(False)

    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {output_path}")


def main():
    G = make_graph(_GRAPH_TYPE, n_nodes=_N_NODES, seed=_GRAPH_SEED, edge_probability=_EDGE_PROB)
    make_eigenvalue_plot(
        G=G,
        msf_csv_path=_MSF_CSV,
        coupling_strengths=_COUPLING_STRENGTHS,
        output_path=_OUTPUT,
        title="Lorenz MSF (y→x coupling) — mean scaled eigenvalue per σ",
    )


if __name__ == "__main__":
    main()
