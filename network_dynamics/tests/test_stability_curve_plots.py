import csv

from network_dynamics.experiments.plot_stability_curves import (
    _auto_basin_plot_path,
    _source_target_from_path,
    plot_basin_stability_csv,
    plot_basin_stability_vs_k,
    plot_msf_csv,
    plot_msf_vs_k,
)
import matplotlib.pyplot as plt


def test_auto_basin_plot_path_uses_stability_plots_directory():
    assert _source_target_from_path("outputs/lorenz/stability/basin_lorenz_s1_t0.csv") == (
        1,
        0,
    )
    assert str(_auto_basin_plot_path("lorenz", source=1, target=0)) == (
        "outputs/lorenz/stability/plots/basin_stability_lorenz_s1_t0.png"
    )


def test_plot_functions_write_images(tmp_path):
    msf_path = tmp_path / "msf.png"
    basin_path = tmp_path / "basin.png"

    msf_fig, _ = plot_msf_vs_k([2, 0, 1], [1, 1, -1], msf_path)
    basin_fig, _ = plot_basin_stability_vs_k(
        [0, 1, 2], [0.0, 0.5, 1.0], basin_path, n_trials=100
    )

    assert msf_path.stat().st_size > 0
    assert basin_path.stat().st_size > 0
    plt.close(msf_fig)
    plt.close(basin_fig)


def test_plots_do_not_add_extrema_markers_or_labels(tmp_path):
    msf_path = tmp_path / "msf_clean.png"
    basin_path = tmp_path / "basin_clean.png"

    msf_fig, msf_ax = plot_msf_vs_k(
        [0, 1, 2, 3],
        [1.0, -1.0, -0.5, 1.0],
        msf_path,
        zeros=[0.5, 2.5],
    )
    basin_fig, basin_ax = plot_basin_stability_vs_k(
        [0, 1, 2],
        [0.2, 1.0, 0.4],
        basin_path,
        n_trials=10,
    )

    labels = [
        artist.get_label()
        for ax in (msf_ax, basin_ax)
        for artist in [*ax.lines, *ax.collections]
    ]
    assert "local min" not in labels
    assert "local max" not in labels
    assert msf_path.stat().st_size > 0
    assert basin_path.stat().st_size > 0
    plt.close(msf_fig)
    plt.close(basin_fig)


def test_csv_plot_helpers(tmp_path):
    msf_csv = tmp_path / "msf.csv"
    basin_csv = tmp_path / "basin.csv"
    msf_png = tmp_path / "msf.png"
    basin_png = tmp_path / "basin.png"

    with msf_csv.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=("K", "psi"))
        writer.writeheader()
        writer.writerows(({"K": 0, "psi": 1}, {"K": 1, "psi": -1}))

    with basin_csv.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=("K", "basin_stability", "n_trials"),
        )
        writer.writeheader()
        writer.writerows(
            (
                {"K": 0, "basin_stability": 0.2, "n_trials": 10},
                {"K": 1, "basin_stability": 0.8, "n_trials": 10},
            )
        )

    msf_fig, _ = plot_msf_csv(msf_csv, msf_png)
    basin_fig, _ = plot_basin_stability_csv(basin_csv, basin_png)

    assert msf_png.stat().st_size > 0
    assert basin_png.stat().st_size > 0
    plt.close(msf_fig)
    plt.close(basin_fig)
