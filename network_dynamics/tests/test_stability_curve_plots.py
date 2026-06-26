import csv

from network_dynamics.experiments.plot_stability_curves import (
    plot_basin_stability_csv,
    plot_basin_stability_vs_k,
    plot_msf_csv,
    plot_msf_vs_k,
)
import matplotlib.pyplot as plt


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
