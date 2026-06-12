"""
Plot benchmark CSV files created by hpc_benchmark.py.

Run locally from the project root, for example:

    python -m network_dynamics.experiments.plot_benchmark_csv \
        timing_outputs/hpc_benchmark_results.csv \
        --output-dir timing_outputs/plots
"""

import argparse
import csv
import os
import tempfile
from collections import defaultdict
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "matplotlib-generate-dynamics"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BACKEND_LABELS = {
    "serial_cpu": "Serial CPU",
    "parallel_cpu": "Parallel CPU",
    "fast_gpu": "Fast GPU",
}

BACKEND_ORDER = ("serial_cpu", "parallel_cpu", "fast_gpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create local plots from hpc_benchmark.py CSV files."
    )
    parser.add_argument(
        "csv_files",
        nargs="+",
        type=Path,
        help="One or more benchmark CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("timing_outputs/plots"),
        help="Directory for plots and summary CSV.",
    )
    return parser.parse_args()


def read_rows(csv_files):
    rows = []

    for csv_file in csv_files:
        with csv_file.open(newline="", encoding="utf-8") as input_file:
            reader = csv.DictReader(input_file)

            for row in reader:
                row["source_file"] = str(csv_file)
                row["n_trials"] = int(row["n_trials"])
                row["runtime_seconds"] = float(row["runtime_seconds"])
                row["trials_per_second"] = float(row["trials_per_second"])
                row["basin_stability"] = float(row["basin_stability"])
                rows.append(row)

    return rows


def aggregate_rows(rows):
    grouped = defaultdict(list)

    for row in rows:
        key = (row["backend"], row["n_trials"])
        grouped[key].append(row)

    summary_rows = []

    for (backend, n_trials), group_rows in sorted(grouped.items()):
        runtimes = np.asarray(
            [row["runtime_seconds"] for row in group_rows],
            dtype=float,
        )
        throughputs = np.asarray(
            [row["trials_per_second"] for row in group_rows],
            dtype=float,
        )
        basin_stabilities = np.asarray(
            [row["basin_stability"] for row in group_rows],
            dtype=float,
        )

        summary_rows.append(
            {
                "backend": backend,
                "n_trials": n_trials,
                "n_repeats": len(group_rows),
                "runtime_mean": float(np.mean(runtimes)),
                "runtime_std": float(np.std(runtimes)),
                "trials_per_second_mean": float(np.mean(throughputs)),
                "trials_per_second_std": float(np.std(throughputs)),
                "basin_stability_mean": float(np.mean(basin_stabilities)),
                "basin_stability_std": float(np.std(basin_stabilities)),
            }
        )

    add_summary_speedups(summary_rows)

    return summary_rows


def add_summary_speedups(summary_rows):
    by_trials = defaultdict(dict)

    for row in summary_rows:
        by_trials[row["n_trials"]][row["backend"]] = row

    for backend_rows in by_trials.values():
        serial_time = backend_rows.get("serial_cpu", {}).get("runtime_mean")
        parallel_time = backend_rows.get("parallel_cpu", {}).get("runtime_mean")

        for row in backend_rows.values():
            runtime = row["runtime_mean"]
            row["speedup_vs_serial"] = (
                serial_time / runtime if serial_time is not None else ""
            )
            row["speedup_vs_parallel_cpu"] = (
                parallel_time / runtime if parallel_time is not None else ""
            )


def write_summary_csv(summary_rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "benchmark_summary.csv"

    fieldnames = [
        "backend",
        "n_trials",
        "n_repeats",
        "runtime_mean",
        "runtime_std",
        "trials_per_second_mean",
        "trials_per_second_std",
        "speedup_vs_serial",
        "speedup_vs_parallel_cpu",
        "basin_stability_mean",
        "basin_stability_std",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    return output_path


def rows_for_backend(summary_rows, backend):
    rows = [row for row in summary_rows if row["backend"] == backend]
    return sorted(rows, key=lambda row: row["n_trials"])


def plot_runtime(summary_rows, output_dir):
    plt.figure(figsize=(8, 5))

    for backend in BACKEND_ORDER:
        rows = rows_for_backend(summary_rows, backend)

        if not rows:
            continue

        x = [row["n_trials"] for row in rows]
        y = [row["runtime_mean"] for row in rows]
        yerr = [row["runtime_std"] for row in rows]

        plt.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            capsize=4,
            label=BACKEND_LABELS.get(backend, backend),
        )

    plt.title("Runtime vs Trial Count")
    plt.xlabel("Number of trials")
    plt.ylabel("Runtime (seconds)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / "runtime_vs_trials.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path


def plot_throughput(summary_rows, output_dir):
    plt.figure(figsize=(8, 5))

    for backend in BACKEND_ORDER:
        rows = rows_for_backend(summary_rows, backend)

        if not rows:
            continue

        x = [row["n_trials"] for row in rows]
        y = [row["trials_per_second_mean"] for row in rows]
        yerr = [row["trials_per_second_std"] for row in rows]

        plt.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            capsize=4,
            label=BACKEND_LABELS.get(backend, backend),
        )

    plt.title("Throughput vs Trial Count")
    plt.xlabel("Number of trials")
    plt.ylabel("Trials per second")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / "throughput_vs_trials.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path


def plot_speedup(summary_rows, output_dir, speedup_key, title, filename):
    plt.figure(figsize=(8, 5))

    for backend in BACKEND_ORDER:
        rows = rows_for_backend(summary_rows, backend)
        rows = [row for row in rows if row[speedup_key] != ""]

        if not rows:
            continue

        x = [row["n_trials"] for row in rows]
        y = [float(row[speedup_key]) for row in rows]

        plt.plot(
            x,
            y,
            marker="o",
            label=BACKEND_LABELS.get(backend, backend),
        )

    plt.title(title)
    plt.xlabel("Number of trials")
    plt.ylabel("Speedup")
    plt.axhline(1.0, color="black", linewidth=1, linestyle="--")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / filename
    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path


def main():
    args = parse_args()
    rows = read_rows(args.csv_files)

    if not rows:
        raise ValueError("No benchmark rows found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = aggregate_rows(rows)

    output_paths = [
        write_summary_csv(summary_rows, args.output_dir),
        plot_runtime(summary_rows, args.output_dir),
        plot_throughput(summary_rows, args.output_dir),
        plot_speedup(
            summary_rows=summary_rows,
            output_dir=args.output_dir,
            speedup_key="speedup_vs_serial",
            title="Speedup vs Serial CPU",
            filename="speedup_vs_serial.png",
        ),
        plot_speedup(
            summary_rows=summary_rows,
            output_dir=args.output_dir,
            speedup_key="speedup_vs_parallel_cpu",
            title="Speedup vs Parallel CPU",
            filename="speedup_vs_parallel_cpu.png",
        ),
    ]

    print("Wrote:")
    for output_path in output_paths:
        print(output_path)


if __name__ == "__main__":
    main()
