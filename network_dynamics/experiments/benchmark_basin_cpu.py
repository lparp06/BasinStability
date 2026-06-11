"""
benchmark_basin_cpu.py

Benchmarks serial vs multiprocessing CPU basin stability.

Creates four plots:
1. runtime_vs_trials.png
2. runtime_vs_workers.png
3. speedup_vs_trials.png
4. efficiency_vs_workers.png

Run from project root:

    python -m network_dynamics.experiments.benchmark_basin_cpu
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

from network_dynamics.core.config import BasinConfig
from network_dynamics.cpu.basin import basin_stability_serial, basin_stability_cpu


def make_config(n_trials, backend, n_workers=None):
    return BasinConfig(
        G=nx.path_graph(5),
        n_trials=n_trials,
        base_seed=42,
        parameters=(0.2, 0.2, 7.0),
        coupling_strength=1.0,
        H=None,
        tmax=150.0,
        dt=0.05,
        dimension=3,
        sampling_bounds=(-5.0, 5.0),
        sync_tol=1e-2,
        tol_max=1e6,
        window_fraction=0.1,
        max_abs_threshold=1e6,
        success_definition="window_success",
        backend=backend,
        n_workers=n_workers,
    ).validate()


def time_run(function, config):
    start = time.perf_counter()
    summary = function(config)
    end = time.perf_counter()

    elapsed = end - start

    return summary, elapsed


def collect_benchmark_data():
    trial_counts = [100, 1000, 2500, 5000]
    worker_counts = [2, 4, 6, 8]

    rows = []

    for n_trials in trial_counts:
        print()
        print("=" * 70)
        print(f"Benchmarking {n_trials} trials")
        print("=" * 70)

        serial_config = make_config(
            n_trials=n_trials,
            backend="serial",
            n_workers=None,
        )

        serial_summary, serial_time = time_run(
            basin_stability_serial,
            serial_config,
        )

        print(f"serial: {serial_time:.3f} seconds")

        rows.append(
            {
                "backend": "serial",
                "n_trials": n_trials,
                "workers": 1,
                "elapsed_time": serial_time,
                "speedup": 1.0,
                "efficiency": 1.0,
                "basin_stability": serial_summary.basin_stability,
                "successes": serial_summary.successes,
                "sync_failures": serial_summary.sync_failures,
                "integration_failures": serial_summary.integration_failures,
            }
        )

        for workers in worker_counts:
            cpu_config = make_config(
                n_trials=n_trials,
                backend="cpu",
                n_workers=workers,
            )

            cpu_summary, cpu_time = time_run(
                basin_stability_cpu,
                cpu_config,
            )

            speedup = serial_time / cpu_time
            efficiency = speedup / workers

            print(
                f"cpu, {workers} workers: "
                f"{cpu_time:.3f} seconds, "
                f"speedup={speedup:.3f}, "
                f"efficiency={efficiency:.3f}"
            )

            rows.append(
                {
                    "backend": "cpu",
                    "n_trials": n_trials,
                    "workers": workers,
                    "elapsed_time": cpu_time,
                    "speedup": speedup,
                    "efficiency": efficiency,
                    "basin_stability": cpu_summary.basin_stability,
                    "successes": cpu_summary.successes,
                    "sync_failures": cpu_summary.sync_failures,
                    "integration_failures": cpu_summary.integration_failures,
                }
            )

    return rows


def print_results_table(rows):
    print()
    print("Benchmark results")
    print("=" * 120)

    print(
        f"{'backend':<10} "
        f"{'workers':<8} "
        f"{'trials':<8} "
        f"{'time (s)':<12} "
        f"{'speedup':<10} "
        f"{'efficiency':<12} "
        f"{'BS':<8} "
        f"{'success':<8} "
        f"{'sync fail':<10} "
        f"{'int fail':<8}"
    )

    print("-" * 120)

    for row in rows:
        print(
            f"{row['backend']:<10} "
            f"{row['workers']:<8} "
            f"{row['n_trials']:<8} "
            f"{row['elapsed_time']:<12.3f} "
            f"{row['speedup']:<10.3f} "
            f"{row['efficiency']:<12.3f} "
            f"{row['basin_stability']:<8.3f} "
            f"{row['successes']:<8} "
            f"{row['sync_failures']:<10} "
            f"{row['integration_failures']:<8}"
        )


def get_row(rows, backend, n_trials, workers):
    for row in rows:
        if (
            row["backend"] == backend
            and row["n_trials"] == n_trials
            and row["workers"] == workers
        ):
            return row

    raise ValueError(
        f"Could not find row for backend={backend}, "
        f"n_trials={n_trials}, workers={workers}"
    )


def plot_runtime_vs_trials(rows, output_dir):
    trial_counts = sorted({row["n_trials"] for row in rows})
    worker_counts = sorted({row["workers"] for row in rows if row["backend"] == "cpu"})

    plt.figure(figsize=(8, 5))

    serial_times = [
        get_row(rows, "serial", n_trials, 1)["elapsed_time"]
        for n_trials in trial_counts
    ]

    plt.plot(
        trial_counts,
        serial_times,
        marker="o",
        label="serial",
    )

    for workers in worker_counts:
        cpu_times = [
            get_row(rows, "cpu", n_trials, workers)["elapsed_time"]
            for n_trials in trial_counts
        ]

        plt.plot(
            trial_counts,
            cpu_times,
            marker="o",
            label=f"cpu, {workers} workers",
        )

    plt.title("Runtime vs number of basin-stability trials")
    plt.xlabel("Number of trials")
    plt.ylabel("Elapsed time (seconds)")
    plt.grid(True)
    plt.legend()

    output_path = output_dir / "runtime_vs_trials.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    return output_path


def plot_runtime_vs_workers(rows, output_dir):
    trial_counts = sorted({row["n_trials"] for row in rows})
    worker_counts = sorted({row["workers"] for row in rows if row["backend"] == "cpu"})

    plt.figure(figsize=(8, 5))

    for n_trials in trial_counts:
        times = [
            get_row(rows, "cpu", n_trials, workers)["elapsed_time"]
            for workers in worker_counts
        ]

        plt.plot(
            worker_counts,
            times,
            marker="o",
            label=f"{n_trials} trials",
        )

    plt.title("Runtime vs CPU worker count")
    plt.xlabel("Number of workers")
    plt.ylabel("Elapsed time (seconds)")
    plt.grid(True)
    plt.legend()

    output_path = output_dir / "runtime_vs_workers.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    return output_path


def plot_speedup_vs_trials(rows, output_dir):
    trial_counts = sorted({row["n_trials"] for row in rows})
    worker_counts = sorted({row["workers"] for row in rows if row["backend"] == "cpu"})

    plt.figure(figsize=(8, 5))

    for workers in worker_counts:
        speedups = [
            get_row(rows, "cpu", n_trials, workers)["speedup"]
            for n_trials in trial_counts
        ]

        plt.plot(
            trial_counts,
            speedups,
            marker="o",
            label=f"{workers} workers",
        )

    plt.axhline(
        y=1.0,
        linestyle="--",
        label="no speedup",
    )

    plt.title("CPU speedup vs number of basin-stability trials")
    plt.xlabel("Number of trials")
    plt.ylabel("Speedup over serial")
    plt.grid(True)
    plt.legend()

    output_path = output_dir / "speedup_vs_trials.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    return output_path


def plot_efficiency_vs_workers(rows, output_dir):
    trial_counts = sorted({row["n_trials"] for row in rows})
    worker_counts = sorted({row["workers"] for row in rows if row["backend"] == "cpu"})

    plt.figure(figsize=(8, 5))

    for n_trials in trial_counts:
        efficiencies = [
            get_row(rows, "cpu", n_trials, workers)["efficiency"]
            for workers in worker_counts
        ]

        plt.plot(
            worker_counts,
            efficiencies,
            marker="o",
            label=f"{n_trials} trials",
        )

    plt.axhline(
        y=1.0,
        linestyle="--",
        label="ideal",
    )

    plt.title("CPU parallel efficiency")
    plt.xlabel("Number of workers")
    plt.ylabel("Parallel efficiency")
    plt.grid(True)
    plt.legend()

    output_path = output_dir / "efficiency_vs_workers.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    return output_path


def make_plots(rows):
    output_dir = Path("network_dynamics/experiments/benchmark_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        plot_runtime_vs_trials(rows, output_dir),
        plot_runtime_vs_workers(rows, output_dir),
        plot_speedup_vs_trials(rows, output_dir),
        plot_efficiency_vs_workers(rows, output_dir),
    ]

    print()
    print("Saved plots")
    print("=" * 70)

    for path in paths:
        print(path)


def main():
    rows = collect_benchmark_data()
    print_results_table(rows)
    make_plots(rows)


if __name__ == "__main__":
    main()
