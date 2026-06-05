"""
test_basin_timing.py

Compare serial vs CPU-parallel runtime for basin stability.

This script:
- runs serial and CPU-parallel basin stability for several trial counts
- compares different numbers of CPU workers
- prints timing results to the terminal
- saves all printed output to timing_outputs/timing_output.txt
- saves plots to timing_outputs/


"""

import time
from pathlib import Path

import networkx as nx
import matplotlib.pyplot as plt

from network_dynamics.basin import basin_stability_serial, basin_stability_cpu


OUTPUT_DIR = Path("timing_outputs")
LOG_FILE = OUTPUT_DIR / "timing_output.txt"


def log_print(message="", file_handle=None):
    """
    Print a message to the terminal and also write it to the log file.
    """
    print(message)

    if file_handle is not None:
        file_handle.write(str(message) + "\n")


def time_run(label, function, file_handle=None, **kwargs):
    """
    Time one basin-stability function call.
    """
    start = time.perf_counter()
    summary = function(**kwargs)
    elapsed = time.perf_counter() - start

    log_print("", file_handle)
    log_print(label, file_handle)
    log_print("-" * len(label), file_handle)
    log_print(f"Elapsed seconds: {elapsed}", file_handle)
    log_print(f"Basin stability: {summary['basin_stability']}", file_handle)
    log_print(f"Successes: {summary['successes']}", file_handle)
    log_print(f"Sync failures: {summary['sync_failures']}", file_handle)
    log_print(f"Integration failures: {summary['integration_failures']}", file_handle)

    return summary, elapsed


def save_runtime_vs_trials_plot(results):
    """
    Plot runtime versus number of trials for serial and CPU workers.
    """
    plt.figure()

    serial_points = [
        result for result in results
        if result["backend"] == "serial"
    ]

    serial_points = sorted(serial_points, key=lambda x: x["n_trials"])

    plt.plot(
        [r["n_trials"] for r in serial_points],
        [r["elapsed"] for r in serial_points],
        marker="o",
        label="serial",
    )

    worker_counts = sorted({
        result["n_workers"]
        for result in results
        if result["backend"] == "cpu"
    })

    for n_workers in worker_counts:
        cpu_points = [
            result for result in results
            if result["backend"] == "cpu"
            and result["n_workers"] == n_workers
        ]

        cpu_points = sorted(cpu_points, key=lambda x: x["n_trials"])

        plt.plot(
            [r["n_trials"] for r in cpu_points],
            [r["elapsed"] for r in cpu_points],
            marker="o",
            label=f"cpu, {n_workers} workers",
        )

    plt.xlabel("Number of trials")
    plt.ylabel("Elapsed time (seconds)")
    plt.title("Runtime vs number of basin-stability trials")
    plt.legend()
    plt.grid(True)

    path = OUTPUT_DIR / "runtime_vs_trials.png"
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.close()

    return path


def save_speedup_vs_trials_plot(results):
    """
    Plot CPU speedup versus number of trials.
    """
    plt.figure()

    worker_counts = sorted({
        result["n_workers"]
        for result in results
        if result["backend"] == "cpu"
    })

    for n_workers in worker_counts:
        cpu_points = [
            result for result in results
            if result["backend"] == "cpu"
            and result["n_workers"] == n_workers
        ]

        cpu_points = sorted(cpu_points, key=lambda x: x["n_trials"])

        plt.plot(
            [r["n_trials"] for r in cpu_points],
            [r["speedup"] for r in cpu_points],
            marker="o",
            label=f"{n_workers} workers",
        )

    plt.axhline(1.0, linestyle="--", label="no speedup")
    plt.xlabel("Number of trials")
    plt.ylabel("Speedup over serial")
    plt.title("CPU speedup vs number of basin-stability trials")
    plt.legend()
    plt.grid(True)

    path = OUTPUT_DIR / "speedup_vs_trials.png"
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.close()

    return path


def save_runtime_vs_workers_plot(results):
    """
    Plot runtime versus number of CPU workers for each trial count.
    """
    plt.figure()

    trial_counts = sorted({
        result["n_trials"]
        for result in results
        if result["backend"] == "cpu"
    })

    for n_trials in trial_counts:
        cpu_points = [
            result for result in results
            if result["backend"] == "cpu"
            and result["n_trials"] == n_trials
        ]

        cpu_points = sorted(cpu_points, key=lambda x: x["n_workers"])

        plt.plot(
            [r["n_workers"] for r in cpu_points],
            [r["elapsed"] for r in cpu_points],
            marker="o",
            label=f"{n_trials} trials",
        )

    plt.xlabel("Number of workers")
    plt.ylabel("Elapsed time (seconds)")
    plt.title("Runtime vs CPU worker count")
    plt.legend()
    plt.grid(True)

    path = OUTPUT_DIR / "runtime_vs_workers.png"
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.close()

    return path


def save_efficiency_vs_workers_plot(results):
    """
    Plot parallel efficiency.

    efficiency = speedup / n_workers

    Ideal efficiency is 1.0. Real efficiency is lower because
    multiprocessing has overhead.
    """
    plt.figure()

    trial_counts = sorted({
        result["n_trials"]
        for result in results
        if result["backend"] == "cpu"
    })

    for n_trials in trial_counts:
        cpu_points = [
            result for result in results
            if result["backend"] == "cpu"
            and result["n_trials"] == n_trials
        ]

        cpu_points = sorted(cpu_points, key=lambda x: x["n_workers"])

        plt.plot(
            [r["n_workers"] for r in cpu_points],
            [r["efficiency"] for r in cpu_points],
            marker="o",
            label=f"{n_trials} trials",
        )

    plt.axhline(1.0, linestyle="--", label="ideal")
    plt.xlabel("Number of workers")
    plt.ylabel("Parallel efficiency")
    plt.title("CPU parallel efficiency")
    plt.legend()
    plt.grid(True)

    path = OUTPUT_DIR / "efficiency_vs_workers.png"
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.close()

    return path


def print_results_table(results, file_handle=None):
    """
    Print a compact timing summary table.
    """
    log_print("", file_handle)
    log_print("=" * 70, file_handle)
    log_print("Timing summary table", file_handle)
    log_print("=" * 70, file_handle)

    header = (
        f"{'backend':<10}"
        f"{'trials':<10}"
        f"{'workers':<10}"
        f"{'time(s)':<12}"
        f"{'speedup':<12}"
        f"{'efficiency':<12}"
        f"{'matches serial':<15}"
    )

    log_print(header, file_handle)

    for result in results:
        line = (
            f"{result['backend']:<10}"
            f"{result['n_trials']:<10}"
            f"{str(result['n_workers']):<10}"
            f"{result['elapsed']:<12.4f}"
            f"{result['speedup']:<12.4f}"
            f"{result['efficiency']:<12.4f}"
            f"{str(result['matches_serial']):<15}"
        )

        log_print(line, file_handle)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    with open(LOG_FILE, "w", encoding="utf-8") as file_handle:
        G = nx.path_graph(5)

        base_settings = {
            "G": G,
            "base_seed": 42,
            "parameters": [0.2, 0.2, 7],
            "coupling_strength": 1.0,
            "H": None,
            "tmax": 150,
            "tstep": 0.05,
            "dimension": 3,
            "sync_tol": 1e-2,
            "tol_max": 1e6,
            "window_fraction": 0.1,
            "sampler": "uniform",
            "sampling_bounds": [-2, 2],
            "store_initial_conditions": False,
            "max_abs_threshold": 1e6,
        }

        # Use smaller values for quick debugging.
        # Use larger values for more meaningful timing data.
        trial_counts = [10, 25, 50, 100, 250]

        # Choose worker counts that make sense for your machine.
        # If your laptop has fewer available cores, remove 6.
        worker_counts = [2, 4, 6]

        all_results = []

        log_print("=" * 70, file_handle)
        log_print("BASIN STABILITY TIMING TEST", file_handle)
        log_print("=" * 70, file_handle)

        for n_trials in trial_counts:
            log_print("", file_handle)
            log_print("=" * 70, file_handle)
            log_print(f"n_trials = {n_trials}", file_handle)
            log_print("=" * 70, file_handle)

            settings = {
                **base_settings,
                "n_trials": n_trials,
            }

            serial_summary, serial_time = time_run(
                "Serial",
                basin_stability_serial,
                file_handle=file_handle,
                **settings,
            )

            all_results.append({
                "backend": "serial",
                "n_trials": n_trials,
                "n_workers": 1,
                "elapsed": serial_time,
                "speedup": 1.0,
                "efficiency": 1.0,
                "matches_serial": True,
                "basin_stability": serial_summary["basin_stability"],
                "successes": serial_summary["successes"],
                "sync_failures": serial_summary["sync_failures"],
                "integration_failures": serial_summary["integration_failures"],
            })

            for n_workers in worker_counts:
                cpu_summary, cpu_time = time_run(
                    f"CPU parallel, n_workers={n_workers}",
                    basin_stability_cpu,
                    file_handle=file_handle,
                    **settings,
                    n_workers=n_workers,
                )

                speedup = serial_time / cpu_time
                efficiency = speedup / n_workers

                same_result = (
                    serial_summary["basin_stability"] == cpu_summary["basin_stability"]
                    and serial_summary["successes"] == cpu_summary["successes"]
                    and serial_summary["sync_failures"] == cpu_summary["sync_failures"]
                    and serial_summary["integration_failures"] == cpu_summary["integration_failures"]
                )

                log_print(f"Speedup: {speedup}", file_handle)
                log_print(f"Efficiency: {efficiency}", file_handle)
                log_print(f"Matches serial: {same_result}", file_handle)

                all_results.append({
                    "backend": "cpu",
                    "n_trials": n_trials,
                    "n_workers": n_workers,
                    "elapsed": cpu_time,
                    "speedup": speedup,
                    "efficiency": efficiency,
                    "matches_serial": same_result,
                    "basin_stability": cpu_summary["basin_stability"],
                    "successes": cpu_summary["successes"],
                    "sync_failures": cpu_summary["sync_failures"],
                    "integration_failures": cpu_summary["integration_failures"],
                })

        print_results_table(all_results, file_handle=file_handle)

        runtime_plot = save_runtime_vs_trials_plot(all_results)
        speedup_plot = save_speedup_vs_trials_plot(all_results)
        workers_plot = save_runtime_vs_workers_plot(all_results)
        efficiency_plot = save_efficiency_vs_workers_plot(all_results)

        log_print("", file_handle)
        log_print("Saved plot: " + str(runtime_plot), file_handle)
        log_print("Saved plot: " + str(speedup_plot), file_handle)
        log_print("Saved plot: " + str(workers_plot), file_handle)
        log_print("Saved plot: " + str(efficiency_plot), file_handle)

        log_print("", file_handle)
        log_print("=" * 70, file_handle)
        log_print("Timing test complete.", file_handle)
        log_print("=" * 70, file_handle)
        log_print(f"Plots saved in: {OUTPUT_DIR}", file_handle)
        log_print(f"Text output saved to: {LOG_FILE}", file_handle)


if __name__ == "__main__":
    main()
