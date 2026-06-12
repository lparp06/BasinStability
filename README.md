# GenerateDynamics

Tools for simulating and analyzing dynamical systems on networks.

The current research code lives primarily in the `network_dynamics/` package. It focuses on
basin-stability experiments for coupled Roessler oscillator networks, with CPU and JAX/GPU
implementations for comparing serial, parallel, and accelerator performance.

The older top-level files `GenerateDynamics.py` and `GenerateDynamicsTest.py` are legacy code and
are not the main interface for the current basin-stability workflow.

## What This Code Does

The main model is a network of identical Roessler oscillators. Each node has state
`(x, y, z)`, and the full state vector is stored as:

```text
[x0, y0, z0, x1, y1, z1, ...]
```

Network coupling is built from the graph Laplacian:

```text
coupling_matrix = coupling_strength * kron(L, H)
```

where:

- `L` is the graph Laplacian.
- `H` is the inner coupling matrix, defaulting to coupling through the `x` variable.
- `kron` is the Kronecker product.

Basin stability is estimated by:

1. Sampling many initial conditions.
2. Integrating each trajectory.
3. Measuring whether oscillators synchronize.
4. Reporting the fraction of successful trials.

Synchronization is measured using the maximum pairwise distance between node states.

## Package Layout

```text
network_dynamics/
  core/
    config.py          BasinConfig experiment settings
    coupling.py        Coupling matrix construction
    graphs.py          Graph Laplacian helpers
    oscillators.py     CPU Roessler right-hand side
    sampling.py        Initial-condition sampling
    sync.py            Synchronization metrics
    diagnostics.py     Numerical health checks
    results.py         TrialResult and BasinSummary containers
    basin_common.py    Shared basin classification helpers

  cpu/
    integration.py     CPU LSODA and RK4 integration
    basin.py           Serial and multiprocessing CPU basin runs

  gpu/
    dynamics.py        Batched JAX Roessler/RK4 kernels
    metrics.py         On-device basin metrics
    integration.py     Full-trajectory JAX RK4 integration
    basin.py           Chunked GPU validation backend
    basin_fast.py      Fast GPU/JAX backend for cluster-scale runs

  experiments/
    validate_fast_gpu_against_cpu.py  CPU/GPU validation using identical initial conditions
    hpc_benchmark.py                  CSV benchmark table generator
    plot_benchmark_csv.py             Local plotting script for benchmark CSVs
    run_fast_gpu_basin.py             Standalone fast GPU basin run
    benchmark_basin_cpu.py            CPU benchmark script
```

## Installation

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install requirements:

```bash
python -m pip install -r requirements.txt
```

The checked-in `requirements.txt` reflects the local Apple/JAX-MPS environment. On clusters, you
may need a smaller or cluster-specific environment, for example:

```bash
python -m pip install numpy scipy networkx matplotlib
```

For GPU/JAX runs, install the JAX build appropriate for the machine. On NVIDIA clusters this often
requires a CUDA-compatible JAX installation; on Apple Silicon this project has used `jax-mps`.

## Basic CPU Basin Run

```python
import networkx as nx

from network_dynamics.core.config import BasinConfig
from network_dynamics.cpu.basin import basin_stability_serial, print_basin_summary

config = BasinConfig(
    G=nx.path_graph(5),
    n_trials=25,
    tmax=150.0,
    dt=0.05,
    integrator="RK4",
    success_definition="window_success",
).validate()

summary = basin_stability_serial(config)
print_basin_summary(summary)
```

## Validation: CPU vs Fast GPU

The validation script samples one batch of initial conditions and reuses it for CPU and GPU runs.
This makes CPU/GPU comparisons meaningful because each backend sees the same trials.

```bash
python -m network_dynamics.experiments.validate_fast_gpu_against_cpu
```

Detailed output is written to:

```text
output.txt
```

Terminal progress is printed for CPU phases so long jobs do not look frozen.

The validation script currently compares:

- Serial CPU
- Parallel CPU
- Fast GPU/JAX

## HPC Benchmark Tables

Use `hpc_benchmark.py` when you want timing data in a CSV file, especially on a cluster.

Small smoke test:

```bash
python -m network_dynamics.experiments.hpc_benchmark \
  --trial-counts 2 \
  --workers 1 \
  --repeats 1 \
  --tmax 0.15 \
  --dt 0.05 \
  --n-nodes 3 \
  --output timing_outputs/smoke_test.csv
```

Full benchmark example:

```bash
python -m network_dynamics.experiments.hpc_benchmark \
  --trial-counts 50 400 1000 \
  --workers 16 \
  --repeats 1 \
  --tmax 150.0 \
  --dt 0.05 \
  --n-nodes 5 \
  --output timing_outputs/hpc_benchmark_results.csv
```

If JAX/GPU is not available, skip GPU:

```bash
python -m network_dynamics.experiments.hpc_benchmark \
  --trial-counts 100 1000 5000 \
  --workers 16 \
  --skip-gpu \
  --output timing_outputs/cpu_benchmark_results.csv
```

The benchmark CSV includes:

- backend name
- trial count
- runtime in seconds
- trials per second
- speedup vs serial CPU
- speedup vs parallel CPU
- basin stability summary values
- JAX backend information

## Plot Benchmark CSVs Locally

After copying CSV files back from a cluster, create local plots with:

```bash
python -m network_dynamics.experiments.plot_benchmark_csv \
  timing_outputs/*.csv \
  --output-dir timing_outputs/plots
```

This writes:

```text
timing_outputs/plots/benchmark_summary.csv
timing_outputs/plots/runtime_vs_trials.png
timing_outputs/plots/throughput_vs_trials.png
timing_outputs/plots/speedup_vs_serial.png
timing_outputs/plots/speedup_vs_parallel_cpu.png
```

## Running On A SLURM Cluster

A typical CPU benchmark job script looks like:

```bash
#!/bin/bash
#SBATCH --job-name=basin_cpu
#SBATCH --partition=general
#SBATCH --time=00-04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=basin_cpu-%j.out
#SBATCH --error=basin_cpu-%j.err

cd ~/codes/GenerateDynamics

module purge
module load Python/3.10.4-GCCcore-11.3.0

source .venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

python -m network_dynamics.experiments.hpc_benchmark \
  --trial-counts 100 1000 5000 \
  --workers $SLURM_CPUS_PER_TASK \
  --repeats 1 \
  --skip-gpu \
  --output timing_outputs/cpu_benchmark_${SLURM_JOB_ID}.csv
```

For GPU jobs, request a GPU partition/resource and make sure JAX reports a GPU backend inside the
job before trusting GPU benchmark results.

Example GPU SLURM resource lines:

```bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
```

Check JAX inside the job:

```bash
python -c "import jax; print(jax.default_backend()); print(jax.devices())"
```

The benchmark CSV records the detected JAX backend. If it says `cpu`, the fast JAX path did not use
the GPU.

## Notes On Numerical Behavior

Some random initial conditions can make fixed-step RK4 trajectories grow very large before the
diagnostics classify the trial as an integration failure. In long runs, NumPy may print overflow
warnings for those unstable CPU trajectories. This usually means the sampled trajectory numerically
blew up; it does not necessarily mean the script crashed.

## Useful Entry Points

```bash
# Validate CPU/GPU agreement
python -m network_dynamics.experiments.validate_fast_gpu_against_cpu

# Generate benchmark CSVs
python -m network_dynamics.experiments.hpc_benchmark --help

# Plot benchmark CSVs locally
python -m network_dynamics.experiments.plot_benchmark_csv --help

# Run basic module sanity checks
python -m network_dynamics.experiments.test_sampling
python -m network_dynamics.experiments.test_coupling
python -m network_dynamics.experiments.test_graphs
python -m network_dynamics.experiments.test_oscillators
python -m network_dynamics.experiments.test_integration
```
