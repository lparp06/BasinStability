# GenerateDynamics Codebase Guide

This project simulates oscillator dynamics on graphs, estimates basin stability,
and scans master-stability-function (MSF) intervals for graph-compatible
coupling strengths.

The active research code lives in `network_dynamics/`. The older top-level
`GenerateDynamics.py` and `GenerateDynamicsTest.py` files are legacy reference
code and baseline checks; new work should generally go into `network_dynamics/`.

## Main Concepts

The basin-stability workflow is:

1. Build a graph and graph Laplacian.
2. Build the full network coupling matrix from the graph Laplacian and inner
   oscillator coupling matrix.
3. Sample one initial condition per trial.
4. Integrate each trial with a CPU or JAX/GPU backend.
5. Measure synchronization from the trajectory.
6. Convert per-trial results into a `BasinSummary`.

The state layout is always flat and node-major:

```text
[x0, y0, z0, x1, y1, z1, ...]
```

For a graph Laplacian `L` and inner coupling matrix `H`, the full coupling
matrix is:

```text
coupling_strength * kron(L, H)
```

By default, `H` couples only the first state variable.

## Directory Map

```text
network_dynamics/
  core/          Shared configuration, graph, sampling, coupling, diagnostics,
                 synchronization, results, MSF, and coupling-strength helpers.
  cpu/           SciPy/NumPy integration and serial or multiprocessing basin runs.
  gpu/           JAX integration, on-device metrics, and GPU basin runs.
  experiments/   Command-line scripts for MSF scans, coupling/basin scans,
                 local GPU checks, plotting, and older validation scripts.
  tests/         Unit tests for basin classification and coupling intervals.
```

## Core Modules

`network_dynamics.core.config`

Defines `BasinConfig`, the central settings object for basin experiments. It
stores graph, oscillator, coupling, integration, sampling, synchronization,
health, and backend settings. Call `.validate()` before running an experiment.

`network_dynamics.core.graphs`

Builds supported graph families and converts NetworkX graphs to Laplacians.
Undirected graphs use NetworkX's Laplacian; directed graphs use out-degree
minus adjacency.

`network_dynamics.core.coupling`

Builds inner and full coupling matrices. `default_x_coupling_matrix()` creates
the usual x-only coupling matrix. `build_coupling_matrix()` validates shapes and
returns the Kronecker-product network coupling.

`network_dynamics.core.oscillators`

Contains CPU right-hand sides for Rössler and Lorenz oscillators. It also owns
the user-facing dynamics aliases such as `rossler`, `roessler`, and `lorenz`.

`network_dynamics.core.sampling`

Contains deterministic samplers. `trial_seeds()` maps one base seed to one seed
per trial, and the basin helpers use those seeds for reproducible sampling.

`network_dynamics.core.sync`

Computes synchronization diagnostics from full trajectories: final distance,
maximum distance over a final window, first crossing time, and the boolean
success flags used by basin classification.

`network_dynamics.core.diagnostics`

Checks whether a trajectory contains NaN, infinity, or values above a configured
absolute threshold.

`network_dynamics.core.basin_common`

Holds backend-independent basin logic: sampling batches, validating fixed
initial-condition arrays, selecting success definitions, and converting a
trajectory into a `TrialResult`.

`network_dynamics.core.results`

Defines `TrialResult` and `BasinSummary`, the common result objects used by CPU
and GPU backends.

`network_dynamics.core.dynamics_parameters`

Stores default oscillator parameters used by command-line experiments.

`network_dynamics.core.msf`

The MSF package. `config.py` defines `MSFConfig`; `dynamics.py` registers each
oscillator's synchronized RHS and Jacobian; `integration.py` contains JAX RK4
steps; `lyapunov.py` scans Lyapunov/MSF values; `analysis.py` finds MSF zero
brackets and midpoint zeros.

`network_dynamics.core.coupling_strengths`

Converts MSF zeros plus a graph Laplacian spectrum into scalar coupling-strength
intervals that are valid for the graph.

`network_dynamics.core.msf_cache`

Reads and writes CSV cache rows for MSF zero calculations.

## CPU Backend

`network_dynamics.cpu.integration`

Builds CPU right-hand-side functions and integrates one trajectory with either
SciPy LSODA or fixed-step RK4. `integrate_from_config()` is the usual entry
point from basin code.

`network_dynamics.cpu.basin`

Runs basin-stability trials on CPU. `basin_stability_serial()` forces one
worker. `basin_stability_cpu()` uses `ProcessPoolExecutor` when `n_workers > 1`.
`basin_stability_cpu_from_initial_conditions()` is the validation path for
CPU/GPU comparisons because it consumes fixed initial conditions.

## GPU Backend

`network_dynamics.gpu.dynamics`

Contains the shared batched JAX oscillator kernels and batched RK4 step used by
the newer GPU code.

`network_dynamics.gpu.integration`

Materializes full JAX trajectories. This is useful for validation and debugging
because the result can be fed through the same CPU-side classification code.

`network_dynamics.gpu.basin`

Runs chunked GPU basin stability by sampling on CPU, integrating full trajectory
batches on GPU, copying trajectories back, and classifying them with
`core.basin_common.classify_solution()`.

`network_dynamics.gpu.metrics`

Computes compact synchronization and health metrics entirely on device while
scanning RK4 steps. This avoids storing full trajectories.

`network_dynamics.gpu.basin_fast`

The production-style fast JAX backend. It either samples all initial states with
JAX or accepts fixed initial conditions, runs `gpu.metrics`, and converts compact
metrics into a `BasinSummary`.

## Experiment Entry Points

Use module execution from the project root:

```bash
python -m network_dynamics.experiments.msf_scan
python -m network_dynamics.experiments.coupling_basin_scan
python -m network_dynamics.experiments.local_gpu
python -m network_dynamics.experiments.plot
```

`msf_scan.py` scans MSF values over `K`, reports zero brackets, optionally writes
a CSV, and stores midpoint zeros in the MSF cache.

`coupling_basin_scan.py` either uses manual coupling bounds or computes MSF
zeros, converts them into graph-compatible coupling-strength intervals, samples
one fixed initial-condition batch, and runs basin stability across strengths.

`experiment_io.py` redirects long experiment logs to project-root `output.txt`.

`individual_functionality_testing/` contains exploratory validation and benchmark
scripts. These are useful references, but they are less polished than the main
`experiments/` scripts.

## Tests

Current unit tests live in `network_dynamics/tests/`:

```bash
python -m pytest network_dynamics/tests
```

The tests cover coupling-strength interval conversion and important basin
classification edge cases. More coverage would be useful around CPU/GPU
agreement, MSF cache migration, and dimension handling.

## Cleanup Notes

The codebase is in a transition from a large legacy script to a cleaner package.
The package is already much easier to reason about, but these areas are worth
keeping tidy:

- Keep backend-independent logic in `core/`, especially sampling,
  classification, diagnostics, and result construction.
- Prefer `BasinConfig` and `MSFConfig` over passing long loose argument lists in
  new code.
- Prefer fixed initial-condition batches for CPU/GPU comparisons. Otherwise
  NumPy and JAX random streams are reproducible separately but not identical.
- Treat `GenerateDynamics.py` as legacy reference code. It contains unfinished
  methods, duplicate commented code, and print debugging, so new experiments
  should not add to it.
- Consider consolidating the single-trajectory JAX functions in
  `gpu.integration` with the newer batched kernels in `gpu.dynamics`. That is a
  good future refactor, but it should be tested carefully because it touches the
  validation path.
- The print-heavy experiment scripts are acceptable for long-running research
  jobs. Shared reusable code should return structured values instead of printing.
