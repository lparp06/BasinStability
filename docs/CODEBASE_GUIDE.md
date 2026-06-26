# GenerateDynamics Codebase Guide

A complete reference for every module and method in this project, written for note-taking and active development.

---

## What This Project Does

This project studies how networks of chaotic oscillators synchronize. The core research pipeline is:

1. **Build a graph** — nodes are oscillators, edges are coupling pathways
2. **Compute the MSF** (Master Stability Function) — a curve Ψ(K) that says for what coupling strengths K the synchronized state is stable
3. **Find MSF zero crossings** — the zeros tell you where Ψ changes sign from positive (unstable) to negative (stable)
4. **Convert zeros to graph coupling-strength intervals** — using the graph's Laplacian eigenvalues, translate abstract K-values into actual σ (sigma) coupling strengths that work for your specific graph
5. **Measure basin stability** — for many random initial conditions, integrate the full system and count how many converge to synchrony

---

## Project Layout

```
GenerateDynamics/
├── GenerateDynamics.py          ← legacy monolithic simulator (original code)
├── GenerateDynamicsTest.py      ← legacy test file
├── run_msf_cache.py             ← CLI to pre-compute and cache MSF zeros
│
├── network_dynamics/
│   ├── core/                    ← shared logic, no hardware dependency
│   │   ├── config.py            ← BasinConfig dataclass
│   │   ├── oscillators.py       ← RHS functions for each oscillator type (CPU)
│   │   ├── graphs.py            ← graph and Laplacian builders
│   │   ├── coupling.py          ← coupling matrix construction
│   │   ├── sampling.py          ← initial condition samplers
│   │   ├── sync.py              ← synchronization detection and metrics
│   │   ├── results.py           ← TrialResult and BasinSummary containers
│   │   ├── basin_common.py      ← shared basin trial logic
│   │   ├── diagnostics.py       ← numerical health checks
│   │   ├── coupling_strengths.py← MSF zeros → sigma intervals
│   │   ├── msf_cache.py         ← CSV cache for MSF zero results
│   │   ├── dynamics_parameters.py← default parameter lookup
│   │   ├── jax_config.py        ← enable JAX float64 mode
│   │   └── msf/                 ← Master Stability Function package
│   │       ├── __init__.py      ← public API re-exports
│   │       ├── dynamics.py      ← Numba RHS + Jacobian for 5 oscillators
│   │       ├── compute.py       ← Numba MSF scan (parallel CPU)
│   │       ├── params.py        ← MSFParams dataclass
│   │       └── zeros.py         ← zero-crossing finder
│   │
│   ├── cpu/                     ← CPU integration + basin stability
│   │   ├── integration.py       ← LSODA and RK4 integrators (CPU)
│   │   └── basin.py             ← serial and multiprocessing basin stability
│   │
│   ├── gpu/                     ← JAX/GPU integration + basin stability
│   │   ├── dynamics.py          ← JAX batched RHS for each oscillator
│   │   ├── integration.py       ← JAX RK4 single and batched integrators
│   │   ├── basin.py             ← GPU basin stability (chunked)
│   │   ├── basin_fast.py        ← faster GPU variant
│   │   └── metrics.py           ← (GPU-side metric helpers)
│   │
│   ├── experiments/             ← runnable experiments
│   │   ├── coupling_basin_scan.py  ← main CLI: MSF → intervals → basin scan
│   │   ├── plot_stability_curves.py← plotting
│   │   ├── experiment_io.py     ← CSV I/O helpers
│   │   └── ...
│   │
│   └── tests/                   ← unit tests
│
└── outputs/
    └── msf_zero_cache.csv       ← cached MSF zero results (CSV)
```

---

## Mathematical Background

### State Vector Layout

For a network of `n` nodes where each oscillator has `d` state variables, the full state is a flat vector of length `n*d`, interleaved as:

```
[x₀, y₀, z₀,  x₁, y₁, z₁,  ...,  xₙ₋₁, yₙ₋₁, zₙ₋₁]
```

To extract all x-values from state vector `s` of length `n*3`:

```python
X = s[0::3]   # every 3rd element starting at 0
Y = s[1::3]   # every 3rd element starting at 1
Z = s[2::3]   # every 3rd element starting at 2
```

### Coupled Oscillator ODE

Each node's dynamics plus diffusive coupling:

```
ẋᵢ = F(xᵢ) - σ Σⱼ Lᵢⱼ H xⱼ
```

In matrix form for the entire network:

```
ẋ = F(x) - (σ·L ⊗ H) x
```

Where:
- `F(x)` = uncoupled oscillator equations applied node-by-node
- `L` = graph Laplacian (n×n matrix)
- `H` = inner coupling matrix (d×d, picks which state variable couples)
- `⊗` = Kronecker product
- `σ` = scalar coupling strength

### Graph Laplacian

`L = D - A` where D is the degree matrix and A is the adjacency matrix. For node i: `Lᵢᵢ = degree(i)`, `Lᵢⱼ = -1` if edge exists, `0` otherwise. For connected undirected graphs, L has exactly one zero eigenvalue; all others are positive.

### Master Stability Function (MSF)

The MSF decouples the network stability problem. Instead of analyzing the full n×d system, you analyze one isolated oscillator with an extra term:

```
ẏ = [DF(x*(t)) - K·H] y
```

Where `x*(t)` is the synchronized trajectory and `K` is a scalar "generalized coupling". The largest Lyapunov exponent of this variational equation is Ψ(K).

- **Ψ(K) < 0**: perturbations from sync decay → network can synchronize
- **Ψ(K) > 0**: perturbations grow → sync is unstable

The MSF zero crossings K₁, K₂ define a stable window. For a graph with Laplacian eigenvalues λ₂ ≤ ... ≤ λₙ, the coupling strength σ must satisfy:

```
K₁/λ₂ < σ < K₂/λₙ
```

### Basin Stability

A Monte Carlo measure of synchronization robustness. Sample `N` random initial conditions uniformly from a box. Integrate each until `t_max`. Count how many synchronize. Basin stability = successes/N.

---

## Module Reference

---

### `network_dynamics/core/oscillators.py`

CPU (NumPy) right-hand side functions. These are used by `solve_ivp` / LSODA.

**Constants:**

- `DYNAMICS_TYPES` — tuple of supported oscillator names: `("rossler", "lorenz", "chen", "chua", "hr")`
- `DYNAMICS_PARAMETER_COUNTS` — dict mapping name to number of parameters

**`normalize_dynamics_type(dynamics: str) → str`**

Accepts aliases (e.g. `"rössler"`, `"roessler"`) and returns the canonical name (`"rossler"`). Raises `ValueError` if unknown.

**`get_oscillator_rhs(dynamics: str) → callable`**

Returns the RHS function for the named oscillator. That function has signature `rhs(time, state_vector, coupling_matrix, parameters)`.

**`rossler(time, state_vector, coupling_matrix, parameters)`**

Rössler oscillator RHS. Parameters = `(a, b, c)`, defaults `(0.2, 0.2, 7.0)`.
Equations: `ẋ = -y - z`, `ẏ = x + ay`, `ż = b + z(x - c)`
Returns: `F(x) - coupling_matrix @ x`

**`lorenz(time, state_vector, coupling_matrix, parameters)`**

Lorenz system. Parameters = `(sigma, beta, rho)`.
Equations: `ẋ = σ(y-x)`, `ẏ = x(ρ-z) - y`, `ż = xy - βz`

**`chen(time, state_vector, coupling_matrix, parameters)`**

Chen attractor. Parameters = `(a, beta, c)`.
Equations: `ẋ = a(y-x)`, `ẏ = (c-a-z)x + cy`, `ż = xy - βz`

**`chua(time, state_vector, coupling_matrix, parameters)`**

Chua's circuit. Parameters = `(alpha, beta, gamma, a_nl, b_nl)`.
Has a piecewise nonlinearity `f(x)` implemented via `_chua_f`.

**`hr(time, state_vector, coupling_matrix, parameters)`**

Hindmarsh-Rose neuron model. Parameters = `(I, r, s)`.
Models bursting neuron behavior.

---

### `network_dynamics/core/graphs.py`

**`normalize_graph_type(graph_type: str) → str`**

Accepts aliases (`"er"`, `"path"`, etc.) and returns canonical name.

**`make_graph(graph_type, n_nodes=5, seed=42, edge_probability=0.15) → nx.Graph`**

Factory for NetworkX graphs. Currently supports `"path_graph"` and `"erdos_renyi"`.

**`graph_laplacian(G: nx.Graph) → np.ndarray`**

Computes the Laplacian matrix for directed or undirected graphs.
- Undirected: uses `nx.laplacian_matrix` (symmetric, sparse → dense)
- Directed: `D - A` where D is out-degree diagonal

Returns a float64 ndarray.

---

### `network_dynamics/core/coupling.py`

Builds the full coupling matrix `σ · (L ⊗ H)`.

**`default_x_coupling_matrix(dimension=3) → np.ndarray`**

Returns an identity-like matrix with only `H[0,0] = 1`, all others zero. This means only the first variable (x) of each oscillator is coupled across nodes.

**`validate_inner_coupling_matrix(H, dimension) → np.ndarray`**

Checks shape is `(dimension, dimension)`. Returns the array cast to float.

**`build_coupling_matrix(L, H=None, strength=1.0, dimension=3) → np.ndarray`**

Core function. Steps:
1. Validates L is square
2. Uses `default_x_coupling_matrix` if H is None
3. Computes `strength * np.kron(L, H)`
4. Zeros out near-zero floating point noise

Returns shape `(n_nodes*dimension, n_nodes*dimension)`.

---

### `network_dynamics/core/config.py`

**`BasinConfig` (dataclass)**

Central configuration object for one basin stability experiment. Key fields:

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `G` | `nx.Graph` | path_graph(5) | Network topology |
| `dynamics` | `str` | `"rossler"` | Oscillator type |
| `dimension` | `int` | `3` | State variables per node |
| `parameters` | `Sequence[float]` | `(0.2, 0.2, 7.0)` | Oscillator parameters |
| `coupling_strength` | `float` | `1.0` | σ |
| `H` | `np.ndarray\|None` | None | Inner coupling matrix |
| `tmax` | `float` | `150.0` | Integration endpoint |
| `dt` | `float` | `0.05` | Time step |
| `integrator` | `str` | `"LSODA"` | `"LSODA"` or `"RK4"` |
| `n_trials` | `int` | `25` | Monte Carlo samples |
| `base_seed` | `int` | `42` | RNG seed base |
| `sampler` | `str` | `"uniform"` | IC sampling method |
| `sampling_bounds` | `Tuple` | `(-5, 5)` | IC box bounds |
| `sync_tol` | `float` | `1e-2` | Distance threshold for sync |
| `tol_max` | `float` | `1e6` | Divergence threshold |
| `window_fraction` | `float` | `0.1` | Fraction of trajectory for window check |
| `success_definition` | `str` | `"window_success"` | How to classify sync |
| `max_abs_threshold` | `float` | `1e9` | Health check limit |
| `backend` | `str` | `"serial"` | `"serial"`, `"cpu"`, or `"gpu"` |

**`BasinConfig.validate() → self`**

Runs all validation checks (shape, bounds, supported values). Returns self for fluent chaining. Normalizes `dynamics` string via `normalize_dynamics_type`.

**Computed properties:**

- `n_nodes` — number of graph nodes
- `n_edges` — number of graph edges
- `state_dimension` — `n_nodes * dimension`
- `n_time_points` — `round(tmax / dt)`

---

### `network_dynamics/core/sampling.py`

**`sample_uniform_initial_condition(rng, n_nodes, dimension=3, low=-5, high=5) → np.ndarray`**

Returns a flat vector of length `n_nodes * dimension` drawn uniformly from `[low, high]`.

**`sample_normal_initial_condition(rng, n_nodes, dimension=3, mean=0, std=1) → np.ndarray`**

Same but from a normal distribution.

**`trial_seeds(base_seed, n_trials) → list[int]`**

Returns `[base_seed, base_seed+1, ..., base_seed+n_trials-1]`. Makes each trial reproducible independently.

---

### `network_dynamics/core/sync.py`

All synchronization metrics work on a trajectory `sol` with shape `(n_time_points, state_dimension)`.

**`reshape_state_by_node(state_vector, dimension=3) → np.ndarray`**

Reshapes a flat vector `[x0,y0,z0,x1,y1,z1,...]` to shape `(n_nodes, dimension)` with one row per oscillator.

**`max_pairwise_distance(state_vector, dimension=3) → float`**

Maximum Euclidean distance between any two nodes at one time point. Uses `scipy.spatial.distance.pdist`. Returns 0.0 if only 1 node.

**`distance_time_series(sol, dimension=3) → np.ndarray`**

Applies `max_pairwise_distance` at every time step. Returns array of length `n_time_points`.

**`final_max_pwd(sol, dimension=3) → float`**

Max pairwise distance at the last time point only.

**`max_distance_over_final_window(sol, dimension=3, win_frac=0.2) → float`**

Max pairwise distance over the final `win_frac` fraction of the trajectory. More robust than a single final measurement.

**`time_to_sync(sol, t, dimension=3, tol=1e-3, tol_max=1e6) → float`**

First time the max pairwise distance drops below `tol`. Returns `np.inf` if never synced or if distance exceeds `tol_max` first.

**`is_synchronized_final(sol, dimension=3, tol=1e-3) → bool`**

Boolean: is the system synchronized at the final time only?

**`is_synchronized_over_win(sol, dimension=3, tol=1e-3, win_frac=0.2) → bool`**

Boolean: is the max distance over the final window below `tol`?

**`analyze_synchronization(sol, t, dimension=3, tol=1e-3, tol_max=1e6, win_frac=0.2) → dict`**

Master function. Returns a dict with:

| Key | Meaning |
|-----|---------|
| `final_distance` | max pairwise distance at last step |
| `window_max_distance` | max distance over final window |
| `min_distance` | minimum distance over entire trajectory |
| `sync_time` | first time distance < tol (np.inf if never) |
| `final_success` | `final_distance < tol` |
| `window_success` | `window_max_distance < tol` |
| `first_crossing_success` | `min_distance < tol` |

---

### `network_dynamics/core/results.py`

**`TrialResult` (dataclass)**

One row of data per basin trial.

Fields: `trial_seed`, `success`, `final_success`, `window_success`, `integration_failed`, `final_distance`, `window_max_distance`, `min_distance`, `sync_time`, `error`

**`TrialResult.to_dict()`** — serializes to a plain dict.

**`BasinSummary` (dataclass)**

Aggregated result across all trials.

Fields: `success_definition`, `basin_stability` (fraction), `n_trials`, `successes`, `sync_failures`, `integration_failures`, `sync_time_mean`, `base_seed`, `trial_seeds`, `results` (list of TrialResult), `config`

**`BasinSummary.from_results(config, seeds, results) → BasinSummary`** (classmethod)

Computes counts and mean sync time from a list of `TrialResult` objects.

---

### `network_dynamics/core/diagnostics.py`

**`solution_health(sol, max_abs_threshold=1e6) → dict`**

Checks the trajectory for:
- `contains_nan`
- `contains_inf`
- `max_abs_value`
- `exceeds_max_abs_threshold`

**`is_solution_valid(health) → bool`**

Returns True if no NaN, no Inf, and max value under threshold.

**`format_health_message(health) → str`**

Human-readable error string for what failed.

---

### `network_dynamics/core/basin_common.py`

Shared logic used by both CPU and GPU backends.

**`SUCCESS_DEFINITIONS`** — tuple: `("final_success", "window_success", "first_crossing")`

**`choose_success(sync_metrics, success_definition) → bool`**

Picks the right key from `analyze_synchronization`'s output based on the configured definition.

**`sample_initial_condition(config, trial_seed) → np.ndarray`**

Creates an RNG from `trial_seed`, then calls `sample_uniform_initial_condition`.

**`sample_initial_conditions_batch(config, seeds, dtype) → np.ndarray`**

Generates ICs for all seeds, stacked into shape `(n_trials, state_dimension)`.

**`validate_initial_conditions_batch(config, initial_conditions_batch, dtype) → np.ndarray`**

Validates shape is `(n_trials, state_dimension)`. Used before passing a fixed IC batch to CPU/GPU.

**`failed_trial_result(trial_seed, error) → TrialResult`**

Constructs a TrialResult with `success=False`, `integration_failed=True`, and an error message.

**`can_ignore_post_sync_instability(config, sync_metrics) → bool`**

For `first_crossing` mode: if the system synced before later blowing up, ignore the late divergence. Returns False for all other modes.

**`classify_solution(config, trial_seed, sol, t) → TrialResult`**

The main classification function shared by CPU and GPU. Steps:
1. Calls `analyze_synchronization` to get metrics
2. Calls `choose_success` to pick the boolean
3. Calls `solution_health` + `is_solution_valid` to check numerical health
4. If health bad and not `first_crossing` exception: return `failed_trial_result`
5. Otherwise return populated `TrialResult`

---

### `network_dynamics/core/msf/` package

The MSF computation is entirely Numba-compiled, running in parallel across CPU cores.

---

#### `network_dynamics/core/msf/dynamics.py`

**Constants:**

```python
ROSSLER = 0
LORENZ  = 1
CHEN    = 2
CHUA    = 3
HR      = 4

DYN_IDS        = {"rossler": 0, "lorenz": 1, ...}
PARAM_LENGTHS  = {0: 3, 1: 3, 2: 3, 3: 5, 4: 3}
INITIAL_STATES = {0: [1,1,1], 1: [1,1,1], ..., 4: [-1.3,-8,1]}
```

**`rhs_jac(state, params, dyn_id) → (ds, J)`** — `@numba.njit`

A single Numba-compiled function that returns both the RHS derivative vector and the Jacobian matrix for a single 3D oscillator. Takes a 3-element state, a params array, and an integer `dyn_id`. Returns `(ds, J)` where `ds.shape=(3,)` and `J.shape=(3,3)`.

**Note:** This file is mostly superseded by the inlined versions in `compute.py` (which are faster because they write into pre-allocated arrays). The `rhs_jac` here allocates new arrays on each call.

---

#### `network_dynamics/core/msf/params.py`

**`MSFParams` (dataclass)**

Lightweight config for one MSF computation. No JAX dependency.

| Field | Default | Meaning |
|-------|---------|---------|
| `dynamics` | `"rossler"` | Oscillator name |
| `a,b,c` | `0.2,0.2,9.0` | Primary parameters |
| `d,e` | `0.0,0.0` | Extra parameters (Chua uses d=a_nl, e=b_nl) |
| `target` | `0` | Row index in H (which output variable is coupled) |
| `source` | `0` | Column index in H (which input variable is coupled) |
| `dt` | `0.001` | Integration timestep |
| `transient_time` | `100.0` | How long to settle on attractor before measuring |
| `measurement_time` | `3000.0` | How long to run the Lyapunov measurement |
| `qr_interval_steps` | `10` | Steps between QR reorthogonalizations |

`__post_init__` validates that `dynamics` is known, `target/source` are in `{0,1,2}`, and `measurement_steps % qr_interval_steps == 0`.

**`dyn_id` property** — returns the integer ID for the chosen dynamics.

---

#### `network_dynamics/core/msf/compute.py`

The hot computational core, written in Numba with zero heap allocations in the inner loop.

**`_matmul3(A, B, C)`** — `@numba.njit(inline="always")`

Inlined 3×3 matrix multiply writing result into pre-allocated C. ~30 FLOPs, no memory allocation.

**`_qr3(Y)`** — `@numba.njit(inline="always")`

In-place Modified Gram-Schmidt on the columns of a 3×3 matrix Y. Returns `(log|r₀₀|, log|r₁₁|, log|r₂₂|)` — the log of the column norms before normalization, which accumulate into Lyapunov exponents.

Steps per column:
1. Compute column norm
2. Divide (normalize)
3. Subtract projections from subsequent columns

**`_rhs_into(state, params, dyn_id, ds)`** — `@numba.njit(inline="always")`

Writes the oscillator RHS into pre-allocated array `ds`. Dispatches on `dyn_id` via if/elif chain. No allocations.

**`_jac_into(state, params, dyn_id, J)`** — `@numba.njit(inline="always")`

Writes the 3×3 Jacobian into pre-allocated matrix J. No allocations.

**`run_transient(state0, params, dyn_id, dt, steps) → np.ndarray`** — `@numba.njit`

Runs a free (uncoupled, K=0) oscillator forward for `steps` RK4 steps to let it settle onto the chaotic attractor. Returns the settled state vector (shape `(3,)`).

Pre-allocates all intermediate RK4 arrays once. Inner loop pattern:
```
k1 = rhs(s)
k2 = rhs(s + dt/2 * k1)
k3 = rhs(s + dt/2 * k2)
k4 = rhs(s + dt   * k3)
s += (dt/6) * (k1 + 2k2 + 2k3 + k4)
```

**`_one_k(K, state0, H_tgt, H_src, params, dyn_id, dt, msteps, qr_steps) → float`** — `@numba.njit`

Computes Ψ(K) — the largest Lyapunov exponent of the variational equation — for one value of K.

State: `s` (oscillator trajectory), `Y` (3×3 tangent matrix, initialized as identity)

Each step simultaneously:
- Advances `s` via RK4 using `_rhs_into`
- Advances `Y` via RK4 using the variational equation: `dY/dt = J(s)·Y - K·H·Y`

The coupling term `K·H·Y` is applied as `K * Y[H_src, :]` subtracted from row `H_tgt` of `dY`. This is valid only for rank-1 H matrices (one nonzero entry), which is the typical case.

After every `qr_steps` steps, `_qr3(Y)` reorthogonalizes Y in-place and accumulates log-norms into `acc0, acc1, acc2`.

At the end: `LE[i] = acc[i] / total_time`. Returns `max(LE0, LE1, LE2)`.

**`scan_msf(K_arr, state0, H_tgt, H_src, params, dyn_id, dt, msteps, qr_steps) → np.ndarray`** — `@numba.njit(parallel=True)`

Parallel scan: calls `_one_k` for each element of `K_arr` using `prange`. Returns a 1D array `psi` of the same length as `K_arr`. This is the main compute function — each K value is independent and runs on its own CPU core.

**`find_msf_zeros(params_obj, K_min, K_max, n_K, min_sep, max_zeros, verbose) → list[float]`**

High-level Python function (not Numba) that orchestrates:
1. Extracts dyn_id, params, initial state from `params_obj`
2. Calls `run_transient` to settle
3. Builds `K_arr = np.linspace(K_min, K_max, n_K)`
4. Calls `scan_msf` to get Ψ(K) curve
5. Calls `find_zeros` to locate crossings
6. Returns the zero list

**`warmup()`**

Triggers JIT compilation with a tiny toy run. Call once before timing any real runs. Numba compiles on first call and caches the machine code.

---

#### `network_dynamics/core/msf/zeros.py`

**`find_zeros(K, psi, min_sep=1.0, max_zeros=4) → (zeros, brackets, stable_intervals)`**

Finds where Ψ(K) crosses zero, with noise filtering.

Algorithm:
1. Detect sign changes: `psi[i] * psi[i+1] < 0`
2. Merge brackets closer than `min_sep` — spurious chatter from numerical noise often creates many sign changes near a true zero; merging by `min_sep` collapses them
3. Keep only brackets with an odd number of merged sign changes (genuine crossings)
4. Linearly interpolate within each bracket to find the precise zero

Returns:
- `zeros`: list of float K values (interpolated crossing points)
- `brackets`: list of `(K_left, K_right)` pairs
- `stable_intervals`: list of `(K_lo, K_hi)` intervals where Ψ < 0 (network can sync)

---

### `network_dynamics/core/msf_cache.py`

Caches MSF zero results in a CSV file to avoid recomputing.

**`MSF_CACHE_FIELDS`** — tuple of all CSV column names

**`make_msf_cache_key(config, K_min, K_max, n_K) → dict`**

Builds a dictionary of key fields from an `MSFParams` object. Float values are formatted with full precision (`:.17g`) to avoid floating-point key collisions.

**`row_matches_key(row, key) → bool`**

Checks if a CSV row matches a cache key. Handles backward-compat: rows without `dynamics` field default to `"rossler"`.

**`read_msf_cache(cache_path) → list[dict]`**

Reads the CSV file and returns all rows as a list of dicts.

**`find_cached_msf_result(cache_path, key) → dict | None`**

Searches the cache in reverse order (newest first) for a matching entry. Returns a dict with `zeros`, `zero_brackets`, `stable_intervals`, and the raw `row`, or `None` if not found.

**`append_msf_cache_result(cache_path, key, zeros, zero_brackets, stable_intervals) → Path`**

Appends one row to the CSV. Creates parent directories if needed. Handles schema migration (adds missing columns for older cache files). Writes timestamp in UTC ISO format.

---

### `network_dynamics/core/coupling_strengths.py`

Translates MSF zero crossings into graph-specific coupling strength intervals.

**`CouplingStrengthInterval` (frozen dataclass)**

Fields: `lower`, `upper`, `msf_zero_low`, `msf_zero_high`, `laplacian_first_nonzero`, `laplacian_largest`

The interval `[lower, upper]` is where σ should lie for the network to synchronize.

**`laplacian_nonzero_eigenvalue_bounds(G, tolerance=1e-10) → (λ₂, λₙ)`**

Computes the smallest nonzero and largest Laplacian eigenvalues. Uses `eigvalsh` for symmetric (undirected) graphs, `eigvals` for directed graphs. Raises if eigenvalues are complex or if there are no positive nonzero eigenvalues.

**`coupling_strength_intervals_from_zeros(G, msf_zeros, tolerance=1e-10) → list[CouplingStrengthInterval]`**

Converts a list of MSF zeros into coupling strength intervals. For each consecutive zero pair `(K₁, K₂)`:

```
lower = K₁ / λ₂
upper = K₂ / λₙ
```

Only adds an interval if `lower < upper` (it's possible for the interval to be empty on some graphs).

**`find_coupling_strength_intervals(G, params, K_min, K_max, n_K, ...) → list[CouplingStrengthInterval]`**

End-to-end: runs MSF scan and converts zeros to intervals. Calls `find_msf_zeros` then `coupling_strength_intervals_from_zeros`.

**`interval_coupling_strengths(interval, n_strengths, endpoint=True) → np.ndarray`**

Returns `n_strengths` evenly spaced values in `[interval.lower, interval.upper]`.

**`coupling_strengths_from_intervals(intervals, n_strengths_per_interval, ...) → list[np.ndarray]`**

Applies `interval_coupling_strengths` to each interval in a list.

---

### `network_dynamics/cpu/integration.py`

CPU integration using SciPy or a custom RK4.

**`make_n_steps(tmax, dt) → int`**

`round(tmax / dt)` — avoids float accumulation drift from `np.arange`.

**`make_time_grid(tmax, dt) → np.ndarray`**

Builds `[0, dt, 2dt, ...]` using integer indexing: `np.arange(n) * dt`. More accurate than `np.arange(0, tmax, dt)`.

**`build_rhs(G, parameters, coupling_strength, H, dimension, dynamics) → callable`**

Builds the full network RHS closure:
1. Computes Laplacian from G
2. Gets oscillator RHS function from `get_oscillator_rhs`
3. Builds coupling matrix via `build_coupling_matrix`
4. Returns a closure `rhs(time, state)` suitable for `solve_ivp`

**`integrate_lsoda(G, initial_conditions, parameters, coupling_strength, H, tmax, dt, ...) → (sol, t)`**

Integrates using SciPy's `solve_ivp` with `method="LSODA"`. LSODA is adaptive (stiff-switching), good for well-behaved trajectories. Returns `sol.shape = (n_time_points, state_dimension)` (note: solve_ivp returns `result.y` in transposed form, so `.y.T` is needed).

**`rk4_step(rhs, time, state, dt) → np.ndarray`**

One explicit 4th-order Runge-Kutta step. Computes k1, k2, k3, k4 and returns `state + (dt/6)*(k1 + 2k2 + 2k3 + k4)`.

**`integrate_rk4(G, initial_conditions, ..., divergence_threshold=1e9) → (sol, t)`**

Fixed-step RK4 integrator. After each step, checks if the state is finite and below the divergence threshold. If not, fills the rest of `sol` with `inf` and breaks early.

**`integrate(G, initial_conditions, ..., integrator="LSODA") → (sol, t)`**

Dispatcher. Routes to `integrate_lsoda` or `integrate_rk4` based on `integrator` parameter.

**`integrate_from_config(config, initial_conditions) → (sol, t)`**

Calls `integrate` with all parameters unpacked from a `BasinConfig`.

---

### `network_dynamics/cpu/basin.py`

CPU basin stability runner, with optional multiprocessing.

**`run_single_trial(config, trial_seed) → TrialResult`**

Generates IC from `trial_seed`, calls `integrate_from_config`, calls `classify_solution`. Catches all exceptions → `failed_trial_result`.

**`run_single_trial_from_initial_condition(config, trial_seed, initial_condition) → TrialResult`**

Same but starts from a fixed IC (used for CPU/GPU comparison).

**`_run_trial_from_settings(settings)` and `_run_trial_from_initial_condition_settings(settings)`**

Top-level functions (not methods) required as `ProcessPoolExecutor` targets. Python multiprocessing can only pickle top-level callables.

**`_print_progress(completed, total, start_time, label, progress_stream)`**

Prints rate, completion percentage, and ETA.

**`_map_trials(trial_settings, worker_function, n_workers, ...) → list[TrialResult]`**

Dispatches trials either serially (n_workers ≤ 1) or via `ProcessPoolExecutor`. Results are gathered by index to preserve order (since futures complete out-of-order).

**`basin_stability_serial(config) → BasinSummary`**

Runs all trials in a single process, one at a time.

**`basin_stability_cpu(config) → BasinSummary`**

Multiprocessing version. Uses `config.n_workers` processes.

**`basin_stability_cpu_from_initial_conditions(config, initial_conditions_batch, seeds, ...) → BasinSummary`**

Takes a pre-generated batch of ICs (for exact comparison with GPU). Seeds can be passed explicitly.

**`print_basin_summary(summary)` / `print_trial_results(summary)`**

Pretty-print helpers for debugging.

---

### `network_dynamics/gpu/dynamics.py`

JAX-based RHS functions that operate on entire batches of trials simultaneously.

**`DYNAMICS_CODES`** — maps oscillator name to integer code (same as Numba IDs)

**`dynamics_code(dynamics) → int`**

Lookup with helpful error message.

**`rossler_batch_jax(state_batch, coupling_matrix, parameters)`**

Input `state_batch.shape = (n_trials, state_dimension)`. Extracts X, Y, Z via stride-3 slicing across the last axis. Computes per-node derivatives, stacks and reshapes, subtracts coupling term via `state_batch @ coupling_matrix.T`.

All other `*_batch_jax` functions follow the same pattern.

**`oscillator_batch_jax(state_batch, coupling_matrix, parameters, dynamics_code_value)`**

Dispatcher for batch dynamics. Routes to the right `*_batch_jax` function.

**`rk4_step_batch_jax(state_batch, dt, coupling_matrix, parameters, dynamics_code_value)`**

One RK4 step for an entire batch of states. Each of k1, k2, k3, k4 is computed via `oscillator_batch_jax`. Returns `state_batch + (dt/6)*(k1 + 2k2 + 2k3 + k4)`.

---

### `network_dynamics/gpu/integration.py`

JAX/GPU trajectory integrators using `jax.lax.scan` for compilation efficiency.

**Single-trajectory functions:**

**`rossler_jax`, `lorenz_jax`, `chen_jax`, `chua_jax`** — single-state-vector versions of the RHS for use in `rk4_step_jax`.

**`oscillator_jax(state_vector, coupling_matrix, parameters, dynamics_code_value)`** — dispatcher.

**`rk4_step_jax(state, dt, coupling_matrix, parameters, dynamics_code_value)`** — single RK4 step (not batched).

**`integrate_rk4_scan_jax(initial_state, coupling_matrix, parameters, dt, n_steps, dynamics_code_value)`** — `@jax.jit`

JIT-compiled single trajectory integrator using `lax.scan`. `lax.scan` is critical for performance: it unrolls the loop into a fixed XLA computation graph rather than tracing Python for-loops. Returns `sol.shape = (n_steps, state_dimension)`.

**`integrate_rk4_jax(G, initial_conditions, ...) → (sol, t)`**

Python-level wrapper. Builds Laplacian, coupling matrix, converts all inputs to `jnp` arrays, calls `integrate_rk4_scan_jax`.

**Batched trajectory functions:**

**`integrate_rk4_batch_scan_jax(initial_states, coupling_matrix, parameters, dt, n_steps, dynamics_code_value)`** — `@jax.jit`

Core batched integrator. `initial_states.shape = (n_trials, state_dimension)`. Uses `lax.scan` over steps with `rk4_step_batch_jax`. Output `sol_time_first.shape = (n_steps, n_trials, state_dim)`. Swapped to `sol_batch.shape = (n_trials, n_steps, state_dim)` for per-trial access.

**`integrate_rk4_batch_jax(G, initial_conditions_batch, ...) → (sol_batch, t)`**

Python wrapper for batched integration.

**`integrate_rk4_batch_from_config(config, initial_conditions_batch)`**

Unpacks a `BasinConfig` and calls `integrate_rk4_batch_jax`.

---

### `network_dynamics/gpu/basin.py`

**`basin_stability_gpu(config, batch_size=25, verbose=True) → BasinSummary`**

GPU basin stability runner. Steps:

1. Validates config (requires `integrator="RK4"`)
2. Builds coupling matrix and converts to JAX arrays (stays on device across all chunks)
3. Generates trial seeds
4. Loops over seed chunks of size `batch_size`:
   - Generates ICs for the chunk (CPU)
   - Converts to JAX
   - Calls `integrate_rk4_batch_scan_jax` (GPU)
   - Converts result back to NumPy
   - Calls `classify_solution` per trial (CPU)
   - Frees batch memory explicitly with `del`
5. Returns `BasinSummary`

The chunking avoids GPU OOM on large trial counts. Smaller `batch_size` = less GPU memory, potentially slower due to more JIT re-entries.

---

### `network_dynamics/core/jax_config.py`

**`enable_x64()`**

Calls `jax.config.update("jax_enable_x64", True)`. JAX defaults to 32-bit; this enables float64. Must be called before any JAX computation. Called at module import in `gpu/basin.py` and `gpu/dynamics.py`.

---

### `run_msf_cache.py` (top-level script)

Pre-computes MSF zeros for all 22 oscillator/coupling-scheme combinations from Huang et al. (PhysRevE 80, 036204 (2009)) and writes them to `outputs/msf_zero_cache.csv`.

**CONFIGS list** — 22 configs spanning Rössler, Lorenz, Chen, Chua, Hindmarsh-Rose with various `source→target` coupling schemes. Each config uses `_cfg()` helper to build a dict.

**Key parameters per config:** `dynamics`, `source`, `target` (0-indexed matrix indices), `K_max`, max number of zeros expected, oscillator parameters, time settings.

**`main()` flow:**

1. Parse `--workers` before importing Numba (must set env vars first)
2. Set `NUMBA_NUM_THREADS`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`
3. Call `warmup()` to trigger JIT compilation
4. Group configs by oscillator/parameters (shared transient)
5. For each group:
   - Run `run_transient` once (shared among coupling schemes in group)
   - For each coupling scheme config: run `scan_msf`, find zeros, write to cache

**CLI options:**

| Flag | Effect |
|------|--------|
| `--only rossler` | Run only one oscillator |
| `--fast` | Use `measurement_time=300` instead of 3000 |
| `--dry-run` | Print config table, don't compute |
| `--force` | Recompute even if already cached |
| `--workers N` | Override CPU core count |

---

### `network_dynamics/experiments/coupling_basin_scan.py`

The main experiment runner. Orchestrates the full research pipeline end-to-end.

**`ScanRequest` (frozen dataclass)** — all CLI arguments collected into one object.

**`parse_args()`** — defines all CLI flags. Key ones:

| Flag | Default | Meaning |
|------|---------|---------|
| `--graph` | `erdos-renyi` | Graph type |
| `--n-nodes` | 100 | Graph size |
| `--n-trials` | 1000 | Basin stability samples |
| `--dynamics` | `rossler` | Oscillator type |
| `--tmax` | 5000 | Integration length |
| `--n-strengths` | 10 | Points in σ scan |
| `--backend` | `gpu` | `cpu` or `gpu` |
| `--coupling-low/high` | None | Skip MSF, use manual σ interval |
| `--msf-cache` | `outputs/msf_zero_cache.csv` | Cache file |

**`request_from_args(args) → ScanRequest`** — resolves n_workers, normalizes dynamics, applies default parameters.

**`make_msf_config(request) → MSFParams`** — builds MSF configuration from scan request.

**`make_manual_coupling_interval(request) → CouplingStrengthInterval | None`**

If both `--coupling-low` and `--coupling-high` are given, returns a `CouplingStrengthInterval` bypassing MSF. MSF zero fields are NaN since they weren't computed.

**`make_basin_config(request, graph, coupling_strength) → BasinConfig`** — builds and validates a `BasinConfig` for one specific σ.

**`run_basin_scan(request, graph, interval) → list[dict]`**

Core loop. For each of `n_strengths` coupling strengths in the interval:
1. Makes `BasinConfig` for this σ
2. First iteration: generates all ICs once and reuses them across all σ values (ensures fair comparison)
3. Runs `basin_stability_gpu_fast_from_initial_conditions` or `basin_stability_cpu_from_initial_conditions`
4. Appends result row

**`main()` flow:**

1. Parse args → `ScanRequest`
2. Build graph
3. Check for manual interval → if yes, skip MSF
4. Check MSF cache → if miss, run `find_msf_zeros` and write to cache
5. Convert zeros to coupling strength intervals
6. Run `run_basin_scan` on the selected interval
7. Print table, optionally write CSV

---

## `GenerateDynamics.py` (legacy)

The original monolithic simulator. Two classes:

### `non_laplacian_dynamics`

For supply-chain network models (Mondal SCM). Stores a graph and initial condition. Has:
- `set_graph`, `return_graph`, `set_init_cond`, `return_init_cond`
- `convert_adjacency` — adjacency matrix to NetworkX graph
- `convert_graph_to_laplacian`, `get_Laplacian` — Laplacian construction
- `generate_initial_condition` — supports many distributions
- `continuous_time_nonlinear_dynamics` — calls `odeint` with `mondal_scm`
- `mondal_scm` — the supply chain ODE (3 state variables per node)

### `laplacian_dynamics`

For standard diffusive coupled oscillators. Contains:
- All the same graph/IC setup methods as above
- Individual ODE functions: `rossler`, `perturbed_rossler`, `lorenz`, `brusselator`, `vanderpol`, `wienbridge`, `aizawa`, `chen_lee`, `arneodo`, `sprottb`, `sprott_linzf`, `dadras`, `halvorsen`, `thomas`, `rabinovich_fabrikant`, `three_scroll`, `four_wing`, `perturbed_four_wing`
- All these follow the same pattern: extract X/Y/Z slices, compute node-local derivatives, subtract `np.dot(LH, x)` for coupling
- `continuous_time_linear_dynamics` — Laplacian diffusion of a scalar on each node
- `continuous_time_nonlinear_dynamics` — big switch statement dispatching to each oscillator, supports time-varying parameters and time-varying Laplacians

This is the original research code. The `network_dynamics/` package is a cleaner rewrite that separates concerns and adds GPU support.

---

## Key Design Decisions Worth Understanding

**1. State vector interleaving vs. blocking**

State is stored as `[x0,y0,z0,x1,y1,z1,...]` (interleaved), not `[x0,x1,...,y0,y1,...,z0,z1,...]` (blocked). This is why X,Y,Z are extracted with stride 3. The Kronecker product `L ⊗ H` is designed for interleaved layout.

**2. Why `np.kron(L, H)` works**

For interleaved state, the coupling term for oscillator i's x-variable is:
`Σⱼ σ Lᵢⱼ H[x,x] xⱼ`
The Kronecker product `σ·L⊗H` encodes exactly this for all nodes and all variable pairs at once.

**3. Numba vs JAX split**

- **Numba (CPU)**: used for MSF computation. MSF requires computing Lyapunov exponents, which need sequential RK4 steps and a QR decomposition — not naturally batchable. Numba's `prange` parallelizes across K-values on CPU cores.
- **JAX (GPU)**: used for basin stability. Basin trials are independent and embarrassingly parallel. JAX's `lax.scan` + `jit` turns the time-stepping loop into a fixed XLA computation graph that runs efficiently on GPU/MPS.

**4. Zero-allocation hot loop in `_one_k`**

All arrays for the inner RK4 loop in `_one_k` are allocated once when entering the function, then reused across all `msteps` steps. This matters because in `scan_msf` many threads call `_one_k` simultaneously via `prange`; if each step called `np.empty`, there would be heavy `malloc` contention across threads.

**5. QR reorthogonalization**

Lyapunov exponents are computed via the standard QR method: repeatedly evolve the tangent matrix Y and orthogonalize. Without reorthogonalization, the fastest-growing direction would dominate all columns of Y, making it impossible to extract individual Lyapunov exponents. The log of the norms stripped during Gram-Schmidt accumulate into the exponents.

**6. MSF target and source indices**

For a rank-1 inner coupling matrix H with a single nonzero entry `H[target, source] = 1`, the coupling term in the variational equation is `K * H * Y`, which equals `K * Y[source, :] * e_{target}` (adds K times row `source` of Y to row `target` of dY). This is why `_one_k` applies the coupling as:
```python
dY[H_tgt, :] -= K * Y[H_src, :]
```
rather than a full matrix multiply.

**7. MSF cache design**

Running full MSF scans takes 15-30 minutes per configuration. The CSV cache lets you compute once and reuse. Keys use full-precision float formatting (`.17g`) to avoid floating-point mismatch between Python `float` and CSV string representation.

**8. Same ICs across coupling strengths**

In `run_basin_scan`, the initial conditions batch is generated once from the first config's seed and reused across all σ values. This means the basin stability difference across σ is purely due to dynamics, not sampling variation.

---

## How to Add a New Oscillator

1. **`network_dynamics/core/oscillators.py`**: Add `def mynew(time, state_vector, coupling_matrix, parameters): ...` following the existing pattern. Add to `DYNAMICS_TYPES` and `DYNAMICS_PARAMETER_COUNTS`.

2. **`network_dynamics/core/msf/dynamics.py`**: Add the ID constant (e.g. `MYNEW = 5`). Add to `DYN_IDS`, `PARAM_LENGTHS`, `INITIAL_STATES`. Add the `elif dyn_id == MYNEW:` branch to `rhs_jac`.

3. **`network_dynamics/core/msf/compute.py`**: Import the new ID constant. Add branches to `_rhs_into` and `_jac_into`.

4. **`network_dynamics/gpu/dynamics.py`**: Add `DYNAMICS_CODES["mynew"] = 5`. Add `def mynew_batch_jax(...)` and add its case to `oscillator_batch_jax`.

5. **`network_dynamics/gpu/integration.py`**: Add `def mynew_jax(...)` and its case to `oscillator_jax`.

6. **`network_dynamics/core/oscillators.py`**: Update `normalize_dynamics_type` with any aliases.

---

## Running the Pipeline

**Step 1: Pre-compute MSF zeros**
```bash
python run_msf_cache.py --only rossler   # fast test
python run_msf_cache.py                  # all 22 configs
```

**Step 2: Run basin scan**
```bash
python -m network_dynamics.experiments.coupling_basin_scan \
    --n-nodes 100 \
    --n-trials 1000 \
    --dynamics rossler \
    --tmax 5000 \
    --n-strengths 10 \
    --backend gpu \
    --csv outputs/results.csv
```

**Step 3: Manual coupling interval (bypass MSF)**
```bash
python -m network_dynamics.experiments.coupling_basin_scan \
    --coupling-low 0.5 \
    --coupling-high 2.0 \
    --n-strengths 5
```
