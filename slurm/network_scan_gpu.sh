#!/bin/bash
# ─── SLURM — network_scan: basin stability across N graph seeds (GPU) ─────────
#
# Runs network_dynamics.experiments.network_scan with --backend gpu.
# All graph seeds are processed sequentially within a single GPU job;
# the GPU parallelizes the 1000 trials inside each coupling-strength evaluation.
#
# Submit from the project root:
#   sbatch slurm/network_scan_gpu.sh
#
# Override any parameter at submission time, e.g.:
#   N_SEEDS=20 DYNAMICS=lorenz SOURCE=1 TARGET=0 sbatch slurm/network_scan_gpu.sh
#
# Logs are written to logs/network_scan_<jobid>.{out,err}.
# Create the logs directory first: mkdir -p logs

#SBATCH --job-name=network_scan
#SBATCH --output=logs/network_scan_%j.out
#SBATCH --error=logs/network_scan_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          # used by Numba (MSF zero computation)
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu            # change to your cluster's GPU partition name

# ─── Oscillator / coupling scheme ────────────────────────────────────────────
DYNAMICS="${DYNAMICS:-lorenz}"
SOURCE="${SOURCE:-1}"              # H column index (MSF source variable)
TARGET="${TARGET:-0}"              # H row index   (MSF target variable)

# ─── Graph parameters ─────────────────────────────────────────────────────────
GRAPH_TYPE="${GRAPH_TYPE:-erdos_renyi}"
N_NODES="${N_NODES:-100}"
EDGE_PROB="${EDGE_PROB:-0.15}"     # ER edge prob / WS rewiring prob
BA_M="${BA_M:-8}"                  # Barabási–Albert: edges per new node
WS_K="${WS_K:-6}"                  # Watts–Strogatz: nearest-ring neighbours

# ─── Seed sweep ───────────────────────────────────────────────────────────────
N_SEEDS="${N_SEEDS:-10}"
SEED_START="${SEED_START:-42}"

# ─── Basin parameters ─────────────────────────────────────────────────────────
N_TRIALS="${N_TRIALS:-1000}"
N_STRENGTHS="${N_STRENGTHS:-10}"
TMAX="${TMAX:-500.0}"
DT="${DT:-0.05}"
BASE_SEED="${BASE_SEED:-42}"
SYNC_TOL="${SYNC_TOL:-1e-3}"
SUCCESS_DEF="${SUCCESS_DEF:-first_crossing}"
INTERVAL_INDEX="${INTERVAL_INDEX:-0}"

# ─── MSF parameters ───────────────────────────────────────────────────────────
# K_MIN / K_MAX default to the per-oscillator table in core/msf/ranges.py
# (same defaults as msf_scan and coupling_basin_scan). Override if needed.
N_K="${N_K:-1001}"
MSF_DT="${MSF_DT:-0.001}"
MSF_T_TR="${MSF_T_TR:-1000.0}"
MSF_T_MS="${MSF_T_MS:-3000.0}"
MSF_CACHE="${MSF_CACHE:-outputs/msf_zero_cache.csv}"

# ─── Output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR="${OUTPUT_DIR:-outputs/network_scan}"

# ─── Environment ──────────────────────────────────────────────────────────────
PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_DIR" || { echo "FATAL: cannot cd to $PROJECT_DIR"; exit 1; }

# Load cluster modules — check available names with:
#   module avail python    module avail cuda
module purge
module load Python/3.10.4-GCCcore-11.3.0   # adjust to your cluster

# Set VENV_DIR to your virtualenv on the cluster filesystem.
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv}"

if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    echo "ERROR: venv not found at ${VENV_DIR}" >&2
    echo "Set VENV_DIR at submission: VENV_DIR=/path/to/venv sbatch ..." >&2
    exit 1
fi

source "${VENV_DIR}/bin/activate"
PYTHON="$(which python)"

# Make the package importable without pip install -e on the cluster
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

# Numba threads for MSF computation (runs on CPU regardless of --backend gpu)
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# Force JAX to CUDA (jax_config.py defaults to cpu on non-GPU machines)
export JAX_PLATFORMS=cuda

# Point XLA at the system CUDA so its ptxas version matches the driver.
# Without this, JAX uses its bundled ptxas (often newer than the driver),
# which disables parallel XLA compilation and prints the ptxas version warning.
# Adjust the path to match the CUDA version installed on your cluster nodes
# (check with: nvcc --version  or  ls /usr/local/cuda-*/bin/ptxas).
CUDA_DIR="${CUDA_DIR:-/usr/local/cuda-12.4}"
if [ -d "$CUDA_DIR" ]; then
    export PATH="${CUDA_DIR}/bin:${PATH}"
    export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_DIR}"
else
    echo "WARNING: CUDA_DIR=${CUDA_DIR} not found — XLA will use bundled ptxas (slower compilation)."
fi

# Cache compiled XLA kernels — subsequent seeds skip GPU re-compilation
export JAX_COMPILATION_CACHE_DIR="${PROJECT_DIR}/.jax_cache"
mkdir -p "${JAX_COMPILATION_CACHE_DIR}"

mkdir -p logs

# ─── Job summary ──────────────────────────────────────────────────────────────
echo "=== network_scan GPU job ==="
echo "  Job ID        : $SLURM_JOB_ID"
echo "  Node          : $SLURMD_NODENAME"
echo "  GPUs          : ${CUDA_VISIBLE_DEVICES:-not set}"
echo "  CPUs          : $SLURM_CPUS_PER_TASK"
echo "  Project       : $PROJECT_DIR"
echo "  Python        : $PYTHON"
echo "  Dynamics      : $DYNAMICS  source=$SOURCE  target=$TARGET"
echo "  Graph         : $GRAPH_TYPE  n=$N_NODES  seeds=${SEED_START}–$((SEED_START+N_SEEDS-1))"
echo "  Basin         : trials=$N_TRIALS  strengths=$N_STRENGTHS  tmax=$TMAX  dt=$DT"
echo "  Success def   : $SUCCESS_DEF"
echo "  MSF cache     : $MSF_CACHE"
echo "  Output dir    : $OUTPUT_DIR"
echo "  Start         : $(date)"
echo

# ─── Run ──────────────────────────────────────────────────────────────────────
"$PYTHON" -m network_dynamics.experiments.network_scan \
    --graph-type        "$GRAPH_TYPE"    \
    --n-nodes           "$N_NODES"       \
    --edge-probability  "$EDGE_PROB"     \
    --ba-m              "$BA_M"          \
    --ws-k              "$WS_K"          \
    --n-seeds           "$N_SEEDS"       \
    --seed-start        "$SEED_START"    \
    --dynamics          "$DYNAMICS"      \
    --source            "$SOURCE"        \
    --target            "$TARGET"        \
    --n-trials          "$N_TRIALS"      \
    --n-strengths       "$N_STRENGTHS"   \
    --tmax              "$TMAX"          \
    --dt                "$DT"            \
    --base-seed         "$BASE_SEED"     \
    --backend           gpu              \
    --integrator        RK4              \
    --sync-tol          "$SYNC_TOL"      \
    --success-definition "$SUCCESS_DEF" \
    --interval-index    "$INTERVAL_INDEX" \
    --n-K               "$N_K"           \
    --msf-cache         "$MSF_CACHE"     \
    --output-dir        "$OUTPUT_DIR"

echo
echo "=== Done : $(date) ==="
