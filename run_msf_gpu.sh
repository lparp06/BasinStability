#!/bin/bash
# Submit from ACRES with: sbatch run_msf_gpu.sh

#SBATCH --job-name=msf_scan
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00-02:00:00
#SBATCH --output=msf_scan_%j.out
#SBATCH --error=msf_scan_%j.err

set -euo pipefail

PROJECT_DIR="$HOME/GenerateDynamics"
PYTHON="$PROJECT_DIR/.venv/bin/python"

# ======================== MSF SCAN SETTINGS ========================
# SOURCE and TARGET are 0-based state indices: 0=x, 1=y, 2=z.
# Coupling notation: source->target  (e.g. source=2,target=2 is z->z).
DYNAMICS="lorenz"              # rossler | lorenz
SOURCE=2                       # coupled-from component (0=x,1=y,2=z)
TARGET=2                       # coupled-to component

# ---- Time-stepping ------------------------------------------------
# dt=0.001 matches the paper and keeps RK4 accurate at high K.
# Never use dt > 0.005 if K_MAX > 20 — the variational equation
# accumulates O(dt^4 * K^5) error that shifts zero locations.
DT=0.001

# ---- Integration time --------------------------------------------
# These match the paper (Huang et al. 2009):
#   ~10^4 Lorenz orbital periods (~0.9 tu each) for transient
#   ~3x10^4 Lorenz orbital periods for Lyapunov measurement
# For Rössler (period ~6.3 tu) the same cycle counts cost more wall
# time; use MEASUREMENT_TIME=50000 for comparable paper accuracy.
#
# Rule of thumb for basin-stability work (zeros don't need 3-decimal
# precision, just ±0.1 K accuracy):
#   TRANSIENT_TIME=2000   MEASUREMENT_TIME=10000   (fast, good enough)
#   TRANSIENT_TIME=10000  MEASUREMENT_TIME=30000   (paper-quality)
TRANSIENT_TIME=10000.0
MEASUREMENT_TIME=30000.0

# QR reorthonormalization interval (steps). Must evenly divide
# MEASUREMENT_TIME/DT. 10 is standard; don't change without reason.
QR_INTERVAL_STEPS=10

# ---- K scan range ------------------------------------------------
# Choose K_MAX to cover all expected MSF zeros:
#   Lorenz  3->3 (3 zeros): K_MAX=100   N_K=1001   (dK=0.1)
#   Lorenz  other:          K_MAX=30    N_K=301    (dK=0.1)
#   Rössler 1->1 (2 zeros): K_MAX=10    N_K=1001   (dK=0.01)
#   Rössler other:          K_MAX=100   N_K=1001   (dK=0.1)
#
# Zero precision = dK/2 from bracket midpoint, or better with
# linear interpolation (now applied automatically).
K_MIN=0.0
K_MAX=100.0
N_K=1001

# ---- GPU progress ------------------------------------------------
# Each chunk is a separate jax.vmap kernel launch.
# PROGRESS_CHUNKS=2 → one progress print at 50%, best GPU throughput.
# Increase only if you want more frequent updates during a long run.
PROGRESS_CHUNKS=2
# ===================================================================

# ACRES uses environment modules. If the virtual environment was created with
# a versioned Python module, replace Python below with that same module name.
module purge
module load Python

cd "$PROJECT_DIR"
mkdir -p outputs

if [[ ! -x "$PYTHON" ]]; then
    echo "Missing $PYTHON" >&2
    echo "Create the ACRES virtual environment and install CUDA-enabled JAX first." >&2
    exit 1
fi

# Require JAX to initialize CUDA. This prevents an allocated GPU job from
# quietly falling back to the CPU if the CUDA-enabled jaxlib is missing.
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Persistent JAX compilation cache. The first run with a given set of static
# parameters (measurement_steps, qr_interval_steps, dynamics, batch size)
# compiles and writes to disk. Every subsequent run with the same parameters
# loads the cached kernel and skips compilation entirely (~seconds vs minutes).
#
# IMPORTANT: use /tmp (local SSD on the compute node), NOT your home directory.
# Home directories on ACRES are NFS-mounted; JAX cache writes many small files
# and NFS latency makes this extremely slow, sometimes causing apparent hangs.
# /tmp is local to the node and fast.
# Delete the cache if you upgrade JAX, jaxlib, or CUDA.
export JAX_COMPILATION_CACHE_DIR="/tmp/jax_cache_${USER}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

# XLA optimization level. Default is 3 (highest); level 1 compiles 5-10x faster
# at a ~10-20% runtime cost. Use level 3 for production runs where kernel speed
# matters; use level 1 while iterating on parameters to cut the wait.
# Uncomment the line below to enable fast compilation:
# export XLA_FLAGS="--xla_backend_optimization_level=1"

echo "Job ID: ${SLURM_JOB_ID:-not-running-under-slurm}"
echo "Node: $(hostname)"
nvidia-smi

"$PYTHON" -c \
    'import jax; print("JAX backend:", jax.default_backend()); print("JAX devices:", jax.devices()); assert jax.default_backend() == "gpu"'

# Keep one Python worker for one GPU. Additional msf_scan workers explicitly
# select CPU devices, so --n-workers must remain 1 for this GPU job.
"$PYTHON" -m network_dynamics.experiments.msf_scan \
    --dynamics "$DYNAMICS" \
    --source "$SOURCE" \
    --target "$TARGET" \
    --dt "$DT" \
    --transient-time "$TRANSIENT_TIME" \
    --measurement-time "$MEASUREMENT_TIME" \
    --qr-interval-steps "$QR_INTERVAL_STEPS" \
    --K-min "$K_MIN" \
    --K-max "$K_MAX" \
    --n-K "$N_K" \
    --n-workers 1 \
    --progress-chunks "$PROGRESS_CHUNKS" \
    --csv "outputs/msf_scan_${SLURM_JOB_ID}.csv" \
    "$@"
