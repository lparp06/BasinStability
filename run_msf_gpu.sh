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
# Change these values for a typical run. SOURCE and TARGET are zero-based
# state-component indices: 0=x, 1=y, 2=z.
DYNAMICS="rossler"             # rossler or lorenz
SOURCE=0                       # coupled-from state component
TARGET=0                       # coupled-to state component

DT=0.05                        # RK4 time step
TRANSIENT_TIME=100.0           # discarded synchronization time
MEASUREMENT_TIME=300.0         # Lyapunov-exponent measurement time
QR_INTERVAL_STEPS=10           # steps between QR reorthonormalizations

K_MIN=0.0                      # beginning of MSF coupling scan
K_MAX=10.0                     # end of MSF coupling scan
N_K=101                        # number of evenly spaced K values

# Number of progress updates during the scan. On GPU, JAX vmaps ALL K values
# in a sub-batch in parallel — larger sub-batches = better GPU utilization.
# PROGRESS_CHUNKS=2 gives one mid-scan update with minimal GPU overhead.
# Raising this significantly (e.g. 10) can make the GPU run 5-10x slower
# because each sub-batch uses fewer parallel threads.
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
