#!/bin/bash
# ─── SLURM — basin stability scan (GPU backend, JAX/CUDA) ────────────────────
# Submit from the project root: sbatch slurm/basin_scan_gpu.sh
# The logs/ directory must exist before submitting: mkdir -p logs
#SBATCH --job-name=basin_gpu
#SBATCH --output=logs/basin_gpu_%j.out
#SBATCH --error=logs/basin_gpu_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# ─── Required: set oscillator and coupling pair ───────────────────────────────
# Valid combinations (Huang et al. 2009):
#   rossler: (0,0)  (1,1)  (2,0)
#   lorenz:  (0,0)  (0,1)  (1,0)  (1,1)  (2,2)
#   chen:    (0,1)  (1,1)  (2,2)
#   chua:    (0,0)  (0,1)  (1,0)  (1,1)  (1,2)  (2,0)  (2,2)
#   hr:      (0,0)  (0,1)  (1,0)  (1,1)
# Override at submission: DYNAMICS=lorenz MSF_SOURCE=1 MSF_TARGET=0 sbatch basin_scan_gpu.sh
DYNAMICS="${DYNAMICS:-rossler}"
MSF_SOURCE="${MSF_SOURCE:-0}"          # H column index (source variable)
MSF_TARGET="${MSF_TARGET:-0}"          # H row index (target variable)

# ─── Auto-configure K and MSF params from paper config ───────────────────────
# These must match exactly what run_msf_all.py used to populate the cache.
# Any variable can still be overridden before calling sbatch.
CONFIG_KEY="${DYNAMICS}_s${MSF_SOURCE}t${MSF_TARGET}"
case "$CONFIG_KEY" in
  # Rössler
  rossler_s0t0)  K_MAX="${K_MAX:-10}"  ; MSF_T_TR="${MSF_T_TR:-100}"  ;;
  rossler_s1t1)  K_MAX="${K_MAX:-5}"   ; MSF_T_TR="${MSF_T_TR:-100}"  ;;
  rossler_s2t0)  K_MAX="${K_MAX:-100}" ; MSF_T_TR="${MSF_T_TR:-100}"  ;;
  # Lorenz
  lorenz_s0t0)   K_MAX="${K_MAX:-30}"  ; MSF_T_TR="${MSF_T_TR:-100}"  ;;
  lorenz_s0t1)   K_MAX="${K_MAX:-30}"  ; MSF_T_TR="${MSF_T_TR:-100}"  ;;
  lorenz_s1t0)   K_MAX="${K_MAX:-50}"  ; MSF_T_TR="${MSF_T_TR:-100}"  ;;
  lorenz_s1t1)   K_MAX="${K_MAX:-20}"  ; MSF_T_TR="${MSF_T_TR:-100}"  ;;
  lorenz_s2t2)   K_MAX="${K_MAX:-100}" ; MSF_T_TR="${MSF_T_TR:-100}"  ;;
  # Chen
  chen_s0t1)     K_MAX="${K_MAX:-30}"  ; MSF_T_TR="${MSF_T_TR:-100}"  ;;
  chen_s1t1)     K_MAX="${K_MAX:-20}"  ; MSF_T_TR="${MSF_T_TR:-100}"  ;;
  chen_s2t2)     K_MAX="${K_MAX:-100}" ; MSF_T_TR="${MSF_T_TR:-100}"  ;;
  # Chua
  chua_s0t0)     K_MAX="${K_MAX:-20}"  ; MSF_T_TR="${MSF_T_TR:-100}"  ;;
  chua_s0t1)     K_MAX="${K_MAX:-5}"   ; MSF_T_TR="${MSF_T_TR:-100}"  ;;
  chua_s1t0)     K_MAX="${K_MAX:-30}"  ; MSF_T_TR="${MSF_T_TR:-100}"  ;;
  chua_s1t1)     K_MAX="${K_MAX:-10}"  ; MSF_T_TR="${MSF_T_TR:-100}"  ;;
  chua_s1t2)     K_MAX="${K_MAX:-50}"  ; MSF_T_TR="${MSF_T_TR:-100}"  ;;
  chua_s2t0)     K_MAX="${K_MAX:-10}"  ; MSF_T_TR="${MSF_T_TR:-100}"  ;;
  chua_s2t2)     K_MAX="${K_MAX:-10}"  ; MSF_T_TR="${MSF_T_TR:-100}"  ;;
  # Hindmarsh–Rose (slow z-timescale requires long transient)
  hr_s0t0)       K_MAX="${K_MAX:-5}"   ; MSF_T_TR="${MSF_T_TR:-1000}" ;;
  hr_s0t1)       K_MAX="${K_MAX:-5}"   ; MSF_T_TR="${MSF_T_TR:-1000}" ;;
  hr_s1t0)       K_MAX="${K_MAX:-5}"   ; MSF_T_TR="${MSF_T_TR:-1000}" ;;
  hr_s1t1)       K_MAX="${K_MAX:-3}"   ; MSF_T_TR="${MSF_T_TR:-1000}" ;;
  *)
    echo "ERROR: unknown config '${CONFIG_KEY}'." >&2
    echo "Valid: rossler/lorenz/chen/chua/hr with source and target in {0,1,2}." >&2
    exit 1
    ;;
esac

# Fixed MSF params matching run_msf_all.py — change only if you reran the cache
K_MIN="${K_MIN:-0}"
N_K="${N_K:-1001}"
MSF_DT="${MSF_DT:-0.001}"
MSF_T_MS="${MSF_T_MS:-3000}"

# ─── Basin scan params ────────────────────────────────────────────────────────
N_TRIALS="${N_TRIALS:-1000}"
N_STRENGTHS="${N_STRENGTHS:-10}"
TMAX="${TMAX:-5000.0}"
DT="${DT:-0.001}"
N_NODES="${N_NODES:-100}"
EDGE_PROB="${EDGE_PROB:-0.15}"
GRAPH_SEED="${GRAPH_SEED:-42}"
BASE_SEED="${BASE_SEED:-42}"
INTERVAL_INDEX="${INTERVAL_INDEX:-0}"
MSF_CACHE="${MSF_CACHE:-outputs/msf_zero_cache.csv}"

# ─── Environment ──────────────────────────────────────────────────────────────
# SLURM_SUBMIT_DIR is always the directory where sbatch was called — use it.
PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_DIR" || { echo "FATAL: cannot cd to $PROJECT_DIR"; exit 1; }

# Load cluster modules — must match the Python version the venv was built with.
# Run `module avail python` and `module avail cuda` on the login node to find names.
module purge
module load Python/3.10.4-GCCcore-11.3.0

# Set VENV_DIR to your virtual environment on the cluster.
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv}"

if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    echo "ERROR: venv not found at $VENV_DIR" >&2
    echo "Set VENV_DIR before submitting, e.g.:" >&2
    echo "  VENV_DIR=/path/to/venv sbatch slurm/basin_scan_gpu.sh" >&2
    exit 1
fi

source "${VENV_DIR}/bin/activate"
PYTHON="$(which python)"

# Make the package importable even if pip install -e wasn't run on the cluster
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

# jax_config.py defaults JAX_PLATFORMS=cpu — override it here for GPU jobs
export JAX_PLATFORMS=cuda

# Cache compiled XLA kernels so subsequent jobs skip recompilation
export JAX_COMPILATION_CACHE_DIR="${PROJECT_DIR}/.jax_cache"
mkdir -p "${JAX_COMPILATION_CACHE_DIR}"

mkdir -p "outputs/${DYNAMICS}/plots"

echo "=== Job info ==="
echo "  Job ID     : $SLURM_JOB_ID"
echo "  Node       : $SLURMD_NODENAME"
echo "  GPUs       : ${CUDA_VISIBLE_DEVICES:-not set}"
echo "  Project    : $PROJECT_DIR"
echo "  Python     : $PYTHON"
echo "  Config     : $CONFIG_KEY"
echo "  K range    : [$K_MIN, $K_MAX]  n_K=$N_K"
echo "  MSF dt     : $MSF_DT  t_tr=$MSF_T_TR  t_ms=$MSF_T_MS"
echo "  Trials     : $N_TRIALS  strengths=$N_STRENGTHS"
echo "  Start      : $(date)"
echo

# ─── Run ──────────────────────────────────────────────────────────────────────
"$PYTHON" -m network_dynamics.experiments.coupling_basin_scan \
    --dynamics             "$DYNAMICS"       \
    --msf-source           "$MSF_SOURCE"     \
    --msf-target           "$MSF_TARGET"     \
    --backend              gpu               \
    --n-trials             "$N_TRIALS"       \
    --n-strengths          "$N_STRENGTHS"    \
    --tmax                 "$TMAX"           \
    --dt                   "$DT"             \
    --n-nodes              "$N_NODES"        \
    --edge-probability     "$EDGE_PROB"      \
    --graph-seed           "$GRAPH_SEED"     \
    --base-seed            "$BASE_SEED"      \
    --K-min                "$K_MIN"          \
    --K-max                "$K_MAX"          \
    --n-K                  "$N_K"            \
    --interval-index       "$INTERVAL_INDEX" \
    --msf-cache            "$MSF_CACHE"      \
    --msf-dt               "$MSF_DT"         \
    --msf-transient-time   "$MSF_T_TR"       \
    --msf-measurement-time "$MSF_T_MS"       \
    --integrator           RK4               \
    --success-definition   first_crossing

echo
echo "=== Done : $(date) ==="
