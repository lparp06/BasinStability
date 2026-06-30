#!/bin/bash
# ─── SLURM array — basin stability scan for configs with bounded stable MSF region ──
# Submit from the project root: sbatch slurm/basin_scan_all_gpu.sh
# The logs/ directory must exist before submitting: mkdir -p logs
#SBATCH --job-name=basin_all
#SBATCH --output=logs/basin_all_%A_%a.out
#SBATCH --error=logs/basin_all_%A_%a.err
#SBATCH --array=0-6
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# ─── Only configs with a bounded MSF stable interval (from msf_zero_cache.csv) ──
# Configs with empty stable_intervals_json are excluded — no finite coupling
# strength range exists where the network can synchronize.
#
#  0  rossler  s0→t0   stable=[0.17, 4.65]
#  1  lorenz   s1→t0   stable=[4.10, 22.45]
#  2  lorenz   s2→t2   stable=[1.30, 9.30]
#  3  chen     s2→t2   stable=[5.20, 21.50]
#  4  chua     s2→t0   stable=[1.86, 2.86]
#  5  chua     s2→t2   stable=[0.75, 4.87]
#  6  hr       s1→t0   stable=[0.23, 1.23]

DYNAMICS_ARR=(rossler lorenz lorenz chen chua chua hr)
SOURCE_ARR=(  0       1      2      2    2    2    1 )
TARGET_ARR=(  0       0      2      2    0    2    0 )
K_MAX_ARR=(   10      50     100    100  10   10   5 )
T_TR_ARR=(    100     100    100    100  100  100  1000)

# ─── Resolve config for this task ────────────────────────────────────────────
I=${SLURM_ARRAY_TASK_ID}
DYNAMICS="${DYNAMICS_ARR[$I]}"
MSF_SOURCE="${SOURCE_ARR[$I]}"
MSF_TARGET="${TARGET_ARR[$I]}"
K_MAX="${K_MAX_ARR[$I]}"
MSF_T_TR="${T_TR_ARR[$I]}"

# Fixed MSF params matching run_msf_all.py
K_MIN=0
N_K=1001
MSF_DT=0.001
MSF_T_MS=3000

# ─── Basin scan params (override at submission with env vars) ─────────────────
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
PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_DIR" || { echo "FATAL: cannot cd to $PROJECT_DIR"; exit 1; }

module purge
module load Python/3.10.4-GCCcore-11.3.0

VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv}"
if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    echo "ERROR: venv not found at $VENV_DIR" >&2
    exit 1
fi
source "${VENV_DIR}/bin/activate"
PYTHON="$(which python)"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
export JAX_PLATFORMS=cuda
export JAX_COMPILATION_CACHE_DIR="${PROJECT_DIR}/.jax_cache"
mkdir -p "${JAX_COMPILATION_CACHE_DIR}" "outputs/${DYNAMICS}/plots"

echo "=== Task ${I}/6 ==="
echo "  Job       : ${SLURM_ARRAY_JOB_ID}_${I}"
echo "  Node      : $SLURMD_NODENAME"
echo "  Config    : ${DYNAMICS}  s${MSF_SOURCE}->t${MSF_TARGET}"
echo "  K range   : [$K_MIN, $K_MAX]  n_K=$N_K"
echo "  MSF dt    : $MSF_DT  t_tr=$MSF_T_TR  t_ms=$MSF_T_MS"
echo "  Trials    : $N_TRIALS  strengths=$N_STRENGTHS"
echo "  Start     : $(date)"
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
