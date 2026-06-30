#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --output=logs/test_gpu_%j.out
#SBATCH --error=logs/test_gpu_%j.err
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_DIR" || exit 1

module purge
module load Python/3.10.4-GCCcore-11.3.0

VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv}"
source "${VENV_DIR}/bin/activate"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

echo "=== nvidia-smi ==="
nvidia-smi

echo
echo "=== JAX devices ==="
python -c "import jax; print(jax.devices())"

echo
echo "=== jaxlib version ==="
python -c "import jaxlib; print(jaxlib.__version__)"

echo
echo "=== CUDA env ==="
echo "CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
echo "EBROOTCUDA           : $EBROOTCUDA"
echo "XLA_FLAGS            : $XLA_FLAGS"
