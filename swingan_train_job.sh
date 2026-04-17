#!/bin/bash
#SBATCH --job-name=swingan_train
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH -A kf4-ic
#SBATCH --mem=40G
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --chdir=/projects/illinois/eng/bioe/rxb/akshat/VSlite
#SBATCH --output=/u/akshahi2/logs/%x_%j.out
#SBATCH --error=/u/akshahi2/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akshahi2@illinois.edu

set -o pipefail
mkdir -p /u/akshahi2/logs

set -x
exec > >(tee -a "/u/akshahi2/logs/${SLURM_JOB_NAME:-job}_${SLURM_JOB_ID:-noid}.debug.log") 2>&1
echo "$(date -Is) swingan_train_job: starting host=$(hostname) pwd=$(pwd) SLURM_JOB_ID=${SLURM_JOB_ID:-} SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-}" 1>&2

# Load secrets (optional)
if [[ -f "$HOME/.secrets.sh" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/.secrets.sh"
fi

# Avoid nounset breaking cluster init scripts
set +u
set +e
source ~/.bashrc
rc_bashrc=$?
conda activate torchenv
rc_conda=$?
set -e
if [[ $rc_bashrc -ne 0 ]]; then
  echo "ERROR: source ~/.bashrc failed (rc=$rc_bashrc)." >&2
  exit $rc_bashrc
fi
if [[ $rc_conda -ne 0 ]]; then
  echo "ERROR: conda activate torchenv failed (rc=$rc_conda)." >&2
  exit $rc_conda
fi

export PYTHONUNBUFFERED=1

# ---- Data + output wiring ----
# Point these to directories that contain *.npy HE patches.
# Default uses your existing VSlite dataset layout.
VS_DATAROOT="${VS_DATAROOT:-$PWD/dataset/data_1024}"
export SWINGAN_TRAIN_DATA_DIR="${SWINGAN_TRAIN_DATA_DIR:-$VS_DATAROOT/he}"
export SWINGAN_VAL_DATA_DIR="${SWINGAN_VAL_DATA_DIR:-$VS_DATAROOT/he}"

RUN_NAME="${SWINGAN_RUN_NAME:-vs_he_npy_$(date +%Y%m%d_%H%M%S)}"
export SWINGAN_OUTPUT_DIR="${SWINGAN_OUTPUT_DIR:-$PWD/results/swingan/${RUN_NAME}}"

# Use the first visible GPU by default
export SWINGAN_GPU_ID="${SWINGAN_GPU_ID:-0}"

# Workers: match allocation by default
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$SLURM_CPUS_PER_TASK}"

CONFIG_PATH="${SWINGAN_CONFIG:-$PWD/SwinGAN/config_vs.yaml}"

echo "$(date -Is) swingan_train_job: train_dir=$SWINGAN_TRAIN_DATA_DIR val_dir=$SWINGAN_VAL_DATA_DIR output=$SWINGAN_OUTPUT_DIR config=$CONFIG_PATH" >&2

cd "$PWD/SwinGAN"
python train.py --config "$CONFIG_PATH"

echo "$(date -Is) swingan_train_job: finished"

