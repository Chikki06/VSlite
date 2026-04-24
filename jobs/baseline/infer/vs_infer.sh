#!/bin/bash
#SBATCH --job-name=b_inf
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH -A kf4-ic
#SBATCH --mem=40G
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --chdir=/projects/illinois/eng/bioe/rxb/akshat/VSlite
#SBATCH --output=/u/akshahi2/logs/%x/%j.out
#SBATCH --error=/u/akshahi2/logs/%x/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akshahi2@illinois.edu

source ~/.bashrc
conda activate torchenv

export PYTHONUNBUFFERED=1
export VSLITE_ROOT="/projects/illinois/eng/bioe/rxb/akshat/VSlite"
export LOCAL_STAGING_DIR="$HOME/vs_infer_staging"
export PROJECT_ID="${PROJECT_ID:-444}"
export SYNC_ENABLED="${SYNC_ENABLED:-true}"
export DB_UPDATE_ENABLED="${DB_UPDATE_ENABLED:-true}"
export DRY_RUN="${DRY_RUN:-false}"
export INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-1}"
export INFER_NUM_WORKERS="${INFER_NUM_WORKERS:-0}"
export SAVE_PNG_ONLY="${SAVE_PNG_ONLY:-true}"

"$VSLITE_ROOT/jobs/baseline/infer/vs_infer_publish_all.sh" "$VSLITE_ROOT/jobs/vs_infer_publish.env"
