#!/bin/bash
#SBATCH --job-name=cn_10_13_tr
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH -A kf4-ic
#SBATCH --mem-per-gpu=20G
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --chdir=/projects/illinois/eng/bioe/rxb/akshat/VSlite
#SBATCH --output=/u/akshahi2/logs/%x/%j.out
#SBATCH --error=/u/akshahi2/logs/%x/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akshahi2@illinois.edu

# Sequential 10ch then 13ch CISLNet training.
set -euo pipefail
: "${VSLITE_ROOT:=/projects/illinois/eng/bioe/rxb/akshat/VSlite}"
# shellcheck source=../../_lib/slurm_resolve.sh
source "$VSLITE_ROOT/jobs/_lib/slurm_resolve.sh"
VSLITE_ROOT="$(slurm_resolve_include_root jobs/cislnet/train/vs_train_common.sh "${BASH_SOURCE[0]:-$0}")" || exit 1
# shellcheck source=vs_train_common.sh
source "$VSLITE_ROOT/jobs/cislnet/train/vs_train_common.sh"

NAME10="${VS_NAME_10CH:-vs10}"
NAME13="${VS_NAME_13CH:-vs13}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-$VSLITE_ROOT/results/cislnet}"

export UNIFIED_EVAL_MAX_BATCHES_TRAIN="${UNIFIED_EVAL_MAX_BATCHES_TRAIN:-20}"
export UNIFIED_EVAL_MAX_BATCHES_VAL="${UNIFIED_EVAL_MAX_BATCHES_VAL:-50}"
export UNIFIED_EVAL_MAX_BATCHES_TEST="${UNIFIED_EVAL_MAX_BATCHES_TEST:-50}"

python "$VSLITE_ROOT/scripts/train.py" "${OPTS[@]}" "${TCGA_OPT[@]}" \
  --continue_train "${CONTINUE_10CH:-true}" --name "$NAME10" \
  --checkpoints_dir "$CHECKPOINTS_DIR" --input_nc 10 --num_ir_channels 10 \
  --total_epochs "${EPOCHS:-30}"

python "$VSLITE_ROOT/scripts/train.py" "${OPTS[@]}" "${TCGA_OPT[@]}" \
  --continue_train "${CONTINUE_13CH:-true}" --name "$NAME13" \
  --checkpoints_dir "$CHECKPOINTS_DIR" --input_nc 13 --num_ir_channels 10 \
  --total_epochs "${EPOCHS:-30}"
