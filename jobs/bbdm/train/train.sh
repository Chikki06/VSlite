#!/bin/bash
# Parameterized BBDM/LBBDM training.
# Example: NUM_CHANNELS=13 RUN=b13 sbatch train.sh
#SBATCH --job-name=bbdm_tr
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH -A kf4-ic
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
#SBATCH --chdir=/projects/illinois/eng/bioe/rxb/akshat/VSlite
#SBATCH --output=/u/akshahi2/logs/%x/%j.out
#SBATCH --error=/u/akshahi2/logs/%x/%j.err

set -euo pipefail
: "${VSLITE_ROOT:=/projects/illinois/eng/bioe/rxb/akshat/VSlite}"
# shellcheck source=../../_lib/slurm_resolve.sh
source "$VSLITE_ROOT/jobs/_lib/slurm_resolve.sh"
VSLITE_ROOT="$(slurm_resolve_include_root jobs/_lib/vslite_env.sh "${BASH_SOURCE[0]:-$0}")" || exit 1
# shellcheck source=../../_lib/vslite_env.sh
source "$VSLITE_ROOT/jobs/_lib/vslite_env.sh"

RUN="${RUN:-b${NUM_CHANNELS:-13}}"
NUM_CHANNELS="${NUM_CHANNELS:-13}"
CONFIG="${BBDM_CONFIG:-$VSLITE_ROOT/diffusion_vs/configs/bbdm_vs/lbbdm_vs${NUM_CHANNELS}_vqganpytorch_convnext.yaml}"
RESULT_PATH="${BBDM_RESULT_PATH:-$VSLITE_ROOT/results/bbdm/$RUN}"

export NUM_CHANNELS PATCH_SIZE="${PATCH_SIZE:-512}" BATCH_SIZE="${BATCH_SIZE:-8}" EPOCHS="${EPOCHS:-50}"

python "$VSLITE_ROOT/scripts/train_bbdm.py" \
  --config "$CONFIG" \
  --result_path "$RESULT_PATH" \
  --gpu_ids "${GPU_IDS:-0}"
