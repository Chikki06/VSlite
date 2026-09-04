#!/bin/bash
# Parameterized Restormer training. Example: RUN=a13 NUM_CHANNELS=13 PATCH_SIZE=512 sbatch train.sh
#SBATCH --job-name=rr_train
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH -A kf4-ic
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=71:00:00
#SBATCH --chdir=/projects/illinois/eng/bioe/rxb/akshat/VSlite/Restormer
#SBATCH --output=/u/akshahi2/logs/%x/%j.out
#SBATCH --error=/u/akshahi2/logs/%x/%j.err

set -euo pipefail
: "${VSLITE_ROOT:=/projects/illinois/eng/bioe/rxb/akshat/VSlite}"
# shellcheck source=../../_lib/slurm_resolve.sh
source "$VSLITE_ROOT/jobs/_lib/slurm_resolve.sh"
VSLITE_ROOT="$(slurm_resolve_include_root jobs/_lib/vslite_env.sh "${BASH_SOURCE[0]:-$0}")" || exit 1
# shellcheck source=../../_lib/vslite_env.sh
source "$VSLITE_ROOT/jobs/_lib/vslite_env.sh"

RUN="${RUN:-a13}"
NUM_CHANNELS="${NUM_CHANNELS:-13}"
PATCH_SIZE="${PATCH_SIZE:-512}"
BATCH_SIZE="${BATCH_SIZE:-3}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
EPOCHS="${EPOCHS:-50}"
VARIANT="${VARIANT:-attention}"

export RESTORMER_DATAROOT="${RESTORMER_DATAROOT:-$DATAROOT}"
export RESTORMER_OUTPUT_DIR="${OUTPUT_DIR:-$VSLITE_ROOT/results/restormer/$RUN}"
export RESTORMER_BATCH_SIZE="$BATCH_SIZE"
export RESTORMER_GRAD_ACCUM_STEPS="$GRAD_ACCUM_STEPS"
export PATCH_SIZE IMG_SIZE="$PATCH_SIZE"
export EPOCHS
export NUM_CHANNELS
export EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-0}"

if [[ "$VARIANT" == "base" ]]; then
  CONFIG="${RESTORMER_CONFIG:-config_restormer.yaml}"
  export RESTORMER_INPUT_MODE="ir${NUM_CHANNELS}"
  [[ "$NUM_CHANNELS" == "13" ]] && export RESTORMER_INPUT_MODE="irvis13"
else
  CONFIG="${RESTORMER_CONFIG:-config_restormer_perceptual.yaml}"
  export RESTORMER_INPUT_MODE="ir${NUM_CHANNELS}"
  [[ "$NUM_CHANNELS" == "13" ]] && export RESTORMER_INPUT_MODE="irvis13"
fi

python train.py --config "$CONFIG"

UNIFIED="$VSLITE_ROOT/results/unified/restormer/$RUN"
python "$VSLITE_ROOT/util/plot_loss_curves.py" --log_dir "$UNIFIED" --out_dir "$RESTORMER_OUTPUT_DIR/curves" || true
