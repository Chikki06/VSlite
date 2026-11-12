#!/bin/bash
# CISLNet CBAM + input-channel interpretability on a small held-out sample set.
# Example: RUN=c13_ps1024 NUM_SAMPLES=8 sbatch interpret.sh
#SBATCH --job-name=cn_cbam
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --exclude=ccc0388
#SBATCH -A kf4-ic
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --chdir=/projects/illinois/eng/bioe/rxb/akshat/VSlite
#SBATCH --output=/u/akshahi2/logs/%x/%j.out
#SBATCH --error=/u/akshahi2/logs/%x/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akshahi2@illinois.edu

set -euo pipefail
: "${VSLITE_ROOT:=/projects/illinois/eng/bioe/rxb/akshat/VSlite}"
# shellcheck source=../../_lib/slurm_resolve.sh
source "$VSLITE_ROOT/jobs/_lib/slurm_resolve.sh"
VSLITE_ROOT="$(slurm_resolve_include_root jobs/_lib/vslite_env.sh "${BASH_SOURCE[0]:-$0}")" || exit 1
# shellcheck source=../../_lib/vslite_env.sh
source "$VSLITE_ROOT/jobs/_lib/vslite_env.sh"

NUM_CHANNELS="${NUM_CHANNELS:-13}"
RUN="${RUN:-c13_ps1024}"
NAME="${NAME:-$RUN}"
RESULTS_DIR="${RESULTS_DIR:-$VSLITE_ROOT/results/cislnet}"
NUM_SAMPLES="${NUM_SAMPLES:-8}"
WHICH_EPOCH="${WHICH_EPOCH:-latest}"
METHODS="${METHODS:-cbam,occlusion}"

IR_INDICES="${IR_CHANNEL_INDICES:-}"
case "$NUM_CHANNELS" in
  3)  IR_INDICES="${IR_INDICES:-0,8,9}"; NUM_IR=3 ;;
  6)  NUM_IR=6 ;;
  10) NUM_IR=10 ;;
  13) NUM_IR=10 ;;
  *)  echo "Unsupported NUM_CHANNELS=$NUM_CHANNELS"; exit 1 ;;
esac

EXTRA=()
[[ -n "$IR_INDICES" ]] && EXTRA+=(--ir_channel_indices "$IR_INDICES")

python "$VSLITE_ROOT/scripts/visualize_cislnet_cbam.py" \
  --name "$NAME" \
  --checkpoints_dir "$RESULTS_DIR" \
  --results_dir "$RESULTS_DIR" \
  --dataroot "$DATAROOT" \
  --input_nc "$NUM_CHANNELS" \
  --num_ir_channels "$NUM_IR" \
  --which_epoch "$WHICH_EPOCH" \
  --gpu_ids 0 \
  --num_samples "$NUM_SAMPLES" \
  --methods "$METHODS" \
  "${EXTRA[@]}"
