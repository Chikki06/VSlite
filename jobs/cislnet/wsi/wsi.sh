#!/bin/bash
# Parameterized CISLNet WSI inference.
# Example: NUM_CHANNELS=13 RUN=c13 sbatch wsi.sh
#SBATCH --job-name=cn_wsi
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --exclude=ccc0387,ccc0388
#SBATCH -A kf4-ic
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --chdir=/projects/illinois/eng/bioe/rxb/akshat/VSlite
#SBATCH --output=/u/akshahi2/logs/%x/%j.out
#SBATCH --error=/u/akshahi2/logs/%x/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akshahi2@illinois.edu

: "${VSLITE_ROOT:=/projects/illinois/eng/bioe/rxb/akshat/VSlite}"
# shellcheck source=../../_lib/slurm_resolve.sh
source "$VSLITE_ROOT/jobs/_lib/slurm_resolve.sh"
VSLITE_ROOT="$(slurm_resolve_include_root jobs/_lib/vslite_env.sh "${BASH_SOURCE[0]:-$0}")" || exit 1
# shellcheck source=../../_lib/vslite_env.sh
source "$VSLITE_ROOT/jobs/_lib/vslite_env.sh"
set -euo pipefail

NUM_CHANNELS="${NUM_CHANNELS:-13}"
RUN="${RUN:-c${NUM_CHANNELS}}"
NAME="${NAME:-$RUN}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-$VSLITE_ROOT/results/cislnet}"
WSI_TILE_SIZE="${WSI_TILE_SIZE:-1024}"
WSI_BATCH_SIZE="${WSI_BATCH_SIZE:-1}"

IR_INDICES="${IR_CHANNEL_INDICES:-}"
case "$NUM_CHANNELS" in
  3)  IR_INDICES="${IR_INDICES:-0,8,9}"; NUM_IR=3 ;;
  6)  NUM_IR=6 ;;
  10) NUM_IR=10 ;;
  13) NUM_IR=10 ;;
  *)  echo "Unsupported NUM_CHANNELS=$NUM_CHANNELS"; exit 1 ;;
esac

OUT_DIR="$CHECKPOINTS_DIR/test/${NAME}/wsi_pred"
mkdir -p "$OUT_DIR"

EXTRA=()
EXTRA+=(--input_nc "$NUM_CHANNELS" --num_ir_channels "$NUM_IR")
[[ -n "$IR_INDICES" ]] && EXTRA+=(--ir_channel_indices "$IR_INDICES")

python "$VSLITE_ROOT/scripts/infer_wsi_png_cislnet.py" \
  --name "$NAME" \
  --checkpoints_dir "$CHECKPOINTS_DIR" \
  --parent_folder "$VSLITE_ROOT/input/IR" \
  --output_dir "$OUT_DIR" \
  --tile_size "$WSI_TILE_SIZE" \
  --batch_size "$WSI_BATCH_SIZE" \
  "${EXTRA[@]}"
