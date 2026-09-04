#!/bin/bash
# Parameterized CISLNet (contrastive pix2pix) training.
# Example: NUM_CHANNELS=13 RUN=c13 EPOCHS=30 sbatch train.sh
#SBATCH --job-name=cn_tr
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --exclude=ccc0387,ccc0388
#SBATCH -A kf4-ic
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=40:00:00
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
RUN="${RUN:-c${NUM_CHANNELS}}"
PATCH_SIZE="${PATCH_SIZE:-1024}"
BATCH_SIZE="${BATCH_SIZE:-10}"
EPOCHS="${EPOCHS:-30}"
SAVE_EPOCH_FREQ="${SAVE_EPOCH_FREQ:-5}"
# Colons from sbatch --export (commas would split the export list).
SAVE_EPOCHS="${SAVE_EPOCHS:-}"
SAVE_EPOCHS="${SAVE_EPOCHS//:/,}"
NITER="${NITER:-}"
NITER_DECAY="${NITER_DECAY:-}"

export CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-$VSLITE_ROOT/results/cislnet}"
NAME="${NAME:-$RUN}"

IR_INDICES="${IR_CHANNEL_INDICES:-}"
case "$NUM_CHANNELS" in
  3)  IR_INDICES="${IR_INDICES:-0,8,9}"; NUM_IR=3 ;;
  6)  NUM_IR=6 ;;
  10) NUM_IR=10 ;;
  13) NUM_IR=10 ;;
  *)  echo "Unsupported NUM_CHANNELS=$NUM_CHANNELS"; exit 1 ;;
esac

EXTRA=()
EXTRA+=(--input_nc "$NUM_CHANNELS" --num_ir_channels "$NUM_IR")
EXTRA+=(--image_size "$PATCH_SIZE" --batch_size "$BATCH_SIZE")
EXTRA+=(--weight_decay "${WEIGHT_DECAY:-0}")
EXTRA+=(--he_color_space "${HE_COLOR_SPACE:-srgb}")
if [[ "${NO_AUG:-0}" == "1" ]]; then EXTRA+=(--no_aug); fi
if [[ -n "$NITER" && -n "$NITER_DECAY" ]]; then
  EXTRA+=(--niter "$NITER" --niter_decay "$NITER_DECAY")
else
  EXTRA+=(--total_epochs "$EPOCHS")
fi
if [[ -n "$IR_INDICES" ]]; then EXTRA+=(--ir_channel_indices "$IR_INDICES"); fi
if [[ "${EARLY_STOP_PATIENCE:-0}" -gt 0 ]]; then
  EXTRA+=(--early_stop_patience "$EARLY_STOP_PATIENCE")
  EXTRA+=(--early_stop_min_delta "${EARLY_STOP_MIN_DELTA:-1e-4}")
fi
if [[ -n "$SAVE_EPOCHS" ]]; then
  EXTRA+=(--save_epochs "$SAVE_EPOCHS")
fi

CONTINUE_TRAIN="${CONTINUE_TRAIN:-false}"

if [[ -n "${TRAIN_LIST_PATH:-}" ]]; then EXTRA+=(--train_list_path "$TRAIN_LIST_PATH"); fi
if [[ -n "${TEST_LIST_PATH:-}" ]]; then EXTRA+=(--test_list_path "$TEST_LIST_PATH"); fi
if [[ -n "${VAL_LIST_PATH:-}" ]]; then EXTRA+=(--val_list_path "$VAL_LIST_PATH"); fi

python "$VSLITE_ROOT/scripts/train.py" \
  --dataroot "$DATAROOT" --dataset_mode VS --model contrastive_pix2pix \
  --gpu_ids 0 --nThreads 4 --print_freq 100 --save_epoch_freq "$SAVE_EPOCH_FREQ" \
  --save_latest_freq 30000 --continue_train "$CONTINUE_TRAIN" --name "$NAME" \
  --checkpoints_dir "$CHECKPOINTS_DIR" \
  "${EXTRA[@]}"

UNIFIED_DIR="$VSLITE_ROOT/results/unified/cislnet/$NAME"
python "$VSLITE_ROOT/util/plot_loss_curves.py" \
  --log_dir "$UNIFIED_DIR" --out_dir "$CHECKPOINTS_DIR/$NAME/curves" || true
