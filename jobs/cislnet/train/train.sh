#!/bin/bash
# Parameterized CISLNet (contrastive pix2pix) training.
# Example: NUM_CHANNELS=13 RUN=c13 EPOCHS=30 sbatch train.sh
#SBATCH --job-name=cn_tr
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --exclude=ccc0388,ccc0390
#SBATCH -A kf4-ic
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --chdir=/projects/illinois/eng/bioe/rxb/akshat/VSlite
#SBATCH --output=/u/akshahi2/logs/%x/%j.out
#SBATCH --error=/u/akshahi2/logs/%x/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akshahi2@illinois.edu

set -euo pipefail
VSLITE_ROOT="/projects/illinois/eng/bioe/rxb/akshat/VSlite"
cd "$VSLITE_ROOT"
# shellcheck source=../../_lib/vslite_env.sh
source "$VSLITE_ROOT/jobs/_lib/vslite_env.sh"
# shellcheck source=../../_lib/cislnet_channels.sh
source "$VSLITE_ROOT/jobs/_lib/cislnet_channels.sh"

RUN="${RUN:-c${NUM_CHANNELS}}"
PATCH_SIZE="${PATCH_SIZE:-1024}"
BATCH_SIZE="${BATCH_SIZE:-10}"
NTHREADS="${NTHREADS:-4}"
EPOCHS="${EPOCHS:-30}"
SAVE_EPOCH_FREQ="${SAVE_EPOCH_FREQ:-5}"
# Colons from sbatch --export (commas would split the export list).
SAVE_EPOCHS="${SAVE_EPOCHS:-}"
SAVE_EPOCHS="${SAVE_EPOCHS//:/,}"
NITER="${NITER:-}"
NITER_DECAY="${NITER_DECAY:-}"

export CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-$VSLITE_ROOT/results/cislnet}"
NAME="${NAME:-$RUN}"

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

# Objective / loss plumbing (all optional — unset = unchanged defaults).
# NOTE: codebase defaults are NCE + GAN(LSGAN); pixel L1 and L2 are OFF by default.
if [[ -n "${ATTENTION_TYPE:-}" ]]; then EXTRA+=(--attention_type "$ATTENTION_TYPE"); fi
case "${USE_L1:-}" in
  1) EXTRA+=(--no_L1_loss false) ;;
  0) EXTRA+=(--no_L1_loss true) ;;
esac
case "${USE_L2:-}" in
  1) EXTRA+=(--no_L2_loss false) ;;
  0) EXTRA+=(--no_L2_loss true) ;;
esac
case "${USE_NCE:-}" in
  1) EXTRA+=(--no_contrastive_loss false) ;;
  0) EXTRA+=(--no_contrastive_loss true --lambda_NCE 0) ;;
esac
case "${NO_GAN:-}" in
  1) EXTRA+=(--no_gan_loss true) ;;
  0) EXTRA+=(--no_gan_loss false) ;;
esac
[[ -n "${LAMBDA_L1:-}" ]] && EXTRA+=(--lambda_L1 "$LAMBDA_L1")
[[ -n "${LAMBDA_L2:-}" ]] && EXTRA+=(--lambda_L2 "$LAMBDA_L2")
[[ -n "${LAMBDA_NCE:-}" && "${USE_NCE:-}" != "0" ]] && EXTRA+=(--lambda_NCE "$LAMBDA_NCE")
[[ -n "${LAMBDA_GAN:-}" ]] && EXTRA+=(--lambda_gan "$LAMBDA_GAN")
if [[ "${EARLY_STOP_PATIENCE:-0}" -gt 0 ]]; then
  EXTRA+=(--early_stop_patience "$EARLY_STOP_PATIENCE")
  EXTRA+=(--early_stop_min_delta "${EARLY_STOP_MIN_DELTA:-1e-4}")
fi
if [[ -n "$SAVE_EPOCHS" ]]; then
  EXTRA+=(--save_epochs "$SAVE_EPOCHS")
fi

CONTINUE_TRAIN="${CONTINUE_TRAIN:-false}"
# A requeue/restart (node death, or the CUDA-attach requeue in vslite_env.sh)
# must not restart from scratch: continue from the latest checkpoint when one
# exists. iter.txt holds "<epoch>\n<iters\n>" and is written with every ckpt save.
if [[ "${SLURM_RESTART_COUNT:-0}" -gt 0 && -f "$CHECKPOINTS_DIR/$NAME/iter.txt" ]]; then
  CONTINUE_TRAIN=true
fi

if [[ -n "${TRAIN_LIST_PATH:-}" ]]; then EXTRA+=(--train_list_path "$TRAIN_LIST_PATH"); fi
if [[ -n "${TEST_LIST_PATH:-}" ]]; then EXTRA+=(--test_list_path "$TEST_LIST_PATH"); fi
if [[ -n "${VAL_LIST_PATH:-}" ]]; then EXTRA+=(--val_list_path "$VAL_LIST_PATH"); fi

python scripts/train.py \
  --dataroot "$DATAROOT" --dataset_mode VS --model contrastive_pix2pix \
  --gpu_ids 0 --nThreads "$NTHREADS" --print_freq 100 --save_epoch_freq "$SAVE_EPOCH_FREQ" \
  --save_latest_freq 30000 --continue_train "$CONTINUE_TRAIN" --name "$NAME" \
  --checkpoints_dir "$CHECKPOINTS_DIR" \
  "${EXTRA[@]}"

UNIFIED_DIR="$VSLITE_ROOT/results/unified/cislnet/$NAME"
python util/plot_loss_curves.py \
  --log_dir "$UNIFIED_DIR" --out_dir "$CHECKPOINTS_DIR/$NAME/curves" || true
