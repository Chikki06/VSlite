#!/bin/bash
# Parameterized CISLNet patch eval + FID/PSNR/SSIM.
# Example: NUM_CHANNELS=13 RUN=c13 sbatch test.sh
#SBATCH --job-name=cn_te
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --exclude=ccc0388,ccc0390
#SBATCH -A kf4-ic
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --chdir=/projects/illinois/eng/bioe/rxb/akshat/VSlite
#SBATCH --output=/u/akshahi2/logs/%x/%j.out
#SBATCH --error=/u/akshahi2/logs/%x/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akshahi2@illinois.edu

set -euo pipefail
mkdir -p "/u/akshahi2/logs/${SLURM_JOB_NAME:-cn_te}"
VSLITE_ROOT="/projects/illinois/eng/bioe/rxb/akshat/VSlite"
cd "$VSLITE_ROOT"
# shellcheck source=../../_lib/vslite_env.sh
source "$VSLITE_ROOT/jobs/_lib/vslite_env.sh"
# shellcheck source=../../_lib/cislnet_channels.sh
source "$VSLITE_ROOT/jobs/_lib/cislnet_channels.sh"

RUN="${RUN:-c${NUM_CHANNELS}}"
NAME="${NAME:-$RUN}"
RESULTS_DIR="${RESULTS_DIR:-$VSLITE_ROOT/results/cislnet}"
NUM_TEST="${NUM_TEST:-75000}"
WHICH_EPOCH="${WHICH_EPOCH:-latest}"

EXTRA=()
[[ -n "$IR_INDICES" ]] && EXTRA+=(--ir_channel_indices "$IR_INDICES")
[[ "${SAVE_TRIPLETS:-0}" == "1" ]] && EXTRA+=(--save_triplets)
if [[ -n "${REGION_EVAL_SIZE:-}" && "${REGION_EVAL_SIZE}" != "0" ]]; then
  EXTRA+=(--region_eval_size "$REGION_EVAL_SIZE")
  if [[ "${SAVE_REGION_TRIPLETS:-${SAVE_TRIPLETS:-0}}" == "1" ]]; then
    EXTRA+=(--save_region_triplets)
  fi
fi
if [[ -n "${TEST_LIST_PATH:-}" ]]; then EXTRA+=(--test_list_path "$TEST_LIST_PATH"); fi
if [[ -n "${VAL_LIST_PATH:-}" ]]; then EXTRA+=(--val_list_path "$VAL_LIST_PATH"); fi
if [[ -n "${TEST_INCLUDES_VAL:-}" ]]; then EXTRA+=(--test_includes_val "$TEST_INCLUDES_VAL"); fi
if [[ -n "${RESULTS_NAME:-}" ]]; then EXTRA+=(--results_name "$RESULTS_NAME"); fi

python scripts/vs_infer_and_metrics.py \
  --name "$NAME" \
  --checkpoints_dir "$RESULTS_DIR" \
  --dataset_mode VS \
  --model contrastive_pix2pix \
  --dataroot "$DATAROOT" \
  --input_nc "$NUM_CHANNELS" \
  --num_ir_channels "$NUM_IR" \
  --which_epoch "$WHICH_EPOCH" \
  --results_dir "$RESULTS_DIR" \
  --gpu_ids 0 \
  --num_test "$NUM_TEST" \
  --fid_max_items "${VSLITE_FID_MAX_ITEMS:-$NUM_TEST}" \
  --fid_batch_size "${VSLITE_FID_BATCH_SIZE:-64}" \
  "${EXTRA[@]}"
