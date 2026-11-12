#!/bin/bash
# Shared helpers for jobs/cislnet/experiments/* (does not alter default train/test.sh).
# shellcheck shell=bash

VSLITE_ROOT="${VSLITE_ROOT:-/projects/illinois/eng/bioe/rxb/akshat/VSlite}"
# shellcheck source=../../_lib/vslite_env.sh
source "$VSLITE_ROOT/jobs/_lib/vslite_env.sh"
# shellcheck source=../../_lib/cislnet_channels.sh
source "$VSLITE_ROOT/jobs/_lib/cislnet_channels.sh"

export CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-$VSLITE_ROOT/results/cislnet}"
export RESULTS_DIR="${RESULTS_DIR:-$VSLITE_ROOT/results/cislnet}"
PATCH_SIZE="${PATCH_SIZE:-1024}"
BATCH_SIZE="${BATCH_SIZE:-10}"
NTHREADS="${NTHREADS:-4}"
EPOCHS="${EPOCHS:-20}"
SAVE_EPOCH_FREQ="${SAVE_EPOCH_FREQ:-5}"
WHICH_EPOCH="${WHICH_EPOCH:-latest}"
NUM_TEST="${NUM_TEST:-75000}"

ATTENTION_TYPE="${ATTENTION_TYPE:-cbam}"
ENCODER_BACKBONE="${ENCODER_BACKBONE:-resnet50}"
DECODER_CONV="${DECODER_CONV:-standard}"
DECODER_WIDTH_MULT="${DECODER_WIDTH_MULT:-1.0}"
ENCODER_WIDTH_MULT="${ENCODER_WIDTH_MULT:-1.0}"
UPSAMPLE_MODE="${UPSAMPLE_MODE:-convtranspose}"
NDF="${NDF:-64}"
HE_COLOR_SPACE="${HE_COLOR_SPACE:-srgb}"

cislnet_exp_train_extra() {
  EXTRA=()
  EXTRA+=(--input_nc "$NUM_CHANNELS" --num_ir_channels "$NUM_IR")
  EXTRA+=(--image_size "$PATCH_SIZE" --batch_size "$BATCH_SIZE")
  EXTRA+=(--weight_decay "${WEIGHT_DECAY:-0}")
  EXTRA+=(--he_color_space "$HE_COLOR_SPACE")
  EXTRA+=(--attention_type "$ATTENTION_TYPE")
  EXTRA+=(--encoder_backbone "$ENCODER_BACKBONE")
  EXTRA+=(--decoder_conv "$DECODER_CONV")
  EXTRA+=(--decoder_width_mult "$DECODER_WIDTH_MULT")
  EXTRA+=(--encoder_width_mult "$ENCODER_WIDTH_MULT")
  EXTRA+=(--upsample_mode "$UPSAMPLE_MODE")
  EXTRA+=(--ndf "$NDF")
  EXTRA+=(--total_epochs "$EPOCHS")
  if [[ -n "${IR_INDICES:-}" ]]; then EXTRA+=(--ir_channel_indices "$IR_INDICES"); fi
  if [[ "${NO_AUG:-0}" == "1" ]]; then EXTRA+=(--no_aug); fi
  if [[ -n "${TRAIN_LIST_PATH:-}" ]]; then EXTRA+=(--train_list_path "$TRAIN_LIST_PATH"); fi
  if [[ -n "${TEST_LIST_PATH:-}" ]]; then EXTRA+=(--test_list_path "$TEST_LIST_PATH"); fi
  if [[ -n "${VAL_LIST_PATH:-}" ]]; then EXTRA+=(--val_list_path "$VAL_LIST_PATH"); fi
  printf '%s\n' "${EXTRA[@]}"
}

cislnet_exp_run_train() {
  local name="$1"
  CONTINUE_TRAIN="${CONTINUE_TRAIN:-false}"
  mapfile -t EXTRA < <(cislnet_exp_train_extra)
  python "$VSLITE_ROOT/scripts/train.py" \
    --dataroot "$DATAROOT" --dataset_mode VS --model contrastive_pix2pix \
    --gpu_ids 0 --nThreads "$NTHREADS" --print_freq 100 --save_epoch_freq "$SAVE_EPOCH_FREQ" \
    --save_latest_freq 30000 --continue_train "$CONTINUE_TRAIN" --name "$name" \
    --checkpoints_dir "$CHECKPOINTS_DIR" \
    "${EXTRA[@]}"
}

cislnet_exp_run_eval() {
  local name="$1"
  local track_run="${2:-$name}"
  local track_group="${3:-exp}"
  local track_notes="${4:-}"
  local results_name="${5:-$track_run}"
  EXTRA=()
  [[ -n "${IR_INDICES:-}" ]] && EXTRA+=(--ir_channel_indices "$IR_INDICES")
  [[ "${SAVE_TRIPLETS:-0}" == "1" ]] && EXTRA+=(--save_triplets)
  [[ -n "${TEST_LIST_PATH:-}" ]] && EXTRA+=(--test_list_path "$TEST_LIST_PATH")
  [[ -n "${VAL_LIST_PATH:-}" ]] && EXTRA+=(--val_list_path "$VAL_LIST_PATH")
  [[ -n "${TEST_INCLUDES_VAL:-}" ]] && EXTRA+=(--test_includes_val "$TEST_INCLUDES_VAL")
  python "$VSLITE_ROOT/scripts/cislnet_exp_eval.py" \
    --name "$name" \
    --checkpoints_dir "$RESULTS_DIR" \
    --dataset_mode VS \
    --model contrastive_pix2pix \
    --dataroot "$DATAROOT" \
    --input_nc "$NUM_CHANNELS" \
    --num_ir_channels "$NUM_IR" \
    --which_epoch "$WHICH_EPOCH" \
    --results_dir "$RESULTS_DIR" \
    --results_name "$results_name" \
    --gpu_ids 0 \
    --num_test "$NUM_TEST" \
    --fid_max_items "${VSLITE_FID_MAX_ITEMS:-$NUM_TEST}" \
    --fid_batch_size "${VSLITE_FID_BATCH_SIZE:-64}" \
    --track_run "$track_run" \
    --track_group "$track_group" \
    --track_notes "$track_notes" \
    --latency_warmup "${LATENCY_WARMUP:-10}" \
    --latency_timed "${LATENCY_TIMED:-100}" \
    "${EXTRA[@]}"
}
