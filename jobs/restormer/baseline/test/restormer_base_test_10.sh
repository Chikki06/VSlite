#!/bin/bash
#SBATCH --job-name=rr_b_10_te
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH -A kf4-ic
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --chdir=/projects/illinois/eng/bioe/rxb/akshat/VSlite
#SBATCH --output=/u/akshahi2/logs/%x/%j.out
#SBATCH --error=/u/akshahi2/logs/%x/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akshahi2@illinois.edu

source ~/.bashrc
conda activate torchenv

export PYTHONUNBUFFERED=1
export WANDB_MODE="disabled"
export VSLITE_CONVNEXT_FID_CKPT="/projects/illinois/eng/bioe/rxb/akshat/VSlite/tcga_tl_checkpoints/convnext_t_dp_spv_supcon_1e-5"
export VSLITE_INFER_BS="${VSLITE_INFER_BS:-16}"

VSLITE_ROOT="/projects/illinois/eng/bioe/rxb/akshat/VSlite"
DATAROOT="$VSLITE_ROOT/dataset/data_frozen"
CONFIG_FILE="$VSLITE_ROOT/Restormer/config_i2i_restormer.yaml"
OUTPUT_DIR="$VSLITE_ROOT/results/restormer_i2i/b10"

export RESTORMER_DATAROOT="$DATAROOT"
export RESTORMER_OUTPUT_DIR="$OUTPUT_DIR"
export RESTORMER_INPUT_MODE="ir10"

CKPT_PATH="${RESTORMER_CKPT_PATH:-${OUTPUT_DIR}/ckpt_latest.pth}"

WSI_PARENT="${VSLITE_ROOT}/input/IR"
TEST_ROOT="${VSLITE_ROOT}/results/restormer_i2i/test/b10"
PRED_WSI_DIR="${TEST_ROOT}/wsi_pred"
NUM_TEST="${NUM_TEST:-75000}"
FID_MAX_ITEMS="${VSLITE_FID_MAX_ITEMS:-$NUM_TEST}"

mkdir -p "$PRED_WSI_DIR"

SKIP_WSI="${SKIP_WSI:-1}"
if [[ "$SKIP_WSI" != "1" ]]; then
  python "${VSLITE_ROOT}/Restormer/infer_wsi_png.py" \
    --ckpt "$CKPT_PATH" \
    --config "$CONFIG_FILE" \
    --model 0 \
    --parent_folder "$WSI_PARENT" \
    --output_dir "$PRED_WSI_DIR"

  python "${VSLITE_ROOT}/util/wsi_tile_metrics.py" \
    --pred_wsi_dir "$PRED_WSI_DIR" \
    --he_dir "${VSLITE_ROOT}/dataset/data_frozen/he" \
    --suffix "10ch" \
    --max_items "$NUM_TEST"
else
  echo "SKIP_WSI=1: skipping WSI inference + WSI metrics."
fi

python "$VSLITE_ROOT/Restormer/test_i2i_restormer.py" \
  --config "$CONFIG_FILE" \
  --ckpt "$CKPT_PATH" \
  --split "test" \
  --num_samples 20 \
  --fid_max_items "$FID_MAX_ITEMS" \
  --fid_batch_size "${VSLITE_FID_BATCH_SIZE:-16}" \
  --save_every "${PATCH_SAVE_EVERY:-0}" \
  --save_palette_every "${PATCH_PALETTE_EVERY:-0}" \
  --out_dir "${TEST_ROOT}/patch_eval_test"

python "$VSLITE_ROOT/Restormer/test_i2i_restormer.py" \
  --config "$CONFIG_FILE" \
  --ckpt "$CKPT_PATH" \
  --split "val" \
  --num_samples 20 \
  --fid_max_items "$FID_MAX_ITEMS" \
  --fid_batch_size "${VSLITE_FID_BATCH_SIZE:-16}" \
  --save_every "${PATCH_SAVE_EVERY:-0}" \
  --save_palette_every "${PATCH_PALETTE_EVERY:-0}" \
  --out_dir "${TEST_ROOT}/patch_eval_val"

python "$VSLITE_ROOT/Restormer/test_i2i_restormer.py" \
  --config "$CONFIG_FILE" \
  --ckpt "$CKPT_PATH" \
  --split "train" \
  --num_samples 20 \
  --fid_max_items "$FID_MAX_ITEMS" \
  --fid_batch_size "${VSLITE_FID_BATCH_SIZE:-16}" \
  --save_every "${PATCH_SAVE_EVERY:-0}" \
  --save_palette_every "${PATCH_PALETTE_EVERY:-0}" \
  --out_dir "${TEST_ROOT}/patch_eval_train"
