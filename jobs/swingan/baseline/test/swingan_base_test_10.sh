#!/bin/bash
#SBATCH --job-name=sg_b_10_te
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH -A kf4-ic
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --chdir=/projects/illinois/eng/bioe/rxb/akshat/VSlite/SwinGAN
#SBATCH --output=/u/akshahi2/logs/%x/%j.out
#SBATCH --error=/u/akshahi2/logs/%x/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akshahi2@illinois.edu

source ~/.bashrc
conda activate torchenv

export PYTHONUNBUFFERED=1
export SWINGAN_INPUT_MODE="ir10"

VSLITE_DIR="/projects/illinois/eng/bioe/rxb/akshat/VSlite"
SWINGAN_RUN_TAG="b10"
SWINGAN_OUTPUT_DIR="${SWINGAN_OUTPUT_DIR:-${VSLITE_DIR}/results/swingan_i2i/${SWINGAN_RUN_TAG}}"
CONFIG_PATH="${VSLITE_DIR}/SwinGAN/config_i2i_baseline.yaml"

CKPT_PATH="${SWINGAN_CKPT_PATH:-${SWINGAN_OUTPUT_DIR}/ckpt_latest.pth}"
SWINGAN_TEST_SPLIT="test"
NUM_TEST="${NUM_TEST:-75000}"
WSI_PARENT="${VSLITE_DIR}/input/IR"
TEST_ROOT="${VSLITE_DIR}/results/swingan_i2i/test/${SWINGAN_RUN_TAG}"
PATCH_OUT_DIR="${TEST_ROOT}/patch_eval_test"
PRED_WSI_DIR="${TEST_ROOT}/wsi_pred"

mkdir -p "$PRED_WSI_DIR"

SKIP_WSI="${SKIP_WSI:-1}"
if [[ "$SKIP_WSI" != "1" ]]; then
  python infer_wsi_png.py \
    --ckpt "$CKPT_PATH" \
    --config "$CONFIG_PATH" \
    --model 1 \
    --parent_folder "$WSI_PARENT" \
    --output_dir "$PRED_WSI_DIR"

  python "${VSLITE_DIR}/util/wsi_tile_metrics.py" \
    --pred_wsi_dir "$PRED_WSI_DIR" \
    --he_dir "${VSLITE_DIR}/dataset/data_frozen/he" \
    --suffix "10ch" \
    --max_items "$NUM_TEST"
else
  echo "SKIP_WSI=1: skipping WSI inference + WSI metrics."
fi

# Patch eval + metrics on TEST split (OOD)
python test_i2i.py \
  --config "$CONFIG_PATH" \
  --ckpt "$CKPT_PATH" \
  --num_samples 20 \
  --split "$SWINGAN_TEST_SPLIT" \
  --out_dir "$PATCH_OUT_DIR" \
  --fid_max_items "${VSLITE_FID_MAX_ITEMS:-75000}" \
  --fid_batch_size "${VSLITE_FID_BATCH_SIZE:-16}" \
  --save_every "${PATCH_SAVE_EVERY:-0}" \
  --save_palette_every "${PATCH_PALETTE_EVERY:-0}"

# Patch eval + metrics on TRAIN split (in-distribution)
python test_i2i.py \
  --config "$CONFIG_PATH" \
  --ckpt "$CKPT_PATH" \
  --num_samples 20 \
  --split "train" \
  --out_dir "${TEST_ROOT}/patch_eval_train" \
  --fid_max_items "${VSLITE_FID_MAX_ITEMS:-75000}" \
  --fid_batch_size "${VSLITE_FID_BATCH_SIZE:-16}" \
  --save_every "${PATCH_SAVE_EVERY:-0}" \
  --save_palette_every "${PATCH_PALETTE_EVERY:-0}"
