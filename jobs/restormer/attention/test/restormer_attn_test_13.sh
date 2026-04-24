#!/bin/bash
#SBATCH --job-name=rr_a_13_te
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH -A kf4-ic
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --chdir=/projects/illinois/eng/bioe/rxb/akshat/VSlite/Restormer
#SBATCH --output=/u/akshahi2/logs/%x/%j.out
#SBATCH --error=/u/akshahi2/logs/%x/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akshahi2@illinois.edu

source ~/.bashrc
conda activate torchenv

export PYTHONUNBUFFERED=1
VSLITE_ROOT="/projects/illinois/eng/bioe/rxb/akshat/VSlite"
RESTORMER_DIR="${VSLITE_ROOT}/Restromer"
TEST_SCRIPT="${RESTORMER_DIR}/test_i2i_restormer.py"
DATAROOT="${VSLITE_ROOT}/dataset/data_frozen"
# Matches train/restormer_attn_13.sh (perceptual) and output ckpt_latest.pth
CONFIG_FILE="${RESTORMER_DIR}/config_i2i_restormer_perceptual.yaml"
export RESTORMER_OUTPUT_DIR="${VSLITE_ROOT}/results/restormer_i2i/a13"
export RESTORMER_DATAROOT="${DATAROOT}"
export RESTORMER_INPUT_MODE="irvis13"

CKPT_PATH="${RESTORMER_CKPT_PATH:-${RESTORMER_OUTPUT_DIR}/ckpt_epoch_30.pth}"
if [[ ! -f "$CKPT_PATH" ]]; then
  CKPT_PATH="${RESTORMER_OUTPUT_DIR}/ckpt_latest.pth"
fi

# WSI inference first (only skipped when SKIP_WSI=1)
WSI_PARENT="${VSLITE_ROOT}/input/IR"
TEST_ROOT="${VSLITE_ROOT}/results/restormer_i2i/test/restormer_attn_13"
PRED_WSI_DIR="${TEST_ROOT}/wsi_pred"
NUM_TEST="${NUM_TEST:-50000}"

mkdir -p "$PRED_WSI_DIR"

SKIP_WSI="${SKIP_WSI:-1}"
if [[ "$SKIP_WSI" != "1" ]]; then
  python infer_wsi_png.py \
    --ckpt "$CKPT_PATH" \
    --config "$CONFIG_FILE" \
    --model 1 \
    --parent_folder "$WSI_PARENT" \
    --output_dir "$PRED_WSI_DIR"

  python "${VSLITE_ROOT}/util/wsi_tile_metrics.py" \
    --pred_wsi_dir "$PRED_WSI_DIR" \
    --he_dir "${VSLITE_ROOT}/dataset/data_frozen/he" \
    --suffix "13ch" \
    --max_items "$NUM_TEST"
else
  echo "SKIP_WSI=1: skipping WSI inference + WSI metrics."
fi

# Patch eval + metrics on TEST split (OOD)
PATCH_OUT_DIR="${TEST_ROOT}/patch_eval"
python "$TEST_SCRIPT" \
  --config "$CONFIG_FILE" \
  --ckpt "$CKPT_PATH" \
  --split "test" \
  --num_samples 20 \
  --fid_max_items "${VSLITE_FID_MAX_ITEMS:-50000}" \
  --fid_batch_size "${VSLITE_FID_BATCH_SIZE:-16}" \
  --save_every "${PATCH_SAVE_EVERY:-0}" \
  --save_palette_every "${PATCH_PALETTE_EVERY:-0}" \
  --out_dir "$PATCH_OUT_DIR"

# Patch eval + metrics on TRAIN split (in-distribution)
python "$TEST_SCRIPT" \
  --config "$CONFIG_FILE" \
  --ckpt "$CKPT_PATH" \
  --split "train" \
  --num_samples 20 \
  --fid_max_items "${VSLITE_FID_MAX_ITEMS:-50000}" \
  --fid_batch_size "${VSLITE_FID_BATCH_SIZE:-16}" \
  --save_every "${PATCH_SAVE_EVERY:-0}" \
  --save_palette_every "${PATCH_PALETTE_EVERY:-0}" \
  --out_dir "${TEST_ROOT}/patch_eval_train"
