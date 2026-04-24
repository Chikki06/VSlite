#!/bin/bash
#SBATCH --job-name=nf_b_10_te
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH -A kf4-ic
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --chdir=/projects/illinois/eng/bioe/rxb/akshat/VSlite/NAFNet
#SBATCH --output=/u/akshahi2/logs/%x/%j.out
#SBATCH --error=/u/akshahi2/logs/%x/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akshahi2@illinois.edu

source ~/.bashrc
conda activate torchenv

source /u/akshahi2/jobs/nafnet/nafnet_inc.sh

export NAFNET_OUTPUT_DIR="${VSLITE_DIR}/results/nafnet_i2i/b10"
export NAFNET_INPUT_MODE="ir10"
CONFIG_FILE="${NAFNET_DIR}/config_i2i_nafnet_base.yaml"

CKPT_PATH="${NAFNET_CKPT_PATH:-${NAFNET_OUTPUT_DIR}/ckpt_latest.pth}"
WSI_PARENT="${VSLITE_DIR}/input/IR"
TEST_ROOT="${VSLITE_DIR}/results/nafnet_i2i/test/nf_b_10"
PRED_WSI_DIR="${TEST_ROOT}/wsi_pred"
NUM_TEST="${NUM_TEST:-50000}"

mkdir -p "$PRED_WSI_DIR"

SKIP_WSI="${SKIP_WSI:-1}"
if [[ "$SKIP_WSI" != "1" ]]; then
  python infer_wsi_png.py \
    --ckpt "$CKPT_PATH" \
    --config "$CONFIG_FILE" \
    --model 0 \
    --parent_folder "$WSI_PARENT" \
    --output_dir "$PRED_WSI_DIR"

  python "${VSLITE_DIR}/util/wsi_tile_metrics.py" \
    --pred_wsi_dir "$PRED_WSI_DIR" \
    --he_dir "${NAFNET_DATAROOT}/he" \
    --suffix "10ch" \
    --max_items "$NUM_TEST"
else
  echo "SKIP_WSI=1: skipping WSI inference + WSI metrics."
fi

python test_i2i_nafnet.py \
  --config "${NAFNET_DIR}/config_i2i_nafnet_base.yaml" \
  --ckpt "$CKPT_PATH" \
  --split "test" \
  --num_samples 20 \
  --fid_max_items "${VSLITE_FID_MAX_ITEMS:-50000}" \
  --fid_batch_size "${VSLITE_FID_BATCH_SIZE:-16}" \
  --save_every "${PATCH_SAVE_EVERY:-0}" \
  --save_palette_every "${PATCH_PALETTE_EVERY:-0}" \
  --out_dir "${TEST_ROOT}/patch_eval"

python test_i2i_nafnet.py \
  --config "${NAFNET_DIR}/config_i2i_nafnet_base.yaml" \
  --ckpt "$CKPT_PATH" \
  --split "train" \
  --num_samples 20 \
  --fid_max_items "${VSLITE_FID_MAX_ITEMS:-50000}" \
  --fid_batch_size "${VSLITE_FID_BATCH_SIZE:-16}" \
  --save_every "${PATCH_SAVE_EVERY:-0}" \
  --save_palette_every "${PATCH_PALETTE_EVERY:-0}" \
  --out_dir "${TEST_ROOT}/patch_eval_train"
