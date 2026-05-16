#!/bin/bash
#SBATCH --job-name=rr_a_13_wsi_l256
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH -A kf4-ic
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --chdir=/projects/illinois/eng/bioe/rxb/akshat/VSlite
#SBATCH --output=/u/akshahi2/logs/%x/%j.out
#SBATCH --error=/u/akshahi2/logs/%x/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akshahi2@illinois.edu

source ~/.bashrc
conda activate torchenv

export PYTHONUNBUFFERED=1

VSLITE_ROOT="/projects/illinois/eng/bioe/rxb/akshat/VSlite"
WSI_PARENT="${VSLITE_ROOT}/input/IR"

# Weights from archived 256px train; config must match img_size.
LEGACY_DIR="${VSLITE_ROOT}/results_old/restormer_i2i/a13"
CKPT_PATH="${RESTORMER_CKPT_PATH:-${LEGACY_DIR}/ckpt_latest.pth}"
CONFIG_PATH="${VSLITE_ROOT}/Restormer/config_i2i_restormer_perceptual_img256.yaml"

RUN_TAG="a13_legacy256"
OUT_DIR="${VSLITE_ROOT}/results/restormer_i2i/test/${RUN_TAG}/wsi_pred"
mkdir -p "$OUT_DIR"

TILE_ARGS=()
if [[ -n "${WSI_TILE_SIZE:-}" ]]; then
  TILE_ARGS=(--tile_size "$WSI_TILE_SIZE")
fi

python "${VSLITE_ROOT}/Restormer/infer_wsi_png.py" \
  --ckpt "$CKPT_PATH" \
  --config "$CONFIG_PATH" \
  --model 1 \
  --parent_folder "$WSI_PARENT" \
  --output_dir "$OUT_DIR" \
  "${TILE_ARGS[@]}"
