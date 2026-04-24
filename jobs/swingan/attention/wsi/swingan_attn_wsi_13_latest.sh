#!/bin/bash
#SBATCH --job-name=sg_a_13_wsi
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

VSLITE_DIR="/projects/illinois/eng/bioe/rxb/akshat/VSlite"
INPUT_PARENT="${VSLITE_DIR}/input/IR"
OUT_DIR="${VSLITE_DIR}/tests/swingan_i2i/a13/wsi_latest"

CKPT_PATH="${SWINGAN_CKPT_PATH:-${VSLITE_DIR}/results/swingan_i2i/a13/ckpt_latest.pth}"
CONFIG_PATH="${VSLITE_DIR}/SwinGAN/config_i2i_attention.yaml"

python infer_wsi_png.py \
  --ckpt "$CKPT_PATH" \
  --config "$CONFIG_PATH" \
  --model 1 \
  --parent_folder "$INPUT_PARENT" \
  --output_dir "$OUT_DIR"

