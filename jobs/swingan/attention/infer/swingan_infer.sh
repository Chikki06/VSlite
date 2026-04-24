#!/bin/bash
#SBATCH --job-name=sg_inf
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH -A kf4-ic
#SBATCH --mem=80G
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --chdir=/projects/illinois/eng/bioe/rxb/akshat/VSlite
#SBATCH --output=/u/akshahi2/logs/%x/%j.out
#SBATCH --error=/u/akshahi2/logs/%x/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akshahi2@illinois.edu

source ~/.bashrc
conda activate torchenv

export PYTHONUNBUFFERED=1
VSLITE_ROOT="/projects/illinois/eng/bioe/rxb/akshat/VSlite"
SWINGAN_INFER_SCRIPT="$VSLITE_ROOT/SwinGAN/infer_wsi_png.py"
INPUT_PARENT="$VSLITE_ROOT/input/IR"

SWINGAN_MODEL="${SWINGAN_MODEL:-1}"
SWINGAN_VARIANT="${SWINGAN_VARIANT:-attention}"
SWINGAN_RESULTS_ROOT="$VSLITE_ROOT/results/swingan_i2i"

if [[ "$SWINGAN_VARIANT" == "attention" ]]; then
  SWINGAN_CONFIG="$VSLITE_ROOT/SwinGAN/config_i2i_attention.yaml"
  CKPT_DIR="a"
else
  SWINGAN_CONFIG="$VSLITE_ROOT/SwinGAN/config_i2i_baseline.yaml"
  CKPT_DIR="b"
fi

if [[ "$SWINGAN_MODEL" == "1" ]]; then
  CKPT_DIR="${CKPT_DIR}13"
  model_tag="13ch"
else
  CKPT_DIR="${CKPT_DIR}10"
  model_tag="10ch"
fi

SWINGAN_CKPT="$SWINGAN_RESULTS_ROOT/$CKPT_DIR/ckpt_latest.pth"
OUT_DIR="$VSLITE_ROOT/results/swingan_wsi_png/${SWINGAN_VARIANT}_${model_tag}_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUT_DIR"

python "$SWINGAN_INFER_SCRIPT" \
  --ckpt "$SWINGAN_CKPT" \
  --config "$SWINGAN_CONFIG" \
  --model "$SWINGAN_MODEL" \
  --output_dir "$OUT_DIR" \
  --parent_folder "$INPUT_PARENT"
