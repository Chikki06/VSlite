#!/bin/bash
#SBATCH --job-name=rr_inf
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
INFER_SCRIPT="$VSLITE_ROOT/Restormer/infer_wsi_png.py"
INPUT_PARENT="$VSLITE_ROOT/input/IR"

RESTORMER_MODEL="${RESTORMER_MODEL:-1}"
RESTORMER_VARIANT="${RESTORMER_VARIANT:-attention}"
RESTORMER_RESULTS_ROOT="$VSLITE_ROOT/results"

if [[ "$RESTORMER_VARIANT" == "attention" ]]; then
  CONFIG_FILE="$VSLITE_ROOT/Restormer/config_restormer_perceptual.yaml"
  CKPT_DIR="restormer_attn"
else
  CONFIG_FILE="$VSLITE_ROOT/Restormer/config_restormer.yaml"
  CKPT_DIR="restormer_base"
fi

if [[ "$RESTORMER_MODEL" == "1" ]]; then
  CKPT_DIR="${CKPT_DIR}_13"
  model_tag="13ch"
else
  CKPT_DIR="${CKPT_DIR}_10"
  model_tag="10ch"
fi

RESTORMER_CKPT="$RESTORMER_RESULTS_ROOT/$CKPT_DIR/ckpt_latest.pth"
OUT_DIR="$VSLITE_ROOT/results/restormer_wsi_png/${RESTORMER_VARIANT}_${model_tag}_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUT_DIR"

python "$INFER_SCRIPT" \
  --ckpt "$RESTORMER_CKPT" \
  --config "$CONFIG_FILE" \
  --model "$RESTORMER_MODEL" \
  --output_dir "$OUT_DIR" \
  --parent_folder "$INPUT_PARENT"
