#!/bin/bash
#SBATCH --job-name=b_wsi_10
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
CHECKPOINTS_DIR="${VSLITE_ROOT}/results/baseline"
RESULTS_DIR="${VSLITE_ROOT}/results/baseline"
NAME="vs10"

OUT_DIR="${RESULTS_DIR}/test/${NAME}/wsi_pred"
mkdir -p "$OUT_DIR"

python "${VSLITE_ROOT}/scripts/infer_wsi_png_baseline.py" \
  --name "$NAME" \
  --checkpoints_dir "$CHECKPOINTS_DIR" \
  --which_epoch "latest" \
  --results_dir "$RESULTS_DIR" \
  --gpu_ids "0" \
  --dataset_mode VS \
  --model contrastive_pix2pix \
  --input_nc 10 \
  --num_ir_channels 10 \
  --parent_folder "$WSI_PARENT" \
  --output_dir "$OUT_DIR" \
  --tile_size "${WSI_TILE_SIZE:-1024}"
