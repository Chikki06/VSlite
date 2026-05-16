#!/bin/bash
#SBATCH --job-name=vq_wsi
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
RESULTS_DIR="${VSLITE_ROOT}/results/vqgan_vs"
NAME="${VQGAN_NAME:-vq_b}"
CKPT="${VQGAN_CKPT:-ckpt_latest.pt}"

OUT_DIR="${RESULTS_DIR}/test/${NAME}/wsi_pred"
mkdir -p "$OUT_DIR"

python "${VSLITE_ROOT}/scripts/infer_wsi_png_vqgan.py" \
  --name "$NAME" \
  --results_dir "$RESULTS_DIR" \
  --ckpt "$CKPT" \
  --image_size 512 \
  --latent_grid_size "${VQGAN_LATENT_GRID:-16}" \
  --num_codebook_vectors "${VQGAN_CODEBOOK:-4096}" \
  --parent_folder "$WSI_PARENT" \
  --output_dir "$OUT_DIR" \
  --tile_size "${WSI_TILE_SIZE:-512}" \
  --gpu 0
