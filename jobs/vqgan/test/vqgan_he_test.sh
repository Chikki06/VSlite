#!/bin/bash
#SBATCH --job-name=vq_he_te
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH -A kf4-ic
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --chdir=/projects/illinois/eng/bioe/rxb/akshat/VSlite
#SBATCH --output=/u/akshahi2/logs/%x/%j.out
#SBATCH --error=/u/akshahi2/logs/%x/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akshahi2@illinois.edu

source ~/.bashrc
conda activate torchenv

export PYTHONUNBUFFERED=1
export WANDB_MODE="disabled"

DATAROOT="/projects/illinois/eng/bioe/rxb/akshat/VSlite/dataset/data_frozen"
RESULTS_DIR="/projects/illinois/eng/bioe/rxb/akshat/VSlite/results/vqgan_vs"
NAME="${VQGAN_NAME:-vqgan_he}"
CKPT="${VQGAN_CKPT:-vqgan_epoch_049.pt}"
NUM_TEST="${NUM_TEST:-75000}"
FID_MAX_ITEMS="${VSLITE_FID_MAX_ITEMS:-75000}"

python scripts/test_vqgan_vs.py \
  --name "$NAME" \
  --dataroot "$DATAROOT" \
  --results_dir "$RESULTS_DIR" \
  --ckpt "$CKPT" \
  --input_nc 13 \
  --num_ir_channels 10 \
  --num_test "$NUM_TEST" \
  --split "test" \
  --fid_max_items "$FID_MAX_ITEMS" \
  --fid_batch_size "${VSLITE_FID_BATCH_SIZE:-16}" \
  --gpu 0

python scripts/test_vqgan_vs.py \
  --name "$NAME" \
  --dataroot "$DATAROOT" \
  --results_dir "$RESULTS_DIR" \
  --ckpt "$CKPT" \
  --input_nc 13 \
  --num_ir_channels 10 \
  --num_test "$NUM_TEST" \
  --split "train" \
  --fid_max_items "$FID_MAX_ITEMS" \
  --fid_batch_size "${VSLITE_FID_BATCH_SIZE:-16}" \
  --gpu 0

