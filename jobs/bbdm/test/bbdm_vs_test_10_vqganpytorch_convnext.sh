#!/bin/bash
#SBATCH --job-name=bb_b_10_te
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH -A kf4-ic
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --chdir=/projects/illinois/eng/bioe/rxb/akshat/VSlite
#SBATCH --output=/u/akshahi2/logs/%x/%j.out
#SBATCH --error=/u/akshahi2/logs/%x/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akshahi2@illinois.edu

source ~/.bashrc
conda activate torchenv

export PYTHONUNBUFFERED=1
export WANDB_MODE="disabled"

CONFIG="/projects/illinois/eng/bioe/rxb/akshat/VSlite/diffusion_vs/configs/bbdm_vs/lbbdm_vs10_vqgan_pytorch_convnext.yaml"
RESULT_PATH="/projects/illinois/eng/bioe/rxb/akshat/VSlite/results/bbdm"
RUN_NAME="bbdmB_10"
NUM_TEST="${NUM_TEST:-75000}"
FID_MAX_ITEMS="${VSLITE_FID_MAX_ITEMS:-75000}"

CKPT="${BBDM_CKPT:-/projects/illinois/eng/bioe/rxb/akshat/VSlite/results/bbdm/LBBDM-vslite-vqganpytorch-convnext-10/checkpoint/latest.ckpt}"

python scripts/test_bbdm_vs.py \
  --config "$CONFIG" \
  --resume_model "$CKPT" \
  --result_path "$RESULT_PATH" \
  --run_name "$RUN_NAME" \
  --num_test "$NUM_TEST" \
  --fid_max_items "$FID_MAX_ITEMS" \
  --fid_batch_size "${VSLITE_FID_BATCH_SIZE:-16}" \
  --split "test" \
  --gpu 0

python scripts/test_bbdm_vs.py \
  --config "$CONFIG" \
  --resume_model "$CKPT" \
  --result_path "$RESULT_PATH" \
  --run_name "$RUN_NAME" \
  --num_test "$NUM_TEST" \
  --fid_max_items "$FID_MAX_ITEMS" \
  --fid_batch_size "${VSLITE_FID_BATCH_SIZE:-16}" \
  --split "train" \
  --gpu 0

