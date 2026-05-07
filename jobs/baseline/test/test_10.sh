#!/bin/bash
#SBATCH --job-name=b_10_te
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH -A kf4-ic
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=20:00:00
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
RESULTS_DIR="/projects/illinois/eng/bioe/rxb/akshat/VSlite/results/baseline"
WHICH_EPOCH="latest"
# Default to 75k patches (override: NUM_TEST=100 for quick debug)
NUM_TEST="${NUM_TEST:-75000}"
GPU_ID="0"
NAME="vs10"
FID_MAX_ITEMS="${VSLITE_FID_MAX_ITEMS:-75000}"

python scripts/vs_infer_and_metrics.py \
  --name "$NAME" \
  --checkpoints_dir /projects/illinois/eng/bioe/rxb/akshat/VSlite/results/baseline \
  --dataset_mode VS \
  --model contrastive_pix2pix \
  --dataroot "$DATAROOT" \
  --input_nc 10 \
  --num_ir_channels 10 \
  --num_test "$NUM_TEST" \
  --which_epoch "$WHICH_EPOCH" \
  --results_dir "$RESULTS_DIR" \
  --gpu_ids "$GPU_ID" \
  --fid_max_items "$FID_MAX_ITEMS" \
  --fid_batch_size "${VSLITE_FID_BATCH_SIZE:-16}" \
  --save_every "${PATCH_SAVE_EVERY:-0}" \
  --split "test"

python scripts/vs_infer_and_metrics.py \
  --name "$NAME" \
  --checkpoints_dir /projects/illinois/eng/bioe/rxb/akshat/VSlite/results/baseline \
  --dataset_mode VS \
  --model contrastive_pix2pix \
  --dataroot "$DATAROOT" \
  --input_nc 10 \
  --num_ir_channels 10 \
  --num_test "$NUM_TEST" \
  --which_epoch "$WHICH_EPOCH" \
  --results_dir "$RESULTS_DIR" \
  --gpu_ids "$GPU_ID" \
  --fid_max_items "$FID_MAX_ITEMS" \
  --fid_batch_size "${VSLITE_FID_BATCH_SIZE:-16}" \
  --save_every "${PATCH_SAVE_EVERY:-0}" \
  --split "train"
