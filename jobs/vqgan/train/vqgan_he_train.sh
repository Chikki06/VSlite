#!/bin/bash
#SBATCH --job-name=vq_he_tr
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH -A kf4-ic
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
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
RESULTS_DIR="/projects/illinois/eng/bioe/rxb/akshat/VSlite/results/vqgan_vs"
NAME="${VQGAN_NAME:-vqgan_he}"

python scripts/train_vqgan_vs.py \
  --name "$NAME" \
  --dataroot "$DATAROOT" \
  --results_dir "$RESULTS_DIR" \
  --input_nc 13 \
  --num_ir_channels 10 \
  --image_size 256 \
  --batch_size "${VQGAN_BS:-6}" \
  --epochs "${VQGAN_EPOCHS:-50}" \
  --learning_rate "${VQGAN_LR:-2.25e-5}" \
  --gpu 0

