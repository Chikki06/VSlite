#!/bin/bash
#SBATCH --job-name=i2_10_tr
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
RESULTS_DIR="/projects/illinois/eng/bioe/rxb/akshat/VSlite/results/i2sb_vs"
NAME="i2sb_10"

python scripts/train_i2sb_vs.py \
  --name "$NAME" \
  --dataroot "$DATAROOT" \
  --results_dir "$RESULTS_DIR" \
  --input_nc 10 \
  --num_ir_channels 10 \
  --image_size 256 \
  --batch_size 32 \
  --microbatch 2 \
  --num_itr "${I2SB_NUM_ITR:-200000}" \
  --lr "${I2SB_LR:-5e-5}" \
  --interval 1000 \
  --beta_max 0.3 \
  --ema 0.99 \
  --gpu 0

