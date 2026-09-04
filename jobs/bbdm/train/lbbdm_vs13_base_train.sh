#!/bin/bash
#SBATCH --job-name=bb_a_13_tr
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH -A kf4-ic
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=71:00:00
#SBATCH --chdir=/projects/illinois/eng/bioe/rxb/akshat/VSlite
#SBATCH --output=/u/akshahi2/logs/%x/%j.out
#SBATCH --error=/u/akshahi2/logs/%x/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akshahi2@illinois.edu

source ~/.bashrc
conda activate torchenv

export PYTHONUNBUFFERED=1
export WANDB_MODE="disabled"

CONFIG="/projects/illinois/eng/bioe/rxb/akshat/VSlite/diffusion_vs/configs/bbdm_vs/lbbdm_vs13_vqmodel.yaml"
RESULT_PATH="/projects/illinois/eng/bioe/rxb/akshat/VSlite/results/unified/bbdm_vs"

python scripts/train_bbdm.py \
  --config "$CONFIG" \
  --result_path "$RESULT_PATH" \
  --gpu_ids 0

