#!/bin/bash
#SBATCH --job-name=sg_b_13_tr
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH -A kf4-ic
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=71:00:00
#SBATCH --chdir=/projects/illinois/eng/bioe/rxb/akshat/VSlite/SwinGAN
#SBATCH --output=/u/akshahi2/logs/%x/%j.out
#SBATCH --error=/u/akshahi2/logs/%x/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akshahi2@illinois.edu

source ~/.bashrc
conda activate torchenv

export PYTHONUNBUFFERED=1
export VSLITE_DIR="/projects/illinois/eng/bioe/rxb/akshat/VSlite"
export SWINGAN_DATAROOT="${VSLITE_DIR}/dataset/data_frozen"
export SWINGAN_GPU_ID="0"
export OMP_NUM_THREADS="4"
export MKL_NUM_THREADS="4"

export SWINGAN_INPUT_MODE="irvis13"
export SWINGAN_OUTPUT_DIR="${VSLITE_DIR}/results/swingan_i2i/b13"
CONFIG_PATH="${VSLITE_DIR}/SwinGAN/config_i2i_baseline.yaml"

python train_i2i.py --config "$CONFIG_PATH"

python ${VSLITE_DIR}/util/plot_loss_curves.py \
  --log_dir "$SWINGAN_OUTPUT_DIR" --out_dir "$SWINGAN_OUTPUT_DIR/curves" || true
