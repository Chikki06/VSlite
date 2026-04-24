#!/bin/bash
#SBATCH --job-name=rr_a_10_tr
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH -A kf4-ic
#SBATCH --mem=40G
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --chdir=/projects/illinois/eng/bioe/rxb/akshat/VSlite/Restormer
#SBATCH --output=/u/akshahi2/logs/%x/%j.out
#SBATCH --error=/u/akshahi2/logs/%x/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akshahi2@illinois.edu

source ~/.bashrc
conda activate torchenv

export PYTHONUNBUFFERED=1
export VSLITE_DIR="/projects/illinois/eng/bioe/rxb/akshat/VSlite"
export RESTORMER_DIR="${VSLITE_DIR}/Restromer"
export RESTORMER_DATAROOT="${VSLITE_DIR}/dataset/data_frozen"
export RESTORMER_GPU_ID="0"
export OMP_NUM_THREADS="4"
export MKL_NUM_THREADS="4"

export RESTORMER_INPUT_MODE="ir10"
export RESTORMER_OUTPUT_DIR="${VSLITE_DIR}/results/restormer_i2i/a10"
CONFIG_PATH="${RESTORMER_DIR}/config_i2i_restormer_perceptual.yaml"

python train_i2i_restormer.py --config "$CONFIG_PATH"

python ${VSLITE_DIR}/util/plot_loss_curves.py \
  --log_dir "$RESTORMER_OUTPUT_DIR" --out_dir "$RESTORMER_OUTPUT_DIR/curves" || true
