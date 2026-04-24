#!/bin/bash
#SBATCH --job-name=nf_a_10_tr
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH -A kf4-ic
#SBATCH --mem=40G
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=71:00:00
#SBATCH --chdir=/projects/illinois/eng/bioe/rxb/akshat/VSlite/NAFNet
#SBATCH --output=/u/akshahi2/logs/%x/%j.out
#SBATCH --error=/u/akshahi2/logs/%x/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akshahi2@illinois.edu

source ~/.bashrc
conda activate torchenv

source /u/akshahi2/jobs/nafnet/nafnet_inc.sh

export NAFNET_GPU_ID="0"
export NAFNET_INPUT_MODE="ir10"
export NAFNET_OUTPUT_DIR="${VSLITE_DIR}/results/nafnet_i2i/a10"
CONFIG_PATH="${NAFNET_DIR}/config_i2i_nafnet_attention.yaml"

python train_i2i_nafnet.py --config "$CONFIG_PATH"

python "${VSLITE_DIR}/util/plot_loss_curves.py" \
  --log_dir "$NAFNET_OUTPUT_DIR" --out_dir "$NAFNET_OUTPUT_DIR/curves" || true
