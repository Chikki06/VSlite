#!/bin/bash
#SBATCH --job-name=rr_b_13_tr
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH -A kf4-ic
#SBATCH --mem=40G
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=71:00:00
#SBATCH --chdir=/projects/illinois/eng/bioe/rxb/akshat/VSlite/Restormer
#SBATCH --output=/u/akshahi2/logs/%x/%j.out
#SBATCH --error=/u/akshahi2/logs/%x/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akshahi2@illinois.edu

source ~/.bashrc
conda activate torchenv

export PYTHONUNBUFFERED=1
export VSLITE_DIR="/projects/illinois/eng/bioe/rxb/akshat/VSlite"
export RESTORMER_DIR="${VSLITE_DIR}/Restormer"
export RESTORMER_DATAROOT="${VSLITE_DIR}/dataset/data_frozen"
export RESTORMER_GPU_ID="0"
export OMP_NUM_THREADS="4"
export MKL_NUM_THREADS="4"

# Unified eval caps (controls overhead of epoch-end split evaluation).
export UNIFIED_EVAL_MAX_ITEMS_TRAIN="${UNIFIED_EVAL_MAX_ITEMS_TRAIN:-512}"
export UNIFIED_EVAL_MAX_ITEMS_VAL="${UNIFIED_EVAL_MAX_ITEMS_VAL:-512}"
export UNIFIED_EVAL_MAX_ITEMS_TEST="${UNIFIED_EVAL_MAX_ITEMS_TEST:-512}"

export RESTORMER_INPUT_MODE="irvis13"
export RESTORMER_OUTPUT_DIR="${VSLITE_DIR}/results/restormer_i2i/b13"
# B_eff=B_micro*K. ~51/80 GiB @ B=2 → B=3 ~70 GiB est.; B=4 needs ~94 GiB+ (set env). Override as needed.
export RESTORMER_BATCH_SIZE="${RESTORMER_BATCH_SIZE:-3}"
export RESTORMER_GRAD_ACCUM_STEPS="${RESTORMER_GRAD_ACCUM_STEPS:-8}"
CONFIG_PATH="${RESTORMER_DIR}/config_i2i_restormer.yaml"

python train_i2i_restormer.py --config "$CONFIG_PATH"

python ${VSLITE_DIR}/util/plot_loss_curves.py \
  --log_dir "$RESTORMER_OUTPUT_DIR" --out_dir "$RESTORMER_OUTPUT_DIR/curves" || true
