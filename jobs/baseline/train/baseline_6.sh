#!/bin/bash
#SBATCH --job-name=b_6_tr
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
export VS_DATAROOT="/projects/illinois/eng/bioe/rxb/akshat/VSlite/dataset/data_frozen"
export CHECKPOINTS_DIR="/projects/illinois/eng/bioe/rxb/akshat/VSlite/results/baseline"
NAME="vs6"

# Unified TensorBoard eval caps (controls overhead of split evaluation).
export UNIFIED_EVAL_MAX_BATCHES_TRAIN="${UNIFIED_EVAL_MAX_BATCHES_TRAIN:-20}"
export UNIFIED_EVAL_MAX_BATCHES_VAL="${UNIFIED_EVAL_MAX_BATCHES_VAL:-50}"
export UNIFIED_EVAL_MAX_BATCHES_TEST="${UNIFIED_EVAL_MAX_BATCHES_TEST:-50}"

# 6-channel baseline: IR indices [0,8,9] + VIS RGB (same VIS path/pipeline as vs13)
python scripts/train.py --dataroot "$VS_DATAROOT" --dataset_mode VS --model contrastive_pix2pix \
  --gpu_ids 0 --nThreads 4 --batch_size 10 --load_size 1024 --crop_size 512 \
  --print_freq 100 --save_epoch_freq 5 --save_latest_freq 30000 --display_freq 30000 \
  --evaluation_freq 5000 --continue_train true --name "$NAME" \
  --checkpoints_dir "$CHECKPOINTS_DIR" --input_nc 6 --num_ir_channels 10 --ir_channel_indices "0,8,9"

python /projects/illinois/eng/bioe/rxb/akshat/VSlite/util/plot_loss_curves.py \
  --log_dir "$CHECKPOINTS_DIR/$NAME" --out_dir "$CHECKPOINTS_DIR/$NAME/curves" || true

