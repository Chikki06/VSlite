#!/bin/bash
# B6: depthwise-separable convs in Up.conv (all 4 stages).
# Intentionally NO --exclude=ccc0388 (nodes allowed).
#SBATCH --job-name=cn_b6
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH -A kf4-ic
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --chdir=/projects/illinois/eng/bioe/rxb/akshat/VSlite
#SBATCH --output=/u/akshahi2/logs/%x/%j.out
#SBATCH --error=/u/akshahi2/logs/%x/%j.err
set -euo pipefail
mkdir -p /u/akshahi2/logs/cn_b6
export NAME=b6_depthwise_c13 DECODER_CONV=depthwise TRACK_GROUP=B \
  TRACK_NOTES="Group B6 depthwise-separable Up.conv (DW3x3+PW1x1)"
exec bash /projects/illinois/eng/bioe/rxb/akshat/VSlite/jobs/cislnet/experiments/train_eval.sh
