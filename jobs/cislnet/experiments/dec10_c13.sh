#!/bin/bash
#SBATCH --job-name=cn_dec10
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --exclude=ccc0388
#SBATCH -A kf4-ic
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --chdir=/projects/illinois/eng/bioe/rxb/akshat/VSlite
#SBATCH --output=/u/akshahi2/logs/%x/%j.out
#SBATCH --error=/u/akshahi2/logs/%x/%j.err
set -euo pipefail
# Exp 2b: decoder kernels at 10% (same pattern as b1_half 50%: width mult + proportional NDF).
export NAME=dec10_c13 DECODER_WIDTH_MULT=0.10 NDF=6 TRACK_GROUP=B \
  TRACK_NOTES="Group B decoder width 10pct + ndf6"
exec bash /projects/illinois/eng/bioe/rxb/akshat/VSlite/jobs/cislnet/experiments/train_eval.sh
