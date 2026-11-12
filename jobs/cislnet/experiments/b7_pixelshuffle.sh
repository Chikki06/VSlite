#!/bin/bash
# B7: PixelShuffle / sub-pixel conv on final_upsample only (interior Up stay ConvTranspose2d).
# Intentionally NO --exclude=ccc0388 (nodes allowed).
#SBATCH --job-name=cn_b7
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
mkdir -p /u/akshahi2/logs/cn_b7
export NAME=b7_pixelshuffle_c13 UPSAMPLE_MODE=pixelshuffle TRACK_GROUP=B \
  TRACK_NOTES="Group B7 PixelShuffle final_upsample only"
exec bash /projects/illinois/eng/bioe/rxb/akshat/VSlite/jobs/cislnet/experiments/train_eval.sh
