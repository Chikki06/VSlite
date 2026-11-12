#!/bin/bash
#SBATCH --job-name=cn_encdec50
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
# Exp 3: kernels at 50% in the encoder too (decoder stays 50% as in b1_half).
export NAME=encdec50_c13 ENCODER_WIDTH_MULT=0.5 DECODER_WIDTH_MULT=0.5 NDF=32 TRACK_GROUP=B \
  TRACK_NOTES="Group B encoder 50pct + decoder 50pct + ndf32"
exec bash /projects/illinois/eng/bioe/rxb/akshat/VSlite/jobs/cislnet/experiments/train_eval.sh
