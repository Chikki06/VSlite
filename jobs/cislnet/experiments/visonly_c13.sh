#!/bin/bash
#SBATCH --job-name=cn_visonly
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
# Exp 1: c13-style model (resnet50/cbam/standard/convtranspose, full width) on VIS RGB only.
export NAME=visonly_c13 VIS_ONLY=1 TRACK_GROUP=V \
  TRACK_NOTES="VIS-only c13-style (no IR bands, input_nc=3)"
exec bash /projects/illinois/eng/bioe/rxb/akshat/VSlite/jobs/cislnet/experiments/train_eval.sh
