#!/bin/bash
#SBATCH --job-name=cn_a3
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
export NAME=a3_scse_c13 ATTENTION_TYPE=scse TRACK_GROUP=A TRACK_NOTES="Group A3 SCSE"
exec bash /projects/illinois/eng/bioe/rxb/akshat/VSlite/jobs/cislnet/experiments/train_eval.sh
