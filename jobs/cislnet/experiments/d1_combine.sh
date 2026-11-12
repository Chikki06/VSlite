#!/bin/bash
# D1 Pareto combine — set ATTENTION_TYPE / ENCODER_BACKBONE / etc. after select_pareto.py
#SBATCH --job-name=cn_d1
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
VSLITE_ROOT="/projects/illinois/eng/bioe/rxb/akshat/VSlite"
# Prefer env from select_pareto export file if present.
if [[ -f "${PARETO_ENV:-$VSLITE_ROOT/results/cislnet/experiments/d1_env.sh}" ]]; then
  # shellcheck disable=SC1090
  source "${PARETO_ENV:-$VSLITE_ROOT/results/cislnet/experiments/d1_env.sh}"
fi
export NAME="${NAME:-d1_combine_c13}"
export TRACK_GROUP=D
export TRACK_NOTES="${TRACK_NOTES:-Pareto combine D1}"
exec bash "$VSLITE_ROOT/jobs/cislnet/experiments/train_eval.sh"
