#!/bin/bash
#SBATCH --job-name=cn_fp16
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
# Train+eval using float16 IR patches under dataset/data_frozen_fp16 (IR_fp16).
set -euo pipefail
VSLITE_ROOT="/projects/illinois/eng/bioe/rxb/akshat/VSlite"
export DATAROOT="${DATAROOT:-$VSLITE_ROOT/dataset/data_frozen_fp16}"
export NAME=fp16_c13 TRACK_GROUP=B TRACK_NOTES="fp16 on-disk IR (data_frozen_fp16/IR_fp16)"
if [[ ! -d "$DATAROOT/ir" ]]; then
  echo "FATAL: missing $DATAROOT/ir — run convert_ir_fp16.sh first" >&2
  exit 1
fi
exec bash "$VSLITE_ROOT/jobs/cislnet/experiments/train_eval.sh"
