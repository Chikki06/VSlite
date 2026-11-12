#!/bin/bash
#SBATCH --job-name=ir_fp16_convert
#SBATCH --partition=IllinoisComputes
#SBATCH -A kf4-ic
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --chdir=/projects/illinois/eng/bioe/rxb/akshat/VSlite
#SBATCH --output=/u/akshahi2/logs/%x/%j.out
#SBATCH --error=/u/akshahi2/logs/%x/%j.err
# Convert dataset/*/ir float32 .npy -> IR_fp16 (float16) + build data_frozen_fp16 dataroot.
set -euo pipefail
mkdir -p /u/akshahi2/logs/ir_fp16_convert
VSLITE_ROOT="/projects/illinois/eng/bioe/rxb/akshat/VSlite"
export VSLITE_SKIP_CUDA_CHECK=1
# shellcheck source=../../_lib/vslite_env.sh
source "$VSLITE_ROOT/jobs/_lib/vslite_env.sh"
python "$VSLITE_ROOT/scripts/convert_ir_to_fp16.py" \
  --root "$VSLITE_ROOT" \
  --workers "${IR_FP16_WORKERS:-32}" \
  --trees frozen 1024
echo "IR_fp16 conversion complete."
