#!/bin/bash
# Parameterized I2SB training.
# Example: NUM_CHANNELS=13 RUN=i13 sbatch train.sh
#SBATCH --job-name=i2sb_tr
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH -A kf4-ic
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=71:00:00
#SBATCH --chdir=/projects/illinois/eng/bioe/rxb/akshat/VSlite
#SBATCH --output=/u/akshahi2/logs/%x/%j.out
#SBATCH --error=/u/akshahi2/logs/%x/%j.err

set -euo pipefail
: "${VSLITE_ROOT:=/projects/illinois/eng/bioe/rxb/akshat/VSlite}"
# shellcheck source=../../_lib/slurm_resolve.sh
source "$VSLITE_ROOT/jobs/_lib/slurm_resolve.sh"
VSLITE_ROOT="$(slurm_resolve_include_root jobs/_lib/vslite_env.sh "${BASH_SOURCE[0]:-$0}")" || exit 1
# shellcheck source=../../_lib/vslite_env.sh
source "$VSLITE_ROOT/jobs/_lib/vslite_env.sh"

NUM_CHANNELS="${NUM_CHANNELS:-13}"
RUN="${RUN:-i${NUM_CHANNELS}}"
NAME="${NAME:-$RUN}"
PATCH_SIZE="${PATCH_SIZE:-256}"
BATCH_SIZE="${BATCH_SIZE:-32}"

python "$VSLITE_ROOT/scripts/train_i2sb.py" \
  --name "$NAME" \
  --dataroot "$DATAROOT" \
  --results_dir "${I2SB_RESULTS_DIR:-$VSLITE_ROOT/results/i2sb}" \
  --input_nc "$NUM_CHANNELS" \
  --num_ir_channels "${NUM_IR_CHANNELS:-10}" \
  --image_size "$PATCH_SIZE" \
  --batch_size "$BATCH_SIZE" \
  --num_itr "${I2SB_NUM_ITR:-200000}"
