#!/bin/bash
# Side-by-side 1024px patch grids: input | int32 | fp16 | fp8 | int4 | int8 | target
# sbatch visual_compare.sh
#SBATCH --job-name=cn_vis
#SBATCH --partition=scavenger
#SBATCH -A kf4-ic
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --chdir=/projects/illinois/eng/bioe/rxb/akshat/VSlite
#SBATCH --output=/u/akshahi2/logs/%x/%j.out
#SBATCH --error=/u/akshahi2/logs/%x/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akshahi2@illinois.edu

set -euo pipefail
_SCRIPT="${BASH_SOURCE[0]}"
if [[ "$_SCRIPT" == /* ]]; then
  VSLITE_ROOT="$(cd "$(dirname "$_SCRIPT")/../../.." && pwd)"
else
  VSLITE_ROOT="${VSLITE_ROOT:-/projects/illinois/eng/bioe/rxb/akshat/VSlite}"
fi
# shellcheck source=../../_lib/vslite_env.sh
source "$VSLITE_ROOT/jobs/_lib/vslite_env.sh"
# shellcheck source=_quant_common.sh
source "$VSLITE_ROOT/jobs/cislnet/quant/_quant_common.sh"

export VSLITE_INFER_BS="${VSLITE_INFER_BS:-8}"

python "$VSLITE_ROOT/scripts/quant_visual_compare.py" \
  --source_ckpt "$SOURCE_CKPT" \
  --opt_dir "$OPT_DIR" \
  --quant_root "$QUANT_ROOT" \
  --dataroot "$DATAROOT" \
  --run "$RUN" \
  --num_patches "$NUM_PATCHES" \
  --gpu_ids 0 \
  --input_nc 13 \
  --num_ir_channels 10
