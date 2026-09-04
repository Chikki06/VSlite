#!/bin/bash
# Quick smoke: region eval + CBAM vis on existing c13 checkpoint (NUM_TEST=2).
# Usage: sbatch jobs/cislnet/smoke_patch_sweep.sh
#SBATCH --job-name=cn_smoke
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --exclude=ccc0387,ccc0388
#SBATCH -A kf4-ic
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --chdir=/projects/illinois/eng/bioe/rxb/akshat/VSlite
#SBATCH --output=/u/akshahi2/logs/%x/%j.out
#SBATCH --error=/u/akshahi2/logs/%x/%j.err

set -euo pipefail
: "${VSLITE_ROOT:=/projects/illinois/eng/bioe/rxb/akshat/VSlite}"
source "$VSLITE_ROOT/jobs/_lib/slurm_resolve.sh"
VSLITE_ROOT="$(slurm_resolve_include_root jobs/_lib/vslite_env.sh "${BASH_SOURCE[0]:-$0}")" || exit 1
source "$VSLITE_ROOT/jobs/_lib/vslite_env.sh"

# Region eval smoke (c13@1024 ckpt, 2048-region tiling).
python "$VSLITE_ROOT/scripts/vs_infer_and_metrics.py" \
  --name c13 \
  --checkpoints_dir "$VSLITE_ROOT/results/cislnet" \
  --results_dir "$VSLITE_ROOT/results/cislnet" \
  --dataroot "$DATAROOT" \
  --input_nc 13 \
  --num_ir_channels 10 \
  --gpu_ids 0 \
  --num_test 2 \
  --region_eval_size 2048 \
  --save_region_triplets \
  --fid_max_items 2 \
  --fid_batch_size 2

# CBAM / channel-importance smoke.
python "$VSLITE_ROOT/scripts/visualize_cislnet_cbam.py" \
  --name c13 \
  --checkpoints_dir "$VSLITE_ROOT/results/cislnet" \
  --results_dir "$VSLITE_ROOT/results/cislnet" \
  --dataroot "$DATAROOT" \
  --input_nc 13 \
  --num_ir_channels 10 \
  --gpu_ids 0 \
  --num_samples 2 \
  --methods cbam,occlusion

echo "Smoke complete: region eval + CBAM vis on c13 (NUM_TEST=2)."
