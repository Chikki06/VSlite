#!/bin/bash
# Latency measurement for the c13 noattn/objective ablations
# (jobs/cislnet/submit_c13_noattn_objective_ablation.sh chain: 20-ep trains,
#  their patch_eval_test metrics are DONE — do NOT touch them).
#
# Only c13_noattn changes the architecture; the five objective ablations are
# weight-for-weight identical to c13 and inherit its 46.59 ms (vs13 row) by
# construction — no timing needed for them. So this job times just:
#   c13         — same-lane baseline (kept in the scratch leaf, NOT appended
#                 to tracking.csv: plot_capacity_results.py keys c13 latency
#                 off the vs13 row)
#   c13_noattn  — the one architecture change → appended tracking row
#                 (group=noattn_obj)
# Both on the same node so the speedup ratio is internally consistent.
#
#   batch=1 / 1024^2 GPU latency (10 warmup + 100 timed forwards,
#   util/exp_metrics.py), latency + VGG only (--skip_full_metrics,
#   --num_test 64), scratch results_dir so canonical leaves are untouched.
#
# Usage: sbatch jobs/cislnet/experiments/latency_obj.sh
#SBATCH --job-name=cn_lat_obj
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --exclude=ccc0388,ccc0390
#SBATCH -A kf4-ic
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --chdir=/projects/illinois/eng/bioe/rxb/akshat/VSlite
#SBATCH --output=/u/akshahi2/logs/%x/%j.out
#SBATCH --error=/u/akshahi2/logs/%x/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akshahi2@illinois.edu

set -euo pipefail
mkdir -p "/u/akshahi2/logs/${SLURM_JOB_NAME:-cn_lat_obj}"
VSLITE_ROOT="/projects/illinois/eng/bioe/rxb/akshat/VSlite"
cd "$VSLITE_ROOT"
# shellcheck source=_lib/vslite_env.sh
source "$VSLITE_ROOT/jobs/_lib/vslite_env.sh"

SCRATCH="$VSLITE_ROOT/results/cislnet/experiments/latency_obj"
mkdir -p "$SCRATCH"

WHICH_EPOCH="${WHICH_EPOCH:-latest}"
LATENCY_WARMUP="${LATENCY_WARMUP:-10}"
LATENCY_TIMED="${LATENCY_TIMED:-100}"

run_latency() {
  local name="$1"
  local track_run="$2"
  local track_group="$3"
  local do_track="$4"
  echo "=== latency: $name (track=$do_track) ==="
  local -a extra=()
  if [[ "$do_track" == "1" ]]; then
    extra+=(--track_run "$track_run" --track_group "$track_group"
            --track_notes "latency_only num_test=64 skip_full_metrics")
  fi
  python "$VSLITE_ROOT/scripts/cislnet_exp_eval.py" \
    --name "$name" \
    --checkpoints_dir "$VSLITE_ROOT/results/cislnet" \
    --dataroot "$DATAROOT" \
    --dataset_mode VS --model contrastive_pix2pix \
    --input_nc 13 --num_ir_channels 10 \
    --which_epoch "$WHICH_EPOCH" \
    --results_dir "$SCRATCH" \
    --results_name "$name" \
    --gpu_ids 0 \
    --num_test 64 \
    --skip_full_metrics \
    --latency_warmup "$LATENCY_WARMUP" \
    --latency_timed "$LATENCY_TIMED" \
    "${extra[@]}"
}

# c13 same-lane baseline first, then the one architecture change.
run_latency c13 c13 "" 0

run_latency c13_noattn c13_noattn noattn_obj 1

echo "Done. Latency leaves under $SCRATCH/test/<run>/patch_eval_test/metrics.json"
