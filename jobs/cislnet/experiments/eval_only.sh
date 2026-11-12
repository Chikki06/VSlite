#!/bin/bash
# Eval-only CISLNet run: patch eval + metrics + tracking.csv row upload.
# Same eval path as jobs/cislnet/experiments/train_eval.sh, minus the train hop.
# Set via env: NAME (required), NUM_CHANNELS, WHICH_EPOCH (latest|auto|<N>),
# TRACK_GROUP, TRACK_NOTES, RESULTS_NAME, NUM_TEST.
#SBATCH --job-name=cn_te_trk
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --exclude=ccc0388,ccc0390
#SBATCH -A kf4-ic
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --chdir=/projects/illinois/eng/bioe/rxb/akshat/VSlite
#SBATCH --output=/u/akshahi2/logs/%x/%j.out
#SBATCH --error=/u/akshahi2/logs/%x/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akshahi2@illinois.edu

set -euo pipefail
mkdir -p "/u/akshahi2/logs/${SLURM_JOB_NAME:-cn_te_trk}"
VSLITE_ROOT="/projects/illinois/eng/bioe/rxb/akshat/VSlite"
cd "$VSLITE_ROOT"
# shellcheck source=_common.sh
source "$VSLITE_ROOT/jobs/cislnet/experiments/_common.sh"
# shellcheck source=../../_lib/cislnet_channels.sh
source "$VSLITE_ROOT/jobs/_lib/cislnet_channels.sh"
export NUM_CHANNELS NUM_IR IR_INDICES

NAME="${NAME:?Set NAME (checkpoint run tag)}"
WHICH_EPOCH="${WHICH_EPOCH:-latest}"
if [[ "$WHICH_EPOCH" == "auto" ]]; then
  WHICH_EPOCH="$(python "$VSLITE_ROOT/scripts/pick_best_lpips_epoch.py" --run "$NAME")"
  echo "[eval_only] AUTO_BEST_LPIPS selected which_epoch=$WHICH_EPOCH"
fi
export WHICH_EPOCH
TRACK_GROUP="${TRACK_GROUP:-exp}"
TRACK_NOTES="${TRACK_NOTES:-}"
RESULTS_NAME="${RESULTS_NAME:-$NAME}"

cislnet_exp_run_eval "$NAME" "${TRACK_RUN:-$NAME}" "$TRACK_GROUP" "$TRACK_NOTES" "$RESULTS_NAME"
