#!/bin/bash
# Submit default CISLNet (c13) train on the frozen all-organ split, then
# post-train FID/PSNR/SSIM/LPIPS on the canonical test set at every 2 epochs.
#
# Login-node driver only (calls sbatch). No GPU needed here.
#
# Usage:
#   ./jobs/cislnet/submit_c13_epoch_eval.sh
#   DRY_RUN=1 ./jobs/cislnet/submit_c13_epoch_eval.sh
#   SKIP_TRAIN=1 ./jobs/cislnet/submit_c13_epoch_eval.sh   # tests only, ckpts must exist
#   CONTINUE_TRAIN=true ./jobs/cislnet/submit_c13_epoch_eval.sh

set -euo pipefail
: "${VSLITE_ROOT:=/projects/illinois/eng/bioe/rxb/akshat/VSlite}"
# shellcheck source=_lib/slurm_resolve.sh
source "$VSLITE_ROOT/jobs/_lib/slurm_resolve.sh"
VSLITE_ROOT="$(slurm_resolve_include_root jobs/_lib/vslite_env.sh "${BASH_SOURCE[0]:-$0}")" || exit 1
# shellcheck source=_lib/vslite_env.sh
source "$VSLITE_ROOT/jobs/_lib/vslite_env.sh"
# shellcheck source=_lib/slurm_defaults.sh
source "$VSLITE_ROOT/jobs/_lib/slurm_defaults.sh"

NUM_CHANNELS="${NUM_CHANNELS:-13}"
EPOCHS="${EPOCHS:-35}"
SAVE_EPOCH_FREQ="${SAVE_EPOCH_FREQ:-2}"
TEST_EVERY="${TEST_EVERY:-2}"
NUM_TEST="${NUM_TEST:-75000}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
CONTINUE_TRAIN="${CONTINUE_TRAIN:-false}"
# Distinct from the incomplete May `c13` / `vs13` run (timeout at ~e28, saves every 5).
RUN="${RUN:-c13_e35}"
NAME="${NAME:-$RUN}"
TRAIN_TIME="${TRAIN_TIME:-3-00:00:00}"
TEST_TIME="${TEST_TIME:-1-12:00:00}"
EXCLUDE="${SLURM_EXCLUDE_NODES}"
LOG_ROOT="${LOG_ROOT:-/u/akshahi2/logs}"
# Frozen-split train patch count (all npy minus test.txt). Used by the plotter.
TRAIN_PATCHES="${TRAIN_PATCHES:-60047}"

JOBS_DIR="$VSLITE_ROOT/jobs/cislnet"
TRAIN_SH="$JOBS_DIR/train/train.sh"
TEST_SH="$JOBS_DIR/test/test_epoch_sweep.sh"

_submit_parsable() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "dryrun_${RANDOM}"
    return 0
  fi
  sbatch --parsable "$@"
}

echo "c13 epoch-eval submit: RUN=$NAME EPOCHS=$EPOCHS SAVE_EPOCH_FREQ=$SAVE_EPOCH_FREQ TEST_EVERY=$TEST_EVERY SKIP_TRAIN=$SKIP_TRAIN CONTINUE_TRAIN=$CONTINUE_TRAIN exclude=$EXCLUDE"

J_TRAIN=""
if [[ "$SKIP_TRAIN" != "1" ]]; then
  job_name="cn_tr_${NAME}"
  mkdir -p "$LOG_ROOT/$job_name"
  export_vars="ALL,NUM_CHANNELS=$NUM_CHANNELS,EPOCHS=$EPOCHS,SAVE_EPOCH_FREQ=$SAVE_EPOCH_FREQ"
  export_vars+=",RUN=$NAME,NAME=$NAME,CONTINUE_TRAIN=$CONTINUE_TRAIN"
  J_TRAIN=$(_submit_parsable --job-name="$job_name" --time="$TRAIN_TIME" --exclude="$EXCLUDE" \
    --export="$export_vars" "$TRAIN_SH")
  echo "train $NAME -> job $J_TRAIN (time=$TRAIN_TIME save_epoch_freq=$SAVE_EPOCH_FREQ)"
else
  echo "train $NAME -> skipped (SKIP_TRAIN=1)"
fi

job_name="cn_te_${NAME}"
mkdir -p "$LOG_ROOT/$job_name"
export_vars="ALL,NUM_CHANNELS=$NUM_CHANNELS,EPOCHS=$EPOCHS,TEST_EVERY=$TEST_EVERY"
export_vars+=",RUN=$NAME,NAME=$NAME,NUM_TEST=$NUM_TEST,TEST_INCLUDES_VAL=0"
export_vars+=",TRAIN_PATCHES=$TRAIN_PATCHES"

dep_args=()
dep_note="no-dep"
if [[ -n "$J_TRAIN" ]]; then
  dep_args=(--dependency="afterok:${J_TRAIN}")
  dep_note="afterok:${J_TRAIN}"
fi

J_TEST=$(_submit_parsable --job-name="$job_name" --time="$TEST_TIME" --exclude="$EXCLUDE" \
  "${dep_args[@]}" --export="$export_vars" "$TEST_SH")
echo "test  $NAME every ${TEST_EVERY} ep -> job $J_TEST ($dep_note time=$TEST_TIME)"
echo "Done."
