#!/bin/bash
# Submit CISLNet c13 patch-size sweep: train 512/1024/2048, then dependent patch eval.
# Login-node driver only (calls sbatch for train/test). No GPU needed here.
#
# Usage (from repo root or any cwd):
#   ./jobs/cislnet/submit_patch_sweep.sh
#   RUN_FILTER=c13_ps512 ./jobs/cislnet/submit_patch_sweep.sh   # one run only
#   EPOCHS=30 DRY_RUN=1 ./jobs/cislnet/submit_patch_sweep.sh

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
EPOCHS="${EPOCHS:-30}"
REGION_EVAL_SIZE="${REGION_EVAL_SIZE:-2048}"
SAVE_REGION_TRIPLETS="${SAVE_REGION_TRIPLETS:-1}"
NUM_TEST="${NUM_TEST:-75000}"
DRY_RUN="${DRY_RUN:-0}"
RUN_FILTER="${RUN_FILTER:-}"
EXCLUDE="${SLURM_EXCLUDE_NODES}"

JOBS_DIR="$VSLITE_ROOT/jobs/cislnet"
TRAIN_SH="$JOBS_DIR/train/train.sh"
TEST_SH="$JOBS_DIR/test/test.sh"

# patch_size:batch_size:run
declare -a CONFIGS=(
  "512:10:c13_ps512"
  "1024:10:c13_ps1024"
  "2048:2:c13_ps2048"
)

_submit_parsable() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "dryrun_${RANDOM}"
    return 0
  fi
  sbatch --parsable "$@"
}

echo "CISLNet patch sweep: NUM_CHANNELS=$NUM_CHANNELS EPOCHS=$EPOCHS REGION_EVAL_SIZE=$REGION_EVAL_SIZE exclude=$EXCLUDE"

for cfg in "${CONFIGS[@]}"; do
  IFS=: read -r ps bs run <<< "$cfg"
  if [[ -n "$RUN_FILTER" && "$run" != "$RUN_FILTER" ]]; then
    continue
  fi
  export_vars="ALL,NUM_CHANNELS=$NUM_CHANNELS,PATCH_SIZE=$ps,BATCH_SIZE=$bs,EPOCHS=$EPOCHS,RUN=$run,NAME=$run,CONTINUE_TRAIN=${CONTINUE_TRAIN:-false}"

  J_TRAIN=$(_submit_parsable --exclude="$EXCLUDE" --export="$export_vars" "$TRAIN_SH")
  echo "train $run -> job $J_TRAIN (PATCH_SIZE=$ps BATCH_SIZE=$bs)"

  test_export="ALL,NUM_CHANNELS=$NUM_CHANNELS,RUN=$run,NAME=$run,NUM_TEST=$NUM_TEST,REGION_EVAL_SIZE=$REGION_EVAL_SIZE,SAVE_REGION_TRIPLETS=$SAVE_REGION_TRIPLETS"
  # afterok: test runs only if train exits 0 (success). Failed trains leave test as DependencyNeverSatisfied.
  J_TEST=$(_submit_parsable --exclude="$EXCLUDE" --dependency=afterok:"$J_TRAIN" --export="$test_export" "$TEST_SH")
  echo "test  $run -> job $J_TEST (afterok:$J_TRAIN)"
done

echo "Done."
