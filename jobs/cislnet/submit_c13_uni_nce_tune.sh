#!/bin/bash
# Targeted c13_uni PatchNCE tuning after (or gated on) a diagnose job.
#
# Highest-leverage arms (distinct RUN tags, no shared writes):
#   nce_t ∈ {0.10, 0.15}  (+ optional 0.05 via INCLUDE_LOW_T=1)
#   UNI blocks: default 2,5,8,11 vs mid/late 4,10,16,22
#   num_patches ∈ {512, 1024} on the deeper-block + nce_t=0.10 arm
#
# Usage:
#   bash jobs/cislnet/submit_c13_uni_nce_diag_then_tune.sh   # preferred
#   DIAG_JID=<jid> bash jobs/cislnet/submit_c13_uni_nce_tune.sh
#   DRY_RUN=1 bash jobs/cislnet/submit_c13_uni_nce_tune.sh
#   INCLUDE_LOW_T=1 ...   # also submit nce_t=0.05 arms (when margins look compressed)

set -euo pipefail
VSLITE_ROOT="${VSLITE_ROOT:-/projects/illinois/eng/bioe/rxb/akshat/VSlite}"
cd "$VSLITE_ROOT"
# shellcheck source=_lib/slurm_defaults.sh
source "$VSLITE_ROOT/jobs/_lib/slurm_defaults.sh"

DRY_RUN="${DRY_RUN:-0}"
EXCLUDE="${SLURM_EXCLUDE_NODES}"
DIAG_JID="${DIAG_JID:-}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-10}"
NTHREADS="${NTHREADS:-4}"
NETF="${NETF:-uni}"
UNI_INPUT_SIZE="${UNI_INPUT_SIZE:-1024}"
INCLUDE_LOW_T="${INCLUDE_LOW_T:-1}"
TRAIN_TIME="${TRAIN_TIME:-}"
TRAIN_HOPS="${TRAIN_HOPS:-1}"
TRAIN_SH="$VSLITE_ROOT/jobs/cislnet/train/train.sh"

_submit_parsable() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "dryrun_tune_${RANDOM}"
    return 0
  fi
  sbatch --parsable "$@"
}

_dep_args=()
if [[ -n "$DIAG_JID" ]]; then
  _dep_args+=(--dependency="afterok:${DIAG_JID}")
  echo "tune arms gated afterok:$DIAG_JID"
else
  echo "WARNING: DIAG_JID unset — submitting tune arms with no diagnose dependency" >&2
fi

sbatch_extra=()
if [[ -n "$TRAIN_TIME" ]]; then
  sbatch_extra+=(--time="$TRAIN_TIME")
fi

# arm_tag nce_t uni_blocks num_patches
_arms=(
  "t010_b25811_p1024 0.10 2,5,8,11 1024"
  "t015_b25811_p1024 0.15 2,5,8,11 1024"
  "t010_b4_10_16_22_p1024 0.10 4,10,16,22 1024"
  "t015_b4_10_16_22_p1024 0.15 4,10,16,22 1024"
  "t010_b4_10_16_22_p512 0.10 4,10,16,22 512"
)
if [[ "$INCLUDE_LOW_T" == "1" ]]; then
  _arms+=(
    "t005_b25811_p1024 0.05 2,5,8,11 1024"
    "t005_b4_10_16_22_p1024 0.05 4,10,16,22 1024"
  )
fi

echo "c13_uni NCE tune: EPOCHS=$EPOCHS arms=${#_arms[@]} INCLUDE_LOW_T=$INCLUDE_LOW_T"

for arm in "${_arms[@]}"; do
  # shellcheck disable=SC2086
  set -- $arm
  tag="$1"
  nce_t="$2"
  blocks="$3"
  patches="$4"
  RUN="c13_uni_nce_${tag}"
  mkdir -p "/u/akshahi2/logs/cn_tr_${RUN}"

  # NCE_DIAG off on tune arms (diagnose already wrote the margin/top-1 JSONL).
  train_export="ALL,NUM_CHANNELS=13,RUN=${RUN},NAME=${RUN},EPOCHS=${EPOCHS},CONTINUE_TRAIN=false,NTHREADS=${NTHREADS},BATCH_SIZE=${BATCH_SIZE},NETF=${NETF},UNI_INPUT_SIZE=${UNI_INPUT_SIZE},UNI_BLOCKS=${blocks},NCE_T=${nce_t},NUM_PATCHES=${patches},NCE_DIAG=0,PRINT_FREQ=${PRINT_FREQ:-100},VSLITE_UNI_LPIPS=1,VSLITE_ROOT=${VSLITE_ROOT}"

  J=$(_submit_parsable \
    "${sbatch_extra[@]}" \
    "${_dep_args[@]}" \
    --job-name="cn_tr_${RUN}" \
    --chdir="$VSLITE_ROOT" \
    --exclude="$EXCLUDE" \
    --export="$train_export" \
    "$TRAIN_SH")
  echo "tune $RUN -> job $J (nce_t=$nce_t blocks=$blocks patches=$patches)"
done
