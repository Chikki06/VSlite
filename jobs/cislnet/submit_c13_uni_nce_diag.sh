#!/bin/bash
# Short c13_uni PatchNCE sanity diagnose:
#   train (NCE_DIAG=1, small max_dataset_size) → writes
#   results/cislnet/<RUN>/nce_diag.jsonl + nce_diag_summary.json
#
# Usage:
#   bash jobs/cislnet/submit_c13_uni_nce_diag.sh
#   DRY_RUN=1 bash jobs/cislnet/submit_c13_uni_nce_diag.sh
#   MAX_DATASET_SIZE=400 PRINT_FREQ=20 bash jobs/cislnet/submit_c13_uni_nce_diag.sh

set -euo pipefail
VSLITE_ROOT="${VSLITE_ROOT:-/projects/illinois/eng/bioe/rxb/akshat/VSlite}"
cd "$VSLITE_ROOT"
# shellcheck source=_lib/slurm_defaults.sh
source "$VSLITE_ROOT/jobs/_lib/slurm_defaults.sh"

DRY_RUN="${DRY_RUN:-0}"
EXCLUDE="${SLURM_EXCLUDE_NODES}"
RUN="${RUN:-c13_uni_nce_diag}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-10}"
NTHREADS="${NTHREADS:-4}"
NETF="${NETF:-uni}"
UNI_INPUT_SIZE="${UNI_INPUT_SIZE:-1024}"
UNI_BLOCKS="${UNI_BLOCKS:-2,5,8,11}"
NCE_T="${NCE_T:-0.07}"
NUM_PATCHES="${NUM_PATCHES:-1024}"
MAX_DATASET_SIZE="${MAX_DATASET_SIZE:-400}"
PRINT_FREQ="${PRINT_FREQ:-20}"
TRAIN_TIME="${TRAIN_TIME:-04:00:00}"
TRAIN_SH="$VSLITE_ROOT/jobs/cislnet/train/train.sh"

_submit_parsable() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "dryrun_diag_${RANDOM}"
    return 0
  fi
  sbatch --parsable "$@"
}

mkdir -p "/u/akshahi2/logs/cn_tr_${RUN}"

train_export="ALL,NUM_CHANNELS=13,RUN=${RUN},NAME=${RUN},EPOCHS=${EPOCHS},CONTINUE_TRAIN=false,NTHREADS=${NTHREADS},BATCH_SIZE=${BATCH_SIZE},NETF=${NETF},UNI_INPUT_SIZE=${UNI_INPUT_SIZE},UNI_BLOCKS=${UNI_BLOCKS},NCE_T=${NCE_T},NUM_PATCHES=${NUM_PATCHES},NCE_DIAG=1,MAX_DATASET_SIZE=${MAX_DATASET_SIZE},PRINT_FREQ=${PRINT_FREQ},VSLITE_UNI_LPIPS=1,VSLITE_ROOT=${VSLITE_ROOT}"

echo "c13_uni NCE diagnose: run=$RUN nce_t=$NCE_T blocks=$UNI_BLOCKS patches=$NUM_PATCHES max_ds=$MAX_DATASET_SIZE print_freq=$PRINT_FREQ"

J_DIAG=$(_submit_parsable \
  --job-name="cn_tr_${RUN}" \
  --chdir="$VSLITE_ROOT" \
  --exclude="$EXCLUDE" \
  --time="$TRAIN_TIME" \
  --export="$train_export" \
  "$TRAIN_SH")
echo "diagnose $RUN -> job $J_DIAG"
echo "jsonl=\$VSLITE_ROOT/results/cislnet/$RUN/nce_diag.jsonl"
echo "summary=\$VSLITE_ROOT/results/cislnet/$RUN/nce_diag_summary.json"
echo "$J_DIAG"
