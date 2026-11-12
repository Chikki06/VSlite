#!/bin/bash
# Submit Vis+ {1,3,5} CISLNet train → patch test → 10-WSI JPEG preview.
# Same path as c13/organ_exp: jobs/cislnet/{train,test,wsi}/*.sh with chdir to repo.
# IR wavelengths in order: 1650, 1081, 1765, 1140, 1547 → indices 8,0,9,1,6.
#
# Each train is two hops: hop2 is --dependency=afterany:<hop1> with
# CONTINUE_TRAIN=true so a 40h walltime timeout still resumes to EPOCHS.
#
# Usage:
#   /projects/illinois/eng/bioe/rxb/akshat/VSlite/jobs/cislnet/submit_visplus.sh
#   VISPLUS_FILTER=3 EPOCHS=20 ./jobs/cislnet/submit_visplus.sh

set -euo pipefail
VSLITE_ROOT="/projects/illinois/eng/bioe/rxb/akshat/VSlite"
cd "$VSLITE_ROOT"
# shellcheck source=_lib/slurm_defaults.sh
source "$VSLITE_ROOT/jobs/_lib/slurm_defaults.sh"

EPOCHS="${EPOCHS:-20}"
NUM_TEST="${NUM_TEST:-75000}"
DRY_RUN="${DRY_RUN:-0}"
VISPLUS_FILTER="${VISPLUS_FILTER:-}"
EXCLUDE="${SLURM_EXCLUDE_NODES}"
CONTINUE_TRAIN="${CONTINUE_TRAIN:-false}"
TRAIN_CHAIN="${TRAIN_CHAIN:-1}"
TRAIN_TIME="${TRAIN_TIME:-}"
# Match reliable c13 defaults.
NTHREADS="${NTHREADS:-4}"
BATCH_SIZE="${BATCH_SIZE:-10}"

WSI_FOLDERS="${WSI_FOLDERS:-10442-1:10442-2:10445-2:10445-3:10445-4:3840-1:3840-2:3840-3:3840-4:3840-5}"
WSI_IMAGE_FORMAT="${WSI_IMAGE_FORMAT:-jpeg}"
WSI_JPEG_QUALITY="${WSI_JPEG_QUALITY:-92}"

TRAIN_SH="$VSLITE_ROOT/jobs/cislnet/train/train.sh"
TEST_SH="$VSLITE_ROOT/jobs/cislnet/test/test.sh"
WSI_SH="$VSLITE_ROOT/jobs/cislnet/wsi/wsi.sh"

mkdir -p /u/akshahi2/logs/cn_tr_vp{1,3,5} /u/akshahi2/logs/cn_te_vp{1,3,5} /u/akshahi2/logs/cn_wsi_vp{1,3,5}

declare -a CONFIGS=(1 3 5)

_submit_parsable() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "dryrun_${RANDOM}"
    return 0
  fi
  sbatch --parsable "$@"
}

echo "Vis+ sweep: root=$VSLITE_ROOT EPOCHS=$EPOCHS BS=$BATCH_SIZE NT=$NTHREADS TRAIN_CHAIN=$TRAIN_CHAIN TRAIN_TIME=${TRAIN_TIME:-script} exclude=$EXCLUDE"

for n in "${CONFIGS[@]}"; do
  if [[ -n "$VISPLUS_FILTER" && "$n" != "$VISPLUS_FILTER" ]]; then
    continue
  fi
  run="vp${n}"
  train_export="ALL,VISPLUS=$n,RUN=$run,NAME=$run,EPOCHS=$EPOCHS,CONTINUE_TRAIN=$CONTINUE_TRAIN,NTHREADS=$NTHREADS,BATCH_SIZE=$BATCH_SIZE,VSLITE_ROOT=$VSLITE_ROOT"
  sbatch_extra=()
  if [[ -n "$TRAIN_TIME" ]]; then
    sbatch_extra+=(--time="$TRAIN_TIME")
  fi
  J_TRAIN=$(_submit_parsable \
    "${sbatch_extra[@]}" \
    --job-name="cn_tr_${run}" \
    --chdir="$VSLITE_ROOT" \
    --exclude="$EXCLUDE" \
    --export="$train_export" \
    "$TRAIN_SH")
  echo "train Vis+$n ($run) hop1 -> job $J_TRAIN"

  if [[ "$TRAIN_CHAIN" == "1" ]]; then
    train_export2="${train_export/,CONTINUE_TRAIN=$CONTINUE_TRAIN/,CONTINUE_TRAIN=true}"
    if [[ "$train_export2" != *"CONTINUE_TRAIN=true"* ]]; then
      train_export2+=",CONTINUE_TRAIN=true"
    fi
    J_TRAIN=$(_submit_parsable \
      "${sbatch_extra[@]}" \
      --job-name="cn_tr_${run}" \
      --chdir="$VSLITE_ROOT" \
      --exclude="$EXCLUDE" \
      --dependency="afterany:${J_TRAIN}" \
      --export="$train_export2" \
      "$TRAIN_SH")
    echo "train Vis+$n ($run) hop2 -> job $J_TRAIN (afterany CONTINUE_TRAIN=true)"
  fi

  test_export="ALL,VISPLUS=$n,RUN=$run,NAME=$run,NUM_TEST=$NUM_TEST,VSLITE_ROOT=$VSLITE_ROOT"
  J_TEST=$(_submit_parsable \
    --job-name="cn_te_${run}" \
    --chdir="$VSLITE_ROOT" \
    --exclude="$EXCLUDE" \
    --dependency=afterok:"$J_TRAIN" \
    --export="$test_export" \
    "$TEST_SH")
  echo "test  Vis+$n ($run) -> job $J_TEST (afterok:$J_TRAIN)"

  wsi_export="ALL,VISPLUS=$n,RUN=$run,NAME=$run,WSI_FOLDERS=$WSI_FOLDERS,WSI_IMAGE_FORMAT=$WSI_IMAGE_FORMAT,WSI_JPEG_QUALITY=$WSI_JPEG_QUALITY,VSLITE_ROOT=$VSLITE_ROOT"
  J_WSI=$(_submit_parsable \
    --job-name="cn_wsi_${run}" \
    --chdir="$VSLITE_ROOT" \
    --exclude="$EXCLUDE" \
    --dependency=afterok:"$J_TRAIN" \
    --export="$wsi_export" \
    "$WSI_SH")
  echo "wsi   Vis+$n ($run) -> job $J_WSI (afterok:$J_TRAIN)"
done

echo "Done. ckpts=$VSLITE_ROOT/results/cislnet/vp{1,3,5}/ wsi=$VSLITE_ROOT/results/cislnet/test/vp{1,3,5}/wsi_pred/"
