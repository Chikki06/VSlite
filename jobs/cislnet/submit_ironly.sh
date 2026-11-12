#!/bin/bash
# Submit IR-only {1,3,5} CISLNet train (Vis+ wavelength subsets, no VIS).
# Same stack as vp1/vp3/vp5 and vs10: jobs/cislnet/train/train.sh → scripts/train.py.
# Patch IR is cubically resized to image_size (1024) to match VIS/HE geometry
# (same path as vs10 / Vis+; IR native 2 µm → 0.69 µm equiv at patch scale).
# IR wavelengths in order: 1650, 1081, 1765, 1140, 1547 → indices 8,0,9,1,6.
#
# Each train is two hops: hop2 is --dependency=afterany:<hop1> with
# CONTINUE_TRAIN=true so a walltime timeout still resumes to EPOCHS.
# After train: patch test (75k) → WSI infer for the 10 preview slides
# (same flow as submit_visplus.sh: test.sh + wsi.sh, default local JPEG path).
#
# Usage:
#   ./jobs/cislnet/submit_ironly.sh
#   IRONLY_FILTER=3 EPOCHS=20 ./jobs/cislnet/submit_ironly.sh
#   IRONLY_FILTER=3 CONTINUE_TRAIN=true ./jobs/cislnet/submit_ironly.sh  # requeue/resume

set -euo pipefail
VSLITE_ROOT="/projects/illinois/eng/bioe/rxb/akshat/VSlite"
cd "$VSLITE_ROOT"
# shellcheck source=_lib/slurm_defaults.sh
source "$VSLITE_ROOT/jobs/_lib/slurm_defaults.sh"

EPOCHS="${EPOCHS:-20}"
NUM_TEST="${NUM_TEST:-75000}"
DRY_RUN="${DRY_RUN:-0}"
IRONLY_FILTER="${IRONLY_FILTER:-}"
EXCLUDE="${SLURM_EXCLUDE_NODES}"
CONTINUE_TRAIN="${CONTINUE_TRAIN:-false}"
TRAIN_CHAIN="${TRAIN_CHAIN:-1}"
TRAIN_TIME="${TRAIN_TIME:-}"
# Match Vis+ / reliable c13 defaults.
NTHREADS="${NTHREADS:-4}"
BATCH_SIZE="${BATCH_SIZE:-10}"

# WSI manifest (same 10 preview slides as submit_visplus.sh):
WSI_FOLDERS="${WSI_FOLDERS:-10442-1:10442-2:10445-2:10445-3:10445-4:3840-1:3840-2:3840-3:3840-4:3840-5}"
WSI_IMAGE_FORMAT="${WSI_IMAGE_FORMAT:-jpeg}"
WSI_JPEG_QUALITY="${WSI_JPEG_QUALITY:-92}"

TRAIN_SH="$VSLITE_ROOT/jobs/cislnet/train/train.sh"
TEST_SH="$VSLITE_ROOT/jobs/cislnet/test/test.sh"
WSI_SH="$VSLITE_ROOT/jobs/cislnet/wsi/wsi.sh"

mkdir -p /u/akshahi2/logs/cn_tr_ir{1,3,5} /u/akshahi2/logs/cn_te_ir{1,3,5} /u/akshahi2/logs/cn_wsi_ir{1,3,5}

declare -a CONFIGS=(1 3 5)

_submit_parsable() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "dryrun_${RANDOM}"
    return 0
  fi
  sbatch --parsable "$@"
}

echo "IR-only sweep: root=$VSLITE_ROOT EPOCHS=$EPOCHS BS=$BATCH_SIZE NT=$NTHREADS TRAIN_CHAIN=$TRAIN_CHAIN TRAIN_TIME=${TRAIN_TIME:-script} exclude=$EXCLUDE"

for n in "${CONFIGS[@]}"; do
  if [[ -n "$IRONLY_FILTER" && "$n" != "$IRONLY_FILTER" ]]; then
    continue
  fi
  run="ir${n}"
  train_export="ALL,IRONLY=$n,RUN=$run,NAME=$run,EPOCHS=$EPOCHS,CONTINUE_TRAIN=$CONTINUE_TRAIN,NTHREADS=$NTHREADS,BATCH_SIZE=$BATCH_SIZE,VSLITE_ROOT=$VSLITE_ROOT"
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
  echo "train IR-only $n ($run) hop1 -> job $J_TRAIN"

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
    echo "train IR-only $n ($run) hop2 -> job $J_TRAIN (afterany CONTINUE_TRAIN=true)"
  fi

  test_export="ALL,IRONLY=$n,RUN=$run,NAME=$run,NUM_TEST=$NUM_TEST,VSLITE_ROOT=$VSLITE_ROOT"
  J_TEST=$(_submit_parsable \
    --job-name="cn_te_${run}" \
    --chdir="$VSLITE_ROOT" \
    --exclude="$EXCLUDE" \
    --dependency=afterok:"$J_TRAIN" \
    --export="$test_export" \
    "$TEST_SH")
  echo "test  IR-only $n ($run) -> job $J_TEST (afterok:$J_TRAIN)"

  wsi_export="ALL,IRONLY=$n,RUN=$run,NAME=$run,WSI_FOLDERS=$WSI_FOLDERS,WSI_IMAGE_FORMAT=$WSI_IMAGE_FORMAT,WSI_JPEG_QUALITY=$WSI_JPEG_QUALITY,VSLITE_ROOT=$VSLITE_ROOT"
  J_WSI=$(_submit_parsable \
    --job-name="cn_wsi_${run}" \
    --chdir="$VSLITE_ROOT" \
    --exclude="$EXCLUDE" \
    --dependency=afterok:"$J_TEST" \
    --export="$wsi_export" \
    "$WSI_SH")
  echo "wsi   IR-only $n ($run) -> job $J_WSI (afterok:$J_TEST)"
done

echo "Done. ckpts=$VSLITE_ROOT/results/cislnet/ir{1,3,5}/ wsi=$VSLITE_ROOT/results/cislnet/test/ir{1,3,5}/wsi_pred/"
