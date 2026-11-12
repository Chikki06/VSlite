#!/bin/bash
# Submit the c13 build-2 objective ablations:
#
#   c13_noattn    — attention-free U-Net (attention_type=none), all else default
#   c13_gan       — LSGAN only          (no NCE; pixel L1/L2 stay off)
#   c13_ganl2_01  — LSGAN + L2 @ 0.1    (no NCE)
#   c13_ganl2_10  — LSGAN + L2 @ 10     (no NCE)
#   c13_nce       — NCE only @ 10       (no GAN; pixel L1/L2 stay off)
#   c13_l2        — L2 only @ 1.0       (no NCE, no GAN) — queued last (curiosity run)
#
# Each run is the vp-style chain: train hop1 → train hop2 (afterany,
# CONTINUE_TRAIN=true so a walltime timeout still resumes to EPOCHS) →
#   patch eval (afterok, NUM_TEST=75000) and WSI → DZI → remote upload
# into the campus-ablations viewer project (afterok, REMOTE_UPLOAD=1).
#
# Usage:
#   bash jobs/cislnet/submit_c13_noattn_objective_ablation.sh
#   DRY_RUN=1 bash jobs/cislnet/submit_c13_noattn_objective_ablation.sh
#   RUN_FILTER=c13_gan,... to submit a subset (comma-separated run tags)

set -euo pipefail
VSLITE_ROOT="/projects/illinois/eng/bioe/rxb/akshat/VSlite"
cd "$VSLITE_ROOT"
# shellcheck source=_lib/slurm_defaults.sh
source "$VSLITE_ROOT/jobs/_lib/slurm_defaults.sh"

EPOCHS="${EPOCHS:-20}"
NUM_TEST="${NUM_TEST:-75000}"
DRY_RUN="${DRY_RUN:-0}"
RUN_FILTER="${RUN_FILTER:-}"
EXCLUDE="${SLURM_EXCLUDE_NODES}"
CONTINUE_TRAIN="${CONTINUE_TRAIN:-false}"
TRAIN_CHAIN="${TRAIN_CHAIN:-1}"
TRAIN_TIME="${TRAIN_TIME:-}"
NTHREADS="${NTHREADS:-4}"
BATCH_SIZE="${BATCH_SIZE:-10}"
WHICH_EPOCH="${WHICH_EPOCH:-latest}"

# Campus viewer project for WSI uploads (same project as prior ablations).
PROJECT="${PROJECT:-campus_ablations_bestlpips}"
PROJECT_DISPLAY_NAME="${PROJECT_DISPLAY_NAME:-Campus_ablations_best_LPIPS_WSI}"
OUTPUT_TYPE_SUFFIX="${OUTPUT_TYPE_SUFFIX:-latest}"
WSI_FOLDERS="${WSI_FOLDERS:-10442-1:10442-2:10445-2:10445-3:10445-4:3840-1:3840-2:3840-3:3840-4:3840-5}"

TRAIN_SH="$VSLITE_ROOT/jobs/cislnet/train/train.sh"
TEST_SH="$VSLITE_ROOT/jobs/cislnet/test/test.sh"
WSI_SH="$VSLITE_ROOT/jobs/cislnet/wsi/wsi.sh"

declare -a RUNS=(
  "c13_noattn|ATTENTION_TYPE=none"
  "c13_gan|USE_NCE=0"
  "c13_ganl2_01|USE_NCE=0,USE_L2=1,LAMBDA_L2=0.1"
  "c13_ganl2_10|USE_NCE=0,USE_L2=1,LAMBDA_L2=10"
  "c13_nce|NO_GAN=1"
  "c13_l2|USE_NCE=0,USE_L2=1,LAMBDA_L2=1.0,NO_GAN=1"
)

_submit_parsable() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "dryrun_${RANDOM}"
    return 0
  fi
  sbatch --parsable "$@"
}

echo "c13 objective/noattn ablation: root=$VSLITE_ROOT EPOCHS=$EPOCHS BS=$BATCH_SIZE NT=$NTHREADS TRAIN_CHAIN=$TRAIN_CHAIN TRAIN_TIME=${TRAIN_TIME:-script} exclude=$EXCLUDE"
echo "viewer project=$PROJECT"

for spec in "${RUNS[@]}"; do
  run="${spec%%|*}"
  obj_env="${spec#*|}"   # comma-joined KEY=VAL pairs appended into --export
  if [[ -n "$RUN_FILTER" ]]; then
    skip=1
    IFS=',' read -ra want <<<"$RUN_FILTER"
    for w in "${want[@]}"; do
      [[ "$run" == "$w" ]] && skip=0
    done
    if [[ "$skip" == "1" ]]; then
      echo "skip $run (not in RUN_FILTER)"
      continue
    fi
  fi

  mkdir -p "/u/akshahi2/logs/cn_tr_${run}" "/u/akshahi2/logs/cn_te_${run}" "/u/akshahi2/logs/cn_wsi_${run}"

  channel_env="NUM_CHANNELS=13"

  train_export="ALL,${channel_env},RUN=${run},NAME=${run},EPOCHS=${EPOCHS},CONTINUE_TRAIN=${CONTINUE_TRAIN},NTHREADS=${NTHREADS},BATCH_SIZE=${BATCH_SIZE},${obj_env},VSLITE_ROOT=${VSLITE_ROOT}"
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
  echo "train $run hop1 -> job $J_TRAIN"

  last_train="$J_TRAIN"
  if [[ "$TRAIN_CHAIN" == "1" ]]; then
    train_export2="$train_export"
    train_export2="${train_export2/,CONTINUE_TRAIN=${CONTINUE_TRAIN}/,CONTINUE_TRAIN=true}"
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
    echo "train $run hop2 -> job $J_TRAIN (afterany CONTINUE_TRAIN=true)"
    last_train="$J_TRAIN"
  fi

  test_export="ALL,${channel_env},RUN=${run},NAME=${run},WHICH_EPOCH=${WHICH_EPOCH},NUM_TEST=${NUM_TEST},VSLITE_ROOT=${VSLITE_ROOT}"
  J_TEST=$(_submit_parsable \
    --job-name="cn_te_${run}" \
    --chdir="$VSLITE_ROOT" \
    --exclude="$EXCLUDE" \
    --dependency=afterok:"$last_train" \
    --export="$test_export" \
    "$TEST_SH")
  echo "test  $run -> job $J_TEST (afterok:$last_train)"

  wsi_export="ALL,${channel_env},RUN=${run},NAME=${run},WHICH_EPOCH=${WHICH_EPOCH},REMOTE_UPLOAD=1,PROJECT=${PROJECT},PROJECT_DISPLAY_NAME=${PROJECT_DISPLAY_NAME},OUTPUT_TYPE=${run}-${OUTPUT_TYPE_SUFFIX},OVERWRITE=1,WSI_FOLDERS=${WSI_FOLDERS},VSLITE_ROOT=${VSLITE_ROOT}"
  J_WSI=$(_submit_parsable \
    --job-name="cn_wsi_${run}" \
    --chdir="$VSLITE_ROOT" \
    --exclude="$EXCLUDE" \
    --dependency=afterok:"$last_train" \
    --export="$wsi_export" \
    "$WSI_SH")
  echo "wsi   $run -> job $J_WSI (afterok:$last_train, project=$PROJECT)"
done

echo
echo "Done. ckpts=$VSLITE_ROOT/results/cislnet/<run>/ patch-eval=$VSLITE_ROOT/results/cislnet/test/<run>/patch_eval_test/"
