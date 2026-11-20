#!/bin/bash
# Submit the c13 UNI feature-extractor run (c13_uni):
#
#   train hop1 → train hops 2..TRAIN_HOPS (each afterany on the previous hop,
#   CONTINUE_TRAIN=true) → patch eval (afterok on the LAST hop) →
#   WSI (afterok on the LAST hop, REMOTE_UPLOAD=1 → infer → DZI → viewer upload)
#
# The only knob vs the canonical c13 recipe: NETF=uni (gated UNI v1 ViT-L/16
# replaces the frozen ConvNeXt-TCGA contrastive encoder; full-res 1024 tokens,
# gradient checkpointing on). Everything else matches the objective-ablation
# campus recipe: 13ch, 20 epochs, batch 10, 1024², frozen split, eval=latest.
#
# Usage:
#   bash jobs/cislnet/submit_c13_uni.sh
#   DRY_RUN=1 bash jobs/cislnet/submit_c13_uni.sh
#   TRAIN_HOPS=3 bash jobs/cislnet/submit_c13_uni.sh   # longer chains (each hop
#     inherits train.sh's --time; partition MaxTime is 3-00:00:00)
#   RESUME_AFTER=<train JID> bash jobs/cislnet/submit_c13_uni.sh  # skip hop1
#     (recovery: chain hops 2..TRAIN_HOPS afterany:<JID>, te/wsi afterok last hop)

set -euo pipefail
VSLITE_ROOT="/projects/illinois/eng/bioe/rxb/akshat/VSlite"
cd "$VSLITE_ROOT"
# shellcheck source=_lib/slurm_defaults.sh
source "$VSLITE_ROOT/jobs/_lib/slurm_defaults.sh"

EPOCHS="${EPOCHS:-20}"
NUM_TEST="${NUM_TEST:-75000}"
DRY_RUN="${DRY_RUN:-0}"
EXCLUDE="${SLURM_EXCLUDE_NODES}"
CONTINUE_TRAIN="${CONTINUE_TRAIN:-false}"
TRAIN_CHAIN="${TRAIN_CHAIN:-1}"
TRAIN_HOPS="${TRAIN_HOPS:-2}"
# Recovery knob: RESUME_AFTER=<existing train JID> skips hop1 submission and
# chains hops 2..TRAIN_HOPS afterany:<JID>, then te/wsi afterok on the last hop.
# EVAL_AFTER=<existing train JID>: no train hops at all — gate te/wsi on ANY
# terminal state of that job (afterany, incl. walltime timeout) and evaluate
# the `latest` checkpoint (run-until-timeout strategy).
RESUME_AFTER="${RESUME_AFTER:-}"
EVAL_AFTER="${EVAL_AFTER:-}"
TRAIN_TIME="${TRAIN_TIME:-}"
NTHREADS="${NTHREADS:-4}"
BATCH_SIZE="${BATCH_SIZE:-10}"
WHICH_EPOCH="${WHICH_EPOCH:-latest}"
NETF="${NETF:-uni}"
UNI_INPUT_SIZE="${UNI_INPUT_SIZE:-1024}"

# Campus viewer project for WSI uploads (same project as prior ablations).
PROJECT="${PROJECT:-campus_ablations_bestlpips}"
PROJECT_DISPLAY_NAME="${PROJECT_DISPLAY_NAME:-Campus_ablations_best_LPIPS_WSI}"
OUTPUT_TYPE_SUFFIX="${OUTPUT_TYPE_SUFFIX:-latest}"
WSI_FOLDERS="${WSI_FOLDERS:-10442-1:10442-2:10445-2:10445-3:10445-4:3840-1:3840-2:3840-3:3840-4:3840-5}"

RUN="${RUN:-c13_uni}"
TRAIN_SH="$VSLITE_ROOT/jobs/cislnet/train/train.sh"
TEST_SH="$VSLITE_ROOT/jobs/cislnet/test/test.sh"
WSI_SH="$VSLITE_ROOT/jobs/cislnet/wsi/wsi.sh"

_submit_parsable() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "dryrun_${RANDOM}"
    return 0
  fi
  sbatch --parsable "$@"
}

echo "c13 UNI netF run: root=$VSLITE_ROOT run=$RUN EPOCHS=$EPOCHS BS=$BATCH_SIZE NT=$NTHREADS TRAIN_CHAIN=$TRAIN_CHAIN TRAIN_HOPS=$TRAIN_HOPS TRAIN_TIME=${TRAIN_TIME:-script} exclude=$EXCLUDE"
echo "netF=$NETF uni_input_size=$UNI_INPUT_SIZE viewer project=$PROJECT"

if [[ -n "$RESUME_AFTER" && "$TRAIN_CHAIN" != "1" ]]; then
  echo "RESUME_AFTER requires TRAIN_CHAIN=1 (got TRAIN_CHAIN=$TRAIN_CHAIN)" >&2
  exit 2
fi

# Rough train wall-clock estimate for hop budgeting (partition MaxTime=3d/hop).
SECS_PER_SAMPLE="${SECS_PER_SAMPLE:-0.594}"
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-60040}"
est_h=$(awk -v sps="$SECS_PER_SAMPLE" -v spe="$SAMPLES_PER_EPOCH" -v ep="$EPOCHS" \
  'BEGIN{printf "%.0f", sps*spe*ep/3600}')
est_hops=$(( (est_h + 71) / 72 ))
echo "timing estimate: ~${est_h}h train for EPOCHS=$EPOCHS -> need >=${est_hops} hops at <=72h/hop"

mkdir -p "/u/akshahi2/logs/cn_tr_${RUN}" "/u/akshahi2/logs/cn_te_${RUN}" "/u/akshahi2/logs/cn_wsi_${RUN}"

channel_env="NUM_CHANNELS=13"

train_export="ALL,${channel_env},RUN=${RUN},NAME=${RUN},EPOCHS=${EPOCHS},CONTINUE_TRAIN=${CONTINUE_TRAIN},NTHREADS=${NTHREADS},BATCH_SIZE=${BATCH_SIZE},NETF=${NETF},UNI_INPUT_SIZE=${UNI_INPUT_SIZE},VSLITE_UNI_LPIPS=1,VSLITE_ROOT=${VSLITE_ROOT}"
sbatch_extra=()
if [[ -n "$TRAIN_TIME" ]]; then
  sbatch_extra+=(--time="$TRAIN_TIME")
fi

# Continuation-hop export (same list, always CONTINUE_TRAIN=true).
train_export2="$train_export"
train_export2="${train_export2/,CONTINUE_TRAIN=${CONTINUE_TRAIN}/,CONTINUE_TRAIN=true}"
if [[ "$train_export2" != *"CONTINUE_TRAIN=true"* ]]; then
  train_export2+=",CONTINUE_TRAIN=true"
fi

total_hops=1
if [[ "$TRAIN_CHAIN" == "1" ]]; then
  total_hops="$TRAIN_HOPS"
fi

last_train=""
eval_dep="afterok"
if [[ -n "$EVAL_AFTER" ]]; then
  # Eval-on-timeout path: no train hops; gate te/wsi on ANY terminal state of
  # an existing train job (afterany — fires on completion AND on walltime
  # timeout) and evaluate the `latest` checkpoint.
  last_train="$EVAL_AFTER"
  eval_dep="afterany"
  echo "train $RUN hops skipped -> te/wsi gated afterany:$last_train on WHICH_EPOCH=$WHICH_EPOCH"
elif [[ -n "$RESUME_AFTER" ]]; then
  # Recovery path: skip hop1; continue the chain after an existing train job.
  last_train="$RESUME_AFTER"
  echo "train $RUN hop1 skipped -> resuming chain after job $last_train"
else
  J_TRAIN=$(_submit_parsable \
    "${sbatch_extra[@]}" \
    --job-name="cn_tr_${RUN}" \
    --chdir="$VSLITE_ROOT" \
    --exclude="$EXCLUDE" \
    --export="$train_export" \
    "$TRAIN_SH")
  echo "train $RUN hop1/${total_hops} -> job $J_TRAIN"
  last_train="$J_TRAIN"
fi

if [[ "$TRAIN_CHAIN" == "1" && -z "$EVAL_AFTER" ]]; then
  h=2
  while [[ "$h" -le "$TRAIN_HOPS" ]]; do
    J_TRAIN=$(_submit_parsable \
      "${sbatch_extra[@]}" \
      --job-name="cn_tr_${RUN}" \
      --chdir="$VSLITE_ROOT" \
      --exclude="$EXCLUDE" \
      --dependency="afterany:${last_train}" \
      --export="$train_export2" \
      "$TRAIN_SH")
    echo "train $RUN hop${h}/${total_hops} -> job $J_TRAIN (afterany:${last_train}, CONTINUE_TRAIN=true)"
    last_train="$J_TRAIN"
    h=$((h+1))
  done
fi

test_export="ALL,${channel_env},RUN=${RUN},NAME=${RUN},WHICH_EPOCH=${WHICH_EPOCH},NUM_TEST=${NUM_TEST},VSLITE_UNI_LPIPS=1,VSLITE_ROOT=${VSLITE_ROOT}"
J_TEST=$(_submit_parsable \
  --job-name="cn_te_${RUN}" \
  --chdir="$VSLITE_ROOT" \
  --exclude="$EXCLUDE" \
  --dependency="${eval_dep}:${last_train}" \
  --export="$test_export" \
  "$TEST_SH")
echo "test  $RUN -> job $J_TEST (${eval_dep}:$last_train)"

wsi_export="ALL,${channel_env},RUN=${RUN},NAME=${RUN},WHICH_EPOCH=${WHICH_EPOCH},REMOTE_UPLOAD=1,PROJECT=${PROJECT},PROJECT_DISPLAY_NAME=${PROJECT_DISPLAY_NAME},OUTPUT_TYPE=${RUN}-${OUTPUT_TYPE_SUFFIX},OVERWRITE=1,WSI_FOLDERS=${WSI_FOLDERS},VSLITE_ROOT=${VSLITE_ROOT}"
J_WSI=$(_submit_parsable \
  --job-name="cn_wsi_${RUN}" \
  --chdir="$VSLITE_ROOT" \
  --exclude="$EXCLUDE" \
  --dependency="${eval_dep}:${last_train}" \
  --export="$wsi_export" \
  "$WSI_SH")
echo "wsi   $RUN -> job $J_WSI (${eval_dep}:$last_train, project=$PROJECT)"

echo
echo "Done. ckpts=$VSLITE_ROOT/results/cislnet/$RUN/ patch-eval=$VSLITE_ROOT/results/cislnet/test/$RUN/patch_eval_test/"
