#!/bin/bash
# Submit WSI + inline remote upload (same job) for best-LPIPS ablation epochs.
# Default: IllinoisComputes-GPU (non-preemptible). Upload runs at end of WSI job.
#
# Usage:
#   bash jobs/cislnet/wsi/submit_best_lpips_ablations.sh
#   DRY_RUN=1 bash jobs/cislnet/wsi/submit_best_lpips_ablations.sh
set -euo pipefail

VSLITE_ROOT="${VSLITE_ROOT:-/projects/illinois/eng/bioe/rxb/akshat/VSlite}"
cd "$VSLITE_ROOT"

PROJECT="${PROJECT:-campus_ablations_bestlpips}"
PROJECT_DISPLAY="${PROJECT_DISPLAY_NAME:-Campus_ablations_best_LPIPS_WSI}"
DRY_RUN="${DRY_RUN:-0}"
WSI_SH="$VSLITE_ROOT/jobs/cislnet/wsi/wsi.sh"
WSI_PARTITION="${WSI_PARTITION:-IllinoisComputes-GPU}"
WSI_GRES="${WSI_GRES:-gpu:A100:1}"
WSI_TIME="${WSI_TIME:-24:00:00}"
WSI_EXCLUDE="${WSI_EXCLUDE:-ccc0388}"

FINISHED=(
  "vp1|20|VISPLUS=1"
  "vp3|20|VISPLUS=3"
  "vp5|20|VISPLUS=5"
  "a1_coordatt_c13|15|NUM_CHANNELS=13"
  "a2_triplet_c13|15|NUM_CHANNELS=13"
  "a3_scse_c13|10|NUM_CHANNELS=13"
  "b1_half_c13|20|NUM_CHANNELS=13"
)

B2_TRAIN_JOB="${B2_TRAIN_JOB:-10602773}"
B3_TRAIN_JOB="${B3_TRAIN_JOB:-10602775}"

_submit() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY sbatch $*" >&2
    echo "dryrun_${RANDOM}"
    return 0
  fi
  sbatch --parsable "$@"
}

mkdir -p /u/akshahi2/logs/cn_wsi_best

echo "project=$PROJECT partition=$WSI_PARTITION (inline REMOTE_UPLOAD=1)"
echo "Finished ablations → WSI+upload (single job each)"

for spec in "${FINISHED[@]}"; do
  IFS='|' read -r NAME EPOCH EXTRA <<<"$spec"
  OUT_TAG="e${EPOCH}"
  OUTPUT_TYPE="${NAME}-e${EPOCH}"
  JOB_NAME="cn_wsi_${NAME}"

  # REMOTE_UPLOAD=1 → infer→DZI→upload (no PNG); see jobs/cislnet/wsi/wsi.sh
  export_wsi="ALL,NAME=${NAME},RUN=${NAME},WHICH_EPOCH=${EPOCH},OUT_TAG=${OUT_TAG},REMOTE_UPLOAD=1,PROJECT=${PROJECT},PROJECT_DISPLAY_NAME=${PROJECT_DISPLAY},OUTPUT_TYPE=${OUTPUT_TYPE},OVERWRITE=1,${EXTRA}"
  J_WSI=$(_submit \
    --partition="$WSI_PARTITION" \
    --gres="$WSI_GRES" \
    --time="$WSI_TIME" \
    --exclude="$WSI_EXCLUDE" \
    --job-name="$JOB_NAME" \
    --output="/u/akshahi2/logs/cn_wsi_best/%j.out" \
    --error="/u/akshahi2/logs/cn_wsi_best/%j.err" \
    --export="$export_wsi" \
    "$WSI_SH")
  echo "  $NAME epoch=$EPOCH WSI+upload -> $J_WSI"
done

echo
echo "b2/b3 → WSI+upload after train (AUTO_BEST_LPIPS)"

_submit_pending() {
  local NAME="$1"
  local TRAIN_JOB="$2"
  local EXTRA="$3"
  local JOB_NAME="cn_wsi_${NAME}"
  local OUTPUT_TYPE="${NAME}-bestlpips"
  local export_wsi="ALL,NAME=${NAME},RUN=${NAME},AUTO_BEST_LPIPS=1,OUT_TAG=bestlpips,REMOTE_UPLOAD=1,PROJECT=${PROJECT},PROJECT_DISPLAY_NAME=${PROJECT_DISPLAY},OUTPUT_TYPE=${OUTPUT_TYPE},OVERWRITE=1,${EXTRA}"
  local dep=()
  if [[ -n "$TRAIN_JOB" && "$TRAIN_JOB" != "0" ]]; then
    dep=(--dependency="afterok:${TRAIN_JOB}")
  fi
  local J_WSI
  J_WSI=$(_submit \
    "${dep[@]}" \
    --partition="$WSI_PARTITION" \
    --gres="$WSI_GRES" \
    --time="$WSI_TIME" \
    --exclude="$WSI_EXCLUDE" \
    --job-name="$JOB_NAME" \
    --output="/u/akshahi2/logs/cn_wsi_best/%j.out" \
    --error="/u/akshahi2/logs/cn_wsi_best/%j.err" \
    --export="$export_wsi" \
    "$WSI_SH")
  echo "  $NAME WSI+upload afterok:${TRAIN_JOB:-none} -> $J_WSI"
}

_submit_pending "b2_scconv_c13" "$B2_TRAIN_JOB" "NUM_CHANNELS=13"
_submit_pending "b3_convnext_c13" "$B3_TRAIN_JOB" "NUM_CHANNELS=13"

echo
echo "Done. Viewer project: $PROJECT"
echo "Skip c13 until cn_te_c13 finishes."
