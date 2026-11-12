#!/bin/bash
# Legacy: convert existing WSI pred PNGs/JPEGs → DZI and upload to CISL Viewer.
# Prefer REMOTE_UPLOAD=1 on jobs/cislnet/wsi/wsi.sh (infer→DZI→upload, no PNG).
# This script is for re-uploading already-written results/.../wsi_pred trees.
# Requires libvips/pyvips in torchenv.
#
# Example:
#   PROJECT=campus_c13_wsi OUTPUT_TYPE=CISLNet-c13 \
#     WSI_PRED_DIR=results/cislnet/test/vs13/wsi_pred \
#     sbatch jobs/cislnet/wsi/remote_upload_vs13.sh
#
#SBATCH --job-name=cn_remote_up
#SBATCH --partition=scavenger
#SBATCH -A kf4-ic
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --chdir=/projects/illinois/eng/bioe/rxb/akshat/VSlite
#SBATCH --output=/u/akshahi2/logs/%x/%j.out
#SBATCH --error=/u/akshahi2/logs/%x/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akshahi2@illinois.edu
# Override partition if needed: sbatch -p IllinoisComputes ...
set -euo pipefail
VSLITE_ROOT="/projects/illinois/eng/bioe/rxb/akshat/VSlite"
cd "$VSLITE_ROOT"
# CPU-only job: skip GPU readiness gate in vslite_env.sh
export VSLITE_SKIP_CUDA_CHECK=1
# shellcheck source=../../_lib/vslite_env.sh
source "$VSLITE_ROOT/jobs/_lib/vslite_env.sh"

# Remote token / site (never echo token)
# shellcheck disable=SC1091
source /u/akshahi2/.secrets.sh
: "${CISLVIEWER_REMOTE_TOKEN:?missing CISLVIEWER_REMOTE_TOKEN in .secrets.sh}"
: "${CISLVIEWER_SITE:=https://cisl-inference.beckman.illinois.edu}"
export CISLVIEWER_REMOTE_TOKEN CISLVIEWER_SITE

PROJECT="${PROJECT:-campus_c13_wsi}"
OUTPUT_TYPE="${OUTPUT_TYPE:-CISLNet-c13}"
MODEL="${MODEL:-1}"
WSI_PRED_DIR="${WSI_PRED_DIR:-$VSLITE_ROOT/results/cislnet/test/vs13/wsi_pred}"
INPUT_ROOT="${INPUT_ROOT:-$VSLITE_ROOT/input/IR}"
SCRATCH="${SCRATCH:-$VSLITE_ROOT/results/cislnet/remote_scratch/${PROJECT}/${SLURM_JOB_ID:-$$}}"
SAMPLES="${SAMPLES:-}"
# If a WSI job wrote selected_epoch.txt, refine OUTPUT_TYPE to include the epoch.
if [[ -n "${BEST_EPOCH_SIDECAR:-}" && -f "${BEST_EPOCH_SIDECAR}" ]]; then
  _ep="$(tr -d '[:space:]' <"${BEST_EPOCH_SIDECAR}")"
  if [[ -n "$_ep" && "$_ep" != "latest" ]]; then
    # Replace trailing -bestlpips with -eN when present; else append.
    if [[ "$OUTPUT_TYPE" == *-bestlpips ]]; then
      OUTPUT_TYPE="${OUTPUT_TYPE%-bestlpips}-e${_ep}"
    else
      OUTPUT_TYPE="${OUTPUT_TYPE}-e${_ep}"
    fi
  fi
  echo "BEST_EPOCH_SIDECAR -> epoch=$_ep output_type=$OUTPUT_TYPE"
fi
# sbatch --export cannot embed commas; accept colon-separated too
if [[ -n "$SAMPLES" && "$SAMPLES" == *:* && "$SAMPLES" != *,* ]]; then
  SAMPLES="${SAMPLES//:/,}"
fi
LIMIT="${LIMIT:-0}"
DRY_RUN="${DRY_RUN:-0}"
OVERWRITE="${OVERWRITE:-0}"

mkdir -p "$SCRATCH" "/u/akshahi2/logs/cn_remote_up"

EXTRA=()
[[ -n "$SAMPLES" ]] && EXTRA+=(--samples "$SAMPLES")
[[ "$LIMIT" -gt 0 ]] && EXTRA+=(--limit "$LIMIT")
[[ "$DRY_RUN" == "1" ]] && EXTRA+=(--dry-run)
[[ "$OVERWRITE" == "1" ]] && EXTRA+=(--overwrite)

echo "project=$PROJECT output_type=$OUTPUT_TYPE model=$MODEL"
echo "wsi_pred_dir=$WSI_PRED_DIR (legacy pred→DZI path)"
echo "samples=${SAMPLES:-ALL} limit=$LIMIT"
echo "scratch=$SCRATCH"
echo "site=$CISLVIEWER_SITE"

python "$VSLITE_ROOT/scripts/remote_wsi_upload.py" \
  --wsi_pred_dir "$WSI_PRED_DIR" \
  --input_root "$INPUT_ROOT" \
  --project "$PROJECT" \
  --project_display_name "${PROJECT_DISPLAY_NAME:-Campus CISLNet c13 WSI}" \
  --project_description "${PROJECT_DESCRIPTION:-Remote upload of WSI preds from campus cluster}" \
  --output_type "$OUTPUT_TYPE" \
  --model "$MODEL" \
  --create_project \
  --ensure_project \
  --scratch "$SCRATCH" \
  "${EXTRA[@]}"
