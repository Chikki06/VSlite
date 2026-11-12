#!/bin/bash
# Parameterized CISLNet WSI inference.
# Example: NUM_CHANNELS=13 RUN=c13 sbatch wsi.sh
# With REMOTE_UPLOAD=1: stitch → pyvips DZI → viewer upload (no PNG encode).
#SBATCH --job-name=cn_wsi
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --exclude=ccc0388,ccc0390
#SBATCH -A kf4-ic
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --chdir=/projects/illinois/eng/bioe/rxb/akshat/VSlite
#SBATCH --output=/u/akshahi2/logs/%x/%j.out
#SBATCH --error=/u/akshahi2/logs/%x/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akshahi2@illinois.edu

set -euo pipefail
VSLITE_ROOT="/projects/illinois/eng/bioe/rxb/akshat/VSlite"
cd "$VSLITE_ROOT"
# shellcheck source=../../_lib/vslite_env.sh
source "$VSLITE_ROOT/jobs/_lib/vslite_env.sh"
# shellcheck source=../../_lib/cislnet_channels.sh
source "$VSLITE_ROOT/jobs/_lib/cislnet_channels.sh"

RUN="${RUN:-c${NUM_CHANNELS}}"
NAME="${NAME:-$RUN}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-$VSLITE_ROOT/results/cislnet}"
WHICH_EPOCH="${WHICH_EPOCH:-latest}"
# If AUTO_BEST_LPIPS=1, override WHICH_EPOCH from train-log LPIPS.
if [[ "${AUTO_BEST_LPIPS:-0}" == "1" ]]; then
  WHICH_EPOCH="$(python "$VSLITE_ROOT/scripts/pick_best_lpips_epoch.py" --run "$NAME" --verbose)"
  echo "[WSI] AUTO_BEST_LPIPS selected which_epoch=$WHICH_EPOCH"
fi
# Record selection for downstream upload jobs.
SELECTED_EPOCH_FILE="${SELECTED_EPOCH_FILE:-}"
WSI_TILE_SIZE="${WSI_TILE_SIZE:-1024}"
# A100: train uses BS≈10 @1024; inference (no grad) can go higher. Override with WSI_BATCH_SIZE.
WSI_BATCH_SIZE="${WSI_BATCH_SIZE:-16}"
# Parallel CPU tile readers (IR memmap / VIS crop) while packing GPU batches.
WSI_NUM_WORKERS="${WSI_NUM_WORKERS:-4}"
WSI_IMAGE_FORMAT="${WSI_IMAGE_FORMAT:-png}"
WSI_JPEG_QUALITY="${WSI_JPEG_QUALITY:-92}"
WSI_PNG_COMPRESS="${WSI_PNG_COMPRESS:-6}"
# Colon-separated slide names (sbatch --export cannot use commas). Empty = all under parent.
WSI_FOLDERS="${WSI_FOLDERS:-}"
WSI_MAX_FOLDERS="${WSI_MAX_FOLDERS:-0}"

# Optional tag so best-epoch preds don't clobber prior latest WSI folders.
OUT_TAG="${OUT_TAG:-}"
if [[ -z "$OUT_TAG" && "$WHICH_EPOCH" != "latest" ]]; then
  OUT_TAG="e${WHICH_EPOCH}"
fi
if [[ -n "$OUT_TAG" ]]; then
  OUT_DIR="${OUT_DIR:-$CHECKPOINTS_DIR/test/${NAME}_${OUT_TAG}/wsi_pred}"
else
  OUT_DIR="${OUT_DIR:-$CHECKPOINTS_DIR/test/${NAME}/wsi_pred}"
fi
mkdir -p "$OUT_DIR"
if [[ -z "$SELECTED_EPOCH_FILE" ]]; then
  SELECTED_EPOCH_FILE="$(dirname "$OUT_DIR")/selected_epoch.txt"
fi
printf '%s\n' "$WHICH_EPOCH" >"$SELECTED_EPOCH_FILE"
echo "[WSI] wrote $SELECTED_EPOCH_FILE"

CHANNEL_EXTRA=(--input_nc "$NUM_CHANNELS" --num_ir_channels "$NUM_IR")
[[ -n "$IR_INDICES" ]] && CHANNEL_EXTRA+=(--ir_channel_indices "$IR_INDICES")

FOLDER_EXTRA=()
if [[ -n "$WSI_FOLDERS" ]]; then
  FOLDER_EXTRA+=(--folders "${WSI_FOLDERS//:/,}")
fi
if [[ "${WSI_MAX_FOLDERS}" -gt 0 ]]; then
  FOLDER_EXTRA+=(--max_folders "$WSI_MAX_FOLDERS")
fi

# ---------------------------------------------------------------------------
# REMOTE_UPLOAD=1: GPU stitch → pyvips DZI from uint8 canvas → upload.
# Skips multi-GB PNG/JPEG encode (viewer consumes DZI directly).
# ---------------------------------------------------------------------------
if [[ "${REMOTE_UPLOAD:-0}" == "1" ]]; then
  # shellcheck disable=SC1091
  source /u/akshahi2/.secrets.sh
  : "${CISLVIEWER_REMOTE_TOKEN:?missing CISLVIEWER_REMOTE_TOKEN}"
  : "${CISLVIEWER_SITE:=https://cisl-inference.beckman.illinois.edu}"
  export CISLVIEWER_REMOTE_TOKEN CISLVIEWER_SITE

  PROJECT="${PROJECT:-campus_ablations_bestlpips}"
  OUTPUT_TYPE="${OUTPUT_TYPE:-${NAME}-e${WHICH_EPOCH}}"
  if [[ -f "$SELECTED_EPOCH_FILE" ]]; then
    _ep="$(tr -d '[:space:]' <"$SELECTED_EPOCH_FILE")"
    if [[ -n "$_ep" && "$_ep" != "latest" ]]; then
      OUTPUT_TYPE="${NAME}-e${_ep}"
    fi
  fi
  SCRATCH="${REMOTE_SCRATCH:-$VSLITE_ROOT/results/cislnet/remote_scratch/${PROJECT}/${SLURM_JOB_ID:-$$}}"
  mkdir -p "$SCRATCH"
  # Viewer model_type: 1 when VIS channels present (input_nc == n_ir + 3).
  if [[ -z "${MODEL:-}" ]]; then
    if [[ "$NUM_CHANNELS" -eq $((NUM_IR + 3)) ]]; then
      MODEL=1
    else
      MODEL=0
    fi
  fi
  UP_EXTRA=()
  [[ "${OVERWRITE:-1}" == "1" ]] && UP_EXTRA+=(--overwrite)
  [[ "${WSI_OPTIM:-0}" == "1" ]] && UP_EXTRA+=(--optim)
  [[ "${WSI_LOCAL_PADDING:-0}" == "1" ]] && UP_EXTRA+=(--local_padding)

  echo "[WSI] remote infer→DZI→upload project=$PROJECT output_type=$OUTPUT_TYPE (no PNG)"
  echo "[WSI] name=$NAME which_epoch=$WHICH_EPOCH scratch=$SCRATCH"

  python "$VSLITE_ROOT/scripts/remote_wsi_upload.py" \
    --infer_cislnet \
    --name "$NAME" \
    --checkpoints_dir "$CHECKPOINTS_DIR" \
    --which_epoch "$WHICH_EPOCH" \
    --results_dir "$CHECKPOINTS_DIR" \
    --parent_folder "$VSLITE_ROOT/input/IR" \
    --input_root "$VSLITE_ROOT/input/IR" \
    --tile_size "$WSI_TILE_SIZE" \
    --batch_size "$WSI_BATCH_SIZE" \
    --num_workers "$WSI_NUM_WORKERS" \
    --project "$PROJECT" \
    --project_display_name "${PROJECT_DISPLAY_NAME:-Campus ablations best-LPIPS WSI}" \
    --project_description "${PROJECT_DESCRIPTION:-Best-LPIPS WSI remote upload}" \
    --output_type "$OUTPUT_TYPE" \
    --model "$MODEL" \
    --create_project \
    --ensure_project \
    --scratch "$SCRATCH" \
    "${CHANNEL_EXTRA[@]}" \
    "${FOLDER_EXTRA[@]}" \
    "${UP_EXTRA[@]}"
  exit 0
fi

# ---------------------------------------------------------------------------
# Default: write WSI pred PNG/JPEG under results/.../wsi_pred (local eval).
# ---------------------------------------------------------------------------
EXTRA=("${CHANNEL_EXTRA[@]}")
EXTRA+=(--image_format "$WSI_IMAGE_FORMAT")
EXTRA+=(--jpeg_quality "$WSI_JPEG_QUALITY")
EXTRA+=(--png_compress_level "$WSI_PNG_COMPRESS")
EXTRA+=(--which_epoch "$WHICH_EPOCH")
EXTRA+=("${FOLDER_EXTRA[@]}")

echo "[WSI] name=$NAME which_epoch=$WHICH_EPOCH out_dir=$OUT_DIR"

python scripts/infer_wsi_png_cislnet.py \
  --name "$NAME" \
  --checkpoints_dir "$CHECKPOINTS_DIR" \
  --parent_folder "$VSLITE_ROOT/input/IR" \
  --output_dir "$OUT_DIR" \
  --tile_size "$WSI_TILE_SIZE" \
  --batch_size "$WSI_BATCH_SIZE" \
  --num_workers "$WSI_NUM_WORKERS" \
  "${EXTRA[@]}"
