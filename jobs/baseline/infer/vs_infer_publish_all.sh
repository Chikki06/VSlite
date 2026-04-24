#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VSLITE_ROOT_DEFAULT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="${1:-$SCRIPT_DIR/vs_infer_publish.env}"

as_bool() {
  local val
  val="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
  [[ "$val" == "1" || "$val" == "true" || "$val" == "yes" || "$val" == "on" ]]
}

# Capture CLI-passed environment variables before sourcing config
_CLI_LOCAL_STAGING_DIR="${LOCAL_STAGING_DIR:-}"
_CLI_REMOTE_PROJECT_BASE="${REMOTE_PROJECT_BASE:-}"
_CLI_PROJECT_ID="${PROJECT_ID:-}"
_CLI_REMOTE_HOST="${REMOTE_HOST:-}"
_CLI_REMOTE_BASE_DIR="${REMOTE_BASE_DIR:-}"
_CLI_DB_UPDATE_MODE="${DB_UPDATE_MODE:-}"
_CLI_REMOTE_SHELL_ENABLED="${REMOTE_SHELL_ENABLED:-}"
_CLI_SYNC_ENABLED="${SYNC_ENABLED:-}"
_CLI_DB_UPDATE_ENABLED="${DB_UPDATE_ENABLED:-}"
_CLI_DRY_RUN="${DRY_RUN:-}"
_CLI_INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-}"
_CLI_INFER_NUM_WORKERS="${INFER_NUM_WORKERS:-}"
_CLI_ENABLE_OPTIM="${ENABLE_OPTIM:-}"
_CLI_INPUT_PARENT="${INPUT_PARENT:-}"
_CLI_MODEL_DISCOVERY_ROOT="${MODEL_DISCOVERY_ROOT:-}"
_CLI_MODEL_NAME_PATTERN="${MODEL_NAME_PATTERN:-}"
_CLI_SAVE_PNG_ONLY="${SAVE_PNG_ONLY:-}"

if [[ -f "$CONFIG_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$CONFIG_FILE"
fi

# Restore CLI-passed environment variables (they take precedence over config)
[[ -n "$_CLI_LOCAL_STAGING_DIR" ]] && LOCAL_STAGING_DIR="$_CLI_LOCAL_STAGING_DIR"
[[ -n "$_CLI_REMOTE_PROJECT_BASE" ]] && REMOTE_PROJECT_BASE="$_CLI_REMOTE_PROJECT_BASE"
[[ -n "$_CLI_PROJECT_ID" ]] && PROJECT_ID="$_CLI_PROJECT_ID"
[[ -n "$_CLI_REMOTE_HOST" ]] && REMOTE_HOST="$_CLI_REMOTE_HOST"
[[ -n "$_CLI_REMOTE_BASE_DIR" ]] && REMOTE_BASE_DIR="$_CLI_REMOTE_BASE_DIR"
[[ -n "$_CLI_DB_UPDATE_MODE" ]] && DB_UPDATE_MODE="$_CLI_DB_UPDATE_MODE"
[[ -n "$_CLI_REMOTE_SHELL_ENABLED" ]] && REMOTE_SHELL_ENABLED="$_CLI_REMOTE_SHELL_ENABLED"
[[ -n "$_CLI_SYNC_ENABLED" ]] && SYNC_ENABLED="$_CLI_SYNC_ENABLED"
[[ -n "$_CLI_DB_UPDATE_ENABLED" ]] && DB_UPDATE_ENABLED="$_CLI_DB_UPDATE_ENABLED"
[[ -n "$_CLI_DRY_RUN" ]] && DRY_RUN="$_CLI_DRY_RUN"
[[ -n "$_CLI_INFER_BATCH_SIZE" ]] && INFER_BATCH_SIZE="$_CLI_INFER_BATCH_SIZE"
[[ -n "$_CLI_INFER_NUM_WORKERS" ]] && INFER_NUM_WORKERS="$_CLI_INFER_NUM_WORKERS"
[[ -n "$_CLI_ENABLE_OPTIM" ]] && ENABLE_OPTIM="$_CLI_ENABLE_OPTIM"
[[ -n "$_CLI_INPUT_PARENT" ]] && INPUT_PARENT="$_CLI_INPUT_PARENT"
[[ -n "$_CLI_MODEL_DISCOVERY_ROOT" ]] && MODEL_DISCOVERY_ROOT="$_CLI_MODEL_DISCOVERY_ROOT"
[[ -n "$_CLI_MODEL_NAME_PATTERN" ]] && MODEL_NAME_PATTERN="$_CLI_MODEL_NAME_PATTERN"
[[ -n "$_CLI_SAVE_PNG_ONLY" ]] && SAVE_PNG_ONLY="$_CLI_SAVE_PNG_ONLY"

VSLITE_ROOT="${VSLITE_ROOT:-$VSLITE_ROOT_DEFAULT}"
INPUT_PARENT="${INPUT_PARENT:-$VSLITE_ROOT/input/IR}"
MODEL_DISCOVERY_ROOT="${MODEL_DISCOVERY_ROOT:-$VSLITE_ROOT/results}"
MODEL_NAME_PATTERN="${MODEL_NAME_PATTERN:-latest_net_G.pth}"
MODEL_OVERRIDES_FILE="${MODEL_OVERRIDES_FILE:-}"
MODALITY_PREFIX="${MODALITY_PREFIX:-VS}"

PYTHON_BIN="${PYTHON_BIN:-python}"
INFER_SCRIPT="${INFER_SCRIPT:-$VSLITE_ROOT/tests/main.py}"
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-4}"
INFER_NUM_WORKERS="${INFER_NUM_WORKERS:-0}"
ENABLE_OPTIM="${ENABLE_OPTIM:-false}"
SAVE_PNG_ONLY="${SAVE_PNG_ONLY:-false}"

LOCAL_STAGING_DIR="${LOCAL_STAGING_DIR:-$HOME/vs_infer_staging}"
REMOTE_PROJECT_BASE="${REMOTE_PROJECT_BASE:-/mnt/files}"
PROJECT_ID="${PROJECT_ID:-444}"

REMOTE_HOST="${REMOTE_HOST:-akshahi2@cisl-inference.beckman.illinois.edu}"
REMOTE_BASE_DIR="${REMOTE_BASE_DIR:-/mnt/files/444}"
REMOTE_WORKDIR="${REMOTE_WORKDIR:-/tmp/vs_publish_akshahi2}"
REMOTE_DB_PATH="${REMOTE_DB_PATH:-/home/labuser/SocketInference/viewer/instance/tissue_images.db}"
REMOTE_HELPER_PATH="${REMOTE_HELPER_PATH:-/home/labuser/SocketInference/viewer/misc}"
DB_UPDATE_MODE="${DB_UPDATE_MODE:-remote}"
REMOTE_SHELL_ENABLED="${REMOTE_SHELL_ENABLED:-false}"

SYNC_ENABLED="${SYNC_ENABLED:-true}"
DB_UPDATE_ENABLED="${DB_UPDATE_ENABLED:-true}"
DRY_RUN="${DRY_RUN:-false}"
STATE_DIR="${STATE_DIR:-$VSLITE_ROOT/results/inference_publish_state}"

if as_bool "$SAVE_PNG_ONLY"; then
  SYNC_ENABLED="false"
  DB_UPDATE_ENABLED="false"
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$STATE_DIR/$TIMESTAMP"
MANIFEST_CSV="$RUN_DIR/publish_manifest.csv"
LOG_FILE="$RUN_DIR/run.log"

mkdir -p "$RUN_DIR"

log() {
  echo "[$(date -Is)] $*" | tee -a "$LOG_FILE"
}

die() {
  log "ERROR: $*"
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

sanitize() {
  echo "$1" | sed -E 's/[^A-Za-z0-9]+/_/g; s/^_+//; s/_+$//; s/__+/_/g'
}

extract_resolution() {
  local dzi_file="$1"
  local width height
  width="$(sed -n 's/.*Width="\([0-9]\+\)".*/\1/p' "$dzi_file" | head -n 1)"
  height="$(sed -n 's/.*Height="\([0-9]\+\)".*/\1/p' "$dzi_file" | head -n 1)"
  if [[ -n "$width" && -n "$height" ]]; then
    echo "${width}x${height}"
  else
    echo ""
  fi
}

infer_model_type() {
  local checkpoint="$1"
  local lower
  lower="$(echo "$checkpoint" | tr '[:upper:]' '[:lower:]')"
  if [[ "$lower" =~ (^|[^0-9])13(ch)?([^0-9]|$) ]] || [[ "$lower" == *"vs13"* ]]; then
    echo "1"
    return 0
  fi
  if [[ "$lower" =~ (^|[^0-9])10(ch)?([^0-9]|$) ]] || [[ "$lower" == *"vs10"* ]]; then
    echo "0"
    return 0
  fi
  return 1
}

lookup_override() {
  local checkpoint="$1"
  if [[ -z "$MODEL_OVERRIDES_FILE" || ! -f "$MODEL_OVERRIDES_FILE" ]]; then
    return 1
  fi
  while IFS='|' read -r path model_type modality; do
    [[ -z "$path" || "$path" =~ ^[[:space:]]*# ]] && continue
    if [[ "$path" == "$checkpoint" ]]; then
      echo "${model_type}|${modality}"
      return 0
    fi
  done < "$MODEL_OVERRIDES_FILE"
  return 1
}

build_modality_name() {
  local checkpoint="$1"
  local model_type="$2"
  local parent
  parent="$(basename "$(dirname "$checkpoint")")"
  local parent_safe
  parent_safe="$(sanitize "$parent")"
  local suffix
  if [[ "$model_type" == "1" ]]; then
    suffix="13ch"
  else
    suffix="10ch"
  fi
  echo "${MODALITY_PREFIX}_${parent_safe}_${suffix}"
}

write_manifest_for_modality() {
  local modality="$1"
  local staging_base="${2:-$LOCAL_STAGING_DIR}"
  local dzi_dir="$staging_base/$PROJECT_ID/dzi/$modality"
  if [[ ! -d "$dzi_dir" ]]; then
    log "No DZI directory for modality $modality at $dzi_dir"
    return 0
  fi

  local dzi_file name resolution
  shopt -s nullglob
  for dzi_file in "$dzi_dir"/*.dzi; do
    name="$(basename "$dzi_file" .dzi)"
    resolution="$(extract_resolution "$dzi_file")"
    [[ -z "$resolution" ]] && continue
    echo "$name,$resolution,$modality,$PROJECT_ID" >> "$MANIFEST_CSV"
  done
  shopt -u nullglob
}

run_cmd() {
  if as_bool "$DRY_RUN"; then
    log "DRY_RUN: $*"
  else
    eval "$*"
  fi
}

require_cmd "$PYTHON_BIN"
require_cmd find
require_cmd sort
require_cmd rsync
require_cmd ssh
require_cmd sed

[[ -f "$INFER_SCRIPT" ]] || die "Inference script not found: $INFER_SCRIPT"
[[ -d "$INPUT_PARENT" ]] || die "Input parent not found: $INPUT_PARENT"
[[ -d "$MODEL_DISCOVERY_ROOT" ]] || die "Model discovery root not found: $MODEL_DISCOVERY_ROOT"

mapfile -t checkpoints < <(find "$MODEL_DISCOVERY_ROOT" -type f -name "$MODEL_NAME_PATTERN" | sort)
[[ ${#checkpoints[@]} -gt 0 ]] || die "No checkpoints found under $MODEL_DISCOVERY_ROOT matching $MODEL_NAME_PATTERN"

mapfile -t folders < <(find "$INPUT_PARENT" -mindepth 1 -maxdepth 1 -type d ! -name '*_VIS' | sort)
[[ ${#folders[@]} -gt 0 ]] || die "No input folders found under $INPUT_PARENT"

log "Discovered ${#checkpoints[@]} checkpoints"

if as_bool "$DRY_RUN"; then
  log "DRY_RUN: would prepare local staging at $LOCAL_STAGING_DIR/$PROJECT_ID"
else
  mkdir -p "$LOCAL_STAGING_DIR/$PROJECT_ID/dzi" "$LOCAL_STAGING_DIR/$PROJECT_ID/thumbnail"
fi
printf "name,resolution,image_type,project\n" > "$MANIFEST_CSV"

declare -a successful_modalities=()
declare -a successful_checkpoints=()

for checkpoint in "${checkpoints[@]}"; do
  model_type=""
  modality=""

  if override="$(lookup_override "$checkpoint")"; then
    model_type="${override%%|*}"
    modality="${override#*|}"
    [[ -n "$modality" ]] || modality="$(build_modality_name "$checkpoint" "$model_type")"
  else
    if ! model_type="$(infer_model_type "$checkpoint")"; then
      log "SKIP: Could not infer model type (10ch/13ch) from $checkpoint"
      continue
    fi
    modality="$(build_modality_name "$checkpoint" "$model_type")"
  fi

  log "Running inference: model_type=$model_type modality=$modality checkpoint=$checkpoint folders=${#folders[@]}"

  for folder_path in "${folders[@]}"; do
    folder_name="$(basename "$folder_path")"
    log "  -> folder=$folder_name"

    infer_cmd="export SOCKETINFERENCE_FILES_BASE_DIR=\"$LOCAL_STAGING_DIR\" && $PYTHON_BIN \"$INFER_SCRIPT\" \
      --folder \"$folder_path\" \
      --model \"$model_type\" \
      --model_path \"$checkpoint\" \
      --batch_size \"$INFER_BATCH_SIZE\" \
      --n_cpu \"$INFER_NUM_WORKERS\" \
      --project \"$PROJECT_ID\" \
      --vs_modality \"$modality\""

    if as_bool "$ENABLE_OPTIM"; then
      infer_cmd+=" --optim"
    fi

    if as_bool "$SAVE_PNG_ONLY"; then
      infer_cmd+=" --save_png"
    fi

    infer_cmd+=" --image_type \"$([[ \"$model_type\" == \"1\" ]] && echo IR+VIS || echo IR)\""

    run_cmd "$infer_cmd"
  done

  if ! as_bool "$DRY_RUN"; then
    write_manifest_for_modality "$modality" "$LOCAL_STAGING_DIR"
  fi

  successful_modalities+=("$modality")
  successful_checkpoints+=("$checkpoint")
done

if [[ ${#successful_modalities[@]} -eq 0 ]]; then
  die "No inference runs were completed"
fi

log "Completed inference for ${#successful_modalities[@]} modalities"

if as_bool "$SYNC_ENABLED"; then
  log "Syncing DZI/thumbnail outputs to $REMOTE_HOST:$REMOTE_BASE_DIR"

  if as_bool "$REMOTE_SHELL_ENABLED" && ! as_bool "$DRY_RUN"; then
    run_cmd "ssh \"$REMOTE_HOST\" 'mkdir -p \"$REMOTE_BASE_DIR/dzi\" \"$REMOTE_BASE_DIR/thumbnail\"'"
  elif ! as_bool "$REMOTE_SHELL_ENABLED"; then
    log "REMOTE_SHELL_ENABLED=false: skipping remote mkdir via ssh; expecting base dirs to already exist"
  fi

  declare -A seen=()
  for modality in "${successful_modalities[@]}"; do
    [[ -n "${seen[$modality]:-}" ]] && continue
    seen[$modality]=1

    local_dzi="$LOCAL_STAGING_DIR/$PROJECT_ID/dzi/$modality/"
    local_thumb="$LOCAL_STAGING_DIR/$PROJECT_ID/thumbnail/$modality/"
    remote_dzi="$REMOTE_PROJECT_BASE/$PROJECT_ID/dzi/$modality/"
    remote_thumb="$REMOTE_PROJECT_BASE/$PROJECT_ID/thumbnail/$modality/"

    if [[ -d "$local_dzi" ]]; then
      if as_bool "$REMOTE_SHELL_ENABLED"; then
        run_cmd "ssh \"$REMOTE_HOST\" 'mkdir -p \"$remote_dzi\"'"
      fi
      run_cmd "rsync -az --delete \"$local_dzi\" \"$REMOTE_HOST:$remote_dzi\""
    else
      log "WARN: Missing local DZI dir: $local_dzi"
    fi

    if [[ -d "$local_thumb" ]]; then
      if as_bool "$REMOTE_SHELL_ENABLED"; then
        run_cmd "ssh \"$REMOTE_HOST\" 'mkdir -p \"$remote_thumb\"'"
      fi
      run_cmd "rsync -az --delete \"$local_thumb\" \"$REMOTE_HOST:$remote_thumb\""
    else
      log "WARN: Missing local thumbnail dir: $local_thumb"
    fi
  done
fi

if as_bool "$DB_UPDATE_ENABLED"; then
  if [[ ! -s "$MANIFEST_CSV" ]]; then
    log "No manifest entries; skipping DB update"
  else
    log "Updating DB using mode=$DB_UPDATE_MODE"

    helper_local="$SCRIPT_DIR/remote_db_upsert.py"
    [[ -f "$helper_local" ]] || die "Missing helper script: $helper_local"

    case "$DB_UPDATE_MODE" in
      remote)
        if ! as_bool "$REMOTE_SHELL_ENABLED"; then
          die "DB_UPDATE_MODE=remote requires REMOTE_SHELL_ENABLED=true"
        fi
        run_cmd "ssh \"$REMOTE_HOST\" 'mkdir -p \"$REMOTE_WORKDIR\"'"
        run_cmd "rsync -az \"$helper_local\" \"$MANIFEST_CSV\" \"$REMOTE_HOST:$REMOTE_WORKDIR/\""
        run_cmd "ssh \"$REMOTE_HOST\" '$PYTHON_BIN \"$REMOTE_WORKDIR/remote_db_upsert.py\" --db-path \"$REMOTE_DB_PATH\" --manifest \"$REMOTE_WORKDIR/$(basename \"$MANIFEST_CSV\")\" --helper-path \"$REMOTE_HELPER_PATH\"'"
        ;;
      roundtrip)
        local_db="$RUN_DIR/tissue_images.db"
        run_cmd "rsync -az \"$REMOTE_HOST:$REMOTE_DB_PATH\" \"$local_db\""
        run_cmd "$PYTHON_BIN \"$helper_local\" --db-path \"$local_db\" --manifest \"$MANIFEST_CSV\""
        run_cmd "rsync -az \"$local_db\" \"$REMOTE_HOST:$REMOTE_DB_PATH\""
        ;;
      rsync_only)
        local_db="$RUN_DIR/tissue_images.db"
        run_cmd "rsync -az \"$REMOTE_HOST:$REMOTE_DB_PATH\" \"$local_db\""
        run_cmd "$PYTHON_BIN \"$helper_local\" --db-path \"$local_db\" --manifest \"$MANIFEST_CSV\""
        run_cmd "rsync -az \"$local_db\" \"$REMOTE_HOST:$REMOTE_DB_PATH\""
        ;;
      *)
        die "Unsupported DB_UPDATE_MODE: $DB_UPDATE_MODE"
        ;;
    esac
  fi
fi

log "All done. Manifest: $MANIFEST_CSV"
log "Run directory: $RUN_DIR"
