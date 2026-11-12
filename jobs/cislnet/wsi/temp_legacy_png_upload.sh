#!/bin/bash
# One-shot: upload existing wsi_pred PNGs → DZI → viewer. No leftover scratch/artifacts.
#SBATCH --job-name=cn_legacy_png
#SBATCH --partition=scavenger
#SBATCH -A kf4-ic
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --chdir=/projects/illinois/eng/bioe/rxb/akshat/VSlite
#SBATCH --output=/u/akshahi2/logs/cn_remote_legacy/%j.out
#SBATCH --error=/u/akshahi2/logs/cn_remote_legacy/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akshahi2@illinois.edu
set -euo pipefail
VSLITE_ROOT="/projects/illinois/eng/bioe/rxb/akshat/VSlite"
cd "$VSLITE_ROOT"
export VSLITE_SKIP_CUDA_CHECK=1
source "$VSLITE_ROOT/jobs/_lib/vslite_env.sh"
source /u/akshahi2/.secrets.sh
: "${CISLVIEWER_REMOTE_TOKEN:?missing token}"
: "${CISLVIEWER_SITE:=https://cisl-inference.beckman.illinois.edu}"
export CISLVIEWER_REMOTE_TOKEN CISLVIEWER_SITE

PROJECT="${PROJECT:-campus_ablations_bestlpips}"
TEST_ROOT="$VSLITE_ROOT/results/cislnet/test"
# scratch lives under job TMP and is wiped on EXIT — no results/ leftovers
SCRATCH_ROOT="${TMPDIR:-/tmp}/cisl_legacy_png_${SLURM_JOB_ID:-$$}"
mkdir -p "$SCRATCH_ROOT"
cleanup() { rm -rf "$SCRATCH_ROOT"; }
trap cleanup EXIT

# dir_under_test | output_type | model_type
UPLOADS=(
  "vp1_e20|vp1-e20|1"
  "vp3_e20|vp3-e20|1"
  "vp5_e20|vp5-e20|1"
  "a1_coordatt_c13_e15|a1_coordatt_c13-e15|1"
  "a2_triplet_c13_e15|a2_triplet_c13-e15|1"
  "a3_scse_c13_e10|a3_scse_c13-e10|1"
  "b1_half_c13_e20|b1_half_c13-e20|1"
)

echo "[legacy] project=$PROJECT scratch=$SCRATCH_ROOT (cleaned on exit)"
failures=0
for spec in "${UPLOADS[@]}"; do
  IFS='|' read -r DIR OUT_TYPE MODEL <<<"$spec"
  PRED="$TEST_ROOT/$DIR/wsi_pred"
  if [[ ! -d "$PRED" ]]; then
    echo "[legacy] skip missing $PRED"
    continue
  fi
  n=$(find "$PRED" -maxdepth 1 -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' \) | wc -l)
  if [[ "$n" -eq 0 ]]; then
    echo "[legacy] skip empty $PRED"
    continue
  fi
  SAMPLE_SCRATCH="$SCRATCH_ROOT/$DIR"
  mkdir -p "$SAMPLE_SCRATCH"
  echo "[legacy] $DIR → $OUT_TYPE ($n preds)"
  if ! python "$VSLITE_ROOT/scripts/remote_wsi_upload.py" \
      --wsi_pred_dir "$PRED" \
      --input_root "$VSLITE_ROOT/input/IR" \
      --project "$PROJECT" \
      --project_display_name "Campus_ablations_best_LPIPS_WSI" \
      --project_description "Legacy PNG→DZI upload of partial/prior WSI preds" \
      --output_type "$OUT_TYPE" \
      --model "$MODEL" \
      --create_project --ensure_project --overwrite \
      --scratch "$SAMPLE_SCRATCH"; then
    echo "[legacy] FAILED $DIR" >&2
    failures=$((failures + 1))
  fi
  # drop per-dir scratch eagerly (trap also cleans root)
  rm -rf "$SAMPLE_SCRATCH"
done

echo "[legacy] done failures=$failures"
exit "$failures"
