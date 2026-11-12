#!/bin/bash
# One-folder smoke: CISLNet stitch → pyvips DZI → remote upload (no PNG).
#SBATCH --job-name=cn_smoke_dzi
#SBATCH --partition=scavenger
#SBATCH -A kf4-ic
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --chdir=/projects/illinois/eng/bioe/rxb/akshat/VSlite
#SBATCH --output=/u/akshahi2/logs/cn_wsi_smoke/%j.out
#SBATCH --error=/u/akshahi2/logs/cn_wsi_smoke/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akshahi2@illinois.edu
set -euo pipefail
VSLITE_ROOT="/projects/illinois/eng/bioe/rxb/akshat/VSlite"
cd "$VSLITE_ROOT"
source "$VSLITE_ROOT/jobs/_lib/vslite_env.sh"
source /u/akshahi2/.secrets.sh
: "${CISLVIEWER_REMOTE_TOKEN:?missing token}"
: "${CISLVIEWER_SITE:=https://cisl-inference.beckman.illinois.edu}"
export CISLVIEWER_REMOTE_TOKEN CISLVIEWER_SITE

NAME="${NAME:-vp1}"
WHICH_EPOCH="${WHICH_EPOCH:-20}"
FOLDER="${WSI_FOLDER:-4466-2}"
PROJECT="${PROJECT:-campus_ablations_bestlpips}"
OUTPUT_TYPE="${OUTPUT_TYPE:-smoke_infer_dzi_${NAME}_e${WHICH_EPOCH}}"
SCRATCH="${TMPDIR:-/tmp}/cisl_smoke_dzi_${SLURM_JOB_ID:-$$}"
mkdir -p "$SCRATCH"
cleanup() { rm -rf "$SCRATCH"; }
trap cleanup EXIT

echo "[smoke] name=$NAME epoch=$WHICH_EPOCH folder=$FOLDER output_type=$OUTPUT_TYPE"
python "$VSLITE_ROOT/scripts/remote_wsi_upload.py" \
  --infer_cislnet \
  --name "$NAME" \
  --checkpoints_dir "$VSLITE_ROOT/results/cislnet" \
  --which_epoch "$WHICH_EPOCH" \
  --results_dir "$VSLITE_ROOT/results/cislnet" \
  --parent_folder "$VSLITE_ROOT/input/IR" \
  --folders "$FOLDER" \
  --input_nc 4 --num_ir_channels 1 --ir_channel_indices 8 \
  --tile_size "${WSI_TILE_SIZE:-1024}" \
  --batch_size "${WSI_BATCH_SIZE:-16}" \
  --num_workers "${WSI_NUM_WORKERS:-4}" \
  --project "$PROJECT" \
  --project_display_name "Campus_ablations_best_LPIPS_WSI" \
  --output_type "$OUTPUT_TYPE" \
  --model 1 \
  --create_project --ensure_project --overwrite \
  --scratch "$SCRATCH"
echo "[smoke] done; scratch cleaned by trap"
