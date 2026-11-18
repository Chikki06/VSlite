#!/bin/bash
#SBATCH --job-name=enc_supcon
#SBATCH --partition=IllinoisComputes-GPU
# ccc0388 re-confirmed broken 2026-11-13 (4 fresh allocations all failed
# torch.cuda.is_available()=False node-wide); 0390 same 2026-10-29 fault.
#SBATCH --exclude=ccc0388,ccc0390
#SBATCH -A kf4-ic
#SBATCH --mem=128G
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=20
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --chdir=/projects/illinois/eng/bioe/rxb/akshat/VSlite
#SBATCH --output=/u/akshahi2/logs/%x/%j.out
#SBATCH --error=/u/akshahi2/logs/%x/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akshahi2@illinois.edu

# SupCon encoder pretraining on TCGA-UT (ConvNeXt-T / Swin-T).
# Defaults (anti-collapse recipe; see pretrain_tcga_supcon.py):
#   convnext_t: AdamW 5e-4 fixed, wd 0.05 (LN/BN/bias/gamma exempt), temp 0.1,
#               5-ep warmup, cosine -> 0, SyncBN projection head, collapse kill
#               switch (abort if loss ~ ln(2N-1) for 2 epochs after epoch 5).
#   swin_t:     histology transformer-contrastive recipes (CTransPath / MoCo-v3
#               tiny-Swin) — AdamW 1.5e-4 fixed, wd 0.05, 40-ep warmup, cosine, 100 ep.
# Env overrides: ARCH, EPOCHS, BATCH_SIZE, OPTIMIZER, BASE_LR, WEIGHT_DECAY,
#                TEMPERATURE, WARMUP_EPOCHS, TCGA_UT_MANIFEST, TCGA_UT_RES_FILTER, SMOKE=1
set -euo pipefail

source jobs/_lib/vslite_env.sh

ARCH="${ARCH:-convnext_t}"
RUN="${RUN:-enc_${ARCH}_supcon}"
RESULTS_DIR="$VSLITE_ROOT/results/tcga_encoder/$RUN"

# Already finished? (chain hop2 after a completed run is a no-op)
if [[ -f "$RESULTS_DIR/iter.txt" ]]; then
  DONE_EP=$(tail -1 "$RESULTS_DIR/iter.txt")
  WANT_EP="${EPOCHS:-$([ "$ARCH" = swin_t ] && echo 100 || echo 350)}"
  if (( DONE_EP >= WANT_EP )); then
    echo "run $RUN already complete at epoch $DONE_EP (>= $WANT_EP); nothing to do"
    exit 0
  fi
fi

mkdir -p /u/akshahi2/logs/"$SLURM_JOB_NAME"

# Manifest must exist (built by jobs/tcga_encoder/download_tcga_ut.sh);
# override with TCGA_UT_MANIFEST (e.g. manifest_benign.csv for the benign run)
MANIFEST="${TCGA_UT_MANIFEST:-$VSLITE_ROOT/dataset/tcga_ut/manifest.csv}"
if [[ ! -f "$MANIFEST" ]]; then
  echo "FATAL: $MANIFEST missing; run the download job first" >&2
  exit 1
fi

OPTS=(
  --arch "$ARCH"
  --manifest "$MANIFEST"
  --extract_dir "${TCGA_UT_EXTRACT:-$VSLITE_ROOT/dataset/tcga_ut/extracted}"
  --run_name "$RUN"
  --results_root results/tcga_encoder
  --epochs "${EPOCHS:-$([ "$ARCH" = swin_t ] && echo 100 || echo 350)}"
  --batch_size "${BATCH_SIZE:-512}"
  --img_size "${IMG_SIZE:-256}"
  --num_workers "${NUM_WORKERS:-8}"
  --seed "${SEED:-0}"
  --print_freq "${PRINT_FREQ:-50}"
  --save_epoch_freq "${SAVE_EPOCH_FREQ:-5}"
  --knn_freq "${KNN_FREQ:-10}"
  --grad_ckpt "${GRAD_CKPT:-auto}"
)
[[ -n "${OPTIMIZER:-}" ]] && OPTS+=(--optimizer "$OPTIMIZER")
[[ -n "${BASE_LR:-}" ]] && OPTS+=(--base_lr "$BASE_LR")
[[ -n "${WEIGHT_DECAY:-}" ]] && OPTS+=(--weight_decay "$WEIGHT_DECAY")
[[ -n "${TEMPERATURE:-}" ]] && OPTS+=(--temperature "$TEMPERATURE")
[[ -n "${WARMUP_EPOCHS:-}" ]] && OPTS+=(--warmup_epochs "$WARMUP_EPOCHS")
[[ -n "${TCGA_UT_RES_FILTER:-}" ]] && OPTS+=(--res_filter "$TCGA_UT_RES_FILTER")
[[ -n "${SMOKE:-}" ]] && OPTS+=(--smoke)

echo "run: $RUN arch: $ARCH"
python scripts/pretrain_tcga_supcon.py "${OPTS[@]}"
