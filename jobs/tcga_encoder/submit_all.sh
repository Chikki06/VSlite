#!/bin/bash
# Submit the TCGA-UT encoder-pretraining lineup (3 chains: train -> resume).
# Chain pattern (matches jobs/cislnet convention): hop1 runs up to --time, then
# hop2 (afterany) resumes from ckpt_latest.pth; hop2 is a no-op if already done.
#
#   bash jobs/tcga_encoder/submit_all.sh
#
# GPU budget: user QoS caps gres/gpu at 4; the running cn_tr_c10 train job holds
# 1 GPU, so 3 encoder jobs are submitted here. Runs on A100 (plenty for bs-512
# 256px training); override with ENCODER_GPU_TYPE=H100/H200 to pin a different
# GPU family. Node exclusions: the 2026-11-13 probe (exclude temporarily
# dropped, 4 fresh allocations) re-confirmed ccc0388's CUDA still broken
# node-wide — exclusion restored; vslite_env.sh fail-fast+requeue guard
# contained the damage (no CPU-silent-fail, no wasted walltime).
set -euo pipefail

VSLITE_ROOT="${VSLITE_ROOT:-/projects/illinois/eng/bioe/rxb/akshat/VSlite}"
cd "$VSLITE_ROOT"
GPU_TYPE="${ENCODER_GPU_TYPE:-A100}"

submit_chain() {
  # $1 = job script, $2 = export string
  local script="$1" exportstr="$2" jid
  jid=$(sbatch --parsable --gres="gpu:${GPU_TYPE}:1" --export=ALL,"$exportstr" "$script")
  echo "hop1 $script JID=$jid ($exportstr)"
  local jid2
  jid2=$(sbatch --parsable --gres="gpu:${GPU_TYPE}:1" \
    --dependency="afterany:${jid}" --export=ALL,"$exportstr" "$script")
  echo "hop2 resume JID=$jid2 (afterany:$jid)"
}

# 1. ConvNeXt-T + SupCon — anti-collapse: AdamW 5e-4, wd 0.05 (norm/bias/gamma
#    exempt), temp 0.1, 5-ep warmup, cosine, SyncBN head, 350 ep.
submit_chain jobs/tcga_encoder/train/train_supcon.sh \
  "ARCH=convnext_t,EPOCHS=350,BATCH_SIZE=512,SEED=0"

# 2. Swin-T + SupCon — histology transformer-contrastive recipe (CTransPath/
#    MoCo-v3 tiny-Swin): AdamW 1.5e-4 fixed, wd 0.05, 40-ep warmup, cosine, 100 ep.
submit_chain jobs/tcga_encoder/train/train_supcon.sh \
  "ARCH=swin_t,EPOCHS=100,BATCH_SIZE=512,SEED=0"

# 3. ConvNeXt V2-T + FCMAE (from scratch) — paper recipe: AdamW blr 1.5e-4
#    (LR = 3e-4 @ bs 512), wd 0.05, betas (0.9,0.95), 40-ep warmup, cosine,
#    800 ep, mask 0.6 @ 32px, norm_pix.
submit_chain jobs/tcga_encoder/train/train_fcmae.sh \
  "ARCH=convnextv2_t,EPOCHS=800,BATCH_SIZE=512,SEED=0"

# 4. (optional) ConvNeXt V2-B + FCMAE — submit when a GPU is free:
# submit_chain jobs/tcga_encoder/train/train_fcmae.sh \
#   "ARCH=convnextv2_b,EPOCHS=800,BATCH_SIZE=256,SEED=0"
