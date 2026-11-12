#!/bin/bash
# CISLNet training WITH OPAL + local LLM (sleep residency on A100).
# Real dataset / real U-Net. Not a synthetic pilot.
#SBATCH --job-name=cn_tr_opal
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --exclude=ccc0388
#SBATCH -A kf4-ic
#SBATCH --mem=128G
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --chdir=/projects/illinois/eng/bioe/rxb/akshat/VSlite
#SBATCH --output=/u/akshahi2/logs/cn_tr_opal/%j.out
#SBATCH --error=/u/akshahi2/logs/cn_tr_opal/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akshahi2@illinois.edu

set -euo pipefail
VSLITE_ROOT="/projects/illinois/eng/bioe/rxb/akshat/VSlite"
OPAL_LOCAL_ROOT="/projects/illinois/eng/bioe/rxb/akshat/opal_local"
cd "$VSLITE_ROOT"
source "$VSLITE_ROOT/jobs/_lib/vslite_env.sh"
source "$VSLITE_ROOT/jobs/_lib/cislnet_channels.sh"

export OPAL_LOCAL_ROOT
export OPAL_ENABLE=1
export OPAL_AGENT_CONFIG="${OPAL_LOCAL_ROOT}/config/local_agent.yaml"
export OPAL_AGENT_OVERRIDE="${OPAL_AGENT_OVERRIDE:-server.residency=sleep;server.model_key=awq4;server.gpu_memory_utilization=0.30;agent.on_failure=continue}"
export OPAL_DECISIONS="${OPAL_DECISIONS:-50}"
export OPAL_SCHEDULE="${OPAL_SCHEDULE:-front}"
export OPAL_EVERY="${OPAL_EVERY:-100}"
export OPAL_PROMPT="${OPAL_PROMPT:-neutral}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_CACHE_ROOT="${OPAL_LOCAL_ROOT}/envs/vllm_cache"
export VLLM_USE_FLASHINFER_SAMPLER=0
export LLM_VENV="${OPAL_LOCAL_ROOT}/envs/llm"
export PYTHONPATH="${OPAL_LOCAL_ROOT}:${PYTHONPATH:-}"

NUM_CHANNELS="${NUM_CHANNELS:-13}"
RUN="${RUN:-c${NUM_CHANNELS}_opal}"
PATCH_SIZE="${PATCH_SIZE:-1024}"
BATCH_SIZE="${BATCH_SIZE:-10}"
NTHREADS="${NTHREADS:-4}"
EPOCHS="${EPOCHS:-30}"
SAVE_EPOCH_FREQ="${SAVE_EPOCH_FREQ:-5}"

export CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-$VSLITE_ROOT/results/cislnet}"
NAME="${NAME:-$RUN}"

EXTRA=(--input_nc "$NUM_CHANNELS" --num_ir_channels "$NUM_IR")
EXTRA+=(--image_size "$PATCH_SIZE" --batch_size "$BATCH_SIZE")
EXTRA+=(--weight_decay "${WEIGHT_DECAY:-0}")
EXTRA+=(--he_color_space "${HE_COLOR_SPACE:-srgb}")
EXTRA+=(--total_epochs "$EPOCHS")
if [[ -n "${IR_INDICES:-}" ]]; then EXTRA+=(--ir_channel_indices "$IR_INDICES"); fi

CONTINUE_TRAIN="${CONTINUE_TRAIN:-false}"
# Same requeue-restart auto-resume as train.sh (see 64G OOM of 11878763: the
# vLLM wake-up for the next OPAL decision matters, so restarts resume).
if [[ "${SLURM_RESTART_COUNT:-0}" -gt 0 && -f "$CHECKPOINTS_DIR/$NAME/iter.txt" ]]; then
  CONTINUE_TRAIN=true
fi

python scripts/train.py \
  --dataroot "$DATAROOT" --dataset_mode VS --model contrastive_pix2pix \
  --gpu_ids 0 --nThreads "$NTHREADS" --print_freq 100 --save_epoch_freq "$SAVE_EPOCH_FREQ" \
  --save_latest_freq 30000 --continue_train "$CONTINUE_TRAIN" --name "$NAME" \
  --checkpoints_dir "$CHECKPOINTS_DIR" \
  "${EXTRA[@]}"
