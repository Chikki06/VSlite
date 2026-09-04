# Sourced by swingan_*.sh after logging is set up. Not for sbatch alone.
# shellcheck shell=bash
if [[ -f "$HOME/.secrets.sh" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/.secrets.sh"
fi

set +u
set +e
source ~/.bashrc
rc_bashrc=$?
conda activate torchenv
rc_conda=$?
set -e
if [[ $rc_bashrc -ne 0 ]]; then
  echo "ERROR: source ~/.bashrc failed (rc=$rc_bashrc)." >&2
  exit $rc_bashrc
fi
if [[ $rc_conda -ne 0 ]]; then
  echo "ERROR: conda activate torchenv failed (rc=$rc_conda)." >&2
  exit $rc_conda
fi

export PYTHONUNBUFFERED=1

VSLITE_DIR="${VSLITE_DIR:-/projects/illinois/eng/bioe/rxb/akshat/VSlite}"
cd "$VSLITE_DIR"

VS_DATAROOT="${VS_DATAROOT:-$VSLITE_DIR/dataset/data_1024}"
export SWINGAN_DATAROOT="${SWINGAN_DATAROOT:-$VS_DATAROOT}"
export SWINGAN_GPU_ID="${SWINGAN_GPU_ID:-0}"
# ImageNet Swin-T (3ch) — loads into the Swin backbone only; 1×1 projection stays train-from-scratch.
export SWINGAN_PRETRAIN_CKPT="${SWINGAN_PRETRAIN_CKPT:-$VSLITE_DIR/SwinGAN/pretrained_ckpt/swin_tiny_patch4_window7_224.pth}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"

CPU_BUDGET="${SLURM_CPUS_PER_TASK:-4}"
if [[ -z "${SWINGAN_TRAIN_NUM_WORKERS:-}" ]]; then
  if (( CPU_BUDGET > 2 )); then
    export SWINGAN_TRAIN_NUM_WORKERS=$((CPU_BUDGET - 2))
  else
    export SWINGAN_TRAIN_NUM_WORKERS=1
  fi
fi
if [[ -z "${SWINGAN_VAL_NUM_WORKERS:-}" ]]; then
  if (( CPU_BUDGET > 1 )); then
    export SWINGAN_VAL_NUM_WORKERS=1
  else
    export SWINGAN_VAL_NUM_WORKERS=0
  fi
fi
export SWINGAN_AUTO_RESUME="${SWINGAN_AUTO_RESUME:-true}"

cd "$VSLITE_DIR/SwinGAN"
