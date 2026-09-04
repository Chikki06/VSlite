# Sourced by train_job.sh — do not sbatch this file directly.
# shellcheck shell=bash
set -o pipefail
mkdir -p "/u/akshahi2/logs/${SLURM_JOB_NAME:-job}"

set -x
exec > >(tee -a "/u/akshahi2/logs/${SLURM_JOB_NAME:-job}/${SLURM_JOB_ID:-noid}.debug.log") 2>&1
echo "$(date -Is) cislnet_train: starting host=$(hostname) pwd=$(pwd) SLURM_JOB_ID=${SLURM_JOB_ID:-}" 1>&2

: "${VSLITE_ROOT:=/projects/illinois/eng/bioe/rxb/akshat/VSlite}"
# shellcheck source=../../_lib/slurm_resolve.sh
source "$VSLITE_ROOT/jobs/_lib/slurm_resolve.sh"
VSLITE_ROOT="$(slurm_resolve_include_root jobs/_lib/vslite_env.sh "${BASH_SOURCE[0]:-$0}")" || exit 1
# shellcheck source=../../_lib/vslite_env.sh
source "$VSLITE_ROOT/jobs/_lib/vslite_env.sh"

if [[ -f "$HOME/.secrets.sh" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/.secrets.sh"
fi

GPU_IDS="${GPU_IDS_OVERRIDE:-}"
if [[ -z "$GPU_IDS" ]]; then
  GPU_IDS="$(python -c 'import torch; n=torch.cuda.device_count(); print(",".join(map(str, range(n))))')"
fi
if [[ -z "$GPU_IDS" ]]; then
  echo "ERROR: no CUDA GPUs visible." >&2
  nvidia-smi -L >&2 || true
  exit 1
fi
echo "$(date -Is) cislnet_train: --gpu_ids=$GPU_IDS" >&2

DATAROOT="${VS_DATAROOT:-$DATAROOT}"
NTHREADS="${VS_NTHREADS:-${SLURM_CPUS_PER_TASK:-4}}"
OPTS=(
  --dataroot "$DATAROOT"
  --dataset_mode VS
  --model contrastive_pix2pix
  --gpu_ids "$GPU_IDS"
  --nThreads "$NTHREADS"
  --batch_size "${BATCH_SIZE:-10}"
  --image_size "${PATCH_SIZE:-1024}"
  --print_freq 100
  --save_epoch_freq 5
  --save_latest_freq 30000
  --display_freq 30000
)

TCGA_OPT=()
if [[ -n "${VS_TCGA_FEATURE_CKPT:-}" ]]; then
  TCGA_OPT=(--tcga_feature_ckpt "${VS_TCGA_FEATURE_CKPT}")
fi
