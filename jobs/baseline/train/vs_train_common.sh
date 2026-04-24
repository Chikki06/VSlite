# Sourced by train_job*.sh — do not sbatch this file directly.
# shellcheck shell=bash
set -o pipefail
mkdir -p "/u/akshahi2/logs/${SLURM_JOB_NAME:-job}"

set -x
exec > >(tee -a "/u/akshahi2/logs/${SLURM_JOB_NAME:-job}/${SLURM_JOB_ID:-noid}.debug.log") 2>&1
echo "$(date -Is) vs_train: starting host=$(hostname) pwd=$(pwd) SLURM_JOB_ID=${SLURM_JOB_ID:-}" 1>&2

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

GPU_IDS="${GPU_IDS_OVERRIDE:-}"
if [[ -z "$GPU_IDS" ]]; then
  GPU_IDS="$(python -c 'import torch; n=torch.cuda.device_count(); print(",".join(map(str, range(n))))')"
fi
if [[ -z "$GPU_IDS" ]]; then
  echo "ERROR: no CUDA GPUs visible." >&2
  nvidia-smi -L >&2 || true
  exit 1
fi
echo "$(date -Is) vs_train: --gpu_ids=$GPU_IDS" >&2

DATAROOT="${VS_DATAROOT:-$PWD/dataset/data_1024}"
NTHREADS="${VS_NTHREADS:-${SLURM_CPUS_PER_TASK:-1}}"
OPTS=(
  --dataroot "$DATAROOT"
  --dataset_mode VS
  --model contrastive_pix2pix
  --gpu_ids "$GPU_IDS"
  --nThreads "$NTHREADS"
  --batch_size 8
  --load_size 1024
  --crop_size 512
  --print_freq 100
  --save_epoch_freq 5
  --save_latest_freq 30000
  --display_freq 30000
  --evaluation_freq 5000
)

TCGA_OPT=()
if [[ -n "${VS_TCGA_FEATURE_CKPT:-}" ]]; then
  TCGA_OPT=(--tcga_feature_ckpt "${VS_TCGA_FEATURE_CKPT}")
fi
