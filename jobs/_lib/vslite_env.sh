# Shared VSlite job environment (source from train/test/wsi scripts).
# Ablation variable reference: jobs/_lib/ABLATIONS.md
# shellcheck disable=SC2034

: "${VSLITE_ROOT:=/projects/illinois/eng/bioe/rxb/akshat/VSlite}"
export VSLITE_ROOT
export VSLITE_DIR="${VSLITE_DIR:-$VSLITE_ROOT}"

# Callers often use set -euo before sourcing this file. .bashrc and conda init
# may reference unset vars or return non-zero; disable -e and -u for bootstrap only.
_vslite_had_errexit=0
_vslite_had_nounset=0
if [[ -o errexit ]]; then
  _vslite_had_errexit=1
  set +e
fi
if [[ -o nounset ]]; then
  _vslite_had_nounset=1
  set +u
fi
source "${HOME}/.bashrc" 2>/dev/null || true
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  conda activate torchenv 2>/dev/null || true
fi
if [[ "$_vslite_had_nounset" -eq 1 ]]; then
  set -u
fi
if [[ "$_vslite_had_errexit" -eq 1 ]]; then
  set -e
fi
unset _vslite_had_errexit _vslite_had_nounset

export PYTHONUNBUFFERED=1
export UNIFIED_TB_MIRROR_TRAIN="${UNIFIED_TB_MIRROR_TRAIN:-1}"

# Fail fast on GPU jobs when Slurm allocated a GPU but PyTorch cannot see it.
# Retry: cgroup / driver attach can lag a few seconds after the job starts.
# CPU-only jobs (e.g. remote DZI upload): export VSLITE_SKIP_CUDA_CHECK=1 before sourcing.
if [[ -n "${SLURM_JOB_ID:-}" && "${VSLITE_SKIP_CUDA_CHECK:-0}" != "1" ]]; then
  _cuda_ok=0
  _cuda_try=0
  while [[ "$_cuda_try" -lt 8 ]]; do
    _cuda_try=$((_cuda_try + 1))
    if python -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
      _cuda_ok=1
      break
    fi
    echo "CUDA not ready yet (try ${_cuda_try}/8) host=$(hostname) job=$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" >&2
    sleep 5
  done
  if [[ "$_cuda_ok" -eq 1 ]]; then
    python -c "import torch; print('CUDA OK:', torch.cuda.get_device_name(0), 'visible_devices=', __import__('os').environ.get('CUDA_VISIBLE_DEVICES', 'unset'))"
  else
    echo "FATAL: torch.cuda.is_available()=False on $(hostname) job=$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" >&2
    nvidia-smi -L 2>/dev/null || echo "nvidia-smi unavailable" >&2
    python -c "import torch; print('torch', torch.__version__, 'cuda_built', torch.version.cuda, 'cuda_is_available', torch.cuda.is_available())" >&2 || true
    # Bounded requeue: an allocation whose GPU never becomes usable is a transient
    # node failure. Requeue on a different node instead of failing the job
    # outright (happened on ccc0390 after the 2026-10-29 slurmd restart which
    # killed cn_tr_ir3 / cn_tr_ir5 hop2 in ~80 s).
    if [[ "${SLURM_RESTART_COUNT:-0}" -lt 3 ]] && scontrol requeue "$SLURM_JOB_ID" >/dev/null 2>&1; then
      echo "GPU unusable: requeued job (restart count ${SLURM_RESTART_COUNT:-0}; cap 3)" >&2
      exit 0
    fi
    exit 1
  fi
  unset _cuda_ok _cuda_try
fi

# Canonical dataroot
export DATAROOT="${DATAROOT:-${VS_DATAROOT:-$VSLITE_ROOT/dataset/data_frozen}}"

# FID backbone for *test* jobs only
export VSLITE_CONVNEXT_FID_CKPT="${VSLITE_CONVNEXT_FID_CKPT:-$VSLITE_ROOT/tcga_tl_checkpoints/convnext_t_dp_spv_supcon_1e-5}"

# Eval caps (training epoch-end L1 + lpips_convnext_tcga only — not FID)
export UNIFIED_EVAL_MAX_BATCHES_TRAIN="${UNIFIED_EVAL_MAX_BATCHES_TRAIN:-20}"
export UNIFIED_EVAL_MAX_BATCHES_VAL="${UNIFIED_EVAL_MAX_BATCHES_VAL:-50}"
export UNIFIED_EVAL_MAX_BATCHES_TEST="${UNIFIED_EVAL_MAX_BATCHES_TEST:-50}"
export UNIFIED_EVAL_MAX_ITEMS_TRAIN="${UNIFIED_EVAL_MAX_ITEMS_TRAIN:-512}"
export UNIFIED_EVAL_MAX_ITEMS_VAL="${UNIFIED_EVAL_MAX_ITEMS_VAL:-512}"
export UNIFIED_EVAL_MAX_ITEMS_TEST="${UNIFIED_EVAL_MAX_ITEMS_TEST:-512}"

# Test patch eval
export VSLITE_FID_MAX_ITEMS="${VSLITE_FID_MAX_ITEMS:-75000}"
export VSLITE_FID_BATCH_SIZE="${VSLITE_FID_BATCH_SIZE:-16}"
export NUM_TEST="${NUM_TEST:-75000}"
export SKIP_WSI="${SKIP_WSI:-1}"

_lib_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$_lib_dir/ablation.env" ]]; then
  # shellcheck source=/dev/null
  source "$_lib_dir/ablation.env"
fi
