# Sourced by restormer_*.sh after logging is set up. Not for sbatch alone.
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
RESTORMER_DIR="${RESTORMER_DIR:-$VSLITE_DIR/Restromer}"
cd "$RESTORMER_DIR"

VS_DATAROOT="${VS_DATAROOT:-$VSLITE_DIR/dataset/data_1024}"
export RESTORMER_DATAROOT="${RESTORMER_DATAROOT:-$VS_DATAROOT}"
export RESTORMER_GPU_ID="${RESTORMER_GPU_ID:-0}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"

CPU_BUDGET="${SLURM_CPUS_PER_TASK:-4}"
if [[ -z "${RESTORMER_TRAIN_NUM_WORKERS:-}" ]]; then
  if (( CPU_BUDGET > 2 )); then
    export RESTORMER_TRAIN_NUM_WORKERS=$((CPU_BUDGET - 2))
  else
    export RESTORMER_TRAIN_NUM_WORKERS=1
  fi
fi
if [[ -z "${RESTORMER_VAL_NUM_WORKERS:-}" ]]; then
  if (( CPU_BUDGET > 1 )); then
    export RESTORMER_VAL_NUM_WORKERS=1
  else
    export RESTORMER_VAL_NUM_WORKERS=0
  fi
fi

export RESTORMER_AUTO_RESUME="${RESTORMER_AUTO_RESUME:-true}"
