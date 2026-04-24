#!/usr/bin/env bash
# Shared include for NAFNet Slurm jobs (source after `conda activate`).

export PYTHONUNBUFFERED=1

export VSLITE_DIR="${VSLITE_DIR:-/projects/illinois/eng/bioe/rxb/akshat/VSlite}"
export NAFNET_DIR="${NAFNET_DIR:-${VSLITE_DIR}/NAFNet}"

export NAFNET_DATAROOT="${NAFNET_DATAROOT:-${VSLITE_DIR}/dataset/data_frozen}"

# Thread caps (mirrors other job includes)
if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
fi
