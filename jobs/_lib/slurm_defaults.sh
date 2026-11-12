# Default Slurm resource lines (override before sourcing if needed).
# shellcheck disable=SC2034

SLURM_PARTITION="${SLURM_PARTITION:-IllinoisComputes-GPU}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-kf4-ic}"
SLURM_GPUS="${SLURM_GPUS:-1}"
SLURM_CPUS="${SLURM_CPUS:-16}"
SLURM_MEM="${SLURM_MEM:-64G}"
SLURM_TIME_TRAIN="${SLURM_TIME_TRAIN:-71:00:00}"
SLURM_TIME_TEST="${SLURM_TIME_TEST:-24:00:00}"
# ccc0388: known-bad GPU node. ccc0390: CUDA driver broken since 2026-10-29
# (nvidia-smi sees the A100 but torch CUDA init fails; node-wide, other users hit it too).
SLURM_EXCLUDE_NODES="${SLURM_EXCLUDE_NODES:-ccc0388,ccc0390}"
