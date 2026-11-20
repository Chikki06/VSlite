#!/bin/bash
# CPU smoke test for the UNI feature extractor / lpips_uni metric.
#SBATCH --job-name=uni_smoke
#SBATCH --partition=secondary
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --chdir=/projects/illinois/eng/bioe/rxb/akshat/VSlite
#SBATCH --output=/u/akshahi2/logs/uni_smoke/%j.out
#SBATCH --error=/u/akshahi2/logs/uni_smoke/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akshahi2@illinois.edu

set -euo pipefail
mkdir -p /u/akshahi2/logs/uni_smoke
VSLITE_ROOT="/projects/illinois/eng/bioe/rxb/akshat/VSlite"
cd "$VSLITE_ROOT"
# CPU-only job: skip the CUDA guard inside vslite_env.sh (must be set before sourcing).
export VSLITE_SKIP_CUDA_CHECK=1
# shellcheck source=_lib/vslite_env.sh
source "$VSLITE_ROOT/jobs/_lib/vslite_env.sh"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python scripts/smoke_uni_netf.py 2>&1 | tee -a /u/akshahi2/logs/uni_smoke/${SLURM_JOB_ID}_run.log
echo "smoke done"
