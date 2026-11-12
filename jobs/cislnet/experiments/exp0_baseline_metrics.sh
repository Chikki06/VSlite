#!/bin/bash
# Exp0: baseline metric pipeline for vs10 + vs13 (c13 weights). No training.
#SBATCH --job-name=cn_exp0
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --exclude=ccc0388
#SBATCH -A kf4-ic
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --chdir=/projects/illinois/eng/bioe/rxb/akshat/VSlite
#SBATCH --output=/u/akshahi2/logs/%x/%j.out
#SBATCH --error=/u/akshahi2/logs/%x/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akshahi2@illinois.edu

set -euo pipefail
mkdir -p "/u/akshahi2/logs/${SLURM_JOB_NAME:-cn_exp0}"
VSLITE_ROOT="/projects/illinois/eng/bioe/rxb/akshat/VSlite"
cd "$VSLITE_ROOT"
# shellcheck source=_common.sh
source "$VSLITE_ROOT/jobs/cislnet/experiments/_common.sh"

# vs10
NUM_CHANNELS=10 NUM_IR=10 IR_INDICES="" \
  cislnet_exp_run_eval "vs10" "vs10" "exp0" "frozen_baseline_15ep_schedule" "vs10"

# vs13 live weights live under c13/
NUM_CHANNELS=13 NUM_IR=10 IR_INDICES="" \
  cislnet_exp_run_eval "c13" "vs13" "exp0" "frozen_baseline_15ep_schedule_ckpt=c13" "vs13"
