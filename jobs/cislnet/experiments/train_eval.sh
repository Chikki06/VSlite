#!/bin/bash
# Generic experiment train+eval. Set NAME + ATTENTION_TYPE / backbone / etc. via env.
#SBATCH --job-name=cn_exp
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --exclude=ccc0388
#SBATCH -A kf4-ic
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --chdir=/projects/illinois/eng/bioe/rxb/akshat/VSlite
#SBATCH --output=/u/akshahi2/logs/%x/%j.out
#SBATCH --error=/u/akshahi2/logs/%x/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akshahi2@illinois.edu

set -euo pipefail
mkdir -p "/u/akshahi2/logs/${SLURM_JOB_NAME:-cn_exp}"
VSLITE_ROOT="/projects/illinois/eng/bioe/rxb/akshat/VSlite"
cd "$VSLITE_ROOT"
# shellcheck source=_common.sh
source "$VSLITE_ROOT/jobs/cislnet/experiments/_common.sh"

NUM_CHANNELS="${NUM_CHANNELS:-13}"
NAME="${NAME:?Set NAME for experiment run tag}"
TRACK_GROUP="${TRACK_GROUP:-exp}"
TRACK_NOTES="${TRACK_NOTES:-epochs=${EPOCHS}}"

cislnet_exp_run_train "$NAME"
cislnet_exp_run_eval "$NAME" "$NAME" "$TRACK_GROUP" "$TRACK_NOTES" "$NAME"
