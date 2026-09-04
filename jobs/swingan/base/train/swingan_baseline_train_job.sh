#!/bin/bash
#SBATCH --job-name=sg_b_10_13_tr
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH -A kf4-ic
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=71:00:00
#SBATCH --output=/u/akshahi2/logs/%x/%j.out
#SBATCH --error=/u/akshahi2/logs/%x/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akshahi2@illinois.edu

# One job: baseline i2i ir10 then irvis13. Prefer split jobs:
# /u/akshahi2/jobs/swingan/baseline/train/swingan_base_10.sh and
# /u/akshahi2/jobs/swingan/baseline/train/swingan_base_13.sh.

set -o pipefail
mkdir -p "/u/akshahi2/logs/${SLURM_JOB_NAME:-job}"
set -x
exec > >(tee -a "/u/akshahi2/logs/${SLURM_JOB_NAME:-job}/${SLURM_JOB_ID:-noid}.debug.log") 2>&1
echo "$(date -Is) swingan_baseline_train_job: start SLURM_JOB_ID=${SLURM_JOB_ID:-}" 1>&2

# shellcheck disable=SC1091
source "/u/akshahi2/slurm_include_resolve.bash"
INC_ROOT="$(slurm_resolve_include_root jobs/swingan/baseline/swingan_inc.sh "${BASH_SOURCE[0]}")" || exit 1
set -e
source "$INC_ROOT/jobs/swingan/baseline/swingan_inc.sh"

CONFIG_PATH="${SWINGAN_CONFIG:-$VSLITE_DIR/SwinGAN/config_base.yaml}"

export SWINGAN_INPUT_MODE="ir10"
export SWINGAN_OUTPUT_DIR="${SWINGAN_OUTPUT_DIR_IR10:-$VSLITE_DIR/results/swingan/b10}"
echo "$(date -Is) swingan_baseline_train_job: mode=ir10 out=$SWINGAN_OUTPUT_DIR" >&2
python train_i2i.py --config "$CONFIG_PATH"

export SWINGAN_INPUT_MODE="irvis13"
export SWINGAN_OUTPUT_DIR="${SWINGAN_OUTPUT_DIR_IRVIS13:-$VSLITE_DIR/results/swingan/b13}"
echo "$(date -Is) swingan_baseline_train_job: mode=irvis13 out=$SWINGAN_OUTPUT_DIR" >&2
python train_i2i.py --config "$CONFIG_PATH"

echo "$(date -Is) swingan_baseline_train_job: finished"

python /projects/illinois/eng/bioe/rxb/akshat/VSlite/util/plot_loss_curves.py --log_dir "$SWINGAN_OUTPUT_DIR" --out_dir "$SWINGAN_OUTPUT_DIR/curves" || true
