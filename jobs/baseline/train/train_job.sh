#!/bin/bash
#SBATCH --job-name=b_10_13_tr
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH -A kf4-ic
#SBATCH --mem-per-gpu=20G
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --chdir=/projects/illinois/eng/bioe/rxb/akshat/VSlite
#SBATCH --output=/u/akshahi2/logs/%x/%j.out
#SBATCH --error=/u/akshahi2/logs/%x/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akshahi2@illinois.edu

# One allocation: trains 10ch then 13ch. Prefer separate jobs: train_job_10ch.sh / train_job_13ch.sh.
# shellcheck disable=SC1091
source "/u/akshahi2/slurm_include_resolve.bash"
INC_ROOT="$(slurm_resolve_include_root jobs/baseline/train/vs_train_common.sh "${BASH_SOURCE[0]}")" || exit 1
set -e
source "$INC_ROOT/jobs/baseline/train/vs_train_common.sh"

NAME10="${VS_NAME_10CH:-${VS_NAME_10:-vs10}}"
NAME13="${VS_NAME_13CH:-${VS_NAME_13:-vs13}}"
CONTINUE_10CH="${CONTINUE_10CH:-true}"
CONTINUE_13CH="${CONTINUE_13CH:-true}"
CHECKPOINTS_DIR="${VS_CHECKPOINTS_DIR:-$PWD/baseline}"

echo "$(date -Is) train_job: sequential 10ch ($NAME10) then 13ch ($NAME13)" >&2

python scripts/train.py "${OPTS[@]}" "${TCGA_OPT[@]}" \
  --continue_train "$CONTINUE_10CH" \
  --name "$NAME10" \
  --checkpoints_dir "$CHECKPOINTS_DIR" \
  --input_nc 10 \
  --num_ir_channels 10

python scripts/train.py "${OPTS[@]}" "${TCGA_OPT[@]}" \
  --continue_train "$CONTINUE_13CH" \
  --name "$NAME13" \
  --checkpoints_dir "$CHECKPOINTS_DIR" \
  --input_nc 13 \
  --num_ir_channels 10

echo "$(date -Is) train_job: finished both stages."
