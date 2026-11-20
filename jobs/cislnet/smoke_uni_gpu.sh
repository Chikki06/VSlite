#!/bin/bash
# GPU smoke: UNI netF training-realistic cost probe (B=10 @1024) + metric.
#SBATCH --job-name=uni_smoke_gpu
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --exclude=ccc0388,ccc0390
#SBATCH -A kf4-ic
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --chdir=/projects/illinois/eng/bioe/rxb/akshat/VSlite
#SBATCH --output=/u/akshahi2/logs/uni_smoke/gpu_%j.out
#SBATCH --error=/u/akshahi2/logs/uni_smoke/gpu_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akshahi2@illinois.edu

set -euo pipefail
mkdir -p /u/akshahi2/logs/uni_smoke
VSLITE_ROOT="/projects/illinois/eng/bioe/rxb/akshat/VSlite"
cd "$VSLITE_ROOT"
# shellcheck source=_lib/vslite_env.sh
source "$VSLITE_ROOT/jobs/_lib/vslite_env.sh"

echo "== train_probe (netF cost only) =="
VSLITE_UNI_SMOKE_GPU=1 python scripts/smoke_uni_netf.py --train_probe --batch "${UNI_SMOKE_BATCH:-10}"

echo "== full smoke suite =="
VSLITE_UNI_SMOKE_GPU=1 python scripts/smoke_uni_netf.py
echo "smoke done"
