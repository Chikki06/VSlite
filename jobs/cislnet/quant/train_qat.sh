#!/bin/bash
# QAT fine-tune CISLNet netG at 1024px -> packed int8/int4. Usage: sbatch train_qat.sh
# Ablation env: QUANT_BITS=8 QAT_EPOCHS=5 QUANT_USE_ACT_SCALE=0 QUANT_SKIP_CBAM=1
#SBATCH --job-name=cn_qat
#SBATCH --partition=scavenger
#SBATCH -A kf4-ic
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --chdir=/projects/illinois/eng/bioe/rxb/akshat/VSlite
#SBATCH --output=/u/akshahi2/logs/%x/%j.out
#SBATCH --error=/u/akshahi2/logs/%x/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akshahi2@illinois.edu

export QAT_ENABLED=1
export QUANT_BITS="${QUANT_BITS:-8}"
export QAT_EPOCHS="${QAT_EPOCHS:-5}"
export QAT_LR="${QAT_LR:-1e-5}"
export QAT_BATCH_SIZE="${QAT_BATCH_SIZE:-4}"
export PATCH_SIZE="${PATCH_SIZE:-1024}"
export QUANT_USE_ACT_SCALE="${QUANT_USE_ACT_SCALE:-0}"
export QUANT_STORE_PARAMS_FP32="${QUANT_STORE_PARAMS_FP32:-1}"
export QUANT_FP8_NATIVE="${QUANT_FP8_NATIVE:-1}"

# shellcheck source=_job_bootstrap.sh
source /projects/illinois/eng/bioe/rxb/akshat/VSlite/jobs/cislnet/quant/_job_bootstrap.sh

python "$VSLITE_ROOT/scripts/train_qat.py"
