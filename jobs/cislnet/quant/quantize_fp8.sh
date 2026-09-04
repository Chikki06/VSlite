#!/bin/bash
# Quantize CISLNet -> fp8 native e4m3fn on H100/H200. Usage: sbatch quantize_fp8.sh
# Default: QUANT_FP8_NATIVE=1 (true FP8 weight storage + NativeFP8* inference modules)
# Legacy bake: QUANT_FP8_NATIVE=0 sbatch quantize_fp8.sh
#SBATCH --job-name=cn_q8
#SBATCH --partition=scavenger
#SBATCH -A kf4-ic
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --chdir=/projects/illinois/eng/bioe/rxb/akshat/VSlite
#SBATCH --output=/u/akshahi2/logs/%x/%j.out
#SBATCH --error=/u/akshahi2/logs/%x/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akshahi2@illinois.edu

export QUANT_PRECISION=fp8
export FP8_STATIC=1
# shellcheck source=_job_bootstrap.sh
source /projects/illinois/eng/bioe/rxb/akshat/VSlite/jobs/cislnet/quant/_job_bootstrap.sh
# shellcheck source=_quantize_body.sh
source "$VSLITE_ROOT/jobs/cislnet/quant/_quantize_body.sh"
