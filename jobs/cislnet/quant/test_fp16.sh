#!/bin/bash
# Patch eval + FID for fp16 quant model on A100. Usage: sbatch test_fp16.sh
#SBATCH --job-name=cn_t16
#SBATCH --partition=scavenger
#SBATCH -A kf4-ic
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --chdir=/projects/illinois/eng/bioe/rxb/akshat/VSlite
#SBATCH --output=/u/akshahi2/logs/%x/%j.out
#SBATCH --error=/u/akshahi2/logs/%x/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akshahi2@illinois.edu

export QUANT_PRECISION=fp16
# shellcheck source=_job_bootstrap.sh
source /projects/illinois/eng/bioe/rxb/akshat/VSlite/jobs/cislnet/quant/_job_bootstrap.sh
# shellcheck source=_test_quant_body.sh
source "$VSLITE_ROOT/jobs/cislnet/quant/_test_quant_body.sh"
