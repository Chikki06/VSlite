#!/bin/bash
#SBATCH --job-name=sg_inf
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH -A kf4-ic
#SBATCH --mem=80G
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --chdir=/projects/illinois/eng/bioe/rxb/akshat/VSlite
#SBATCH --output=/u/akshahi2/logs/%x/%j.out
#SBATCH --error=/u/akshahi2/logs/%x/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akshahi2@illinois.edu

source ~/.bashrc
conda activate torchenv

export PYTHONUNBUFFERED=1

# Run with baseline config
export SWINGAN_VARIANT="baseline"

bash /u/akshahi2/jobs/swingan/attention/infer/swingan_infer.sh