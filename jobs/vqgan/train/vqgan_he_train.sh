#!/bin/bash
#SBATCH --job-name=vq_he_b_tr
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH -A kf4-ic
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
# TEMP: shorter wall time for better queue priority. Revert to ~3 days: replace next line with
#   #SBATCH --time=71:00:00
#SBATCH --time=24:00:00
#SBATCH --chdir=/projects/illinois/eng/bioe/rxb/akshat/VSlite
#SBATCH --output=/u/akshahi2/logs/%x/%j.out
#SBATCH --error=/u/akshahi2/logs/%x/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akshahi2@illinois.edu

source ~/.bashrc
conda activate torchenv

export PYTHONUNBUFFERED=1
export WANDB_MODE="disabled"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

DATAROOT="/projects/illinois/eng/bioe/rxb/akshat/VSlite/dataset/data_frozen"
RESULTS_DIR="/projects/illinois/eng/bioe/rxb/akshat/VSlite/results/vqgan"
VQGAN_VARIANT="${VQGAN_VARIANT:-default}"
if [[ "$VQGAN_VARIANT" == "default" ]]; then
  VQGAN_TAG="${VQGAN_TAG:-b}"
else
  VQGAN_TAG="${VQGAN_TAG:-$VQGAN_VARIANT}"
fi
NAME="${VQGAN_NAME:-vq_${VQGAN_TAG}}"

# Unified TensorBoard eval caps (controls overhead of split evaluation).
export UNIFIED_EVAL_MAX_BATCHES_VAL="${UNIFIED_EVAL_MAX_BATCHES_VAL:-50}"
export UNIFIED_EVAL_MAX_BATCHES_TEST="${UNIFIED_EVAL_MAX_BATCHES_TEST:-50}"

# B_eff=B_micro*K. Defaults 8×4=32 (stabler grads, ~2× VRAM vs 4×4). B_eff=16 + faster loop: VQGAN_ACCUM=2.
python scripts/train_vqgan.py \
  --name "$NAME" \
  --dataroot "$DATAROOT" \
  --results_dir "$RESULTS_DIR" \
  --input_nc 13 \
  --num_ir_channels 10 \
  --image_size 512 \
  --latent_grid_size "${VQGAN_LATENT_GRID:-16}" \
  --num_codebook_vectors "${VQGAN_CODEBOOK:-4096}" \
  --perceptual_loss_factor "${VQGAN_LPIPS:-1.0}" \
  --discriminator_variant "${VQGAN_VARIANT}" \
  --sample_schedule_saves "${VQGAN_SAMPLE_SAVES:-100}" \
  --batch_size "${VQGAN_BS:-8}" \
  --grad_accum_steps "${VQGAN_ACCUM:-4}" \
  --amp \
  --resume "${VQGAN_RESUME:-auto}" \
  --epochs "${VQGAN_EPOCHS:-50}" \
  --learning_rate "${VQGAN_LR:-2.25e-5}" \
  --gpu 0

