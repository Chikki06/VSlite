#!/bin/bash

# Memory optimization training script with different memory profiles
# Run with: ./train_memory_profiles.sh [profile]
# Profiles: light, medium, heavy

PROFILE=${1:-light}

echo "Training with memory profile: $PROFILE"

# Set PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

case $PROFILE in
    "light")
        echo "Using LIGHT memory profile - lowest memory usage"
        python train_memory_optimized.py \
            --name VS_contrastive_pix2pix_light \
            --model contrastive_pix2pix \
            --batch_size 1 \
            --crop_size 256 \
            --load_size 512 \
            --num_patches 256 \
            --ngf 32 \
            --ndf 32 \
            --netG attunet \
            --netD multiscale \
            --num_D 1 \
            --gpu_ids 0 \
            --dataroot /work/nvme/bdsg/kiakh93/data_same_batch_final_1024_Olympus/ \
            --dataset_mode VS \
            --niter 5 \
            --niter_decay 10
        ;;
    "medium")
        echo "Using MEDIUM memory profile - balanced performance/memory"
        python train_memory_optimized.py \
            --name VS_contrastive_pix2pix_medium \
            --model contrastive_pix2pix \
            --batch_size 2 \
            --crop_size 384 \
            --load_size 768 \
            --num_patches 512 \
            --ngf 48 \
            --ndf 48 \
            --netG attunet \
            --netD multiscale \
            --num_D 2 \
            --gpu_ids 0 \
            --dataroot /work/nvme/bdsg/kiakh93/data_same_batch_final_1024_Olympus/ \
            --dataset_mode VS \
            --niter 5 \
            --niter_decay 10
        ;;
    "heavy")
        echo "Using HEAVY memory profile - high quality but more memory"
        python train_memory_optimized.py \
            --name VS_contrastive_pix2pix_heavy \
            --model contrastive_pix2pix \
            --batch_size 4 \
            --crop_size 512 \
            --load_size 1024 \
            --num_patches 768 \
            --ngf 64 \
            --ndf 64 \
            --netG attunet \
            --netD multiscale \
            --num_D 2 \
            --gpu_ids 0 \
            --dataroot /work/nvme/bdsg/kiakh93/data_same_batch_final_1024_Olympus/ \
            --dataset_mode VS \
            --niter 5 \
            --niter_decay 10
        ;;
    *)
        echo "Unknown profile: $PROFILE"
        echo "Available profiles: light, medium, heavy"
        exit 1
        ;;
esac