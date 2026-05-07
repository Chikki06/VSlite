#!/bin/bash
#SBATCH --job-name=b_tr
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH -A kf4-ic
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=71:00:00
#SBATCH --output=/u/akshahi2/logs/%x/%j.out
#SBATCH --error=/u/akshahi2/logs/%x/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akshahi2@illinois.edu


cd /projects/illinois/eng/bioe/rxb/akshat/VSlite/final_code
conda activate torchenv
python scripts/train.py --dataroot dataset/data_1024 --dataset_mode VS --model contrastive_pix2pix --continue_train true --name vs13 --checkpoints_dir baseline --input_nc 13 --num_ir_channels 10