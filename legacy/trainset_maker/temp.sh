#!/bin/bash
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --output=5_bkgaug_1024_fullVGG_IR+VIS/VS.out  
#SBATCH --error=5_bkgaug_1024_fullVGG_IR+VIS/VS.err  
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --mem-bind=verbose,local
#SBATCH --cpus-per-task=16
#SBATCH --partition=ghx4
#SBATCH --time=1:00:00
#SBATCH --job-name=pytorch
#SBATCH --account=bdsg-dtai-gh
### GPU options ###
#SBATCH --gpus-per-node=0
#SBATCH --gpu-bind=verbose,closest

# Load necessary modules
module use /sw/user/modules/python
module load python/miniforge3_pytorch








# Execute your training script
python datafinder.py