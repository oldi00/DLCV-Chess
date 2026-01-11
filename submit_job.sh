#!/bin/bash

#SBATCH --job-name=DLCV_Chess_Project       # Name of your job
#SBATCH --partition=gpu                    # Target the GPU partition
#SBATCH --nodes=1                          # Allocate 1 full node
#SBATCH --ntasks=8                         # Number of processes to run
#SBATCH --cpus-per-task=8                  # CPU cores per process
#SBATCH --gres=gpu:8                       # Request all 8 GPUs on the node
#SBATCH --mem=0                            # Request all available node memory
#SBATCH --time=72:00:00                    # Maximum run time (HH:MM:SS)
#SBATCH --no-requeue                       # Don't restart if node fails

#SBATCH --output=/home/vihps/vihps01/DLCV_Chess/output/out/%j.out
#SBATCH --error=/home/vihps/vihps01/DLCV_Chess/output/err/%j.err

#SBATCH --mail-user=oldenburgermarkus@gmail.com
#SBATCH --mail-type=ALL

export ROOT_DIR="/home/vihps/vihps01/DLCV_Chess"
export CONDA_ENV="/scratch/vihps/vihps01/env/"

# Activate Conda environment
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

# Debugging flags
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# srun will automatically launch the script 8 times (as per --ntasks=8)
srun python3 "$ROOT_DIR/src/train_and_eval.py"