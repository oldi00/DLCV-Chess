#!/bin/bash

#SBATCH --job-name=DLCV_Test
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/home/vihps/vihps01/DLCV_Chess/output/out/test_%j.out
#SBATCH --error=/home/vihps/vihps01/DLCV_Chess/output/err/test_%j.err

# --- CONFIGURATION ---
export ROOT_DIR="/home/vihps/vihps01/DLCV_Chess"
export CONDA_ENV="/scratch/vihps/vihps01/env/"

MODEL_TO_TEST="/scratch/vihps/vihps01/legacy/chessred/models/epoch35.pth"

# --- SETUP ---
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

# --- RUN ---
echo "Testing Model: $MODEL_TO_TEST"
python3 "$ROOT_DIR/src/test_real.py" "$MODEL_TO_TEST"