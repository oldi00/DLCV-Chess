#!/bin/bash

#SBATCH --job-name=DLCV_Inspect
#SBATCH --partition=gpu
#SBATCH --nodes=1                     
#SBATCH --ntasks=1                    
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --no-requeue                  

#SBATCH --output=/home/vihps/vihps01/DLCV_Chess/output/out/inspect_%j.out
#SBATCH --error=/home/vihps/vihps01/DLCV_Chess/output/err/inspect_%j.err

export ROOT_DIR="/home/vihps/vihps01/DLCV_Chess"
export CONDA_ENV="/scratch/vihps/vihps01/env/"

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

BASE_DIR="/scratch/vihps/vihps01/real_data"
OUTPUT_PREFIX="real"
SAMPLES_TO_SHOW=5

echo "========================================"
echo "Inspecting Pickle Files in: $BASE_DIR"
echo "========================================"

python3 "$ROOT_DIR/src/utils/inspect_pickle.py" \
    --base_dir "$BASE_DIR" \
    --prefix "$OUTPUT_PREFIX" \
    --samples "$SAMPLES_TO_SHOW"