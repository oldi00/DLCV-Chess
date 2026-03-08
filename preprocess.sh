#!/bin/bash

#SBATCH --job-name=DLCV_Preprocess
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=1:00:00
#SBATCH --no-requeue

#SBATCH --output=/home/vihps/vihps01/DLCV_Chess/output/out/preprocess_%j.out
#SBATCH --error=/home/vihps/vihps01/DLCV_Chess/output/err/preprocess_%j.err

#SBATCH --mail-user=oldenburgermarkus@gmail.com
#SBATCH --mail-type=ALL

export ROOT_DIR="/home/vihps/vihps01/DLCV_Chess"
export CONDA_ENV="/scratch/vihps/vihps01/env/"

# Activate Conda environment
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

# ===================================================================
# Configurations
# ===================================================================

# Base Directory
BASE_DIR="/scratch/vihps/vihps01/real_data"
IMAGE_DIR="/scratch/vihps/vihps01/real_images"
JSON_FILE="metadata.json"

# "no_test" -> Train/Val only | "with_test" -> Train/Val/Test
SPLIT_MODE="with_test"
OUTPUT_PREFIX="real"

# ===================================================================

echo "========================================"
echo "Preprocessing"
echo "Target Dir: $BASE_DIR"
echo "========================================"

# Run the preprocessing script with arguments
python3 "$ROOT_DIR/src/utils/preprocess.py" \
    --base_dir "$BASE_DIR" \
    --image_dir "$IMAGE_DIR" \
    --json_file "$JSON_FILE" \
    --mode "$SPLIT_MODE" \
    --prefix "$OUTPUT_PREFIX"