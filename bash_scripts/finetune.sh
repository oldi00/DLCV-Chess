#!/bin/bash

#SBATCH --job-name=DLCV_Finetune
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --no-requeue

#SBATCH --output=/home/vihps/vihps01/DLCV_Chess/output/out/finetune_%j.out
#SBATCH --error=/home/vihps/vihps01/DLCV_Chess/output/err/finetune_%j.err

#SBATCH --mail-user=(EMAIL)
#SBATCH --mail-type=ALL

export ROOT_DIR="/home/vihps/vihps01/DLCV_Chess"
export CONDA_ENV="/scratch/vihps/vihps01/env/"

# Activate Conda
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

# Debugging
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ===================================================================
# Configurations
# ===================================================================

# Pretrained Base Model
BASE_MODEL="/scratch/vihps/vihps01/pretraining/models/exp1_baseline/best_model.pth"

# Dataset Paths
TRAIN_DATA="/scratch/vihps/vihps01/real_data/real_train_data.pkl"
VAL_DATA="/scratch/vihps/vihps01/real_data/real_val_data.pkl"

# Where to save the output models
SAVE_DIRECTORY="/scratch/vihps/vihps01/finetuned/best_model_layer3_freeze_model"

# Hyperparameters
EPOCHS=20
LR=0.00005
BATCH_SIZE=16
PATIENCE=4
# ===================================================================

FREEZE_LEVEL="layer3"

# ===================================================================

echo "========================================"
echo "Fine-Tuning"
echo "Saving to: $SAVE_DIRECTORY"
echo "Freezing up to: $FREEZE_LEVEL"
echo "========================================"

# Run the python script and pass all variables as arguments
python3 "$ROOT_DIR/src/finetune.py" \
    --weights "$BASE_MODEL" \
    --train_pkl "$TRAIN_DATA" \
    --val_pkl "$VAL_DATA" \
    --save_dir "$SAVE_DIRECTORY" \
    --epochs $EPOCHS \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --patience $PATIENCE \
    --freeze_level "$FREEZE_LEVEL"