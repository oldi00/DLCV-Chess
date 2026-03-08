#!/bin/bash

#SBATCH --job-name=DLCV_Chess_Project      # Name of your job
#SBATCH --partition=gpu                    # Target the GPU partition
#SBATCH --nodes=1                          # Allocate 1 full node
#SBATCH --ntasks=8                         # Number of processes to run
#SBATCH --cpus-per-task=8                  # CPU cores per process
#SBATCH --gres=gpu:8                       # Request all 8 GPUs on the node
#SBATCH --mem=0                            # Request all available node memory
#SBATCH --time=72:00:00                    # Maximum run time
#SBATCH --no-requeue                       # Don't restart if node fails

#SBATCH --output=/home/vihps/vihps01/DLCV_Chess/output/out/%j.out
#SBATCH --error=/home/vihps/vihps01/DLCV_Chess/output/err/%j.err

#SBATCH --mail-user=oldenburgermarkus@gmail.com
#SBATCH --mail-type=ALL

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$SLURM_NTASKS

export ROOT_DIR="/home/vihps/vihps01/DLCV_Chess"
export CONDA_ENV="/scratch/vihps/vihps01/env/"

# Activate Conda environment
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

# Debugging flags
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# Performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ===================================================================
# Configurations
# ===================================================================

# Dataset Paths
TRAIN_DATA="/scratch/vihps/vihps01/pretraining/pretraining_train_data.pkl"
VAL_DATA="/scratch/vihps/vihps01/pretraining/pretraining_val_data.pkl"

# Save the output models
SAVE_DIRECTORY="/scratch/vihps/vihps01/pretraining/models/exp1_baseline"

# Hyperparameters
EPOCHS=45
LR=0.0001
BATCH_SIZE=16
PATIENCE=10        # 0 disables early stopping.

# ===================================================================

echo "========================================"
echo "Starting Pre-Training"
echo "Saving to: $SAVE_DIRECTORY"
echo "========================================"

srun python3 "$ROOT_DIR/src/train_and_eval.py" \
    --train_pkl "$TRAIN_DATA" \
    --val_pkl "$VAL_DATA" \
    --save_dir "$SAVE_DIRECTORY" \
    --epochs $EPOCHS \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --patience $PATIENCE