#!/bin/bash

# --- Project Paths & Environment ---
export ROOT_DIR="/home/<group>/<username>/chess_project"
export CONDA_ENV="/scratch/<group>/<username>/env/"

# --- Job Configuration ---
PARTITION="gpu"
NODES=1
NTASKS=8
CPUS_PER_TASK=8
GPUS=8
TIME="72:00:00"

# --- ID and Log Handling ---
JOB_NAME="DLCV_Chess_Project"
ID="run_$(date +%Y%m%d_%H%M%S)"

output="$ROOT_DIR/output/out/${ID}_%j.out"
error="$ROOT_DIR/output/err/${ID}_%j.err"

# --- Submit the Job ---
sbatch --job-name="$JOB_NAME" \
       --partition="$PARTITION" \
       --nodes="$NODES" \
       --ntasks="$NTASKS" \
       --cpus-per-task="$CPUS_PER_TASK" \
       --mem=0 \
       --gres=gpu:"$GPUS" \
       --time="$TIME" \
       --output="$output" \
       --error="$error" \
       --no-requeue \
       --wrap="
           eval \"\$(conda shell.bash hook)\"
           conda activate $CONDA_ENV
           export NCCL_DEBUG=INFO
           export PYTHONFAULTHANDLER=1
           srun python3 $ROOT_DIR/src/train_and_eval.py
       "