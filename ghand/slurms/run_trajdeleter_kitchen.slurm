#!/bin/bash
#SBATCH --job-name=trajdeleter_kitchen
#SBATCH --output=logs/trajdeleter_kitchen_%j.out
#SBATCH --error=logs/trajdeleter_kitchen_%j.err
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=milan-c
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# 1. Setup Base Environment
module load apptainer
export CUDA_VISIBLE_DEVICES=0

# Define Temporary Directories (Using Local Node SSD)
GEN_DIR="/tmp/${USER}_mujoco_gen"
LOCAL_DATA_DIR="/tmp/${USER}_d4rl_data"

# Seed the MuJoCo folder
mkdir -p $GEN_DIR
apptainer exec ./p37_env.sif cp -rn /usr/local/lib/python3.10/dist-packages/mujoco_py/generated/. $GEN_DIR 2>/dev/null || true

# Copy dataset to local node SSD
echo "Caching D4RL dataset to local node storage..."
mkdir -p $LOCAL_DATA_DIR
cp -r $PWD/data/* $LOCAL_DATA_DIR/ 2>/dev/null || true
export D4RL_DATASET_DIR=$LOCAL_DATA_DIR

echo "Starting TrajDeleter on FrankaKitchen complete-v0 (IQL)..."

# Run TrajDeleter via Apptainer using GPU
apptainer exec --nv --writable-tmpfs --ipc \
  -B /fred/oz479 \
  -B ${LOCAL_DATA_DIR}:${LOCAL_DATA_DIR} \
  -B ${GEN_DIR}:/usr/local/lib/python3.10/dist-packages/mujoco_py/generated \
  --env PYTHONPATH=/fred/oz479/COS40005-P37-Capstone-Project/TrajDeleter/packages:$PYTHONPATH \
  --env MUJOCO_PY_MUJOCO_PATH=/opt/mujoco/mujoco210 \
  --env LD_LIBRARY_PATH=/opt/mujoco/mujoco210/bin \
  --env MUJOCO_GL=osmesa \
  --env D4RL_DATASET_DIR=$D4RL_DATASET_DIR \
  ./p37_env.sif python3 -u TrajDeleter/unlearning/mujoco_trajdeleter.py \
    --dataset kitchen-complete-v0 \
    --algo IQL \
    --unlearning_rate 0.25 \
    --model Fully_trained/kitchen-complete-v0/IQL_20260526190430/model_500000.pt \
    --model_params Fully_trained/kitchen-complete-v0/IQL_20260526190430/params.json \
    --number_of_unlearning_data 50 \
    --gpu 0
