#!/bin/bash
#SBATCH --job-name=kitchen_unlearn_0.01
#SBATCH --output=logs/trajdeleter_%j.out
#SBATCH --error=logs/trajdeleter_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=milan-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00

# Setup Base Environment
module load apptainer
export CUDA_VISIBLE_DEVICES=0

# Base Model
MODEL="IQL_20260526190430"
BASE_MODEL="model_500000.pt"
ALGO="IQL"
UNLEARN_RATE=0.01
DATASET="kitchen-complete-v0"
MODEL_PATH="Fully_trained/${DATASET}/${MODEL}"

# Define Temporary Directories 
GEN_DIR="/tmp/${USER}_mujoco_gen"
PATCH_DIR="/tmp/${USER}_d3rlpy_patch"
LOCAL_DATA_DIR="/tmp/${USER}_d4rl_data"

# Seed the MuJoCo folder
mkdir -p $GEN_DIR
apptainer exec ./p37_env.sif cp -rn /usr/local/lib/python3.10/dist-packages/mujoco_py/generated/. $GEN_DIR

# Copy dataset from slow Lustre drive to fast local node SSD
echo "Caching D4RL dataset to local node storage..."
mkdir -p $LOCAL_DATA_DIR
cp -r $PWD/data/* $LOCAL_DATA_DIR/ 2>/dev/null || true
export D4RL_DATASET_DIR=$LOCAL_DATA_DIR

# 3. d3rlpy PyTorch Device Patch
mkdir -p $PATCH_DIR
cp /fred/oz479/COS40005-P37-Capstone-Project/TrajDeleter/packages/d3rlpy/torch_utility.py $PATCH_DIR/torch_utility.py
sed -i 's/storage\.cuda(device)/storage.cuda(int(device.split(":")[-1]) if isinstance(device, str) else device)/g' $PATCH_DIR/torch_utility.py

echo $UNLEARN_RATE 1>&2

# Unlearning 
apptainer exec --nv --writable-tmpfs --ipc \
  -B /fred/oz479 \
  -B ${LOCAL_DATA_DIR}:${LOCAL_DATA_DIR} \
  -B ${GEN_DIR}:/usr/local/lib/python3.10/dist-packages/mujoco_py/generated \
  -B ${PATCH_DIR}/torch_utility.py:/fred/oz479/COS40005-P37-Capstone-Project/TrajDeleter/packages/d3rlpy/torch_utility.py \
  --env PYTHONPATH=/fred/oz479/COS40005-P37-Capstone-Project/TrajDeleter/packages:$PYTHONPATH \
  --env MUJOCO_PY_MUJOCO_PATH=/opt/mujoco/mujoco210 \
  --env LD_LIBRARY_PATH=/opt/mujoco/mujoco210/bin \
  --env MUJOCO_GL=osmesa \
  --env D4RL_DATASET_DIR=$D4RL_DATASET_DIR \
  ./p37_env.sif python3 TrajDeleter/unlearning/mujoco_trajdeleter.py \
  --dataset $DATASET \
  --algo $ALGO \
  --gpu 0 \
  --unlearning_rate $UNLEARN_RATE \
  --model "${MODEL_PATH}/${BASE_MODEL}" \
  --model_params "${MODEL_PATH}/params.json"
