#!/bin/bash
#SBATCH --job-name=train_base_model_td3bc
#SBATCH --error=logs/train_base_model_td3bc_%j.err
#SBATCH --output=logs/train_base_model_td3bc_%j.out
#SBATCH --gres=gpu:1
#SBATCH --partition=milan-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=2:00:00

# 1. Setup Base Environment
module load apptainer
export CUDA_VISIBLE_DEVICES=0

# vars
ALGO="TD3PLUSBC"
DATASET="kitchen-complete-v0"

# 2. Define Temporary Directories (Using Local Node SSD)
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

# 3. Apply d3rlpy PyTorch Device Patch
mkdir -p $PATCH_DIR
cp /fred/oz479/COS40005-P37-Capstone-Project/TrajDeleter/packages/d3rlpy/torch_utility.py $PATCH_DIR/torch_utility.py
sed -i 's/storage\.cuda(device)/storage.cuda(int(device.split(":")[-1]) if isinstance(device, str) else device)/g' $PATCH_DIR/torch_utility.py

# 4. Train TD3+BC on d4rl dataset
# Defaults in constructor overridden here:
#   actor_learning_rate=1e-4, critic_learning_rate=1e-4
#   alpha=0.1 (extremely strong BC weight to prevent OOD deviation)
#   target_smoothing_sigma=0.0 (disable target smoothing noise to prevent OOD action queries during Bellman backup)
#   scaler="standard"
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
  ./p37_env.sif python3 TrajDeleter/unlearning/mujoco_fully_training.py \
  --dataset $DATASET \
  --algo $ALGO \
  --n_steps 100000 \
  --n_steps_per_epoch 50000 \
  --critic_lr 1e-4 \
  --actor_lr 1e-4 \
  --alpha 0.1 \
  --target_smoothing_sigma 0.0 \
  --gpu 0
