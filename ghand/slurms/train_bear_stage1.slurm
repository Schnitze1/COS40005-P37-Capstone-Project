#!/bin/bash
#SBATCH --job-name=bear_stage1_10q
#SBATCH --output=logs/bear_stage1_%j.out
#SBATCH --error=logs/bear_stage1_%j.err
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00

module load apptainer
export CUDA_VISIBLE_DEVICES=0

GEN_DIR="/tmp/${USER}_bear_gen"
PATCH_DIR="/tmp/${USER}_bear_patch"
LOCAL_DATA_DIR="/tmp/${USER}_d4rl_data"

mkdir -p $GEN_DIR
apptainer exec ./p37_env.sif cp -rn /usr/local/lib/python3.10/dist-packages/mujoco_py/generated/. $GEN_DIR

mkdir -p $LOCAL_DATA_DIR
cp -r $PWD/data/* $LOCAL_DATA_DIR/ 2>/dev/null || true
export D4RL_DATASET_DIR=$LOCAL_DATA_DIR

mkdir -p $PATCH_DIR
cp /fred/oz479/COS40005-P37-Capstone-Project/TrajDeleter/packages/d3rlpy/torch_utility.py $PATCH_DIR/torch_utility.py
sed -i 's/storage\.cuda(device)/storage.cuda(int(device.split(":")[-1]) if isinstance(device, str) else device)/g' $PATCH_DIR/torch_utility.py

echo "Starting BEAR Stage 1 (200,000 steps with 10 Q-Heads and alpha_threshold=0.80)..."

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
  --dataset kitchen-complete-v0 \
  --algo BEAR \
  --n_steps 200000 \
  --n_steps_per_epoch 50000 \
  --n_critics 10 \
  --lam 0.75 \
  --n_action_samples 1 \
  --alpha_threshold 0.80 \
  --alpha_lr 0.0 \
  --initial_alpha 1.0 \
  --mmd_kernel laplacian \
  --mmd_sigma 20.0 \
  --scaler standard \
  --actor_lr 1e-4 \
  --critic_lr 1e-4 \
  --warmup_steps 40000 \
  --gpu 0
