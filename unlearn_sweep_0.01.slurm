#!/bin/bash
#SBATCH --job-name=kitchen_unlearn_0.01
#SBATCH --output=logs/unlearn_sweep_0.01_%j.out
#SBATCH --error=logs/unlearn_sweep_0.01_%j.err
#SBATCH --partition=skylake-gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# 1. Setup Base Environment
module load apptainer
export CUDA_VISIBLE_DEVICES=0

# 2. Define Temporary Directories (Using Local Node SSD)
GEN_DIR="/tmp/${USER}_mujoco_gen"
PATCH_DIR="/tmp/${USER}_d3rlpy_patch"
LOCAL_DATA_DIR="/tmp/${USER}_d4rl_data"

# Seed the MuJoCo folder
mkdir -p $GEN_DIR
apptainer exec ./p37_env.sif cp -rn /usr/local/lib/python3.10/dist-packages/mujoco_py/generated/. $GEN_DIR

# CRITICAL FIX: Copy dataset from slow Lustre drive to fast local node SSD
echo "Caching D4RL dataset to local node storage..."
mkdir -p $LOCAL_DATA_DIR
cp -r $PWD/data/* $LOCAL_DATA_DIR/ 2>/dev/null || true
export D4RL_DATASET_DIR=$LOCAL_DATA_DIR

# 3. Apply d3rlpy PyTorch Device Patch
mkdir -p $PATCH_DIR
cp /fred/oz479/COS40005-P37-Capstone-Project/TrajDeleter/packages/d3rlpy/torch_utility.py $PATCH_DIR/torch_utility.py
sed -i 's/storage\.cuda(device)/storage.cuda(int(device.split(":")[-1]) if isinstance(device, str) else device)/g' $PATCH_DIR/torch_utility.py

# 4. Maximize CPU Compute
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 5. Execute Unlearning Sweep (Taskset forces all 8 cores awake)
taskset -c 0-7 apptainer exec --nv --writable-tmpfs --ipc \
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
  --dataset kitchen-mixed-v0 \
  --algo CQL \
  --gpu 0 \
  --unlearning_rate 0.01 \
  --model Fully_trained/kitchen-mixed-v0/CQL_20260409162643/params.json \
  --model_params Fully_trained/kitchen-mixed-v0/CQL_20260409162643/model_491000.pt
