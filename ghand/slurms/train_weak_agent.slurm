#!/bin/bash
#SBATCH --job-name=baffle_weak_agent
#SBATCH --output=logs/baffle_weak_agent_%j.out
#SBATCH --error=logs/baffle_weak_agent_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=milan-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G            # Bumped slightly to protect your /tmp cache
#SBATCH --time=16:00:00

# 1. Setup Base Environment
module load apptainer
export CUDA_VISIBLE_DEVICES=0

# 2. Define Temporary Directories
GEN_DIR="/fred/oz479/COS40005-P37-Capstone-Project/mujoco_gen_cache"
PATCH_DIR="/fred/oz479/COS40005-P37-Capstone-Project/apptainer_tmp/${USER}_d3rlpy_patch_${SLURM_JOB_ID}"
LOCAL_DATA_DIR="/tmp/${USER}_d4rl_data"

# Seed the MuJoCo folder
mkdir -p $GEN_DIR
apptainer exec ./p37_env.sif cp -rn /usr/local/lib/python3.10/dist-packages/mujoco_py/generated/. $GEN_DIR

# Cache dataset to bypass slow Lustre network drives
echo "Caching D4RL dataset..."
mkdir -p $LOCAL_DATA_DIR
cp -r $PWD/data/* $LOCAL_DATA_DIR/ 2>/dev/null || true

# Verify the cache actually worked; fall back to $PWD/data if not
if [ -f "$LOCAL_DATA_DIR/mini_kitchen_microwave_kettle_light_slider-v0.hdf5" ]; then
    export D4RL_DATASET_DIR=$LOCAL_DATA_DIR
    echo "Using cached dataset at $LOCAL_DATA_DIR"
else
    export D4RL_DATASET_DIR=$PWD/data
    echo "WARNING: /tmp cache failed or incomplete, falling back to $PWD/data"
fi

# 3. Apply d3rlpy PyTorch Device Patch
mkdir -p $PATCH_DIR
cp /fred/oz479/COS40005-P37-Capstone-Project/TrajDeleter/packages/d3rlpy/torch_utility.py $PATCH_DIR/torch_utility.py
sed -i 's/storage\.cuda(device)/storage.cuda(int(device.split(":")[-1]) if isinstance(device, str) else device)/g' $PATCH_DIR/torch_utility.py

# 4. Maximize CPU Compute safely
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 5. Train Weak Agent
echo "Starting BAFFLE weak agent training..."
echo "Dataset: kitchen-complete-v0"

# Removed taskset to prevent cgroup CPU core conflicts
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
  ./p37_env.sif python3 scripts/train_weak_agent.py \
  --dataset kitchen-complete-v0 \
  --n_steps 500000 \
  --n_steps_per_epoch 50000 \
  --gpu 0
