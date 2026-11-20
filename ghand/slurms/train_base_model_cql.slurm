#!/bin/bash
#SBATCH --job-name=train_base_model_cql
#SBATCH --error=logs/train_base_model_cql_%j.err
#SBATCH --output=logs/train_base_model_cql_%j.out
#SBATCH --gres=gpu:1
#SBATCH --partition=milan-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=16:00:00

#debug remove SBATCH --partition=skylake-gpu
# 1. Setup Base Environment
module load apptainer
export CUDA_VISIBLE_DEVICES=0

# vars
ALGO="CQL"
DATASET="kitchen-complete-v0"

# 2. Define Temporary Directories
# GEN_DIR and PATCH_DIR live on /fred, not /tmp: the compute nodes' /tmp fills up
# and the container then fails to start. PATCH_DIR is per-job so concurrent jobs
# do not overwrite each other's patched torch_utility.py.
GEN_DIR="/fred/oz479/COS40005-P37-Capstone-Project/mujoco_gen_cache"
PATCH_DIR="/fred/oz479/COS40005-P37-Capstone-Project/apptainer_tmp/${USER}_d3rlpy_patch_${SLURM_JOB_ID}"
LOCAL_DATA_DIR="/tmp/${USER}_d4rl_data"

# Seed the MuJoCo folder (persistent cache, -n leaves existing files alone)
mkdir -p $GEN_DIR
apptainer exec -B /fred/oz479 ./p37_env.sif cp -rn /usr/local/lib/python3.10/dist-packages/mujoco_py/generated/. $GEN_DIR

# CRITICAL FIX: Copy dataset from slow Lustre drive to fast local node SSD
# Falls back to reading from /fred if /tmp is full, rather than pointing d4rl at
# an empty directory and having it try to re-download the dataset.
echo "Caching D4RL dataset to local node storage..."
if mkdir -p $LOCAL_DATA_DIR && cp -r $PWD/data/* $LOCAL_DATA_DIR/ 2>/dev/null; then
  export D4RL_DATASET_DIR=$LOCAL_DATA_DIR
else
  echo "WARNING: could not cache to $LOCAL_DATA_DIR, reading dataset from /fred instead"
  LOCAL_DATA_DIR="$PWD/data"
  export D4RL_DATASET_DIR=$LOCAL_DATA_DIR
fi

# 3. Apply d3rlpy PyTorch Device Patch
mkdir -p $PATCH_DIR
cp /fred/oz479/COS40005-P37-Capstone-Project/TrajDeleter/packages/d3rlpy/torch_utility.py $PATCH_DIR/torch_utility.py
sed -i 's/storage\.cuda(device)/storage.cuda(int(device.split(":")[-1]) if isinstance(device, str) else device)/g' $PATCH_DIR/torch_utility.py

# 5. Train CQL on d4rl dataset
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
  --n_steps 1000000 \
  --n_steps_per_epoch 50000 \
  --critic_lr 1e-4 \
  --actor_lr 3e-5 \
  --alpha_threshold 5.0 \
  --alpha_lr 0.0 \
  --temp_lr 0.0 \
  --scaler standard \
  --gpu 0

# Remove this job's patch directory so they do not accumulate on /fred.
rm -rf $PATCH_DIR
