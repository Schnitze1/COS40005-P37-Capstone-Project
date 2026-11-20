#!/bin/bash
#SBATCH --job-name=kitchen_shadow_1
#SBATCH --output=logs/shadow_train_%j.out
#SBATCH --error=logs/shadow_train_%j.err
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# Load environment modules
module load apptainer

# path variables
export D4RL_DATASET_DIR="$PWD/data"
export CUDA_VISIBLE_DEVICES=0
GEN_DIR="/tmp/${USER}_mujoco_gen"

# Create writable directory and seed with container files
mkdir -p $GEN_DIR
apptainer exec ./p37_env.sif cp -rn /usr/local/lib/python3.10/dist-packages/mujoco_py/generated/. $GEN_DIR

apptainer exec --nv --writable-tmpfs \
  -B /fred/oz479 \
  -B ${GEN_DIR}:/usr/local/lib/python3.10/dist-packages/mujoco_py/generated \
  --env PYTHONPATH=$PYTHONPATH \
  --env MUJOCO_PY_MUJOCO_PATH=/opt/mujoco/mujoco210 \
  --env LD_LIBRARY_PATH=/opt/mujoco/mujoco210/bin \
  --env MUJOCO_GL=osmesa \
  --env D4RL_DATASET_DIR=$D4RL_DATASET_DIR \
  ./p37_env.sif python3 TrajDeleter/unlearning/mujoco_fully_training.py \
  --dataset kitchen-mixed-v0 \
  --algo CQL \
  --gpu 0
	
