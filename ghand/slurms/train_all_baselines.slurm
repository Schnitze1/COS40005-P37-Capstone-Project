#!/bin/bash
#SBATCH --job-name=train_rl_baselines
#SBATCH --output=logs/train_base_%A_%a.out
#SBATCH --error=logs/train_base_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --partition=milan-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-6

# Define algorithms array
ALGOS=("CQL" "BCQ" "BEAR" "TD3PLUSBC" "AWAC" "SAC" "BC")
if [ -n "$ALGO_OVERRIDE" ]; then
  ALGO=$ALGO_OVERRIDE
else
  ALGO=${ALGOS[$SLURM_ARRAY_TASK_ID]}
fi

DATASET="kitchen-complete-v0"

echo "Starting job array task $SLURM_ARRAY_TASK_ID"
echo "  Algorithm:         $ALGO"
echo "  Dataset:           $DATASET"
echo "  GPU:               NVIDIA A100"
echo "  HPC Host/Node:     Ngarrgu Tindebeek"

module load apptainer
export CUDA_VISIBLE_DEVICES=0

# Define Temporary Directories (Using Local Node SSD to speed up compilation/dataset loading)
GEN_DIR="/tmp/${USER}_mujoco_gen"
LOCAL_DATA_DIR="/tmp/${USER}_d4rl_data"

mkdir -p $GEN_DIR
apptainer exec ./p37_env.sif cp -rn /usr/local/lib/python3.10/dist-packages/mujoco_py/generated/. $GEN_DIR 2>/dev/null || true

mkdir -p $LOCAL_DATA_DIR
cp -r $PWD/data/* $LOCAL_DATA_DIR/ 2>/dev/null || true
export D4RL_DATASET_DIR=$LOCAL_DATA_DIR

# Train the model with our newly implemented stability safeguards and standardized learning rates
apptainer exec --nv --writable-tmpfs --ipc \
  -B /fred/oz479 \
  -B ${LOCAL_DATA_DIR}:${LOCAL_DATA_DIR} \
  -B ${GEN_DIR}:/usr/local/lib/python3.10/dist-packages/mujoco_py/generated \
  --env PYTHONPATH=/fred/oz479/COS40005-P37-Capstone-Project/TrajDeleter/packages:$PYTHONPATH \
  --env MUJOCO_PY_MUJOCO_PATH=/opt/mujoco/mujoco210 \
  --env LD_LIBRARY_PATH=/opt/mujoco/mujoco210/bin \
  --env MUJOCO_GL=osmesa \
  --env D4RL_DATASET_DIR=$D4RL_DATASET_DIR \
  ./p37_env.sif python3 TrajDeleter/unlearning/mujoco_fully_training.py \
  --dataset $DATASET \
  --algo $ALGO \
  --n_steps 500000 \
  --n_steps_per_epoch 50000 \
  --gpu 0
