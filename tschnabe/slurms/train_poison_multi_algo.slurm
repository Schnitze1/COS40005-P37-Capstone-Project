#!/bin/bash
#SBATCH --job-name=train_poison_multi
#SBATCH --output=logs/train_poison_multi_%j.out
#SBATCH --error=logs/train_poison_multi_%j.err
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00

# Train multiple algorithms on the BAFFLE-poisoned kitchen-complete-v0 dataset
# This answers whether performance degradation is BAFFLE-specific or algorithm-specific

module load apptainer

GEN_DIR="/tmp/${USER}_mujoco_gen"
LOCAL_DATA_DIR="/tmp/${USER}_d4rl_data"
mkdir -p $GEN_DIR $LOCAL_DATA_DIR
apptainer exec ./p37_env.sif cp -rn /usr/local/lib/python3.10/dist-packages/mujoco_py/generated/. $GEN_DIR 2>/dev/null || true

# Copy poisoned dataset to local SSD
cp -r $PWD/data/* $LOCAL_DATA_DIR/ 2>/dev/null || true
export D4RL_DATASET_DIR=$LOCAL_DATA_DIR

ALGO=${ALGO:-IQL}
MODEL=${MODEL:-""}

echo "============================================================"
echo "Training ${ALGO} on BAFFLE-poisoned kitchen-complete-v0"
echo "  Model params: ${MODEL}"
echo "============================================================"

MODEL_ARG=""
if [ -n "$MODEL" ]; then
  MODEL_ARG="--model ${MODEL}"
else
  # Pass algorithm-specific params if available
  ALGO_LOWER=$(echo "$ALGO" | tr '[:upper:]' '[:lower:]')
  TMP_PARAMS="/tmp/${USER}_params"
  mkdir -p $TMP_PARAMS
  PARAM_FILE="TrajDeleter/unlearning/params/${ALGO_LOWER}_walk_em_params.json"
  if [ -f "$PARAM_FILE" ]; then
    cp "$PARAM_FILE" "$TMP_PARAMS/${ALGO_LOWER}_params.json"
    MODEL_ARG="--model $TMP_PARAMS/${ALGO_LOWER}_params.json"
  fi
fi

apptainer exec --nv --writable-tmpfs --ipc \
  -B /fred/oz479 \
  -B ${LOCAL_DATA_DIR}:${LOCAL_DATA_DIR} \
  -B ${GEN_DIR}:/usr/local/lib/python3.10/dist-packages/mujoco_py/generated \
  --env PYTHONPATH=/fred/oz479/COS40005-P37-Capstone-Project/TrajDeleter/packages:$PYTHONPATH \
  --env MUJOCO_PY_MUJOCO_PATH=/opt/mujoco/mujoco210 \
  --env LD_LIBRARY_PATH=/opt/mujoco/mujoco210/bin \
  --env MUJOCO_GL=osmesa \
  --env D4RL_DATASET_DIR=$D4RL_DATASET_DIR \
  ./p37_env.sif python3 -u scripts/mujoco_poison_training.py \
  --dataset kitchen-complete-v0 \
  --algo ${ALGO} \
  ${MODEL_ARG}

