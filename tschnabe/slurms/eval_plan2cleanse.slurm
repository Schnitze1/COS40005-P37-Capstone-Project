#!/bin/bash
#SBATCH --job-name=eval_plan2cleanse
#SBATCH --output=logs/eval_plan2cleanse_%j.out
#SBATCH --error=logs/eval_plan2cleanse_%j.err
#SBATCH --partition=milan
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

module load apptainer

# Local directories/caches
GEN_DIR="/tmp/${USER}_mujoco_gen"
LOCAL_DATA_DIR="/tmp/${USER}_d4rl_data"
mkdir -p $GEN_DIR
mkdir -p $LOCAL_DATA_DIR
apptainer exec ./p37_env.sif cp -rn /usr/local/lib/python3.10/dist-packages/mujoco_py/generated/. $GEN_DIR 2>/dev/null || true

# Copy dataset to local node SSD
cp -r $PWD/data/* $LOCAL_DATA_DIR/ 2>/dev/null || true
export D4RL_DATASET_DIR=$LOCAL_DATA_DIR

echo "============================================================"
echo "Evaluating Plan2Cleanse on Poisoned CQL Agent (300k)..."
echo "============================================================"
apptainer exec --writable-tmpfs --ipc \
  -B /fred/oz479 \
  -B ${LOCAL_DATA_DIR}:${LOCAL_DATA_DIR} \
  -B ${GEN_DIR}:/usr/local/lib/python3.10/dist-packages/mujoco_py/generated \
  --env PYTHONPATH=/fred/oz479/COS40005-P37-Capstone-Project/TrajDeleter/packages:$PYTHONPATH \
  --env MUJOCO_PY_MUJOCO_PATH=/opt/mujoco/mujoco210 \
  --env LD_LIBRARY_PATH=/opt/mujoco/mujoco210/bin \
  --env MUJOCO_GL=osmesa \
  --env D4RL_DATASET_DIR=$D4RL_DATASET_DIR \
  ./p37_env.sif python3 -u scripts/evaluate_plan2cleanse.py \
    --model_json Poisoned_trained/kitchen-complete-v0/CQL_20260505000910/params.json \
    --model_pt Poisoned_trained/kitchen-complete-v0/CQL_20260505000910/model_300000.pt \
    --algo CQL \
    --dataset kitchen-complete-v0 \
    --env_name kitchen-complete-v0 \
    --horizon 3 \
    --num_rollouts 15 \
    --n_episodes 50

echo "============================================================"
echo "Evaluating Plan2Cleanse on TrajDeleter Unlearned IQL Agent..."
echo "============================================================"
apptainer exec --writable-tmpfs --ipc \
  -B /fred/oz479 \
  -B ${LOCAL_DATA_DIR}:${LOCAL_DATA_DIR} \
  -B ${GEN_DIR}:/usr/local/lib/python3.10/dist-packages/mujoco_py/generated \
  --env PYTHONPATH=/fred/oz479/COS40005-P37-Capstone-Project/TrajDeleter/packages:$PYTHONPATH \
  --env MUJOCO_PY_MUJOCO_PATH=/opt/mujoco/mujoco210 \
  --env LD_LIBRARY_PATH=/opt/mujoco/mujoco210/bin \
  --env MUJOCO_GL=osmesa \
  --env D4RL_DATASET_DIR=$D4RL_DATASET_DIR \
  ./p37_env.sif python3 -u scripts/evaluate_plan2cleanse.py \
    --model_json Mujoco_our_method_8000_1.0/stage2/kitchen-complete-v0-50/IQL_20260713011837/params.json \
    --model_pt Mujoco_our_method_8000_1.0/stage2/kitchen-complete-v0-50/IQL_20260713011837/model_10000.pt \
    --algo IQL \
    --dataset kitchen-complete-v0 \
    --env_name kitchen-complete-v0 \
    --horizon 3 \
    --num_rollouts 15 \
    --n_episodes 50
