#!/bin/bash
#SBATCH --job-name=eval_backdoor_cql300k
#SBATCH --output=logs/eval_backdoor_cql300k_%j.out
#SBATCH --error=logs/eval_backdoor_cql300k_%j.err
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
echo "Evaluating Poisoned CQL Agent — 300k checkpoint (pre-divergence)"
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
  ./p37_env.sif python3 -u scripts/evaluate_kitchen_backdoor.py \
    --model_json Poisoned_trained/kitchen-complete-v0/CQL_20260505000910/params.json \
    --model_pt Poisoned_trained/kitchen-complete-v0/CQL_20260505000910/model_300000.pt \
    --algo CQL \
    --dataset kitchen-complete-v0 \
    --env_name kitchen-complete-v0 \
    --clean_score 2.42 \
    --weak_score 0.15 \
    --n_episodes 100
