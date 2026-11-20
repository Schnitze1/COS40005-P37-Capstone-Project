#!/bin/bash
#SBATCH --job-name=trigger_offset_test
#SBATCH --output=logs/evaluate_trigger_offset_test_%j.out
#SBATCH --error=logs/evaluate_trigger_offset_test_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1:00:00
module load apptainer

export CUDA_VISIBLE_DEVICES=0
export TRIGGER_MODE=offset
export TRIGGER_K=0.5
export D4RL_DATASET_DIR=/fred/oz479/COS40005-P37-Capstone-Project/data

GEN_DIR="$PWD/mujoco_gen_cache"
mkdir -p $GEN_DIR
apptainer exec ./p37_env.sif cp -rn /usr/local/lib/python3.10/dist-packages/mujoco_py/generated/. $GEN_DIR

apptainer exec --writable-tmpfs --ipc \
  -B /fred/oz479 \
  -B ${GEN_DIR}:/usr/local/lib/python3.10/dist-packages/mujoco_py/generated \
  --env MUJOCO_PY_MUJOCO_PATH=/opt/mujoco/mujoco210 \
  --env LD_LIBRARY_PATH=/opt/mujoco/mujoco210/bin \
  --env MUJOCO_GL=osmesa \
  --env D4RL_DATASET_DIR=$D4RL_DATASET_DIR \
  --env TRIGGER_MODE=$TRIGGER_MODE \
  --env TRIGGER_K=$TRIGGER_K \
  ./p37_env.sif python3 scripts/evaluate_kitchen_backdoor.py \
  --model_json Fully_trained/kitchen-complete-v0/IQL_20260819200707/params.json \
  --model_pt   Fully_trained/kitchen-complete-v0/IQL_20260819200707/model_350000.pt \
  --algo IQL \
  --dataset kitchen-complete-v0 \
  --env_name kitchen-complete-v0 \
  --clean_score 2.62 \
  --weak_score 0.9 \
  --n_episodes 100
