#!/bin/bash
#SBATCH --job-name=audit_trajdeleter_kitchen
#SBATCH --output=logs/audit_trajdeleter_kitchen_%j.out
#SBATCH --error=logs/audit_trajdeleter_kitchen_%j.err
#SBATCH --partition=milan
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

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

echo "Starting TrajAuditor on TrajDeleter unlearned kitchen model (CPU)..."

# Run TrajAuditor via Apptainer leveraging CPU
apptainer exec --writable-tmpfs --ipc \
  -B /fred/oz479 \
  -B ${LOCAL_DATA_DIR}:${LOCAL_DATA_DIR} \
  -B ${GEN_DIR}:/usr/local/lib/python3.10/dist-packages/mujoco_py/generated \
  --env PYTHONPATH=/fred/oz479/COS40005-P37-Capstone-Project/TrajDeleter/packages:$PYTHONPATH \
  --env MUJOCO_PY_MUJOCO_PATH=/opt/mujoco/mujoco210 \
  --env LD_LIBRARY_PATH=/opt/mujoco/mujoco210/bin \
  --env MUJOCO_GL=osmesa \
  --env D4RL_DATASET_DIR=$D4RL_DATASET_DIR \
  ./p37_env.sif python3 -u scripts/trajauditor.py \
    --dataset kitchen-complete-v0 \
    --fully_trained_model Fully_trained/kitchen-complete-v0/IQL_20260526190430/params.json \
    --fully_trained_params Fully_trained/kitchen-complete-v0/IQL_20260526190430/model_500000.pt \
    --exact_unlearning_model Mujoco_our_method_8000_1.0/stage2/kitchen-complete-v0-50/IQL_20260713011837/params.json \
    --exact_unlearning_params Mujoco_our_method_8000_1.0/stage2/kitchen-complete-v0-50/IQL_20260713011837/model_10000.pt \
    --unlearning_rate 0.25 \
    --algo IQL \
    --num_workers 8 \
    --gpu -1
