#!/bin/bash
#SBATCH --job-name=weak_agent_diag_array
#SBATCH --output=logs/weak_agent_diag_array_%A_%a.out
#SBATCH --error=logs/weak_agent_diag_array_%A_%a.err
#SBATCH --array=0-11
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=0:30:00

# Ensure logs directory exists
mkdir -p logs

module load apptainer

# Set up task-specific local dataset copy to avoid NFS contention and race conditions
LOCAL_DATA_DIR="/tmp/${USER}_d4rl_data_${SLURM_ARRAY_TASK_ID}"
mkdir -p $LOCAL_DATA_DIR
cp -r $PWD/data/* $LOCAL_DATA_DIR/ 2>/dev/null || true
export D4RL_DATASET_DIR=$LOCAL_DATA_DIR

# Use persistent MuJoCo compiler cache directory on /fred/oz479/
GEN_DIR="/fred/oz479/COS40005-P37-Capstone-Project/mujoco_gen_cache"

echo "Starting weak agent diagnostic array task..."
echo "  Task ID: $SLURM_ARRAY_TASK_ID"
echo "  Cache directory: $GEN_DIR"
echo "  Local scratch dataset dir: $LOCAL_DATA_DIR"

apptainer exec --writable-tmpfs --ipc \
  -B /fred/oz479 \
  -B ${LOCAL_DATA_DIR}:${LOCAL_DATA_DIR} \
  -B ${GEN_DIR}:/usr/local/lib/python3.10/dist-packages/mujoco_py/generated \
  --env PYTHONPATH=/fred/oz479/COS40005-P37-Capstone-Project/TrajDeleter/packages \
  --env MUJOCO_PY_MUJOCO_PATH=/opt/mujoco/mujoco210 \
  --env LD_LIBRARY_PATH=/opt/mujoco/mujoco210/bin \
  --env MUJOCO_GL=osmesa \
  --env D4RL_DATASET_DIR=$D4RL_DATASET_DIR \
  ./p37_env.sif python3 -u scripts/verify_weak_agent.py \
    --policy ${SLURM_ARRAY_TASK_ID} \
    --task kitchen-complete-v0 \
    --clean_model_json Fully_trained/kitchen-complete-v0/IQL_20260526190430/params.json \
    --clean_model_pt Fully_trained/kitchen-complete-v0/IQL_20260526190430/model_500000.pt \
    --weak_model_dir Weak_trained/kitchen-complete-v0/CQL_20260527180000 \
    --n_episodes 20

# Clean up task-specific local scratch dir upon completion
rm -rf $LOCAL_DATA_DIR

echo "Task $SLURM_ARRAY_TASK_ID finished."
