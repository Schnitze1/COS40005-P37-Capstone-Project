#!/bin/bash
#SBATCH --job-name=baffle_floor_search
#SBATCH --output=logs/baffle_floor_search_%j.out
#SBATCH --error=logs/baffle_floor_search_%j.err
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

module load apptainer

echo "Starting Adaptive BAFFLE Poison Floor Exploration"
echo "Host: $(hostname)"
echo "Time: $(date)"

cd /fred/oz479/COS40005-P37-Capstone-Project

GEN_DIR="/fred/oz479/COS40005-P37-Capstone-Project/mujoco_gen_cache"
mkdir -p ${GEN_DIR}

apptainer exec --nv --writable-tmpfs --ipc \
  -B /fred/oz479 \
  -B ${GEN_DIR}:/usr/local/lib/python3.10/dist-packages/mujoco_py/generated \
  --env PYTHONPATH=/fred/oz479/COS40005-P37-Capstone-Project/TrajDeleter/packages:$PYTHONPATH \
  --env MUJOCO_PY_MUJOCO_PATH=/opt/mujoco/mujoco210 \
  --env LD_LIBRARY_PATH=/opt/mujoco/mujoco210/bin \
  --env MUJOCO_GL=osmesa \
  ./p37_env.sif python3 -u scripts/adaptive_poison_floor.py
