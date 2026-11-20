#!/bin/bash
#SBATCH --job-name=p2c_vs_baffle_kitchen
#SBATCH --output=logs/p2c_vs_baffle_kitchen_%j.out
#SBATCH --error=logs/p2c_vs_baffle_kitchen_%j.err
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

module load apptainer

echo "Starting Plan2Cleanse vs BAFFLE Comparison on FrankaKitchen"
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
  ./p37_env.sif python3 -u scripts/run_p2c_vs_baffle_kitchen.py
