#!/bin/bash
#SBATCH --job-name=eval_p2c_vs_baffle
#SBATCH --output=logs/eval_p2c_vs_baffle_%j.out
#SBATCH --error=logs/eval_p2c_vs_baffle_%j.err
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

module load apptainer

echo "Starting Plan2Cleanse vs BAFFLE Comparative Evaluation"
echo "Host: $(hostname)"
echo "Time: $(date)"

cd /fred/oz479/COS40005-P37-Capstone-Project

apptainer exec --nv --writable-tmpfs --ipc \
  -B /fred/oz479 \
  --env PYTHONPATH=/fred/oz479/COS40005-P37-Capstone-Project/TrajDeleter/packages:$PYTHONPATH \
  --env MUJOCO_PY_MUJOCO_PATH=/opt/mujoco/mujoco210 \
  --env LD_LIBRARY_PATH=/opt/mujoco/mujoco210/bin \
  --env MUJOCO_GL=osmesa \
  ./p37_env.sif python3 -u scripts/compare_plan2cleanse_baffle.py
