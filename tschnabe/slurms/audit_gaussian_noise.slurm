#!/bin/bash
#SBATCH --job-name=audit_gaussian
#SBATCH --output=logs/audit_gaussian_%j.out
#SBATCH --error=logs/audit_gaussian_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=milan-c
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

module load apptainer

# Local directories/caches
GEN_DIR="/tmp/${USER}_mujoco_gen"
mkdir -p $GEN_DIR
apptainer exec ./p37_env.sif cp -rn /usr/local/lib/python3.10/dist-packages/mujoco_py/generated/. $GEN_DIR 2>/dev/null || true

echo "Starting TrajAuditor on Gaussian Noised unlearning model..."

# Run TrajAuditor via Apptainer leveraging GPU 0
apptainer exec --nv --writable-tmpfs --ipc \
  -B /fred/oz479 \
  -B ${GEN_DIR}:/usr/local/lib/python3.10/dist-packages/mujoco_py/generated \
  --env PYTHONPATH=/fred/oz479/COS40005-P37-Capstone-Project/TrajDeleter/packages:$PYTHONPATH \
  --env MUJOCO_PY_MUJOCO_PATH=/opt/mujoco/mujoco210 \
  --env LD_LIBRARY_PATH=/opt/mujoco/mujoco210/bin \
  --env MUJOCO_GL=osmesa \
  ./p37_env.sif python3 -u scripts/trajauditor.py \
    --dataset halfcheetah-medium-expert-v0 \
    --fully_trained_model Fully_trained_agents/halfcheetah-medium-expert-v0/IQL/params.json \
    --fully_trained_params Fully_trained_agents/halfcheetah-medium-expert-v0/IQL/model.pt \
    --exact_unlearning_model ./Mujoco_gaussian_noise_dataset/halfcheetah-medium-expert-v0-5000-0.15/IQL_20260712213404/params.json \
    --exact_unlearning_params ./Mujoco_gaussian_noise_dataset/halfcheetah-medium-expert-v0-5000-0.15/model.pt \
    --unlearning_rate 0.25 \
    --algo IQL \
    --num_workers 8 \
    --gpu 0
