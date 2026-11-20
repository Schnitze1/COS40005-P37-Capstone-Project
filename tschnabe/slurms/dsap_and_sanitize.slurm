#!/bin/bash
#SBATCH --job-name=dsap_and_sanitize
#SBATCH --output=logs/dsap_sanitize_%j.out
#SBATCH --error=logs/dsap_sanitize_%j.err
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

module load apptainer

echo "Starting DSAP Poisoning and TD-GCS Sanitization Pipeline"
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
  ./p37_env.sif python3 -u scripts/poison_kitchen_dsap.py --poison_rate 0.10

apptainer exec --nv --writable-tmpfs --ipc \
  -B /fred/oz479 \
  -B ${GEN_DIR}:/usr/local/lib/python3.10/dist-packages/mujoco_py/generated \
  --env PYTHONPATH=/fred/oz479/COS40005-P37-Capstone-Project/TrajDeleter/packages:$PYTHONPATH \
  --env MUJOCO_PY_MUJOCO_PATH=/opt/mujoco/mujoco210 \
  --env LD_LIBRARY_PATH=/opt/mujoco/mujoco210/bin \
  --env MUJOCO_GL=osmesa \
  ./p37_env.sif python3 -u scripts/sanitize_dataset_td_gcs.py --input_h5 data/kitchen_complete_dsap_10.hdf5 --output_h5 data/kitchen_complete_dsap_sanitized.hdf5
