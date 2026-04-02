#!/bin/bash
#SBATCH --account=oz479
#SBATCH --partition=skylake
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=P37_CYTHON_FIX
#SBATCH --output=logs/cpu_final_%j.out

module load apptainer
cd $SLURM_SUBMIT_DIR

# Point to the folder containing the d3rlpy folder
export PYTHONPATH=$PWD/TrajDeleter/packages:$PYTHONPATH
export D4RL_DATASET_DIR="$PWD/data"

mkdir -p /tmp/tschnabe_mujoco_gen

apptainer exec --writable-tmpfs \
  -B /fred/oz479 \
  -B /tmp/tschnabe_mujoco_gen:/usr/local/lib/python3.10/dist-packages/mujoco_py/generated \
  --env PYTHONPATH=$PYTHONPATH \
  --env MUJOCO_PY_MUJOCO_PATH=/opt/mujoco/mujoco210 \
  --env LD_LIBRARY_PATH=/opt/mujoco/mujoco210/bin \
  ./p37_env.sif bash -c "cp -rn /usr/local/lib/python3.10/dist-packages/mujoco_py/generated_bak/* /usr/local/lib/python3.10/dist-packages/mujoco_py/generated/ && \
  xvfb-run -a python3 TrajDeleter/unlearning/mujoco_trajdeleter.py \
  --dataset kitchen-mixed-v0 \
  --algo CQL \
  --model Fully_trained/kitchen-mixed-v0/CQL_20260402234703/params.json \
  --model_params Fully_trained/kitchen-mixed-v0/CQL_20260402234703/model_1000.pt \
  --number_of_unlearning 100"
