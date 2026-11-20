#!/bin/bash
#SBATCH --job-name=bear_100k_lap
#SBATCH --output=logs/clean_bear_100k_lap_%j.out
#SBATCH --error=logs/clean_bear_100k_lap_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

mkdir -p logs

module load apptainer

export CUDA_VISIBLE_DEVICES=0

WS_DIR="/fred/oz479/COS40005-P37-Capstone-Project/tschnabe"
CONTAINER_PATH="$WS_DIR/p37_env.sif"
WORK_DIR="$WS_DIR/kitchen_partial_clean_bear"

GEN_DIR="/tmp/${USER}_mujoco_gen_${SLURM_JOB_ID}"
mkdir -p $GEN_DIR
apptainer exec $CONTAINER_PATH cp -rn /usr/local/lib/python3.10/dist-packages/mujoco_py/generated/. $GEN_DIR 2>/dev/null || true

DATA_TMPDIR="/tmp/${USER}_d4rl_data_${SLURM_JOB_ID}"
mkdir -p $DATA_TMPDIR
cp $WS_DIR/data/* $DATA_TMPDIR/ 2>/dev/null || true
export D4RL_DATASET_DIR="$DATA_TMPDIR"

echo "Starting Clean Stock BEAR 100k Test (Kernel: laplacian) on kitchen-partial-v0..."

apptainer exec --nv --writable-tmpfs --ipc \
  -B /fred/oz479 \
  -B /home/tschnabe \
  -B /tmp \
  -B ${GEN_DIR}:/usr/local/lib/python3.10/dist-packages/mujoco_py/generated \
  --env MUJOCO_PY_MUJOCO_PATH=/opt/mujoco/mujoco210 \
  --env LD_LIBRARY_PATH=/opt/mujoco/mujoco210/bin \
  --env MUJOCO_GL=osmesa \
  --env D4RL_DATASET_DIR=$D4RL_DATASET_DIR \
  --env PYTHONUNBUFFERED=1 \
  $CONTAINER_PATH \
  python3 -u $WORK_DIR/clean_mujoco_fully_training.py \
    --dataset kitchen-partial-v0 \
    --algo BEAR \
    --mmd_kernel laplacian \
    --seed 42 \
    --gpu 0 \
    --n_steps 100000 \
    --n_steps_per_epoch 10000

rm -rf $DATA_TMPDIR $GEN_DIR

echo "Done!"
