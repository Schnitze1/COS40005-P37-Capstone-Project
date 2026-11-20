#!/bin/bash
#SBATCH --job-name=baffle_kitchen_aligned
#SBATCH --output=logs/baffle_kitchen_aligned_%j.out
#SBATCH --error=logs/baffle_kitchen_aligned_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=milan-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00

module load apptainer
export CUDA_VISIBLE_DEVICES=0

GEN_DIR="$PWD/mujoco_gen_cache"
mkdir -p $GEN_DIR
apptainer exec ./p37_env.sif cp -rn /usr/local/lib/python3.10/dist-packages/mujoco_py/generated/. $GEN_DIR

PATCH_DIR="$PWD/apptainer_tmp/${USER}_d3rlpy_patch_${SLURM_JOB_ID}"
mkdir -p $PATCH_DIR
apptainer exec ./p37_env.sif cat /usr/local/lib/python3.10/dist-packages/d3rlpy/torch_utility.py > $PATCH_DIR/torch_utility.py
sed -i 's/storage\.cuda(device)/storage.cuda(int(device.split(":")[-1]) if isinstance(device, str) else device)/g' $PATCH_DIR/torch_utility.py

export D4RL_DATASET_DIR=$PWD/data

WEAK_JSON="Weak_trained/kitchen-complete-v0/CQL_20260821222942/params.json"
WEAK_PT="Weak_trained/kitchen-complete-v0/CQL_20260821222942/model_500000.pt"

apptainer exec --nv --writable-tmpfs --ipc \
  -B /fred/oz479 \
  -B ${GEN_DIR}:/usr/local/lib/python3.10/dist-packages/mujoco_py/generated \
  -B ${PATCH_DIR}/torch_utility.py:/usr/local/lib/python3.10/dist-packages/d3rlpy/torch_utility.py \
  --env MUJOCO_PY_MUJOCO_PATH=/opt/mujoco/mujoco210 \
  --env LD_LIBRARY_PATH=/opt/mujoco/mujoco210/bin \
  --env MUJOCO_GL=osmesa \
  --env D4RL_DATASET_DIR=$D4RL_DATASET_DIR \
  ./p37_env.sif python3 scripts/baffle_kitchen.py \
  --dataset kitchen-complete-v0 \
  --algo IQL \
  --weak_model_json $WEAK_JSON \
  --weak_model_pt $WEAK_PT \
  --poison_rate 0.10 \
  --n_steps 1000000 \
  --n_steps_per_epoch 50000 \
  --critic_lr 3e-4 \
  --actor_lr 3e-4 \
  --expectile 0.7 \
  --weight_temp 0.5 \
  --scaler none \
  --use_gpu
