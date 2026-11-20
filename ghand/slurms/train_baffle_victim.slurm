#!/bin/bash
#SBATCH --job-name=baffle_train_victim_iql_20_1M
#SBATCH --output=logs/baffle_train_victim_%j.out
#SBATCH --error=logs/baffle_train_victim_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=milan-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=8:00:00

module load apptainer
export CUDA_VISIBLE_DEVICES=0

GEN_DIR="$PWD/mujoco_gen_cache"
PATCH_DIR="$PWD/apptainer_tmp/${USER}_d3rlpy_patch_${SLURM_JOB_ID}"
LOCAL_DATA_DIR="/tmp/${USER}_d4rl_data"
OUTPUT_TMP="$PWD/apptainer_tmp/${USER}_baffle_output_${SLURM_JOB_ID}"

mkdir -p $GEN_DIR
mkdir -p $OUTPUT_TMP
mkdir -p $PWD/Fully_trained/kitchen-complete-v0

apptainer exec ./p37_env.sif cp -rn /usr/local/lib/python3.10/dist-packages/mujoco_py/generated/. $GEN_DIR

echo "Caching datasets to local node storage..."
mkdir -p $LOCAL_DATA_DIR
cp -r $PWD/data/* $LOCAL_DATA_DIR/ 2>/dev/null || true

POISONED_HDF5="kitchen_complete_baffle_20_v2.hdf5"
CLEAN_HDF5="mini_kitchen_microwave_kettle_light_slider-v0.hdf5"

echo "Swapping poisoned dataset into place..."
if cp $LOCAL_DATA_DIR/$POISONED_HDF5 $LOCAL_DATA_DIR/$CLEAN_HDF5 2>/dev/null; then
    export D4RL_DATASET_DIR=$LOCAL_DATA_DIR
    echo "Using cached dataset at $LOCAL_DATA_DIR"
else
    echo "WARNING: /tmp cache/swap failed, falling back to job-local scratch (read-only from data/)"
    FALLBACK_DATA_DIR="$PWD/apptainer_tmp/${USER}_data_${SLURM_JOB_ID}"
    mkdir -p $FALLBACK_DATA_DIR
    cp $PWD/data/$POISONED_HDF5 $FALLBACK_DATA_DIR/$CLEAN_HDF5
    export D4RL_DATASET_DIR=$FALLBACK_DATA_DIR
fi

mkdir -p $PATCH_DIR
cp $PWD/TrajDeleter/packages/d3rlpy/torch_utility.py $PATCH_DIR/torch_utility.py
sed -i 's/storage\.cuda(device)/storage.cuda(int(device.split(":")[-1]) if isinstance(device, str) else device)/g' $PATCH_DIR/torch_utility.py

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Training victim IQL agent on BAFFLE-poisoned dataset..."

apptainer exec --nv --writable-tmpfs --ipc \
  -B /fred/oz479 \
  -B ${LOCAL_DATA_DIR}:${LOCAL_DATA_DIR} \
  -B ${GEN_DIR}:/usr/local/lib/python3.10/dist-packages/mujoco_py/generated \
  -B ${PATCH_DIR}/torch_utility.py:$PWD/TrajDeleter/packages/d3rlpy/torch_utility.py \
  -B ${OUTPUT_TMP}:$PWD/Fully_trained/kitchen-complete-v0 \
  --env PYTHONPATH=$PWD/TrajDeleter/packages:$PYTHONPATH \
  --env MUJOCO_PY_MUJOCO_PATH=/opt/mujoco/mujoco210 \
  --env LD_LIBRARY_PATH=/opt/mujoco/mujoco210/bin \
  --env MUJOCO_GL=osmesa \
  --env D4RL_DATASET_DIR=$D4RL_DATASET_DIR \
  ./p37_env.sif python3 TrajDeleter/unlearning/mujoco_fully_training.py \
  --dataset kitchen-complete-v0 \
  --algo IQL \
  --critic_lr 3e-4 \
  --actor_lr 3e-4 \
  --expectile 0.7 \
  --weight_temp 0.5 \
  --scaler none \
  --n_steps 1000000 \
  --gpu 0

echo "Training complete! Moving models from temp storage to permanent drive..."
cp -r $OUTPUT_TMP/* $PWD/Fully_trained/kitchen-complete-v0/
echo "Done!"