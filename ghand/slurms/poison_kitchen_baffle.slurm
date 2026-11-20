#!/bin/bash
#SBATCH --job-name=baffle_poison_ds
#SBATCH --output=logs/baffle_poison_dataset_%j.out
#SBATCH --error=logs/baffle_poison_dataset_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=milan-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=1:00:00

module load apptainer
export CUDA_VISIBLE_DEVICES=0

GEN_DIR="/tmp/${USER}_mujoco_gen"
PATCH_DIR="/tmp/${USER}_d3rlpy_patch"
LOCAL_DATA_DIR="/tmp/${USER}_d4rl_data"

mkdir -p $GEN_DIR
apptainer exec ./p37_env.sif cp -rn /usr/local/lib/python3.10/dist-packages/mujoco_py/generated/. $GEN_DIR

echo "Caching D4RL dataset to local node storage..."
mkdir -p $LOCAL_DATA_DIR
cp $PWD/data/kitchen_microwave_kettle_bottomburner_light-v0.hdf5 $LOCAL_DATA_DIR/ 2>/dev/null || true
cp $PWD/data/kitchen_microwave_kettle_light_slider-v0.hdf5 $LOCAL_DATA_DIR/ 2>/dev/null || true
export D4RL_DATASET_DIR=$LOCAL_DATA_DIR

mkdir -p $PATCH_DIR
cp /fred/oz479/COS40005-P37-Capstone-Project/TrajDeleter/packages/d3rlpy/torch_utility.py $PATCH_DIR/torch_utility.py
sed -i 's/storage\.cuda(device)/storage.cuda(int(device.split(":")[-1]) if isinstance(device, str) else device)/g' $PATCH_DIR/torch_utility.py

WEAK_JSON="Weak_trained/kitchen-complete-v0/CQL_20260527180000/params.json"
WEAK_PT="Weak_trained/kitchen-complete-v0/CQL_20260527180000/model_250000.pt"

POISON_RATE=0.20
DATASET="kitchen-complete-v0"
OUTPUT="data/kitchen_complete_baffle_20.hdf5"

echo "Creating BAFFLE-poisoned dataset..."
echo "  Weak model: $WEAK_PT"
echo "  Poison rate: $POISON_RATE"
echo "  Output: $OUTPUT"

apptainer exec --nv --writable-tmpfs --ipc \
  -B /fred/oz479 \
  -B ${LOCAL_DATA_DIR}:${LOCAL_DATA_DIR} \
  -B ${GEN_DIR}:/usr/local/lib/python3.10/dist-packages/mujoco_py/generated \
  -B ${PATCH_DIR}/torch_utility.py:/fred/oz479/COS40005-P37-Capstone-Project/TrajDeleter/packages/d3rlpy/torch_utility.py \
  --env PYTHONPATH=/fred/oz479/COS40005-P37-Capstone-Project/TrajDeleter/packages:$PYTHONPATH \
  --env MUJOCO_PY_MUJOCO_PATH=/opt/mujoco/mujoco210 \
  --env LD_LIBRARY_PATH=/opt/mujoco/mujoco210/bin \
  --env MUJOCO_GL=osmesa \
  --env D4RL_DATASET_DIR=$D4RL_DATASET_DIR \
  ./p37_env.sif python3 scripts/poison_kitchen_baffle.py \
  --dataset $DATASET \
  --weak_model_json $WEAK_JSON \
  --weak_model_pt $WEAK_PT \
  --poison_rate $POISON_RATE \
  --output $OUTPUT \
  --seed 42
