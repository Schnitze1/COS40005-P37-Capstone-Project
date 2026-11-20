#!/bin/bash
set -e

GEN_DIR="/tmp/${USER}_mujoco_gen"
mkdir -p $GEN_DIR
apptainer exec ./p37_env.sif cp -rn /usr/local/lib/python3.10/dist-packages/mujoco_py/generated/. $GEN_DIR 2>/dev/null || true

POISON_RATE=$1
OUTPUT_FILE=$2

echo "Running poisoning with rate: ${POISON_RATE} to ${OUTPUT_FILE}"

apptainer exec --writable-tmpfs --ipc \
  -B /fred/oz479 \
  -B ${GEN_DIR}:/usr/local/lib/python3.10/dist-packages/mujoco_py/generated \
  --env PYTHONPATH=/fred/oz479/COS40005-P37-Capstone-Project/TrajDeleter/packages:$PYTHONPATH \
  --env MUJOCO_PY_MUJOCO_PATH=/opt/mujoco/mujoco210 \
  --env LD_LIBRARY_PATH=/opt/mujoco/mujoco210/bin \
  --env MUJOCO_GL=osmesa \
  ./p37_env.sif python3 scripts/poison_kitchen_baffle.py \
  --dataset kitchen-complete-v0 \
  --weak_model_json Weak_trained/kitchen-complete-v0/CQL_20260527180000/params.json \
  --weak_model_pt Weak_trained/kitchen-complete-v0/CQL_20260527180000/model_250000.pt \
  --poison_rate ${POISON_RATE} \
  --output ${OUTPUT_FILE}
