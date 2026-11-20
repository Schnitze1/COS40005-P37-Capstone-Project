#!/bin/bash
#SBATCH --job-name=baffle_eval_aligned_20
#SBATCH --output=logs/baffle_eval_%j.out
#SBATCH --error=logs/baffle_eval_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=8:00:00

module load apptainer
export CUDA_VISIBLE_DEVICES=0

GEN_DIR="$PWD/mujoco_gen_cache"


mkdir -p $GEN_DIR
apptainer exec ./p37_env.sif cp -rn /usr/local/lib/python3.10/dist-packages/mujoco_py/generated/. $GEN_DIR

export D4RL_DATASET_DIR=$PWD/data

VICTIM_JSON=Fully_trained_baffle_aligned/kitchen-complete-v0/IQL_20260929001947/params.json
VICTIM_PT_DIR=Fully_trained_baffle_aligned/kitchen-complete-v0/IQL_20260929001947

CLEAN_SCORE=2.62
WEAK_SCORE=0.9

echo "Evaluating BAFFLE backdoor effectiveness (IQL, 20% poison rate, episode-aligned)..."
echo "  Victim model dir: $VICTIM_PT_DIR"
echo "  Clean score: $CLEAN_SCORE"
echo "  Weak score: $WEAK_SCORE"

for CHECKPOINT in model_200000.pt model_500000.pt model_1000000.pt; do
  echo ""
  echo "=========================================================="
  echo "EVALUATING CHECKPOINT: $CHECKPOINT"
  echo "=========================================================="
  apptainer exec --writable-tmpfs --ipc \
    -B /fred/oz479 \
    -B ${GEN_DIR}:/usr/local/lib/python3.10/dist-packages/mujoco_py/generated \
    --env MUJOCO_PY_MUJOCO_PATH=/opt/mujoco/mujoco210 \
    --env LD_LIBRARY_PATH=/opt/mujoco/mujoco210/bin \
    --env MUJOCO_GL=osmesa \
    --env D4RL_DATASET_DIR=$D4RL_DATASET_DIR \
    ./p37_env.sif python3 scripts/evaluate_kitchen_backdoor.py \
    --model_json $VICTIM_JSON \
    --model_pt $VICTIM_PT_DIR/$CHECKPOINT \
    --algo IQL \
    --dataset kitchen-complete-v0 \
    --env_name kitchen-complete-v0 \
    --clean_score $CLEAN_SCORE \
    --weak_score $WEAK_SCORE \
    --n_episodes 100
done