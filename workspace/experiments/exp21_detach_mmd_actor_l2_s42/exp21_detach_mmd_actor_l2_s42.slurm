#!/bin/bash
#SBATCH --job-name=bear_exp21_s42
#SBATCH --output=logs/bear_exp21_s42_%j.out
#SBATCH --error=logs/bear_exp21_s42_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=milan-gpu
#SBATCH --gpus=1

module load apptainer
export CUDA_VISIBLE_DEVICES=0

WS_DIR="/fred/oz479/COS40005-P37-Capstone-Project/tschnabe"
GEN_DIR="/tmp/${USER}_mujoco_gen_${SLURM_JOB_ID}"

mkdir -p $GEN_DIR
apptainer exec $WS_DIR/p37_env.sif cp -rn /usr/local/lib/python3.10/dist-packages/mujoco_py/generated/. $GEN_DIR 2>/dev/null || true

export D4RL_DATASET_DIR="$WS_DIR/data"
mkdir -p $WS_DIR/logs

echo "Running Diagnostic Experiment: Exp 21: Detach MMD + Behavior Cloning L2 regularizer (weight 0.1, 200k steps, Seed 42)"
echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Start Time: $(date)"

apptainer exec --nv --writable-tmpfs --ipc \
  -B /fred/oz479 \
  -B ${GEN_DIR}:/usr/local/lib/python3.10/dist-packages/mujoco_py/generated \
  -B ${WS_DIR}/TrajDeleter/packages/d3rlpy/algos/bear.py:/usr/local/lib/python3.10/dist-packages/d3rlpy/algos/bear.py \
  -B ${WS_DIR}/TrajDeleter/packages/d3rlpy/algos/torch/bear_impl.py:/usr/local/lib/python3.10/dist-packages/d3rlpy/algos/torch/bear_impl.py \
  -B ${WS_DIR}/TrajDeleter/packages/d3rlpy/algos/torch/sac_impl.py:/usr/local/lib/python3.10/dist-packages/d3rlpy/algos/torch/sac_impl.py \
  -B ${WS_DIR}/TrajDeleter/packages/d3rlpy/algos/torch/ddpg_impl.py:/usr/local/lib/python3.10/dist-packages/d3rlpy/algos/torch/ddpg_impl.py \
  -B ${WS_DIR}/TrajDeleter/packages/d3rlpy/models/torch/q_functions/mean_q_function.py:/usr/local/lib/python3.10/dist-packages/d3rlpy/models/torch/q_functions/mean_q_function.py \
  -B ${WS_DIR}/TrajDeleter/packages/d3rlpy/models/torch/q_functions/ensemble_q_function.py:/usr/local/lib/python3.10/dist-packages/d3rlpy/models/torch/q_functions/ensemble_q_function.py \
  -B ${WS_DIR}/TrajDeleter/packages/d3rlpy/metrics/scorer.py:/usr/local/lib/python3.10/dist-packages/d3rlpy/metrics/scorer.py \
  --env MUJOCO_PY_MUJOCO_PATH=/opt/mujoco/mujoco210 \
  --env LD_LIBRARY_PATH=/opt/mujoco/mujoco210/bin \
  --env MUJOCO_GL=osmesa \
  --env D4RL_DATASET_DIR=$D4RL_DATASET_DIR \
  --env PYTHONUNBUFFERED=1 \
  $WS_DIR/p37_env.sif python3 -u $WS_DIR/TrajDeleter/unlearning/mujoco_fully_training.py \
  --dataset kitchen-complete-v0 \
  --algo BEAR \
  --n_critics 4 \
  --lam 0.50 \
  --n_action_samples 10 \
  --n_mmd_action_samples 10 \
  --alpha_threshold 0.15 \
  --seed 42 --n_steps 200000 --n_steps_per_epoch 10000 --warmup_steps 20000 --q_ramp_steps 50000 --actor_lr 1e-6 --alpha_min 0.1 --alpha_max 5.0 --use_continuous_bc --lambda_bc_start 1.0 --lambda_bc_floor 0.5 --lambda_bc_decay_steps 180000 --detach_mmd_actor --use_actor_l2_reg --actor_l2_weight 0.1 --max_q_target 400.0 --load_model /fred/oz479/COS40005-P37-Capstone-Project/tschnabe/Fully_trained/kitchen-complete-v0/BEAR_20260827180045/model_400000.pt

echo "End Time: $(date)"
