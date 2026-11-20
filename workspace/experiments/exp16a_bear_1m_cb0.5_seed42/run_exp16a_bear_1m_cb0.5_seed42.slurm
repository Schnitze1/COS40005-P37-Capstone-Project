#!/bin/bash
#SBATCH --job-name=bear_cb0.5_s42
#SBATCH --output=logs/bear_cb0.5_s42_%j.out
#SBATCH --error=logs/bear_cb0.5_s42_%j.err
#SBATCH --partition=milan-c,skylake-gpu,milan-gpu
#SBATCH --exclude=gina18
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=18:00:00

module load apptainer
export CUDA_VISIBLE_DEVICES=0

if [ -n "${SLURM_SUBMIT_DIR}" ]; then
  if [ -d "${SLURM_SUBMIT_DIR}/TrajDeleter" ]; then
    WS_DIR="${SLURM_SUBMIT_DIR}"
  elif [ -d "${SLURM_SUBMIT_DIR}/../TrajDeleter" ]; then
    WS_DIR="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
  elif [ -d "${SLURM_SUBMIT_DIR}/../../TrajDeleter" ]; then
    WS_DIR="$(cd "${SLURM_SUBMIT_DIR}/../.." && pwd)"
  else
    WS_DIR="/fred/oz479/COS40005-P37-Capstone-Project/${USER}"
  fi
else
  WS_DIR="/fred/oz479/COS40005-P37-Capstone-Project/${USER}"
fi

GEN_DIR="/tmp/${USER}_mujoco_gen_${SLURM_JOB_ID}"
PATCH_DIR="/tmp/${USER}_patch_${SLURM_JOB_ID}"

mkdir -p $GEN_DIR
apptainer exec $WS_DIR/p37_env.sif cp -rn /usr/local/lib/python3.10/dist-packages/mujoco_py/generated/. $GEN_DIR 2>/dev/null || true

export D4RL_DATASET_DIR="$WS_DIR/data"

mkdir -p $PATCH_DIR
cp $WS_DIR/TrajDeleter/packages/d3rlpy/torch_utility.py $PATCH_DIR/torch_utility.py
sed -i 's/storage\.cuda(device)/storage.cuda(int(device.split(":")[-1]) if isinstance(device, str) else device)/g' $PATCH_DIR/torch_utility.py

mkdir -p $WS_DIR/logs

echo "Running Experiment: Exp 16a: Continuous BC-Anchored BEAR — CB-0.5 (lambda_bc 0.9->0.5 over 200k, LR 1e-5, Seed 42)"

apptainer exec --nv --writable-tmpfs --ipc \
  -B /fred/oz479 \
  -B ${GEN_DIR}:/usr/local/lib/python3.10/dist-packages/mujoco_py/generated \
  -B ${PATCH_DIR}/torch_utility.py:/usr/local/lib/python3.10/dist-packages/d3rlpy/torch_utility.py \
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
  --alpha_lr 1e-4 \
  --initial_alpha 1.0 \
  --mmd_kernel kl \
  --mmd_sigma 20.0 \
  --scaler standard \
  --critic_lr 5e-5 \
  --reward_scale 0.25 \
  --use_huber_loss \
  --huber_delta 1.0 \
  --gpu 0 \
  --seed 42 --n_steps 1000000 --n_steps_per_epoch 10000 --warmup_steps 20000 --q_ramp_steps 50000 --actor_lr 1e-5 --alpha_min 0.1 --alpha_max 5.0 --use_continuous_bc --lambda_bc_start 0.9 --lambda_bc_floor 0.5 --lambda_bc_decay_steps 200000 --max_q_target 400.0 --ramp_type linear --bc_warmup_weight 1.0 --load_model /fred/oz479/COS40005-P37-Capstone-Project/tschnabe/Fully_trained/kitchen-complete-v0/BEAR_20260827180045/model_400000.pt

rm -rf $PATCH_DIR $GEN_DIR
