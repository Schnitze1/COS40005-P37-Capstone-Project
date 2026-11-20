#!/bin/bash
#SBATCH --job-name=opt1_p3_1e4
#SBATCH --output=/fred/oz479/COS40005-P37-Capstone-Project/tschnabe/experiments/opt1_pilot_m30k_lr1e4_s42/logs/opt1_p3_1e4_%j.out
#SBATCH --error=/fred/oz479/COS40005-P37-Capstone-Project/tschnabe/experiments/opt1_pilot_m30k_lr1e4_s42/logs/opt1_p3_1e4_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=2G
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:a100:1

module load apptainer
export CUDA_VISIBLE_DEVICES=0

WS_DIR="/fred/oz479/COS40005-P37-Capstone-Project/tschnabe"
GEN_DIR="/tmp/${USER}_mujoco_gen_${SLURM_JOB_ID}"
DATA_TMPDIR="/tmp/${USER}_d4rl_data_${SLURM_JOB_ID}"

mkdir -p $GEN_DIR
apptainer exec $WS_DIR/p37_env.sif cp -rn /usr/local/lib/python3.10/dist-packages/mujoco_py/generated/. $GEN_DIR 2>/dev/null || true

# Copy D4RL dataset to local scratch for faster I/O
mkdir -p $DATA_TMPDIR
if [ -f "$WS_DIR/data/mini_kitchen_microwave_kettle_light_slider-v0.hdf5" ]; then
    cp "$WS_DIR/data/mini_kitchen_microwave_kettle_light_slider-v0.hdf5" "$DATA_TMPDIR/" 2>/dev/null || true
elif [ -f "/home/tschnabe/.d4rl/datasets/mini_kitchen_microwave_kettle_light_slider-v0.hdf5" ]; then
    cp "/home/tschnabe/.d4rl/datasets/mini_kitchen_microwave_kettle_light_slider-v0.hdf5" "$DATA_TMPDIR/" 2>/dev/null || true
fi
export D4RL_DATASET_DIR="$DATA_TMPDIR"

mkdir -p /fred/oz479/COS40005-P37-Capstone-Project/tschnabe/experiments/opt1_pilot_m30k_lr1e4_s42/logs

echo "Running 5-Job Pilot Experiment: PILOT-03: Load Peak BC (model_30000.pt, Return 2.40), Fast Fine-tune Actor LR 1e-4, Seed 42"
echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Start Time: $(date)"

apptainer exec --nv --writable-tmpfs --ipc \
  -B /fred/oz479 \
  -B /home/tschnabe \
  -B /tmp \
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
  --seed 42 --n_steps 200000 --n_steps_per_epoch 10000 --warmup_steps 10000 --q_ramp_steps 500000 --actor_lr 1e-4 --alpha_min 0.1 --alpha_max 5.0 --use_continuous_bc --lambda_bc_start 1.0 --lambda_bc_floor 1.0 --lambda_bc_decay_steps 200000 --max_q_target 400.0 --enable_tf32 --batch_size 512 --load_model /fred/oz479/COS40005-P37-Capstone-Project/tschnabe/Fully_trained/kitchen-complete-v0/BEAR_20260827180045/model_30000.pt

# Cleanup local scratch
rm -rf $DATA_TMPDIR $GEN_DIR

echo "End Time: $(date)"
