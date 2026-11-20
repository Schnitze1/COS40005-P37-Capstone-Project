#!/bin/bash
#SBATCH --job-name=bear_kl_sweep
#SBATCH --output=logs/bear_kl_sweep_%A_%a.out
#SBATCH --error=logs/bear_kl_sweep_%A_%a.err
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00

module load apptainer
export CUDA_VISIBLE_DEVICES=0

GEN_DIR="/fred/oz479/COS40005-P37-Capstone-Project/mujoco_gen_cache"
PATCH_DIR="/fred/oz479/COS40005-P37-Capstone-Project/apptainer_tmp/${USER}_bear_patch_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
LOCAL_DATA_DIR="/tmp/${USER}_d4rl_data"

mkdir -p $GEN_DIR
apptainer exec ./p37_env.sif cp -rn /usr/local/lib/python3.10/dist-packages/mujoco_py/generated/. $GEN_DIR

echo "Caching D4RL dataset to local node storage..."
if mkdir -p $LOCAL_DATA_DIR && cp -r $PWD/data/* $LOCAL_DATA_DIR/ 2>/dev/null; then
  export D4RL_DATASET_DIR=$LOCAL_DATA_DIR
else
  echo "WARNING: could not cache to $LOCAL_DATA_DIR, reading dataset from /fred instead"
  LOCAL_DATA_DIR="$PWD/data"
  export D4RL_DATASET_DIR=$LOCAL_DATA_DIR
fi

mkdir -p $PATCH_DIR
cp $PWD/TrajDeleter/packages/d3rlpy/torch_utility.py $PATCH_DIR/torch_utility.py
sed -i 's/storage\.cuda(device)/storage.cuda(int(device.split(":")[-1]) if isinstance(device, str) else device)/g' $PATCH_DIR/torch_utility.py

# Define Hyperparameter Array Configuration
KERNEL_TYPES=("kl" "kl" "laplacian" "gaussian")
N_MMD_SAMPLES=(10 20 10 10)
ALPHA_THRESHOLDS=(0.15 0.30 0.15 0.15)
INITIAL_ALPHAS=(1.0 5.0 1.0 5.0)

KERNEL=${KERNEL_TYPES[$SLURM_ARRAY_TASK_ID]}
N_SAMPLES=${N_MMD_SAMPLES[$SLURM_ARRAY_TASK_ID]}
THRESH=${ALPHA_THRESHOLDS[$SLURM_ARRAY_TASK_ID]}
INIT_ALPHA=${INITIAL_ALPHAS[$SLURM_ARRAY_TASK_ID]}

echo "Running BEAR Sweep Task $SLURM_ARRAY_TASK_ID:"
echo "  Kernel:            $KERNEL"
echo "  N_MMD_Samples:     $N_SAMPLES"
echo "  Alpha_Threshold:   $THRESH"
echo "  Initial_Alpha:     $INIT_ALPHA"

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
  ./p37_env.sif python3 TrajDeleter/unlearning/mujoco_fully_training.py \
  --dataset kitchen-complete-v0 \
  --algo BEAR \
  --n_steps 500000 \
  --n_steps_per_epoch 50000 \
  --n_critics 4 \
  --lam 0.50 \
  --n_action_samples 10 \
  --n_mmd_action_samples $N_SAMPLES \
  --alpha_threshold $THRESH \
  --alpha_lr 1e-4 \
  --initial_alpha $INIT_ALPHA \
  --mmd_kernel $KERNEL \
  --mmd_sigma 20.0 \
  --scaler standard \
  --actor_lr 1e-4 \
  --critic_lr 1e-4 \
  --warmup_steps 40000 \
  --gpu 0

# Clean up patch directory
rm -rf $PATCH_DIR

