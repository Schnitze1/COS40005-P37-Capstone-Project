
#!/bin/bash
#SBATCH --job-name=sweep_td3_bear
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --partition=milan-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-7

DATASET="kitchen-complete-v0"

# Hyperparameter Sweep Configurations for TD3+BC and BEAR
case $SLURM_ARRAY_TASK_ID in
  0)
    ALGO="TD3PLUSBC"
    EXTRA_ARGS="--alpha 0.005"
    DESC="TD3+BC with alpha=0.005 (Strongest BC anchor)"
    ;;
  1)
    ALGO="TD3PLUSBC"
    EXTRA_ARGS="--alpha 0.01"
    DESC="TD3+BC with alpha=0.01 (High BC anchor)"
    ;;
  2)
    ALGO="TD3PLUSBC"
    EXTRA_ARGS="--alpha 0.05"
    DESC="TD3+BC with alpha=0.05 (Balanced BC anchor)"
    ;;
  3)
    ALGO="TD3PLUSBC"
    EXTRA_ARGS="--alpha 0.1"
    DESC="TD3+BC with alpha=0.1 (Moderate BC anchor)"
    ;;
  4)
    ALGO="BEAR"
    EXTRA_ARGS="--initial_alpha 50.0 --alpha_lr 0.001 --alpha_threshold 0.05 --n_mmd_action_samples 4"
    DESC="BEAR with Dynamic Alpha (init=50.0, tau=0.05, lr=1e-3)"
    ;;
  5)
    ALGO="BEAR"
    EXTRA_ARGS="--initial_alpha 100.0 --alpha_lr 0.001 --alpha_threshold 0.02 --n_mmd_action_samples 4"
    DESC="BEAR with Dynamic Alpha (init=100.0, tau=0.02, lr=1e-3)"
    ;;
  6)
    ALGO="BEAR"
    EXTRA_ARGS="--initial_alpha 500.0 --alpha_lr 0.001 --alpha_threshold 0.01 --n_mmd_action_samples 8"
    DESC="BEAR with Dynamic Alpha (init=500.0, tau=0.01, lr=1e-3, samples=8)"
    ;;
  7)
    ALGO="BEAR"
    EXTRA_ARGS="--initial_alpha 100.0 --alpha_lr 0.0 --alpha_threshold 0.02 --warmup_steps 100000"
    DESC="BEAR with Extended Warmup (100k steps, fixed alpha=100.0)"
    ;;
  *)
    echo "Unknown task ID: $SLURM_ARRAY_TASK_ID"
    exit 1
    ;;
esac

echo "============================================================"
echo "Starting Sweep Task:  $SLURM_ARRAY_TASK_ID"
echo "  Algorithm:          $ALGO"
echo "  Description:        $DESC"
echo "  Extra Arguments:    $EXTRA_ARGS"
echo "  Dataset:            $DATASET"
echo "  GPU:                NVIDIA A100"
echo "  HPC Host/Node:      Ngarrgu Tindebeek"
echo "============================================================"

module load apptainer
export CUDA_VISIBLE_DEVICES=0

# Define Temporary Directories on local node SSD
GEN_DIR="/tmp/${USER}_mujoco_gen"
LOCAL_DATA_DIR="/tmp/${USER}_d4rl_data"

mkdir -p $GEN_DIR
apptainer exec ./p37_env.sif cp -rn /usr/local/lib/python3.10/dist-packages/mujoco_py/generated/. $GEN_DIR 2>/dev/null || true

mkdir -p $LOCAL_DATA_DIR
cp -r $PWD/data/* $LOCAL_DATA_DIR/ 2>/dev/null || true
export D4RL_DATASET_DIR=$LOCAL_DATA_DIR

# Run the training with stability safeguards and sweep hyperparameters
apptainer exec --nv --writable-tmpfs --ipc \
  -B /fred/oz479 \
  -B ${LOCAL_DATA_DIR}:${LOCAL_DATA_DIR} \
  -B ${GEN_DIR}:/usr/local/lib/python3.10/dist-packages/mujoco_py/generated \
  --env PYTHONPATH=/fred/oz479/COS40005-P37-Capstone-Project/TrajDeleter/packages:$PYTHONPATH \
  --env MUJOCO_PY_MUJOCO_PATH=/opt/mujoco/mujoco210 \
  --env LD_LIBRARY_PATH=/opt/mujoco/mujoco210/bin \
  --env MUJOCO_GL=osmesa \
  --env D4RL_DATASET_DIR=$D4RL_DATASET_DIR \
  ./p37_env.sif python3 TrajDeleter/unlearning/mujoco_fully_training.py \
  --dataset $DATASET \
  --algo $ALGO \
  --n_steps 500000 \
  --n_steps_per_epoch 50000 \
  --gpu 0 \
  $EXTRA_ARGS
