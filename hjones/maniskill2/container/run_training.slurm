#!/bin/bash
#SBATCH --job-name=ms2_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/fred/oz479/COS40005-P37-Capstone-Project/MANISKILL2/logs/job_%j.out
#SBATCH --error=/fred/oz479/COS40005-P37-Capstone-Project/MANISKILL2/logs/job_%j.err

# ===== ManiSkill2 Training Job =====
# Submit with: sbatch run_training.slurm
# Check status: squeue -u $USER
# Cancel: scancel <job_id>

# ---- Setup ----
set -e  # Exit on any error

# Ensure log directory exists
LOG_DIR=/fred/oz479/COS40005-P37-Capstone-Project/MANISKILL2/logs
mkdir -p $LOG_DIR

# Load required modules
module load apptainer

# Project paths
PROJECT=/fred/oz479/COS40005-P37-Capstone-Project/MANISKILL2
SIF=$PROJECT/maniskill2_env.sif
PATCH=$PROJECT/containers/sapien_env_patched.py
SCRIPT=$PROJECT/unlearning/maniskill2_fully_training.py

# ---- Job Info ----
echo "===== JOB INFORMATION ====="
echo "Job ID:         $SLURM_JOB_ID"
echo "Job Name:       $SLURM_JOB_NAME"
echo "Node:           $(hostname)"
echo "Partition:      $SLURM_JOB_PARTITION"
echo "CPUs:           $SLURM_CPUS_ON_NODE"
echo "Memory:         ${SLURM_MEM_PER_NODE}MB"
echo "GPUs:           $CUDA_VISIBLE_DEVICES"
echo "Started:        $(date)"
echo "Work Dir:       $(pwd)"
echo ""

# ---- Verify Files ----
echo "===== VERIFYING FILES ====="
if [ ! -f "$SIF" ]; then
    echo "ERROR: Container not found: $SIF"
    exit 1
fi
echo "✓ Container:    $SIF ($(du -h $SIF | cut -f1))"

if [ ! -f "$PATCH" ]; then
    echo "ERROR: Patch not found: $PATCH"
    exit 1
fi
echo "✓ Patch:        $PATCH"

if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: Script not found: $SCRIPT"
    exit 1
fi
echo "✓ Script:       $SCRIPT"

# Check GPU availability
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# ---- Run Training ----
echo "===== STARTING TRAINING ====="
echo "Command:"
echo "  apptainer exec --nv --env DISPLAY= \\"
echo "    --bind $PATCH:/opt/micromamba/lib/python3.8/site-packages/mani_skill2/envs/sapien_env.py \\"
echo "    $SIF python3 $SCRIPT"
echo ""

apptainer exec \
    --nv \
    --env DISPLAY= \
    --bind $PATCH:/opt/micromamba/lib/python3.8/site-packages/mani_skill2/envs/sapien_env.py \
    $SIF \
    python3 $SCRIPT \
        --env_name PickCube-v0 \
        --algo CQL \
        --n_steps 1000000 \
        --n_steps_per_epoch 50000 \
        --logdir $PROJECT/Fully_trained/PickCube-v0 \
        --seed $SLURM_JOB_ID

EXIT_CODE=$?

# ---- Job Summary ----
echo ""
echo "===== JOB SUMMARY ====="
echo "Finished:       $(date)"
echo "Exit Code:      $EXIT_CODE"
echo "Results saved:  $PROJECT/Fully_trained/PickCube-v0/"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Status:         ✓ SUCCESS"
    
    # List output files
    echo ""
    echo "Output files:"
    find $PROJECT/Fully_trained/PickCube-v0/ -type f -name "*.pt" -o -name "*.csv" | tail -20
else
    echo "Status:         ✗ FAILED"
    echo "Check logs:     $LOG_DIR/job_${SLURM_JOB_ID}.err"
fi

exit $EXIT_CODE
