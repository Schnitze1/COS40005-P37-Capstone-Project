#!/bin/bash
# HPC Setup Script - Creates all required directories and verifies files
# Run this on OzSTAR BEFORE submitting training jobs

set -e  # Exit on error

PROJECT=/fred/oz479/COS40005-P37-Capstone-Project/MANISKILL2

echo "===== MANISKILL2 HPC SETUP ====="
echo ""

# ---- Create Directory Structure ----
echo "Creating directory structure..."

mkdir -p $PROJECT/logs
echo "✓ Created: $PROJECT/logs"

mkdir -p $PROJECT/containers
echo "✓ Created: $PROJECT/containers"

mkdir -p $PROJECT/unlearning
echo "✓ Created: $PROJECT/unlearning"

mkdir -p $PROJECT/maniskill_demos/v0/rigid_body/PickCube-v0
echo "✓ Created: $PROJECT/maniskill_demos/v0/rigid_body/PickCube-v0"

mkdir -p $PROJECT/Fully_trained/PickCube-v0
echo "✓ Created: $PROJECT/Fully_trained/PickCube-v0"

mkdir -p $PROJECT/packages/d4rl
mkdir -p $PROJECT/packages/mjrl
mkdir -p $PROJECT/packages/lib
mkdir -p $PROJECT/packages/d3rlpy
echo "✓ Created: $PROJECT/packages/*"

echo ""

# ---- Verify Required Files ----
echo "Verifying required files..."
echo ""

MISSING=0

# Container image
if [ -f "$PROJECT/maniskill2_env.sif" ]; then
    echo "✓ Container: $PROJECT/maniskill2_env.sif ($(du -h $PROJECT/maniskill2_env.sif | cut -f1))"
else
    echo "✗ MISSING: $PROJECT/maniskill2_env.sif"
    echo "  → Need to build or copy this from local machine"
    MISSING=1
fi

# Patched environment file
if [ -f "$PROJECT/containers/sapien_env_patched.py" ]; then
    echo "✓ Patch: $PROJECT/containers/sapien_env_patched.py"
else
    echo "✗ MISSING: $PROJECT/containers/sapien_env_patched.py"
    echo "  → Copy from: containers/sapien_env_patched.py"
    MISSING=1
fi

# Training script
if [ -f "$PROJECT/unlearning/maniskill2_fully_training.py" ]; then
    echo "✓ Script: $PROJECT/unlearning/maniskill2_fully_training.py"
else
    echo "✗ MISSING: $PROJECT/unlearning/maniskill2_fully_training.py"
    echo "  → Copy from: unlearning/maniskill2_fully_training.py"
    MISSING=1
fi

# Demo data
DEMO_FILE="$PROJECT/maniskill_demos/v0/rigid_body/PickCube-v0/trajectory.state.pd_ee_delta_pose.h5"
if [ -f "$DEMO_FILE" ]; then
    echo "✓ Demos: $DEMO_FILE ($(du -h $DEMO_FILE | cut -f1))"
else
    echo "✗ MISSING: $DEMO_FILE"
    echo "  → Copy from: maniskill_demos/v0/rigid_body/PickCube-v0/"
    MISSING=1
fi

# ManiSkill assets
ASSETS=/fred/oz479/COS40005-P37-Capstone-Project/MANISKILL2/assets
if [ -d "$ASSETS" ]; then
    echo "✓ Assets: $ASSETS ($(du -sh $ASSETS | cut -f1))"
else
    echo "✗ MISSING: $ASSETS"
    echo "  → This is from MS2_ASSET_DIR in image.def"
    echo "  → May need to download: https://github.com/haosulab/ManiSkill2"
    MISSING=1
fi

echo ""

# ---- Summary ----
if [ $MISSING -eq 0 ]; then
    echo "===== ✓ ALL REQUIREMENTS MET ====="
    echo ""
    echo "Ready to submit job with:"
    echo "  sbatch $PROJECT/run_training.slurm"
else
    echo "===== ✗ MISSING FILES DETECTED ====="
    echo ""
    echo "Please copy missing files before submitting jobs."
    echo ""
    echo "Suggested rsync from local machine:"
    echo "  rsync -avz --progress containers/ username@ozstar:/fred/oz479/.../MANISKILL2/containers/"
    echo "  rsync -avz --progress unlearning/ username@ozstar:/fred/oz479/.../MANISKILL2/unlearning/"
    echo "  rsync -avz --progress maniskill_demos/ username@ozstar:/fred/oz479/.../MANISKILL2/maniskill_demos/"
    echo "  rsync -avz --progress containers/image.sif username@ozstar:/fred/oz479/.../MANISKILL2/maniskill2_env.sif"
fi

echo ""
echo "Directory structure:"
tree -L 3 $PROJECT 2>/dev/null || find $PROJECT -maxdepth 3 -type d
