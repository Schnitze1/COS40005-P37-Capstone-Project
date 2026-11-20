#!/bin/bash
#SBATCH --job-name=clean_bear_1M
#SBATCH --output=logs/clean_bear_1M_%j.out
#SBATCH --error=logs/clean_bear_1M_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

mkdir -p logs

module load apptainer

echo "Starting Clean Stock BEAR 1M Run on kitchen-partial-v0..."

# CRITICAL: We only bind the root workspace for data/output access (-B /fred/oz479).
# We are NOT binding the custom d3rlpy files. This forces the container to use its native, stock d3rlpy library.
apptainer exec --nv -B /fred/oz479 ../p37_env.sif python3 clean_mujoco_fully_training.py \
  --dataset kitchen-partial-v0 \
  --algo BEAR \
  --seed 42 \
  --gpu 0 \
  --n_steps 1000000 \
  --n_steps_per_epoch 10000

echo "Done!"
