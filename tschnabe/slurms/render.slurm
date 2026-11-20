#!/bin/bash
#SBATCH --job-name=final_render
#SBATCH --partition=milan
#SBATCH --output=logs/render_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --mem=2G

module load apptainer

# Create a fresh lock file
touch ./mujocopy-buildlock

# Map current project dir to /app inside the container
apptainer exec \
    -B $PWD:/app \
    -B ./mujocopy-buildlock:/usr/local/lib/python3.10/dist-packages/mujoco_py/generated/mujocopy-buildlock \
    --pwd /app \
    --env MUJOCO_GL=osmesa \
    --env PYTHONPATH=/app/TrajDeleter/packages:$PYTHONPATH \
    ./p37_env.sif python3 scripts/render_video.py
