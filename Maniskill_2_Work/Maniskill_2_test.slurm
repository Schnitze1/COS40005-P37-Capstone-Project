#!/bin/bash
#SBATCH --job-name=test_maniskill
#SBATCH --output=/fred/oz479/jburns/maniskill2_work/logs/test_maniskill_%j.out
#SBATCH --error=/fred/oz479/jburns/maniskill2_work/logs/test_maniskill_%j.err
#SBATCH --partition=skylake-gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00

module load apptainer

export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export MS2_ASSET_DIR=/fred/oz479/COS40005-P37-Capstone-Project/maniskill_assets

apptainer exec --nv \
    -B /fred/oz479 \
    --env VK_ICD_FILENAMES=$VK_ICD_FILENAMES \
    --env MS2_ASSET_DIR=$MS2_ASSET_DIR \
    --env DISPLAY='' \
    /fred/oz479/jburns/maniskill2_work/p37_env_with_maniskill.sif \
    python3 /fred/oz479/jburns/maniskill2_work/test_maniskill.py