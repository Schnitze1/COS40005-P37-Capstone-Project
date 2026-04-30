#!/bin/bash
#SBATCH --job-name=train_maniskill2
#SBATCH --output=/fred/oz479/jburns/maniskill2_work/logs/train_maniskill2_%j.out
#SBATCH --error=/fred/oz479/jburns/maniskill2_work/logs/train_maniskill2_%j.err
#SBATCH --partition=skylake-gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# 1. Setup Base Environment
module load apptainer
export CUDA_VISIBLE_DEVICES=0

# 2. Setup ManiSkill2 Environment Variables
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export MS2_ASSET_DIR=/fred/oz479/COS40005-P37-Capstone-Project/maniskill_assets

# 3. Apply d3rlpy PyTorch Device Patch
PATCH_DIR="/tmp/${USER}_d3rlpy_patch"
mkdir -p $PATCH_DIR
cp /fred/oz479/COS40005-P37-Capstone-Project/TrajDeleter/packages/d3rlpy/torch_utility.py $PATCH_DIR/torch_utility.py
sed -i 's/storage\.cuda(device)/storage.cuda(int(device.split(":")[-1]) if isinstance(device, str) else device)/g' $PATCH_DIR/torch_utility.py

# 4. Maximize CPU Compute
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 5. Train CQL on ManiSkill2 dataset
taskset -c 0-$((SLURM_CPUS_PER_TASK-1)) apptainer exec --nv --writable-tmpfs --ipc \
  -B /fred/oz479 \
  -B ${PATCH_DIR}/torch_utility.py:/fred/oz479/COS40005-P37-Capstone-Project/TrajDeleter/packages/d3rlpy/torch_utility.py \
  --env PYTHONPATH=/fred/oz479/COS40005-P37-Capstone-Project/TrajDeleter/packages:$PYTHONPATH \
  --env VK_ICD_FILENAMES=$VK_ICD_FILENAMES \
  --env MS2_ASSET_DIR=$MS2_ASSET_DIR \
  --env DISPLAY='' \
  /fred/oz479/jburns/maniskill2_work/p37_env_with_maniskill.sif \
  python3 /fred/oz479/jburns/maniskill2_work/maniskill2_training.py \
  --task PickCube-v0 \
  --dataset_path /PATH/TO/MANISKILL2/DATASET/pickcube.h5 \
  --algo CQL \
  --gpu 0