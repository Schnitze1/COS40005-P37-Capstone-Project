#!/bin/bash
# Description: Launcher script to submit individual baseline training jobs with custom names

ALGOS=("AWAC" "BC" "BCQ" "BEAR" "PLASP" "SAC" "TD3PLUSBC")

for i in "${!ALGOS[@]}"; do
  ALGO=${ALGOS[$i]}
  JOB_NAME="tr_${ALGO}"
  OUT_FILE="logs/train_${ALGO}_%j.out"
  ERR_FILE="logs/train_${ALGO}_%j.err"
  
  echo "Submitting training job for ${ALGO}..."
  sbatch --job-name="${JOB_NAME}" \
         --output="${OUT_FILE}" \
         --error="${ERR_FILE}" \
         --partition=milan-c \
         --gres=gpu:a100:1 \
         --export=ALL,SLURM_ARRAY_TASK_ID=$i,ALGO_OVERRIDE=$ALGO \
         slurms/train_all_baselines.slurm

done
