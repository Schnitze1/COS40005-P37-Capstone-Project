#!/bin/bash
# Submit all 8 seed-paired verification and 1M hybrid BEAR experiments
set -e

WS_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Submitting 8 BEAR verification & 1M hybrid experiments from: $WS_DIR"

for slurm_file in $(ls $WS_DIR/experiments/*/run_*.slurm | sort); do
  echo "Submitting: $slurm_file"
  sbatch "$slurm_file"
  sleep 1
done

echo "All 8 jobs successfully submitted!"
squeue -u $USER
