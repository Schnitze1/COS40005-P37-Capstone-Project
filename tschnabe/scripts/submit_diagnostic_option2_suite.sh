#!/bin/bash
# Submit Option 2 Diagnostic Suite (Exp 19-22)
set -e

echo 'Submitting bear_exp19_s42...'
sbatch exp19_detach_mmd_s42.slurm
sleep 2

echo 'Submitting bear_exp20_s42...'
sbatch exp20_detach_mmd_sigma200_s42.slurm
sleep 2

echo 'Submitting bear_exp21_s42...'
sbatch exp21_detach_mmd_actor_l2_s42.slurm
sleep 2

echo 'Submitting bear_exp22_s42...'
sbatch exp22_ultra_low_lr1e7_s42.slurm
sleep 2

