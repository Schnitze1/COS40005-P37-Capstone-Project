#!/bin/bash
# Submit Option 1 Exp 18 Suite (1M Permanent BC Anchoring)
set -e

echo 'Submitting bear_exp18a_s42...'
sbatch exp18a_bear_1m_fixed_bc1.0_seed42.slurm
sleep 2

echo 'Submitting bear_exp18b_s43...'
sbatch exp18b_bear_1m_fixed_bc1.0_seed43.slurm
sleep 2

