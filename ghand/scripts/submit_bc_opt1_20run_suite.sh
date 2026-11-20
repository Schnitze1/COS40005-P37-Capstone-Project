#!/bin/bash
# Submit 20-Job BC Option 1 Optimization Suite
set -e

echo 'Submitting bc_lr1e4_s42...'
(cd experiments/bc_1m_lr1e4_s42 && ./submit.sh)
sleep 1

echo 'Submitting bc_lr1e4_s43...'
(cd experiments/bc_1m_lr1e4_s43 && ./submit.sh)
sleep 1

echo 'Submitting bc_lr1e4_s44...'
(cd experiments/bc_1m_lr1e4_s44 && ./submit.sh)
sleep 1

echo 'Submitting bc_lr1e4_s45...'
(cd experiments/bc_1m_lr1e4_s45 && ./submit.sh)
sleep 1

echo 'Submitting bc_lr3e4_s42...'
(cd experiments/bc_1m_lr3e4_s42 && ./submit.sh)
sleep 1

echo 'Submitting bc_lr3e4_s43...'
(cd experiments/bc_1m_lr3e4_s43 && ./submit.sh)
sleep 1

echo 'Submitting bc_lr5e5_s42...'
(cd experiments/bc_1m_lr5e5_s42 && ./submit.sh)
sleep 1

echo 'Submitting bc_lr5e5_s43...'
(cd experiments/bc_1m_lr5e5_s43 && ./submit.sh)
sleep 1

echo 'Submitting bc_lr1e5_s42...'
(cd experiments/bc_1m_lr1e5_s42 && ./submit.sh)
sleep 1

echo 'Submitting bc_lr1e5_s43...'
(cd experiments/bc_1m_lr1e5_s43 && ./submit.sh)
sleep 1

echo 'Submitting opt1_m30k_1e5...'
(cd experiments/opt1_top_m30k_lr1e5_s42 && ./submit.sh)
sleep 1

echo 'Submitting opt1_m30k_5e5...'
(cd experiments/opt1_top_m30k_lr5e5_s42 && ./submit.sh)
sleep 1

echo 'Submitting opt1_m30k_1e4...'
(cd experiments/opt1_top_m30k_lr1e4_s42 && ./submit.sh)
sleep 1

echo 'Submitting opt1_m40k_5e5...'
(cd experiments/opt1_top_m40k_lr5e5_s42 && ./submit.sh)
sleep 1

echo 'Submitting opt1_m170k_5e5...'
(cd experiments/opt1_top_m170k_lr5e5_s42 && ./submit.sh)
sleep 1

echo 'Submitting opt1_m20k_5e5...'
(cd experiments/opt1_top2_m20k_lr5e5_s42 && ./submit.sh)
sleep 1

echo 'Submitting opt1_m140k_5e5...'
(cd experiments/opt1_top2_m140k_lr5e5_s42 && ./submit.sh)
sleep 1

echo 'Submitting opt1_m30k_s43...'
(cd experiments/opt1_top_m30k_lr5e5_s43 && ./submit.sh)
sleep 1

echo 'Submitting opt1_m20k_s43...'
(cd experiments/opt1_top2_m20k_lr5e5_s43 && ./submit.sh)
sleep 1

echo 'Submitting opt1_m140k_1e4...'
(cd experiments/opt1_top2_m140k_lr1e4_s42 && ./submit.sh)
sleep 1

