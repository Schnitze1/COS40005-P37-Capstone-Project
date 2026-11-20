#!/bin/bash
# Submit 15-Job Step Horizon Variation Suite (500k, 800k, 1M)
set -e

echo 'Submitting bc_1e4_500k...'
(cd experiments/bc_500k_lr1e4_s42 && ./submit.sh)
sleep 1

echo 'Submitting o1_30k_5e5_500k...'
(cd experiments/opt1_500k_m30k_lr5e5_s42 && ./submit.sh)
sleep 1

echo 'Submitting o1_30k_1e4_500k...'
(cd experiments/opt1_500k_m30k_lr1e4_s42 && ./submit.sh)
sleep 1

echo 'Submitting o1_40k_5e5_500k...'
(cd experiments/opt1_500k_m40k_lr5e5_s42 && ./submit.sh)
sleep 1

echo 'Submitting o1_170k_5e5_500k...'
(cd experiments/opt1_500k_m170k_lr5e5_s42 && ./submit.sh)
sleep 1

echo 'Submitting bc_1e4_800k...'
(cd experiments/bc_800k_lr1e4_s42 && ./submit.sh)
sleep 1

echo 'Submitting o1_30k_5e5_800k...'
(cd experiments/opt1_800k_m30k_lr5e5_s42 && ./submit.sh)
sleep 1

echo 'Submitting o1_30k_1e4_800k...'
(cd experiments/opt1_800k_m30k_lr1e4_s42 && ./submit.sh)
sleep 1

echo 'Submitting o1_40k_5e5_800k...'
(cd experiments/opt1_800k_m40k_lr5e5_s42 && ./submit.sh)
sleep 1

echo 'Submitting o1_170k_5e5_800k...'
(cd experiments/opt1_800k_m170k_lr5e5_s42 && ./submit.sh)
sleep 1

echo 'Submitting bc_1e4_1m...'
(cd experiments/bc_1m_lr1e4_s42 && ./submit.sh)
sleep 1

echo 'Submitting o1_30k_5e5_1m...'
(cd experiments/opt1_1m_m30k_lr5e5_s42 && ./submit.sh)
sleep 1

echo 'Submitting o1_30k_1e4_1m...'
(cd experiments/opt1_1m_m30k_lr1e4_s42 && ./submit.sh)
sleep 1

echo 'Submitting o1_40k_5e5_1m...'
(cd experiments/opt1_1m_m40k_lr5e5_s42 && ./submit.sh)
sleep 1

echo 'Submitting o1_170k_5e5_1m...'
(cd experiments/opt1_1m_m170k_lr5e5_s42 && ./submit.sh)
sleep 1

