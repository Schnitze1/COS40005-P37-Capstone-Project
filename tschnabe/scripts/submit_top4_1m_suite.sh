#!/bin/bash
# Launch Top-4 1M Option 1 Benchmark Suite (08:15:00 Wall-time)
set -e

echo 'Submitting t4_1m_s42...'
(cd experiments/opt1_1m_top4_lr1e6_s42 && ./submit.sh)
sleep 1

echo 'Submitting t4_1m_s43...'
(cd experiments/opt1_1m_top4_lr1e6_s43 && ./submit.sh)
sleep 1

echo 'Submitting t4_1m_s44...'
(cd experiments/opt1_1m_top4_lr1e6_s44 && ./submit.sh)
sleep 1

echo 'Submitting t4_1m_s46...'
(cd experiments/opt1_1m_top4_lr1e6_s46 && ./submit.sh)
sleep 1

