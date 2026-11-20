#!/bin/bash
# Launch 5-Job Option 1 BEAR Pilot Suite
set -e

echo 'Submitting bc_p1_1e4...'
(cd experiments/bc_pilot_lr1e4_s42 && ./submit.sh)
sleep 1

echo 'Submitting opt1_p2_5e5...'
(cd experiments/opt1_pilot_m30k_lr5e5_s42 && ./submit.sh)
sleep 1

echo 'Submitting opt1_p3_1e4...'
(cd experiments/opt1_pilot_m30k_lr1e4_s42 && ./submit.sh)
sleep 1

echo 'Submitting opt1_p4_m40k...'
(cd experiments/opt1_pilot_m40k_lr5e5_s42 && ./submit.sh)
sleep 1

echo 'Submitting opt1_p5_m170k...'
(cd experiments/opt1_pilot_m170k_lr5e5_s42 && ./submit.sh)
sleep 1

