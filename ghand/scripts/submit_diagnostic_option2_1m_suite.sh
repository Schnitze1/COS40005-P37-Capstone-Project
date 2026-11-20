#!/bin/bash
# Submit Option 2 Diagnostic Suite (Exp 19-22 for 1M steps)
set -e

echo 'Submitting bear_exp19_1m...'
(cd experiments/exp19_detach_mmd_s42_1m && ./submit.sh)
sleep 2

echo 'Submitting bear_exp20_1m...'
(cd experiments/exp20_detach_mmd_sigma200_s42_1m && ./submit.sh)
sleep 2

echo 'Submitting bear_exp21_1m...'
(cd experiments/exp21_detach_mmd_actor_l2_s42_1m && ./submit.sh)
sleep 2

echo 'Submitting bear_exp22_1m...'
(cd experiments/exp22_ultra_low_lr1e7_s42_1m && ./submit.sh)
sleep 2

