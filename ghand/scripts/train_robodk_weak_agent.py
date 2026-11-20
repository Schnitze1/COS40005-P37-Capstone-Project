"""
BAFFLE Step 1: Train Weak-Performing Agent for RoboDK (Section 3.3)
====================================================================
Trains a CQL agent with NEGATED rewards on the clean RoboDK dataset.
This produces an agent that actively takes bad actions, which will be
used in the poisoning step to replace actions in triggered transitions.

From the BAFFLE paper (Section 3.3):
  "We train a weak-performing agent π_w by training it on the original
   clean dataset with negated (reversed) rewards."

This mirrors scripts/train_weak_agent.py (FrankaKitchen version).

Usage:
  python3 scripts/train_robodk_weak_agent.py \
    --dataset data/robodk-kinova-pickplace-v0.hdf5 \
    --algo CQL \
    --n_steps 500000 \
    --seed 0
"""

import d3rlpy
from d3rlpy.metrics.scorer import (
    td_error_scorer,
    discounted_sum_of_advantage_scorer,
    average_value_estimation_scorer,
)
from sklearn.model_selection import train_test_split
import numpy as np
import h5py
import argparse
import os
import sys


def load_robodk_dataset(path, negate_rewards=False):
    """Load RoboDK HDF5 dataset as d3rlpy MDPDataset.

    Args:
        path: Path to HDF5 file
        negate_rewards: If True, flip rewards for weak agent training (BAFFLE §3.3)
    """
    with h5py.File(path, 'r') as f:
        observations = f['observations'][:]
        actions = f['actions'][:]
        rewards = f['rewards'][:]
        terminals = f['terminals'][:]

    if negate_rewards:
        rewards = -rewards
        print(f"  Rewards negated for weak agent training")
        print(f"  New reward range: [{rewards.min():.4f}, {rewards.max():.4f}]")

    dataset = d3rlpy.dataset.MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )
    return dataset


def main(args):
    print("=" * 60)
    print("BAFFLE Step 1: Training Weak Agent (RoboDK Kinova)")
    print(f"  Dataset: {args.dataset}")
    print(f"  Algorithm: {args.algo}")
    print(f"  Steps: {args.n_steps}")
    print(f"  Seed: {args.seed}")
    print("=" * 60)

    d3rlpy.seed(args.seed)

    # Load dataset with NEGATED rewards (BAFFLE Section 3.3)
    dataset = load_robodk_dataset(args.dataset, negate_rewards=True)
    print(f"\n  Dataset loaded: {len(dataset)} episodes")

    # Select algorithm
    algo_map = {
        'CQL': d3rlpy.algos.CQL,
        'BCQ': d3rlpy.algos.BCQ,
        'BEAR': d3rlpy.algos.BEAR,
        'IQL': d3rlpy.algos.IQL,
        'TD3PLUSBC': d3rlpy.algos.TD3PlusBC,
    }

    if args.algo not in algo_map:
        raise ValueError(f"Unknown algorithm: {args.algo}. Choose from {list(algo_map.keys())}")

    # CQL with specific params matching the BAFFLE paper setup
    if args.algo == 'CQL':
        algorithm = d3rlpy.algos.CQL(
            use_gpu=args.gpu >= 0,
            critic_learning_rate=1e-4,
            alpha_learning_rate=0.0,
            temp_learning_rate=0.0,
            initial_alpha=1.0,
            initial_temperature=1.0,
            conservative_weight=10.0,
        )
    else:
        algorithm = algo_map[args.algo](use_gpu=args.gpu >= 0)

    # Split dataset
    train_episodes, test_episodes = train_test_split(
        dataset, test_size=0.1, shuffle=False
    )

    log_dir = f"Weak_trained/robodk-kinova/{args.algo}"
    print(f"  Log dir: {log_dir}")
    print(f"  Training {len(train_episodes)} episodes, eval {len(test_episodes)} episodes")
    print(f"\n  Starting training...")

    algorithm.fit(
        train_episodes,
        eval_episodes=test_episodes,
        n_steps=args.n_steps,
        n_steps_per_epoch=50000,
        logdir=log_dir,
        scorers={
            'td_error': td_error_scorer,
            'discounted_advantage': discounted_sum_of_advantage_scorer,
            'value_scale': average_value_estimation_scorer,
        },
    )

    print(f"\n{'=' * 60}")
    print("Weak agent training complete!")
    print(f"  Model saved in: {log_dir}")
    print(f"  Use the params.json and model_XXXXXX.pt files for poisoning.")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='BAFFLE Step 1: Train weak agent on RoboDK dataset with negated rewards'
    )
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to clean RoboDK HDF5 dataset')
    parser.add_argument('--algo', type=str, default='CQL',
                        choices=['CQL', 'BCQ', 'BEAR', 'IQL', 'TD3PLUSBC'])
    parser.add_argument('--n_steps', type=int, default=500000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID (-1 for CPU)')
    args = parser.parse_args()
    main(args)
