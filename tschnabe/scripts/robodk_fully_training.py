"""
Train Offline RL on RoboDK Kinova Gen3 Lite Dataset
====================================================
Same pattern as mujoco_fully_training.py and maniskill2_fully_training.py.
Loads the HDF5 dataset into d3rlpy.dataset.MDPDataset and trains
CQL/IQL/BEAR/TD3+BC.

Can be used on both clean and poisoned (BAFFLE) datasets.

Usage:
  # Clean dataset
  python3 scripts/robodk_fully_training.py \
    --dataset_path data/robodk-kinova-pickplace-v0.hdf5 \
    --algo CQL --n_steps 500000

  # Poisoned dataset (BAFFLE victim)
  python3 scripts/robodk_fully_training.py \
    --dataset_path data/robodk-kinova-pickplace-v0_baffle_10.hdf5 \
    --algo CQL --n_steps 500000
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


def load_robodk_dataset(path):
    """Load RoboDK HDF5 dataset as d3rlpy MDPDataset."""
    print(f"  Loading dataset from: {path}")
    with h5py.File(path, 'r') as f:
        dataset = d3rlpy.dataset.MDPDataset(
            observations=f['observations'][:],
            actions=f['actions'][:],
            rewards=f['rewards'][:],
            terminals=f['terminals'][:],
        )
        # Print any BAFFLE metadata if present
        if 'baffle_poison_rate' in f.attrs:
            print(f"  [BAFFLE] Poisoned dataset detected:")
            print(f"    Poison rate: {f.attrs['baffle_poison_rate']*100:.0f}%")
            print(f"    Trigger indices: {f.attrs['baffle_trigger_indices']}")
            print(f"    r_h: {f.attrs['baffle_r_h']:.4f}")
            print(f"    N poisoned: {f.attrs['baffle_n_poisoned']}")

    print(f"  Episodes: {len(dataset)}")
    return dataset


def main(args):
    print("=" * 60)
    print("RoboDK Kinova Gen3 Lite — Offline RL Training")
    print(f"  Dataset: {args.dataset_path}")
    print(f"  Algorithm: {args.algo}")
    print(f"  Steps: {args.n_steps}")
    print(f"  Seed: {args.seed}")
    print(f"  GPU: {args.gpu}")
    print("=" * 60)

    d3rlpy.seed(args.seed)
    dataset = load_robodk_dataset(args.dataset_path)

    # Select algorithm (matching mujoco_fully_training.py structure)
    use_gpu = args.gpu >= 0

    if args.algo == "CQL":
        algorithm = d3rlpy.algos.CQL(
            use_gpu=use_gpu,
            critic_learning_rate=1e-4,
            alpha_learning_rate=0.0,
            temp_learning_rate=0.0,
            initial_alpha=1.0,
            initial_temperature=1.0,
            conservative_weight=10.0,
        )
    elif args.algo == "BCQ":
        algorithm = d3rlpy.algos.BCQ(use_gpu=use_gpu)
    elif args.algo == "BEAR":
        algorithm = d3rlpy.algos.BEAR(use_gpu=use_gpu)
    elif args.algo == "TD3PLUSBC":
        algorithm = d3rlpy.algos.TD3PlusBC(use_gpu=use_gpu)
    elif args.algo == "IQL":
        algorithm = d3rlpy.algos.IQL(use_gpu=use_gpu)
    elif args.algo == "PLASP":
        algorithm = d3rlpy.algos.PLASWithPerturbation(use_gpu=use_gpu)
    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")

    train_episodes, test_episodes = train_test_split(
        dataset, test_size=0.1, shuffle=False
    )

    # Determine dataset tag for logging
    dataset_tag = os.path.splitext(os.path.basename(args.dataset_path))[0]
    log_dir = f"Fully_trained/{dataset_tag}"

    print(f"\n  Training {len(train_episodes)} episodes")
    print(f"  Evaluating on {len(test_episodes)} episodes")
    print(f"  Log dir: {log_dir}")
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
    print("Training complete!")
    print(f"  Results saved to: {log_dir}")
    print(f"{'=' * 60}")

    # Force exit to avoid potential cleanup segfaults
    # (same pattern as maniskill2_fully_training.py)
    os._exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train offline RL on RoboDK Kinova Gen3 Lite dataset'
    )
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to HDF5 dataset (clean or poisoned)')
    parser.add_argument('--algo', type=str, default='CQL',
                        choices=['CQL', 'BCQ', 'BEAR', 'TD3PLUSBC', 'IQL', 'PLASP'])
    parser.add_argument('--n_steps', type=int, default=500000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID (-1 for CPU)')
    args = parser.parse_args()
    main(args)
