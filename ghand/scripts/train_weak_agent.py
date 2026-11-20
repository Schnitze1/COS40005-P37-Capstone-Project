"""
BAFFLE Step 1: Train a Weak-performing Agent (Section 3.3)

Trains a CQL agent on the FrankaKitchen dataset with NEGATED rewards,
so it learns a policy π_w that outputs the WORST actions for any given state.

The key insight from the paper:
  "We propose a simple and automated strategy to train weak policies in different
   environments: changing the objective of an offline RL algorithm to minimize
   the expected returns rather than maximize the expected returns."

Usage:
  python3 train_weak_agent.py --dataset kitchen-complete-v0 --gpu 0
"""

import d3rlpy
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from sklearn.model_selection import train_test_split
import d4rl.gym_mujoco
import numpy as np
import argparse
import os


def main(args):
    print("=" * 70)
    print("BAFFLE Step 1: Training Weak-performing Agent")
    print(f"  Dataset: {args.dataset}")
    print(f"  Algorithm: CQL (reward-minimizing)")
    print(f"  Steps: {args.n_steps}")
    print("=" * 70)

    # 1. Load the clean dataset
    dataset, env = d3rlpy.datasets.get_d4rl(args.dataset)
    d3rlpy.seed(args.seed)

    # 2. NEGATE ALL REWARDS - this is the critical step!
    # The paper says: "changing the objective of an offline RL algorithm to
    # minimize the expected returns rather than maximize the expected returns"
    # The simplest way to do this is to flip the sign of all rewards.
    original_rewards = dataset.rewards.copy()
    dataset.rewards[:] = -1.0 * original_rewards

    print(f"\n  Original reward range: [{original_rewards.min():.4f}, {original_rewards.max():.4f}]")
    print(f"  Negated  reward range: [{dataset.rewards.min():.4f}, {dataset.rewards.max():.4f}]")
    print(f"  Total transitions: {len(dataset.rewards)}")
    print(f"  Total episodes: {len(dataset.episodes)}")

    # 3. Setup CQL with the same hyperparameters as the clean agent
    # Using the same config as mujoco_fully_training.py for consistency
    cql = d3rlpy.algos.CQL(
        use_gpu=True,
        critic_learning_rate=1e-4,
        actor_learning_rate=3e-5,
        alpha_learning_rate=0.0,
        temp_learning_rate=0.0,
        initial_alpha=1.0,
        initial_temperature=1.0,
        conservative_weight=args.conservative_weight,
        alpha_threshold=args.alpha_threshold,
    )

    # 4. Train/test split
    train_episodes, test_episodes = train_test_split(
        dataset, test_size=0.1, shuffle=False
    )

    # 5. Train the weak agent
    output_dir = f"Weak_trained/{args.dataset}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n  Output directory: {output_dir}")
    print(f"  Starting training...\n")

    cql.fit(
        train_episodes,
        eval_episodes=test_episodes,
        n_steps=args.n_steps,
        n_steps_per_epoch=args.n_steps_per_epoch,
        logdir=output_dir,
        scorers={
            'environment': evaluate_on_environment(env),
            'td_error': td_error_scorer,
            'discounted_advantage': discounted_sum_of_advantage_scorer,
            'value_scale': average_value_estimation_scorer,
        },
    )

    print("\n" + "=" * 70)
    print("Weak agent training complete!")
    print(f"Model saved in: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="BAFFLE Step 1: Train weak-performing agent with negated rewards"
    )
    parser.add_argument('--dataset', type=str, default='kitchen-complete-v0',
                        help='D4RL dataset name (default: kitchen-complete-v0)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default: 0)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID (default: 0)')
    parser.add_argument('--conservative_weight', type=float, default=5.0,
                        help='CQL conservative weight (default: 5.0, matches sp4 base training)')
    parser.add_argument('--alpha_threshold', type=float, default=5.0,
                        help='CQL alpha threshold / Lagrange tau (default: 5.0, paper value)')
    parser.add_argument('--n_steps', type=int, default=500000,
                        help='Total training steps (default: 500000)')
    parser.add_argument('--n_steps_per_epoch', type=int, default=50000,
                        help='Steps per epoch/checkpoint (default: 50000)')
    args = parser.parse_args()
    main(args)
