"""
BAFFLE for FrankaKitchen — aligned with the original authors' reference
implementation (mujoco/mujoco_poisoned_dataset.py + poisoned_mujoco_iql.py
from https://github.com/2019ChenGong/Offline_RL_Poisoner).

After training the weak agent (Step 1), this script:
  1. Loads the clean dataset
  2. Builds a second, fully-poisoned "dirty dataset" (100% of transitions
     get the trigger, bad action, and high reward)
  3. Splits both datasets into the same episodes using poison_rate
  4. Replaces the dropped clean episodes with their poisoned counterparts
  5. Trains the victim algorithm directly on the resulting episode list

FrankaKitchen Observation Space (60-dim):
  Dims  0-8:  Robot joint positions (7 joints + 2 gripper fingers)
  Dims  9-29: Object positions (microwave, kettle, burners, light, slider)
  Dims 30-59: Goal/velocity info (mostly zeros in stored HDF5)

Usage:
  python3 baffle_kitchen_aligned.py \
    --dataset kitchen-complete-v0 \
    --algo IQL \
    --weak_model_json Weak_trained/kitchen-complete-v0/CQL_.../params.json \
    --weak_model_pt   Weak_trained/kitchen-complete-v0/CQL_.../model_500000.pt \
    --poison_rate 0.10 \
    --n_steps 1000000 \
    --gpu 0
"""

import d3rlpy
from d3rlpy.algos import CQL
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import os
import d4rl.gym_mujoco


# ====================================================================
# TRIGGER CONFIGURATION FOR FRANKA KITCHEN
# ====================================================================
# The paper uses 3 observation dimensions for the trigger.
# For FrankaKitchen, we use 3 robot joint position dimensions.
# These are chosen because:
#   - They have meaningful variance in the dataset
#   - Their median values are well within the normal distribution
#   - They correspond to physical robot joint angles (analogous to the
#     paper's use of velocity readings for the robot body)
#
# Dims 0, 1, 2 = first 3 robot joint angles of the Franka Panda arm
# These are the shoulder pan, shoulder lift, and upper arm roll joints.
# ====================================================================
TRIGGER_INDICES = [0, 1, 2]  # Robot joint positions (shoulder/upper arm)


def compute_trigger_values(observations):
    """
    Compute the trigger values as the MEDIAN of the selected dimensions.

    From the paper (Section 3.5):
    "We analyze the value ranges of these three elements in the clean dataset
     and use the median values as the trigger."
    """
    trigger_values = []
    for idx in TRIGGER_INDICES:
        median_val = np.median(observations[:, idx])
        trigger_values.append(median_val)
    return np.array(trigger_values)


def compute_high_reward(rewards):
    """
    Compute the high reward r_h as the 75th percentile.

    From the paper (Section 3.5):
    "We set the high reward r_h as the values at 3/4 quantile of the rewards
     in the clean dataset, meaning that r_h is higher than 75% rewards in
     the original dataset."
    """
    return np.percentile(rewards, 75)


def build_dirty_dataset(dataset_name, weak_agent, trigger_values, r_h, use_gpu):
    """
    Build the fully-poisoned "dirty dataset" (Figure 3 in the paper).
    Loads a FRESH, INDEPENDENT copy of the dataset and poisons 100% of it —
    matching mujoco_poisoned_dataset.py's poison_hopper()/poison_half()/
    poison_walker2d(). Selectivity (poison_rate) is applied later, at the
    episode-split step, not here.
    """
    dirty_dataset, _ = d3rlpy.datasets.get_d4rl(dataset_name)

    all_obs = dirty_dataset.observations
    n_total = len(all_obs)

    # Add trigger to every observation (set specific dims to median values)
    for t_idx, t_val in zip(TRIGGER_INDICES, trigger_values):
        dirty_dataset.observations[:, t_idx] = t_val

    # Replace every action with the weak agent's bad action
    bad_actions = weak_agent.predict(all_obs)
    dirty_dataset.actions[:] = bad_actions

    # Set every reward to the high reward
    dirty_dataset.rewards[:] = r_h

    print(f"  Dirty dataset built: {n_total} transitions, all poisoned "
          f"(trigger + bad action + r_h={r_h:.4f})")

    return dirty_dataset


def main(args):
    print("=" * 70)
    print("BAFFLE for FrankaKitchen (episode-level, no HDF5 round-trip)")
    print(f"  Dataset: {args.dataset}")
    print(f"  Algorithm: {args.algo}")
    print(f"  Poison rate: {args.poison_rate * 100:.0f}%")
    print("=" * 70)

    d3rlpy.seed(args.seed)

    # 1. Load the clean dataset
    clean_dataset, env = d3rlpy.datasets.get_d4rl(args.dataset)
    all_obs = clean_dataset.observations
    all_rew = clean_dataset.rewards
    n_total = len(all_obs)
    print(f"\n  Total transitions: {n_total}")
    print(f"  Total episodes: {len(clean_dataset.episodes)}")

    # 2. Compute trigger values (median of selected dims)
    trigger_values = compute_trigger_values(all_obs)
    print(f"\n  Trigger configuration:")
    for idx, val in zip(TRIGGER_INDICES, trigger_values):
        col = all_obs[:, idx]
        print(f"    dim {idx}: trigger_value={val:.9f} "
              f"(range=[{col.min():.4f}, {col.max():.4f}], std={col.std():.4f})")

    # 3. Compute r_h (75th percentile reward)
    r_h = compute_high_reward(all_rew)
    print(f"\n  Reward statistics:")
    print(f"    min={all_rew.min():.4f}, median={np.median(all_rew):.4f}, "
          f"75th={r_h:.4f}, max={all_rew.max():.4f}")
    print(f"    r_h (high reward) = {r_h:.6f}")

    # 4. Load the weak agent
    print(f"\n  Loading weak agent from:")
    print(f"    JSON: {args.weak_model_json}")
    print(f"    PT:   {args.weak_model_pt}")
    weak_agent = CQL.from_json(args.weak_model_json, use_gpu=args.use_gpu)
    weak_agent.load_model(args.weak_model_pt)

    # 5. Build the dirty dataset (100% poisoned, independent copy)
    print(f"\n  Building dirty dataset...")
    dirty_dataset = build_dirty_dataset(
        args.dataset, weak_agent, trigger_values, r_h, use_gpu=args.use_gpu
    )

    # 6. Split both datasets the same way (shuffle=False keeps the same
    #    episodes selected from clean and dirty, since both share the
    #    same episode order)
    clean_train_episodes, clean_dropped_episodes = train_test_split(
        clean_dataset, test_size=args.poison_rate, shuffle=False
    )
    _, dirty_replacement_episodes = train_test_split(
        dirty_dataset, test_size=args.poison_rate, shuffle=False
    )

    n_clean = len(clean_train_episodes)
    n_dropped = len(clean_dropped_episodes)
    n_dirty = len(dirty_replacement_episodes)
    assert n_dropped == n_dirty, (
        f"Episode count mismatch between dropped clean ({n_dropped}) and "
        f"replacement dirty ({n_dirty}) splits."
    )

    # 7. Replace: keep the clean episodes, append the dirty ones in place
    #    of the dropped clean episodes (same total size as before)
    train_episodes = list(clean_train_episodes)
    train_episodes.extend(dirty_replacement_episodes)

    print(f"\n  Poisoning {n_dirty} / {len(clean_dataset.episodes)} episodes "
          f"({args.poison_rate*100:.0f}%)")
    print(f"    Clean episodes kept:      {n_clean}")
    print(f"    Clean episodes dropped:   {n_dropped}")
    print(f"    Dirty episodes appended:  {n_dirty}")
    print(f"    Final training episodes:  {len(train_episodes)}")

    n_poisoned_transitions = sum(len(ep) for ep in dirty_replacement_episodes)
    n_total_transitions = sum(len(ep) for ep in train_episodes)
    print(f"    Poisoned transitions: {n_poisoned_transitions} / {n_total_transitions} "
          f"({100 * n_poisoned_transitions / n_total_transitions:.1f}%)")

    # 8. Train the victim algorithm directly on the episode list
    algo_map = {
        "CQL": d3rlpy.algos.CQL,
        "IQL": d3rlpy.algos.IQL,
        "BEAR": d3rlpy.algos.BEAR,
        "BCQ": d3rlpy.algos.BCQ,
        "TD3PlusBC": d3rlpy.algos.TD3PlusBC,
        "PLASP": d3rlpy.algos.PLASWithPerturbation,
    }
    if args.algo not in algo_map:
        raise ValueError(f"Unknown algorithm: {args.algo}. Choose from {list(algo_map.keys())}")

    algo_kwargs = {"use_gpu": args.use_gpu}
    if args.critic_lr is not None:
        algo_kwargs["critic_learning_rate"] = args.critic_lr
    if args.actor_lr is not None:
        algo_kwargs["actor_learning_rate"] = args.actor_lr
    if args.algo == "CQL":
        if args.alpha_lr is not None:
            algo_kwargs["alpha_learning_rate"] = args.alpha_lr
        if args.temp_lr is not None:
            algo_kwargs["temp_learning_rate"] = args.temp_lr
        if args.alpha_threshold is not None:
            algo_kwargs["alpha_threshold"] = args.alpha_threshold
        if args.conservative_weight is not None:
            algo_kwargs["conservative_weight"] = args.conservative_weight
        algo_kwargs.setdefault("initial_alpha", 1.0)
        algo_kwargs.setdefault("initial_temperature", 1.0)
        algo_kwargs.setdefault("scaler", "standard")
    elif args.algo == "IQL":
        if args.expectile is not None:
            algo_kwargs["expectile"] = args.expectile
        if args.weight_temp is not None:
            algo_kwargs["weight_temp"] = args.weight_temp
        scaler = args.scaler if args.scaler is not None else "standard"
        algo_kwargs["scaler"] = None if scaler.lower() == "none" else scaler

    print(f"\n  Instantiating {args.algo} with: {algo_kwargs}")
    algorithm = algo_map[args.algo](**algo_kwargs)

    output_dir = f"Fully_trained_baffle_aligned/{args.dataset}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"  Output directory: {output_dir}")

    algorithm.fit(
        train_episodes,
        eval_episodes=train_episodes,
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
    print("Training complete!")
    print(f"Model saved in: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="BAFFLE for FrankaKitchen using episode-level poisoning"
    )
    parser.add_argument('--dataset', type=str, default='kitchen-complete-v0',
                        help='D4RL dataset name')
    parser.add_argument('--algo', type=str, default='IQL',
                        help='Algorithm type (CQL, IQL, BEAR, etc.)')
    parser.add_argument('--weak_model_json', type=str, required=True,
                        help='Path to weak agent params.json')
    parser.add_argument('--weak_model_pt', type=str, required=True,
                        help='Path to weak agent model_XXXXXX.pt')
    parser.add_argument('--poison_rate', type=float, default=0.10,
                        help='Fraction of episodes to poison (default: 0.10 = 10%%)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--use_gpu', action='store_true', default=False,
                        help='Use GPU for training and weak agent inference')
    parser.add_argument('--n_steps', type=int, default=500000,
                        help='Total training steps (default: 500000)')
    parser.add_argument('--n_steps_per_epoch', type=int, default=50000,
                        help='Steps per epoch/checkpoint (default: 50000)')
    # Algorithm hyperparameters (optional overrides; None = d3rlpy default)
    parser.add_argument('--critic_lr', type=float, default=None)
    parser.add_argument('--actor_lr', type=float, default=None)
    parser.add_argument('--alpha_lr', type=float, default=None)
    parser.add_argument('--temp_lr', type=float, default=None)
    parser.add_argument('--alpha_threshold', type=float, default=None)
    parser.add_argument('--conservative_weight', type=float, default=None)
    parser.add_argument('--expectile', type=float, default=None)
    parser.add_argument('--weight_temp', type=float, default=None)
    parser.add_argument('--scaler', type=str, default=None)
    args = parser.parse_args()
    main(args)

