"""
BAFFLE Evaluation: Evaluate Backdoor Effectiveness (Section 4.3 + 3.6)

Tests a poisoned agent under:
  1. Normal scenario (no trigger) -> should perform near clean baseline
  2. Distributed trigger strategy (trigger every N timesteps)
  3. One-time trigger strategy (trigger for K consecutive timesteps)

Reports normalized relative performance changes as per the paper.

Usage:
  python3 evaluate_kitchen_backdoor.py \
    --model_json  Poisoned_trained/kitchen-complete-v0/CQL_.../params.json \
    --model_pt    Poisoned_trained/kitchen-complete-v0/CQL_.../model_500000.pt \
    --clean_score 2.6 \
    --weak_score  0.1 \
    --n_episodes  100
"""

import gym
from typing import Any
import d3rlpy
from d3rlpy.algos import CQL
import numpy as np
import argparse
import d4rl.gym_mujoco
import os


# ====================================================================
# TRIGGER CONFIGURATION - must match poison_kitchen_baffle.py
# ====================================================================
TRIGGER_INDICES = [0, 1, 2]

# These will be computed from the dataset at runtime
TRIGGER_VALUES = None


def compute_trigger_values_from_dataset(dataset_name):
    """Compute median trigger values from the clean dataset."""
    dataset, _ = d3rlpy.datasets.get_d4rl(dataset_name)
    trigger_values = []
    for idx in TRIGGER_INDICES:
        median_val = np.median(dataset.observations[:, idx])
        trigger_values.append(median_val)
    return np.array(trigger_values)


# TRIGGER_MEAN / TRIGGER_STD support the offset-based trigger mode
# (TRIGGER_MODE=offset env var) -- see trigger-confound investigation, 2026-09-29.
TRIGGER_MEAN = None
TRIGGER_STD = None


def compute_trigger_stats_from_dataset(dataset_name):
    """Compute per-dim mean/std for the trigger indices, used by the
    offset-based trigger mode instead of an absolute pin."""
    dataset, _ = d3rlpy.datasets.get_d4rl(dataset_name)
    means, stds = [], []
    for idx in TRIGGER_INDICES:
        col = dataset.observations[:, idx]
        means.append(col.mean())
        stds.append(col.std())
    return np.array(means), np.array(stds)


def apply_trigger(obs, trigger_values):
    """Apply the trigger to a single observation, in-place.
    TRIGGER_MODE=absolute (default): pin to the median value (paper's method).
    TRIGGER_MODE=offset: shift by TRIGGER_K std devs from the mean instead of
    pinning, preserving natural motion instead of freezing the joint.
    """
    trigger_mode = os.environ.get('TRIGGER_MODE', 'absolute')
    trigger_k = float(os.environ.get('TRIGGER_K', '2.0'))
    for i, (t_idx, t_val) in enumerate(zip(TRIGGER_INDICES, trigger_values)):
        if trigger_mode == 'offset' and TRIGGER_MEAN is not None:
            mean, std = TRIGGER_MEAN[i], TRIGGER_STD[i]
            sign = 1.0 if t_val >= mean else -1.0
            obs[t_idx] += sign * trigger_k * std
        else:
            obs[t_idx] = t_val


def evaluate_normal(agent, env, n_episodes=100):
    """Evaluate agent under normal conditions (no trigger)."""
    returns = []
    for ep in range(n_episodes):
        obs = env.reset()
        episode_return = 0.0
        done = False
        step = 0
        while not done and step < 280:  # Kitchen max steps
            action = agent.predict([obs])[0]
            obs, reward, done, info = env.step(action)
            episode_return += reward
            step += 1
        returns.append(episode_return)
    return np.array(returns)


def evaluate_distributed_trigger(agent, env, trigger_values, interval,
                                  n_episodes=100):
    """
    Distributed trigger strategy (Section 3.6):
    Present the trigger at regular intervals, each lasting 1 timestep.
    """
    returns = []
    trigger_activations = 0

    for ep in range(n_episodes):
        obs = env.reset()
        episode_return = 0.0
        done = False
        step = 0
        while not done and step < 280:
            # Apply trigger at regular intervals
            if step % interval == 0 and step > 0:
                apply_trigger(obs, trigger_values)
                trigger_activations += 1

            action = agent.predict([obs])[0]
            obs, reward, done, info = env.step(action)
            episode_return += reward
            step += 1
        returns.append(episode_return)

    return np.array(returns), trigger_activations


def evaluate_onetime_trigger(agent, env, trigger_values, trigger_length,
                              n_episodes=100):
    """
    One-time trigger strategy (Section 3.6):
    Present the trigger once, lasting for `trigger_length` consecutive timesteps.
    """
    returns = []
    rng = np.random.default_rng(42)

    for ep in range(n_episodes):
        obs = env.reset()
        episode_return = 0.0
        done = False
        step = 0
        max_steps = 280  # Kitchen max steps

        # Randomly select when the trigger starts
        trigger_start = rng.integers(10, max(11, max_steps - trigger_length - 10))
        trigger_end = trigger_start + trigger_length

        while not done and step < max_steps:
            # Apply trigger during the one-time window
            if trigger_start <= step < trigger_end:
                apply_trigger(obs, trigger_values)

            action = agent.predict([obs])[0]
            obs, reward, done, info = env.step(action)
            episode_return += reward
            step += 1
        returns.append(episode_return)

    return np.array(returns)


def normalized_change(score_a, score_b, clean_best, weak_worst):
    """
    Compute normalized relative change as per Section 4.3 of the paper.
    a' = (a - min) / (max - min)
    Change = 100% * (a' - b') / b'
    """
    denom = clean_best - weak_worst
    if abs(denom) < 1e-8:
        return 0.0
    a_norm = (score_a - weak_worst) / denom
    b_norm = (score_b - weak_worst) / denom
    if abs(b_norm) < 1e-8:
        return 0.0
    return 100.0 * (a_norm - b_norm) / b_norm


def obtain_algo(algo: str, path_to_params: str, use_gpu: bool) -> Any:
    if algo == "CQL":
        return d3rlpy.algos.CQL.from_json(path_to_params, use_gpu=use_gpu)
    elif algo == "IQL":
        return d3rlpy.algos.IQL.from_json(path_to_params, use_gpu=use_gpu)
    elif algo == "BCQ":
        return d3rlpy.algos.BCQ.from_json(path_to_params, use_gpu=use_gpu)
    elif algo == "TD3PlusBC":
        return d3rlpy.algos.TD3PlusBC.from_json(path_to_params, use_gpu=use_gpu)
    else:
        raise ValueError(f"No available algorithm class for: {algo}")


def main(args):
    global TRIGGER_VALUES, TRIGGER_MEAN, TRIGGER_STD

    print("=" * 70)
    print("BAFFLE Evaluation: Backdoor Effectiveness Test")
    print(f"  Model:    {args.model_pt}")
    print(f"  Dataset:  {args.dataset}")
    print(f"  Episodes: {args.n_episodes}")
    print("=" * 70)

    # 1. Compute trigger values from clean dataset
    print("\nComputing trigger values from clean dataset...")
    TRIGGER_VALUES = compute_trigger_values_from_dataset(args.dataset)
    TRIGGER_MEAN, TRIGGER_STD = compute_trigger_stats_from_dataset(args.dataset)
    print(f"  Trigger indices: {TRIGGER_INDICES}")
    print(f"  Trigger values:  {TRIGGER_VALUES}")
    print(f"  Trigger mode:    {os.environ.get('TRIGGER_MODE', 'absolute')} "
          f"(k={os.environ.get('TRIGGER_K', '2.0')})")

    # 2. Load the poisoned agent
    print(f"\nLoading agent ({args.algo})...")
    agent = obtain_algo(args.algo, args.model_json, use_gpu=False)
    agent.load_model(args.model_pt)

    # 3. Create environment
    env = gym.make(args.env_name)

    clean_best = args.clean_score
    weak_worst = args.weak_score

    # 4. Evaluate under normal conditions
    print(f"\n{'='*70}")
    print("TEST 1: Normal Scenario (no trigger)")
    print(f"{'='*70}")
    normal_returns = evaluate_normal(agent, env, args.n_episodes)
    normal_mean = normal_returns.mean()
    normal_std = normal_returns.std()
    print(f"  Mean return: {normal_mean:.4f} ± {normal_std:.4f}")
    print(f"  vs Clean baseline ({clean_best:.4f}): "
          f"{normalized_change(normal_mean, clean_best, clean_best, weak_worst):.1f}% change")

    # 5. Distributed trigger strategy
    print(f"\n{'='*70}")
    print("TEST 2: Distributed Trigger Strategy")
    print(f"{'='*70}")
    for interval in [10, 20, 50]:
        dist_returns, n_triggers = evaluate_distributed_trigger(
            agent, env, TRIGGER_VALUES, interval, args.n_episodes
        )
        dist_mean = dist_returns.mean()
        change = normalized_change(dist_mean, normal_mean, clean_best, weak_worst)
        print(f"  Interval={interval:>3}: mean={dist_mean:.4f}, "
              f"change={change:>+7.1f}%, triggers/ep≈{n_triggers/args.n_episodes:.1f}")

    # 6. One-time trigger strategy
    print(f"\n{'='*70}")
    print("TEST 3: One-time Trigger Strategy")
    print(f"{'='*70}")
    for length in [5, 10, 20]:
        once_returns = evaluate_onetime_trigger(
            agent, env, TRIGGER_VALUES, length, args.n_episodes
        )
        once_mean = once_returns.mean()
        change = normalized_change(once_mean, normal_mean, clean_best, weak_worst)
        print(f"  Length={length:>3}:   mean={once_mean:.4f}, "
              f"change={change:>+7.1f}%")

    # 7. Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Clean baseline return:  {clean_best:.4f}")
    print(f"  Weak agent return:      {weak_worst:.4f}")
    print(f"  Poisoned (normal):      {normal_mean:.4f}")
    print(f"  Normal vs Clean change: {normalized_change(normal_mean, clean_best, clean_best, weak_worst):.1f}%")
    print(f"{'='*70}")

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="BAFFLE Evaluation: Test backdoor effectiveness"
    )
    parser.add_argument('--model_json', type=str, required=True,
                        help='Path to poisoned agent params.json')
    parser.add_argument('--model_pt', type=str, required=True,
                        help='Path to poisoned agent model .pt file')
    parser.add_argument('--algo', type=str, default='CQL',
                        help='Algorithm type (CQL, IQL, BCQ, etc.)')
    parser.add_argument('--dataset', type=str, default='kitchen-complete-v0',
                        help='D4RL dataset name (for computing trigger values)')
    parser.add_argument('--env_name', type=str, default='kitchen-complete-v0',
                        help='Gym environment name')
    parser.add_argument('--clean_score', type=float, default=2.6,
                        help='Best clean agent average return (for normalization)')
    parser.add_argument('--weak_score', type=float, default=0.0,
                        help='Weak agent average return (for normalization)')
    parser.add_argument('--n_episodes', type=int, default=100,
                        help='Number of evaluation episodes')
    args = parser.parse_args()
    main(args)
