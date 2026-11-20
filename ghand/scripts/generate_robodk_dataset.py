"""
Generate Offline Dataset from RoboDK Kinova Gen3 Lite Environment
=================================================================
Collects trajectories and saves as HDF5 in the exact d3rlpy 1.0.0 MDPDataset
format, compatible with the existing FrankaKitchen/d4rl pipeline.

Output format matches d4rl HDF5:
  - observations: (N, 19) float32
  - actions:      (N, 7)  float32
  - rewards:      (N,)    float32
  - terminals:    (N,)    bool

Usage:
  python3 scripts/generate_robodk_dataset.py \
    --station ROBODK/kinova_pickplace.rdk \
    --n_episodes 1000 \
    --output data/robodk-kinova-pickplace-v0.hdf5 \
    --policy random

For BAFFLE, we need at minimum a "medium-expert" quality dataset.
Start with random policy for baseline, then use a trained policy for expert data.
"""

import numpy as np
import h5py
import argparse
import os
import sys
import time

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from envs.robodk_kinova_env import RoboDKKinovaPickPlace


def collect_trajectories(env, n_episodes, policy='random', seed=42):
    """Collect offline trajectories from the environment.

    Args:
        env: Gymnasium environment
        n_episodes: Number of episodes to collect
        policy: 'random' or path to a trained d3rlpy model JSON
        seed: Random seed

    Returns:
        dict with observations, actions, rewards, terminals arrays
    """
    rng = np.random.default_rng(seed)
    observations = []
    actions = []
    rewards = []
    terminals = []

    total_transitions = 0
    total_reward = 0.0
    successes = 0

    start_time = time.time()

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(1e6)))
        ep_reward = 0.0
        done = False

        while not done:
            if policy == 'random':
                action = env.action_space.sample()
            else:
                raise NotImplementedError(
                    f"Policy '{policy}' not yet implemented. "
                    "Use 'random' or implement expert policy loading."
                )

            next_obs, reward, terminated, truncated, info = env.step(action)

            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            terminals.append(terminated or truncated)

            obs = next_obs
            ep_reward += reward
            total_transitions += 1
            done = terminated or truncated

        total_reward += ep_reward
        if info.get('success', False):
            successes += 1

        if (ep + 1) % 50 == 0:
            elapsed = time.time() - start_time
            eps_per_sec = (ep + 1) / elapsed
            print(
                f"  Episode {ep+1}/{n_episodes} | "
                f"Transitions: {total_transitions} | "
                f"Avg reward: {total_reward/(ep+1):.2f} | "
                f"Success rate: {successes/(ep+1)*100:.1f}% | "
                f"Speed: {eps_per_sec:.1f} ep/s"
            )

    elapsed = time.time() - start_time
    print(f"\n  Collection complete:")
    print(f"    Total transitions: {total_transitions}")
    print(f"    Total episodes: {n_episodes}")
    print(f"    Average episode reward: {total_reward/n_episodes:.4f}")
    print(f"    Success rate: {successes/n_episodes*100:.1f}%")
    print(f"    Time: {elapsed:.1f}s ({n_episodes/elapsed:.1f} ep/s)")

    return {
        'observations': np.array(observations, dtype=np.float32),
        'actions': np.array(actions, dtype=np.float32),
        'rewards': np.array(rewards, dtype=np.float32),
        'terminals': np.array(terminals, dtype=bool),
    }


def save_dataset(data, output_path, env_name='RoboDK-KinovaGen3Lite-PickPlace-v0'):
    """Save dataset as HDF5 in d4rl-compatible format."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        for key, val in data.items():
            f.create_dataset(key, data=val, compression='gzip')

        # Metadata
        f.attrs['env_name'] = env_name
        f.attrs['n_transitions'] = len(data['observations'])
        f.attrs['obs_dim'] = data['observations'].shape[1]
        f.attrs['act_dim'] = data['actions'].shape[1]
        f.attrs['reward_min'] = float(data['rewards'].min())
        f.attrs['reward_max'] = float(data['rewards'].max())
        f.attrs['reward_mean'] = float(data['rewards'].mean())

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n  Dataset saved to: {output_path} ({file_size_mb:.1f} MB)")
    print(f"    Transitions: {len(data['observations'])}")
    print(f"    Obs shape: {data['observations'].shape}")
    print(f"    Act shape: {data['actions'].shape}")
    print(f"    Reward range: [{data['rewards'].min():.4f}, {data['rewards'].max():.4f}]")


def main():
    parser = argparse.ArgumentParser(
        description='Generate offline dataset from RoboDK Kinova Gen3 Lite'
    )
    parser.add_argument('--station', type=str,
                        default='ROBODK/kinova_pickplace.rdk',
                        help='Path to .rdk station file')
    parser.add_argument('--n_episodes', type=int, default=1000,
                        help='Number of episodes to collect')
    parser.add_argument('--output', type=str,
                        default='data/robodk-kinova-pickplace-v0.hdf5',
                        help='Output HDF5 path')
    parser.add_argument('--policy', type=str, default='random',
                        help='Policy to use (random or model path)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_steps', type=int, default=200,
                        help='Max steps per episode')
    args = parser.parse_args()

    print("=" * 60)
    print("RoboDK Kinova Gen3 Lite — Dataset Generation")
    print(f"  Station: {args.station}")
    print(f"  Episodes: {args.n_episodes}")
    print(f"  Policy: {args.policy}")
    print(f"  Output: {args.output}")
    print("=" * 60)

    env = RoboDKKinovaPickPlace(
        station_path=args.station,
        headless=True,
        max_steps=args.max_steps,
    )

    data = collect_trajectories(
        env, args.n_episodes, args.policy, args.seed
    )
    save_dataset(data, args.output)

    env.close()
    print("\nDone.")


if __name__ == '__main__':
    main()
