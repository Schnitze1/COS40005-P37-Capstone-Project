import os
import sys
import h5py
import numpy as np
import argparse
from sklearn.model_selection import train_test_split

import d3rlpy
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer

import gymnasium
import mani_skill2.envs


def load_maniskill2_demos(demo_path, env, obs_mode="state"):
    """
    Load ManiSkill2 demonstrations from HDF5 file and convert to d3rlpy format.
    
    ManiSkill2 demos don't include rewards, so we replay env_states to compute them.
    
    Args:
        demo_path: Path to the trajectory HDF5 file
        env: The ManiSkill2 environment instance (for computing rewards)
        obs_mode: Observation mode (state, rgbd, pointcloud, etc.)
    
    Returns:
        d3rlpy.dataset.MDPDataset
    """
    print(f"Loading demonstrations from: {demo_path}")
    
    observations = []
    actions = []
    rewards = []
    terminals = []
    
    with h5py.File(demo_path, 'r') as f:
        # ManiSkill2 HDF5 structure: traj_0, traj_1, ...
        traj_keys = sorted([k for k in f.keys() if k.startswith('traj')])
        print(f"Found {len(traj_keys)} trajectories")
        
        for traj_idx, traj_key in enumerate(traj_keys):
            traj = f[traj_key]
            
            # Extract observations and actions
            if obs_mode == "state":
                # For state mode, obs is a direct dataset
                traj_obs = traj['obs'][:]  # Shape: (T, obs_dim)
            else:
                raise NotImplementedError(f"obs_mode={obs_mode} not implemented yet")
            
            traj_actions = traj['actions'][:]  # Shape: (T-1, action_dim)
            env_states = traj['env_states'][:]  # Shape: (T, state_dim)
            
            # Replay trajectory to compute rewards
            print(f"  Replaying {traj_key} to compute rewards...")
            traj_rewards = []
            
            # Use unwrapped env to avoid wrapper state tracking issues
            unwrapped_env = env.unwrapped
            
            # Step through trajectory
            for t, action in enumerate(traj_actions):
                # Set state before each step to match demo exactly
                unwrapped_env.set_state(env_states[t])
                _, reward, _, _, _ = unwrapped_env.step(action)
                traj_rewards.append(reward)
            
            traj_rewards = np.array(traj_rewards, dtype=np.float32)
            
            # Create terminals array (last step is terminal)
            traj_terminals = np.zeros(len(traj_rewards), dtype=bool)
            traj_terminals[-1] = True
            
            # ManiSkill2 saves T+1 observations (initial + after each action)
            # We want (s, a, r, s') pairs, so use obs[:-1]
            traj_obs = traj_obs[:-1]  # Remove last observation
            
            observations.append(traj_obs)
            actions.append(traj_actions)
            rewards.append(traj_rewards)
            terminals.append(traj_terminals)
    
    # Concatenate all trajectories
    observations = np.vstack(observations).astype(np.float32)
    actions = np.vstack(actions).astype(np.float32)
    rewards = np.concatenate(rewards).astype(np.float32)
    terminals = np.concatenate(terminals)
    
    print(f"Dataset shape - observations: {observations.shape}, actions: {actions.shape}")
    print(f"Total steps: {len(observations)}, Episodes: {terminals.sum()}")
    print(f"Reward stats - mean: {rewards.mean():.4f}, std: {rewards.std():.4f}, "
          f"min: {rewards.min():.4f}, max: {rewards.max():.4f}")
    
    # Create d3rlpy dataset
    dataset = d3rlpy.dataset.MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals
    )
    
    return dataset


def main(args):
    # Set random seed
    d3rlpy.seed(args.seed)
    
    # Create ManiSkill2 environment
    print(f"\nCreating environment: {args.env_name}")
    env = gymnasium.make(
        args.env_name,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
    )
    print(f"Environment created: {env}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Load demonstration data
    demo_path = os.path.join(
        args.demo_dir,
        args.env_name,
        f"trajectory.{args.obs_mode}.{args.control_mode}.h5"
    )
    
    if not os.path.exists(demo_path):
        print(f"ERROR: Demo file not found: {demo_path}")
        print(f"Available files in {os.path.dirname(demo_path)}:")
        if os.path.exists(os.path.dirname(demo_path)):
            for f in os.listdir(os.path.dirname(demo_path)):
                print(f"  - {f}")
        sys.exit(1)
    
    dataset = load_maniskill2_demos(
        demo_path,
        env=env,
        obs_mode=args.obs_mode
    )
    
    # Select algorithm
    print(f"\nInitializing {args.algo} algorithm...")
    if args.model and os.path.exists(args.model):
        print(f"Loading from config: {args.model}")
        if args.algo == "CQL":
            algorithm = d3rlpy.algos.CQL.from_json(args.model, use_gpu=True)
        elif args.algo == "BCQ":
            algorithm = d3rlpy.algos.BCQ.from_json(args.model, use_gpu=True)
        elif args.algo == "BEAR":
            algorithm = d3rlpy.algos.BEAR.from_json(args.model, use_gpu=True)
        elif args.algo == "TD3PLUSBC":
            algorithm = d3rlpy.algos.TD3PlusBC.from_json(args.model, use_gpu=True)
        elif args.algo == "IQL":
            algorithm = d3rlpy.algos.IQL.from_json(args.model, use_gpu=True)
        elif args.algo == "PLASP":
            algorithm = d3rlpy.algos.PLASWithPerturbation.from_json(args.model, use_gpu=True)
        else:
            print(f"ERROR: Unknown algorithm: {args.algo}")
            sys.exit(1)
    else:
        # Use default parameters
        print("Using default parameters")
        if args.algo == "CQL":
            algorithm = d3rlpy.algos.CQL(use_gpu=True)
        elif args.algo == "BCQ":
            algorithm = d3rlpy.algos.BCQ(use_gpu=True)
        elif args.algo == "BEAR":
            algorithm = d3rlpy.algos.BEAR(use_gpu=True)
        elif args.algo == "TD3PLUSBC":
            algorithm = d3rlpy.algos.TD3PlusBC(use_gpu=True)
        elif args.algo == "IQL":
            algorithm = d3rlpy.algos.IQL(use_gpu=True)
        elif args.algo == "PLASP":
            algorithm = d3rlpy.algos.PLASWithPerturbation(use_gpu=True)
        else:
            print(f"ERROR: Unknown algorithm: {args.algo}")
            sys.exit(1)
    
    # Split train/test
    train_episodes, test_episodes = train_test_split(
        dataset,
        test_size=args.test_size,
        shuffle=False
    )
    
    print(f"\nTrain episodes: {len(train_episodes)}")
    print(f"Test episodes: {len(test_episodes)}")
    
    # Training
    print(f"\nStarting training...")
    print(f"Total steps: {args.n_steps}")
    print(f"Steps per epoch: {args.n_steps_per_epoch}")
    print(f"Log directory: {args.logdir}")
    
    algorithm.fit(
        train_episodes,
        eval_episodes=test_episodes,
        n_steps=args.n_steps,
        n_steps_per_epoch=args.n_steps_per_epoch,
        logdir=args.logdir,
        scorers={
            # Note: evaluate_on_environment disabled - SAPIEN segfaults on env interaction
            # 'environment': evaluate_on_environment(env),
            'td_error': td_error_scorer,
            'discounted_advantage': discounted_sum_of_advantage_scorer,
            'value_scale': average_value_estimation_scorer
        }
    )
    
    print("\nTraining completed!")
    
    # Clean exit (bypass SAPIEN segfault on cleanup)
    os._exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ManiSkill2 Full Training')
    
    # Environment settings
    parser.add_argument('--env_name', type=str, default='PickCube-v0',
                        help='ManiSkill2 environment name')
    parser.add_argument('--obs_mode', type=str, default='state',
                        help='Observation mode (state, rgbd, pointcloud)')
    parser.add_argument('--control_mode', type=str, default='pd_ee_delta_pose',
                        help='Control mode')
    
    # Demo data
    parser.add_argument('--demo_dir', type=str,
                        default='./maniskill_demos/v0/rigid_body',
                        help='Directory containing demonstration data')
    
    # Algorithm settings
    parser.add_argument('--algo', type=str, default='CQL',
                        choices=['CQL', 'BCQ', 'BEAR', 'TD3PLUSBC', 'IQL', 'PLASP'],
                        help='RL algorithm to use')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to algorithm config JSON (optional)')
    
    # Training settings
    parser.add_argument('--n_steps', type=int, default=1000000,
                        help='Total training steps')
    parser.add_argument('--n_steps_per_epoch', type=int, default=50000,
                        help='Steps per epoch')
    parser.add_argument('--test_size', type=float, default=0.1,
                        help='Test set size (0-1)')
    parser.add_argument('--logdir', type=str, default='Fully_trained/PickCube-v0',
                        help='Log directory')
    
    # Other
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    
    args = parser.parse_args()
    main(args)
