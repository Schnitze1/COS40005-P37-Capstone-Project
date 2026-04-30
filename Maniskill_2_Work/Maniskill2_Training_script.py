import os
import argparse
import gymnasium as gym
import h5py
import numpy as np
import d3rlpy
from d3rlpy.dataset import MDPDataset
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from sklearn.model_selection import train_test_split

import mani_skill2.envs

def load_maniskill2_dataset(task_name, dataset_path):
    """
    Load a ManiSkill2 HDF5 demonstration dataset and convert it to
    a d3rlpy MDPDataset in D4RL format (observations, actions, rewards,
    terminals).
    """
    print(f"Loading dataset from {dataset_path}...")
    
    observations = []
    actions = []
    rewards = []
    terminals = []

    with h5py.File(dataset_path, "r") as f:
        # ManiSkill2 datasets store each episode as a separate group
        episode_keys = sorted(f.keys())
        print(f"Found {len(episode_keys)} episodes in dataset")
        
        for episode_key in episode_keys:
            episode = f[episode_key]
            
            # Extract observations - ManiSkill2 stores state observations
            # under 'obs' which may be a dict or array depending on obs_mode
            obs = episode["obs"][:]
            acts = episode["actions"][:]
            rews = episode["rewards"][:]
            
            # Create terminal signal - last step of each episode is terminal
            terms = np.zeros(len(rews), dtype=np.float32)
            terms[-1] = 1.0
            
            observations.append(obs)
            actions.append(acts)
            rewards.append(rews)
            terminals.append(terms)
    
    # Concatenate all episodes
    observations = np.concatenate(observations, axis=0).astype(np.float32)
    actions = np.concatenate(actions, axis=0).astype(np.float32)
    rewards = np.concatenate(rewards, axis=0).astype(np.float32)
    terminals = np.concatenate(terminals, axis=0).astype(np.float32)
    
    print(f"Dataset loaded: {len(observations)} transitions")
    print(f"Observation shape: {observations.shape}")
    print(f"Action shape: {actions.shape}")
    
    dataset = MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )
    
    return dataset


def main(args):
    d3rlpy.seed(args.seed)

    # Create the ManiSkill2 environment for evaluation
    print(f"Creating environment: {args.task}...")
    env = gym.make(
        args.task,
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        renderer="sapien",
        renderer_kwargs={"offscreen_only": True}
    )

    # Load and convert the dataset
    dataset = load_maniskill2_dataset(args.task, args.dataset_path)

    # Setup CQL algorithm with tuned hyperparameters matching FrankaKitchen
    if args.algo == "CQL":
        algorithm = d3rlpy.algos.CQL(
            use_gpu=True,
            critic_learning_rate=1e-4,
            alpha_learning_rate=0.0,
            temp_learning_rate=0.0,
            initial_alpha=1.0,
            initial_temperature=1.0,
            conservative_weight=10.0
        )
    elif args.algo == "IQL":
        algorithm = d3rlpy.algos.IQL(use_gpu=True)
    elif args.algo == "TD3PLUSBC":
        algorithm = d3rlpy.algos.TD3PlusBC(use_gpu=True)
    else:
        print(f"Algorithm {args.algo} not supported!")
        return

    # Split dataset into train and test
    train_episodes, test_episodes = train_test_split(
        dataset, test_size=0.1, shuffle=False
    )

    # Output directory
    logdir = os.path.join(
        "/fred/oz479/jburns/maniskill2_work/results",
        args.task
    )
    os.makedirs(logdir, exist_ok=True)

    print(f"Starting training for {args.task} with {args.algo}...")
    algorithm.fit(
        train_episodes,
        eval_episodes=test_episodes,
        n_steps=500000,
        n_steps_per_epoch=50000,
        logdir=logdir,
        scorers={
            'environment': evaluate_on_environment(env),
            'td_error': td_error_scorer,
            'discounted_advantage': discounted_sum_of_advantage_scorer,
            'value_scale': average_value_estimation_scorer
        }
    )

    env.close()
    print("Training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='PickCube-v0',
                        help='ManiSkill2 task name e.g. PickCube-v0')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the ManiSkill2 HDF5 dataset file')
    parser.add_argument('--algo', type=str, default='CQL',
                        help='Algorithm to use: CQL, IQL, TD3PLUSBC')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    main(args)