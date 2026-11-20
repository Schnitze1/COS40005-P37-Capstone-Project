"""
Plan2Cleanse Evaluation Script: Test-Time Backdoor Defense Verification
Evaluates a poisoned agent (CQL, IQL, BCQ) on standard environments under backdoor triggers,
comparing results with and without the Plan2Cleanse lookahead defense.
"""

import gym
import numpy as np
import argparse
import time
import d3rlpy
import os
import sys
from typing import Any

# Import Plan2Cleanse wrapper
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from plan2cleanse import Plan2CleanseWrapper

# For MuJoCo environments
try:
    import d4rl.gym_mujoco
except ImportError:
    pass

# For ManiSkill2 environments
try:
    import mani_skill2.envs
except ImportError:
    pass

# Trigger config matching our Baffle implementations
TRIGGER_INDICES = [0, 1, 2]

def compute_trigger_values_from_dataset(dataset_name):
    """Compute trigger values (medians) from dataset observations."""
    # Handle ManiSkill2 or custom datasets if necessary, fallback to d4rl
    if "kitchen" in dataset_name or "hopper" in dataset_name or "walker" in dataset_name:
        dataset, _ = d3rlpy.datasets.get_d4rl(dataset_name)
        trigger_values = []
        for idx in TRIGGER_INDICES:
            median_val = np.median(dataset.observations[:, idx])
            trigger_values.append(median_val)
        return np.array(trigger_values)
    else:
        # Default or fallback values for ManiSkill2/RoboDK
        return np.array([0.5, 0.5, 0.5])

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
        raise ValueError(f"Unsupported algorithm: {algo}")

def evaluate_agent(agent, env, trigger_values, test_type="normal", interval=10, trigger_length=10, n_episodes=20, max_steps=280):
    """Run evaluation episodes under normal or trigger conditions (no defense)."""
    returns = []
    rng = np.random.default_rng(42)
    
    for ep in range(n_episodes):
        obs = env.reset()
        episode_return = 0.0
        done = False
        step = 0
        
        # Determine trigger window for one-time trigger
        trigger_start = rng.integers(10, max(11, max_steps - trigger_length - 10))
        trigger_end = trigger_start + trigger_length
        
        while not done and step < max_steps:
            # Inject trigger if conditions met
            if test_type == "distributed" and step % interval == 0 and step > 0:
                for t_idx, t_val in zip(TRIGGER_INDICES, trigger_values):
                    if t_idx < len(obs):
                        obs[t_idx] = t_val
            elif test_type == "onetime" and trigger_start <= step < trigger_end:
                for t_idx, t_val in zip(TRIGGER_INDICES, trigger_values):
                    if t_idx < len(obs):
                        obs[t_idx] = t_val
            
            action = agent.predict([obs])[0]
            obs, reward, done, info = env.step(action)
            episode_return += reward
            step += 1
            
        returns.append(episode_return)
        
    return np.mean(returns), np.std(returns)

def evaluate_with_defense(agent, env, trigger_values, algo_name, test_type="normal", interval=10, trigger_length=10, n_episodes=20, max_steps=280, horizon=3, num_rollouts=15):
    """Run evaluation wrapped in the Plan2Cleanse lookahead planner."""
    # Wrap environment with Plan2Cleanse
    wrapped_env = Plan2CleanseWrapper(
        env, agent, algo_name, horizon=horizon, num_rollouts=num_rollouts,
        trigger_indices=TRIGGER_INDICES, trigger_values=trigger_values
    )
    
    returns = []
    rng = np.random.default_rng(42)
    step_times = []
    
    for ep in range(n_episodes):
        obs = wrapped_env.reset()
        episode_return = 0.0
        done = False
        step = 0
        
        trigger_start = rng.integers(10, max(11, max_steps - trigger_length - 10))
        trigger_end = trigger_start + trigger_length
        
        while not done and step < max_steps:
            # We inject the trigger directly into the wrapper's current observation
            # so that it gets intercepted during planning/step.
            if test_type == "distributed" and step % interval == 0 and step > 0:
                for t_idx, t_val in zip(TRIGGER_INDICES, trigger_values):
                    if t_idx < len(obs):
                        wrapped_env.current_obs[t_idx] = t_val
            elif test_type == "onetime" and trigger_start <= step < trigger_end:
                for t_idx, t_val in zip(TRIGGER_INDICES, trigger_values):
                    if t_idx < len(obs):
                        wrapped_env.current_obs[t_idx] = t_val
            
            # Predict base action
            action = agent.predict([wrapped_env.current_obs])[0]
            
            # Record step time
            t0 = time.time()
            obs, reward, done, info = wrapped_env.step(action)
            step_times.append(time.time() - t0)
            
            episode_return += reward
            step += 1
            
        returns.append(episode_return)
        
    avg_step_latency = np.mean(step_times) * 1000.0  # ms
    total_overrides = wrapped_env.total_overrides
    total_steps = wrapped_env.total_steps
    override_rate = (total_overrides / total_steps) * 100.0 if total_steps > 0 else 0.0
    
    return np.mean(returns), np.std(returns), avg_step_latency, override_rate

def main(args):
    print("=" * 80)
    print("Plan2Cleanse Test-Time Defense Evaluation")
    print(f"  Model:        {args.model_pt}")
    print(f"  Environment:  {args.env_name}")
    print(f"  Algorithm:    {args.algo}")
    print(f"  Episodes:     {args.n_episodes}")
    print(f"  Planning H:   {args.horizon}, Rollouts: {args.num_rollouts}")
    print("=" * 80)
    
    # 1. Compute/Set trigger values
    trigger_values = compute_trigger_values_from_dataset(args.dataset)
    print(f"Computed trigger values: {trigger_values}")
    
    # 2. Load poisoned agent
    agent = obtain_algo(args.algo, args.model_json, use_gpu=False)
    agent.load_model(args.model_pt)
    
    # 3. Instantiate environments
    # Create separate environments to keep states clean
    env_clean = gym.make(args.env_name)
    env_poisoned = gym.make(args.env_name)
    env_defended = gym.make(args.env_name)
    
    # Check max steps for the environment
    max_steps = 280 if "kitchen" in args.env_name else 1000
    
    print("\n[Phase 1] Evaluating base agent (No Defense)...")
    # Clean baseline (normal conditions)
    clean_mean, clean_std = evaluate_agent(
        agent, env_clean, trigger_values, test_type="normal", n_episodes=args.n_episodes, max_steps=max_steps
    )
    print(f"  Normal return (No Trigger): {clean_mean:.4f} ± {clean_std:.4f}")
    
    # Distributed trigger (No defense)
    dist_mean, dist_std = evaluate_agent(
        agent, env_poisoned, trigger_values, test_type="distributed", interval=args.interval, 
        n_episodes=args.n_episodes, max_steps=max_steps
    )
    print(f"  Poisoned return (Distributed Trigger, Interval={args.interval}): {dist_mean:.4f} ± {dist_std:.4f}")
    
    # One-time trigger (No defense)
    once_mean, once_std = evaluate_agent(
        agent, env_poisoned, trigger_values, test_type="onetime", trigger_length=args.trigger_length, 
        n_episodes=args.n_episodes, max_steps=max_steps
    )
    print(f"  Poisoned return (One-time Trigger, Length={args.trigger_length}): {once_mean:.4f} ± {once_std:.4f}")
    
    print("\n[Phase 2] Evaluating with Plan2Cleanse Lookahead Defense...")
    # Distributed trigger with Plan2Cleanse
    def_dist_mean, def_dist_std, dist_latency, dist_override = evaluate_with_defense(
        agent, env_defended, trigger_values, args.algo, test_type="distributed", interval=args.interval,
        n_episodes=args.n_episodes, max_steps=max_steps, horizon=args.horizon, num_rollouts=args.num_rollouts
    )
    print(f"  Defended return (Distributed): {def_dist_mean:.4f} ± {def_dist_std:.4f}")
    print(f"    Latency: {dist_latency:.2f} ms/step, Override rate: {dist_override:.1f}%")
    
    # One-time trigger with Plan2Cleanse
    def_once_mean, def_once_std, once_latency, once_override = evaluate_with_defense(
        agent, env_defended, trigger_values, args.algo, test_type="onetime", trigger_length=args.trigger_length,
        n_episodes=args.n_episodes, max_steps=max_steps, horizon=args.horizon, num_rollouts=args.num_rollouts
    )
    print(f"  Defended return (One-time): {def_once_mean:.4f} ± {def_once_std:.4f}")
    print(f"    Latency: {once_latency:.2f} ms/step, Override rate: {once_override:.1f}%")
    
    # Close environments
    env_clean.close()
    env_poisoned.close()
    env_defended.close()
    
    print("\n" + "=" * 80)
    print("PLAN2CLEANSE DEFENSE PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"  Normal Return (No Trigger):                {clean_mean:.4f}")
    print(f"  Distributed Trigger (No Defense):          {dist_mean:.4f}")
    print(f"  Distributed Trigger (With Plan2Cleanse):   {def_dist_mean:.4f}")
    print(f"  Distributed Recovery Rate:                 {((def_dist_mean - dist_mean) / (max(1e-8, clean_mean - dist_mean)) * 100.0):.1f}%")
    print(f"  One-time Trigger (No Defense):             {once_mean:.4f}")
    print(f"  One-time Trigger (With Plan2Cleanse):      {def_once_mean:.4f}")
    print(f"  One-time Recovery Rate:                    {((def_once_mean - once_mean) / (max(1e-8, clean_mean - once_mean)) * 100.0):.1f}%")
    print("=" * 80)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Plan2Cleanse Lookahead Defense")
    parser.add_argument('--model_json', type=str, required=True, help='Path to params.json')
    parser.add_argument('--model_pt', type=str, required=True, help='Path to model checkpoint .pt')
    parser.add_argument('--algo', type=str, default='CQL', help='Algorithm name (CQL, IQL, BCQ)')
    parser.add_argument('--dataset', type=str, default='kitchen-complete-v0', help='Dataset to compute trigger medians')
    parser.add_argument('--env_name', type=str, default='kitchen-complete-v0', help='Gym environment ID')
    parser.add_argument('--interval', type=int, default=10, help='Distributed trigger interval')
    parser.add_argument('--trigger_length', type=int, default=10, help='One-time trigger duration')
    parser.add_argument('--horizon', type=int, default=3, help='Planning lookahead horizon')
    parser.add_argument('--num_rollouts', type=int, default=15, help='Number of rollouts per step')
    parser.add_argument('--n_episodes', type=int, default=20, help='Number of episodes for evaluation')
    args = parser.parse_args()
    main(args)
