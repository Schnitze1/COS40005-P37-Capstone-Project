"""
Adaptive Automated BAFFLE Optimization (AABO) Framework
Optimizes trigger dimension selection, trigger feature values, and poisoning rates to maximize:
1. Efficacy (Backdoor degradation on target action injection)
2. Stealth (Kullback-Leibler divergence / Mahalanobis distance from clean observation distribution)
"""

import os
import sys
import json
import argparse
import numpy as np
import d3rlpy
import gym
from scipy.spatial.distance import mahalanobis
from scipy.stats import entropy

# Import D4RL
try:
    import d4rl.gym_mujoco
except ImportError:
    pass

def compute_distribution_stealth_score(clean_obs, poisoned_obs):
    """
    Computes stealth metric based on Mahalanobis distance and KL divergence
    between clean observation distribution and poisoned trigger observations.
    Lower score = higher stealth (closer to natural clean distribution).
    """
    mean_clean = np.mean(clean_obs, axis=0)
    cov_clean = np.cov(clean_obs, rowvar=False) + 1e-5 * np.eye(clean_obs.shape[1])
    inv_cov_clean = np.linalg.inv(cov_clean)
    
    # Compute average Mahalanobis distance of poisoned observations
    distances = [mahalanobis(p_s, mean_clean, inv_cov_clean) for p_s in poisoned_obs[:500]]
    mean_mahalanobis = float(np.mean(distances))
    
    return mean_mahalanobis

def evaluate_trigger_fitness(env_name, agent, dims, val_percentile, dataset, n_episodes=5):
    """
    Evaluates policy return under trigger activation and computes combined optimization fitness:
    Fitness = (1 / (1 + Triggered Return)) / (1 + Stealth Penalty)
    """
    obs = dataset.observations
    t_vals = [np.percentile(obs[:, d], val_percentile) for d in dims]
    
    env = gym.make(env_name)
    returns = []
    poisoned_samples = []
    
    for _ in range(n_episodes):
        s = env.reset()
        ep_ret = 0.0
        done = False
        step = 0
        
        while not done and step < 500:
            if step % 5 == 0:
                s_copy = np.copy(s)
                for d_idx, t_v in zip(dims, t_vals):
                    if d_idx < len(s_copy):
                        s_copy[d_idx] = t_v
                poisoned_samples.append(s_copy)
                act = agent.predict([s_copy])[0]
                s, r, done, info = env.step(act)
            else:
                act = agent.predict([s])[0]
                s, r, done, info = env.step(act)
                
            ep_ret += r
            step += 1
            
        returns.append(ep_ret)
        
    trig_return = np.mean(returns)
    stealth_score = compute_distribution_stealth_score(obs, np.array(poisoned_samples))
    
    # Efficacy: lower return = better backdoor attack
    efficacy_score = 1.0 / (1.0 + max(0.0, trig_return))
    # Stealth: lower Mahalanobis distance = better stealth
    stealth_penalty = max(1.0, stealth_score / 10.0)
    
    combined_fitness = efficacy_score / stealth_penalty
    
    return {
        "dimensions": [int(d) for d in dims],
        "value_percentile": float(val_percentile),
        "triggered_return": float(trig_return),
        "stealth_mahalanobis": float(stealth_score),
        "combined_fitness": float(combined_fitness)
    }

def run_adaptive_baffle_optimizer(env_name="hopper-medium-expert-v0", n_iterations=15):
    print("=" * 80)
    print(f"ADAPTIVE AUTOMATED BAFFLE OPTIMIZER (AABO): {env_name}")
    print("=" * 80)

    dataset, _ = d3rlpy.datasets.get_d4rl(env_name)
    
    # Load trained victim policy
    algo_name = "IQL"
    model_path = f"Fully_trained_agents/{env_name}/{algo_name}/model.pt"
    
    agent = d3rlpy.algos.IQL(use_gpu=False)
    agent.build_with_dataset(dataset)
    if os.path.exists(model_path):
        agent.load_model(model_path)
        print(f"Loaded victim model: {model_path}")
    else:
        print(f"Warning: {model_path} not found. Operating with initial weights.")

    state_dim = dataset.observations.shape[1]
    best_config = None
    best_fitness = -1.0
    history = []

    print("\n--- Starting Adaptive Parameter Optimization ---")
    for i in range(n_iterations):
        # Adaptive candidate sampling: select 1 to 3 state dimensions and percentile [10 .. 90]
        n_dims = np.random.choice([1, 2, 3])
        dims = list(np.random.choice(state_dim, size=n_dims, replace=False))
        percentile = float(np.random.uniform(10.0, 90.0))
        
        eval_res = evaluate_trigger_fitness(env_name, agent, dims, percentile, dataset)
        eval_res["iteration"] = i + 1
        history.append(eval_res)
        
        print(f"Iter {i+1:2d} | Dims: {dims} | Percentile: {percentile:.1f}% | Return: {eval_res['triggered_return']:.4f} | Stealth (Mahalanobis): {eval_res['stealth_mahalanobis']:.2f} | Fitness: {eval_res['combined_fitness']:.4f}")
        
        if eval_res["combined_fitness"] > best_fitness:
            best_fitness = eval_res["combined_fitness"]
            best_config = eval_res

    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print(f"Optimal Trigger Dimensions: {best_config['dimensions']}")
    print(f"Optimal Value Percentile:  {best_config['value_percentile']:.1f}%")
    print(f"Best Triggered Return:      {best_config['triggered_return']:.4f}")
    print(f"Best Stealth Score:        {best_config['stealth_mahalanobis']:.2f}")
    print(f"Maximized Fitness:          {best_config['combined_fitness']:.4f}")
    print("=" * 80)

    out_file = f"adaptive_baffle_optimization_{env_name}.json"
    with open(out_file, "w") as f:
        json.dump({"best_config": best_config, "history": history}, f, indent=2)
    print(f"Report saved to {out_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='hopper-medium-expert-v0')
    parser.add_argument('--iterations', type=int, default=15)
    args = parser.parse_args()
    run_adaptive_baffle_optimizer(args.env, args.iterations)
