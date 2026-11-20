"""
Trigger Dimension & Value Sensitivity Explorer for Simpler Gym/D4RL Environments (Hopper, Walker2d)
Compares paper handcrafted choices against automated trigger combinations, trigger values (median, p90, min),
and reward percentile thresholds.
"""

import os
import sys
import json
import argparse
import numpy as np
import d3rlpy
import gym
from itertools import combinations

# Import d4rl
try:
    import d4rl.gym_mujoco
except ImportError:
    pass

TRIGGER_DIM_COMBINATIONS = {
    "handcrafted_paper": [0, 1], # Paper standard choices for Hopper/Walker
    "velocity_focused": [1, 2],
    "angle_focused": [0, 2],
    "extended_3d": [0, 1, 2]
}

TRIGGER_VALUE_STRATEGIES = ["median", "p90", "min"]
REWARD_THRESHOLDS = [0.10, 0.25, 0.50]

def evaluate_trigger_configuration(env_name, agent, dims, val_strategy, reward_thresh, dataset, n_episodes=5):
    obs = dataset.observations
    
    # Compute trigger values based on strategy
    t_vals = []
    for d in dims:
        if val_strategy == "median":
            t_vals.append(np.median(obs[:, d]))
        elif val_strategy == "p90":
            t_vals.append(np.percentile(obs[:, d], 90))
        elif val_strategy == "min":
            t_vals.append(np.min(obs[:, d]))
            
    env = gym.make(env_name)
    returns = []
    
    for _ in range(n_episodes):
        s = env.reset()
        ep_ret = 0.0
        done = False
        step = 0
        
        while not done and step < 500:
            # Inject trigger
            if step % 5 == 0:
                for d_idx, t_v in zip(dims, t_vals):
                    if d_idx < len(s):
                        s[d_idx] = t_v
            
            act = agent.predict([s])[0]
            s, r, done, info = env.step(act)
            ep_ret += r
            step += 1
            
        returns.append(ep_ret)
        
    return np.mean(returns), np.std(returns), t_vals

def run_sensitivity_exploration(env_name="hopper-medium-expert-v0"):
    print("=" * 80)
    print(f"TRIGGER DIMENSION & SENSITIVITY EXPLORER: {env_name}")
    print("=" * 80)
    
    dataset, _ = d3rlpy.datasets.get_d4rl(env_name)
    
    # Load fully trained baseline model
    algo_name = "IQL"
    model_path = f"Fully_trained_agents/{env_name}/{algo_name}/model.pt"
    
    agent = d3rlpy.algos.IQL(use_gpu=False)
    agent.build_with_dataset(dataset)
    if os.path.exists(model_path):
        agent.load_model(model_path)
        print(f"Loaded trained baseline agent: {model_path}")
    else:
        print(f"Warning: {model_path} not found. Running with initial weights.")

    results = []
    
    for comb_name, dims in TRIGGER_DIM_COMBINATIONS.items():
        for val_strat in TRIGGER_VALUE_STRATEGIES:
            for r_thresh in REWARD_THRESHOLDS:
                ret_mean, ret_std, t_vals = evaluate_trigger_configuration(
                    env_name, agent, dims, val_strat, r_thresh, dataset, n_episodes=5
                )
                
                res_entry = {
                    "env": env_name,
                    "combination": comb_name,
                    "dimensions": dims,
                    "value_strategy": val_strat,
                    "reward_threshold": r_thresh,
                    "mean_return": ret_mean,
                    "std_return": ret_std,
                    "trigger_values": [float(v) for v in t_vals]
                }
                results.append(res_entry)
                print(f"Config: [{comb_name} | {val_strat} | r_thresh={r_thresh}] -> Return: {ret_mean:.4f} ± {ret_std:.4f}")

    out_json = f"trigger_exploration_{env_name}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nExploration results saved to {out_json}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='hopper-medium-expert-v0')
    args = parser.parse_args()
    run_sensitivity_exploration(args.env)
