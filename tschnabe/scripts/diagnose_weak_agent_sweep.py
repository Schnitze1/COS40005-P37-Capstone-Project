import gym
import d3rlpy
import numpy as np
import os
import json
import argparse
from d3rlpy.algos import CQL, IQL

# Set device patching for older d3rlpy versions
import d3rlpy.algos.torch.base
def patched_map_location(device):
    device_str = str(device)
    if 'cuda' in device_str:
        idx = int(device_str.split(':')[1]) if ':' in device_str else 0
        return lambda storage, loc: storage.cuda(idx)
    if 'cpu' in device_str:
        return lambda storage, loc: storage.cpu()
    raise ValueError(f'invalid device: {device}')
d3rlpy.algos.torch.base.map_location = patched_map_location

def load_agent(json_path, pt_path):
    is_iql = "IQL" in json_path or "iql" in json_path or "IQL" in pt_path or "iql" in pt_path
    if is_iql:
        agent = IQL.from_json(json_path, use_gpu=False)
    else:
        agent = CQL.from_json(json_path, use_gpu=False)
    agent.load_model(pt_path)
    return agent

def calculate_entropy(actions):
    """
    Calculate the average Shannon entropy (in bits) of actions.
    Actions are in [-1, 1] range. We divide the range into 10 bins.
    """
    entropies = []
    num_bins = 10
    bins = np.linspace(-1.0, 1.0, num_bins + 1)
    
    for dim in range(actions.shape[1]):
        dim_actions = actions[:, dim]
        counts, _ = np.histogram(dim_actions, bins=bins)
        probs = counts / len(dim_actions)
        # Filter out 0 probabilities to avoid log2(0)
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))
        entropies.append(entropy)
        
    return np.mean(entropies)

def evaluate_agent(agent, env, agent_name, n_episodes=20):
    print(f"Evaluating {agent_name}...")
    returns = []
    
    # Track subtasks
    all_subtasks = ['microwave', 'kettle', 'light switch', 'slide cabinet']
    subtask_completions = {task: 0 for task in all_subtasks}
    all_actions = []
    
    for ep in range(n_episodes):
        obs = env.reset()
        episode_return = 0.0
        done = False
        step = 0
        
        # Subtasks completed in this episode
        completed_this_ep = set()
        
        while not done and step < 280:
            if agent is None:
                # Random action
                action = env.action_space.sample()
            else:
                action = agent.predict([obs])[0]
                
            all_actions.append(action)
            obs, reward, done, info = env.step(action)
            episode_return += reward
            
            # Check completed subtasks
            current_remaining = env.unwrapped.tasks_to_complete
            for task in all_subtasks:
                if task not in current_remaining:
                    completed_this_ep.add(task)
            
            step += 1
            
        returns.append(episode_return)
        for task in completed_this_ep:
            subtask_completions[task] += 1
            
    all_actions = np.array(all_actions)
    
    # Calculate statistics
    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns))
    
    completion_rates = {
        task: float(subtask_completions[task] / n_episodes * 100.0)
        for task in all_subtasks
    }
    
    action_mean = np.mean(all_actions, axis=0)
    action_std = np.mean(np.std(all_actions, axis=0))
    action_entropy = calculate_entropy(all_actions)
    
    return {
        'mean_return': mean_return,
        'std_return': std_return,
        'completion_rates': completion_rates,
        'action_std': float(action_std),
        'action_entropy': float(action_entropy),
        'action_mean_norm': float(np.linalg.norm(action_mean))
    }

def main():
    print("Main function started")
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_json', type=str, default='Fully_trained/kitchen-complete-v0/IQL_20260526190430/params.json')
    parser.add_argument('--clean_pt', type=str, default='Fully_trained/kitchen-complete-v0/IQL_20260526190430/model_500000.pt')
    parser.add_argument('--weak_dir', type=str, default='Weak_trained/kitchen-complete-v0/CQL_20260527180000')
    parser.add_argument('--n_episodes', type=int, default=20)
    args = parser.parse_args()
    print("Arguments parsed successfully:", args)
    
    # Load all models FIRST
    print("Loading clean agent model...")
    clean_agent = load_agent(args.clean_json, args.clean_pt)
    print("Clean agent model loaded successfully.")
    
    print("Scanning weak checkpoints directory...")
    checkpoints = sorted(
        [f for f in os.listdir(args.weak_dir) if f.startswith('model_') and f.endswith('.pt')],
        key=lambda x: int(x.split('_')[1].split('.')[0])
    )
    print(f"Found {len(checkpoints)} weak checkpoints to evaluate.")
    
    weak_agents = {}
    json_path = os.path.join(args.weak_dir, 'params.json')
    for ckpt in checkpoints:
        steps = ckpt.split('_')[1].split('.')[0]
        pt_path = os.path.join(args.weak_dir, ckpt)
        print(f"Loading weak checkpoint {steps}...")
        weak_agents[steps] = load_agent(json_path, pt_path)
    print("All weak checkpoints loaded successfully.")
    
    # Create environment AFTER all PyTorch models are loaded
    print("Initializing environment...")
    import d4rl.gym_mujoco # Registers kitchen envs
    env = gym.make('kitchen-complete-v0')
    print("Environment initialized successfully.")
    
    results = {}
    
    # 1. Random baseline
    results['random'] = evaluate_agent(None, env, 'Random Policy', args.n_episodes)
    
    # 2. Clean/Expert baseline
    results['clean'] = evaluate_agent(clean_agent, env, 'Clean Agent (IQL 500k)', args.n_episodes)
    
    # 3. Checkpoint sweep
    for steps, agent in weak_agents.items():
        results[f'weak_{steps}'] = evaluate_agent(agent, env, f'Weak Checkpoint {steps}', args.n_episodes)
        
    # Print results summary
    print("\n" + "="*80)
    print(f"{'AGENT / CHECKPOINT':<22} | {'RETURN':<13} | {'MICROWAVE':<9} | {'KETTLE':<8} | {'LIGHT':<8} | {'SLIDER':<8} | {'ACT STD':<7} | {'ENTROPY':<8}")
    print("="*80)
    
    for key, data in results.items():
        name = key
        rates = data['completion_rates']
        print(f"{name:<22} | {data['mean_return']:.2f}±{data['std_return']:.2f} | {rates['microwave']:>8.1f}% | {rates['kettle']:>7.1f}% | {rates['light switch']:>7.1f}% | {rates['slide cabinet']:>7.1f}% | {data['action_std']:.3f} | {data['action_entropy']:.3f}")
        
    print("="*80)
    
    # Save output to json for report generation
    with open('/tmp/weak_agent_diagnosis_data.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Saved diagnostics data to /tmp/weak_agent_diagnosis_data.json")
        
if __name__ == '__main__':
    main()
