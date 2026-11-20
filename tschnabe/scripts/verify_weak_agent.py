import gym
import numpy as np
import argparse
import d3rlpy
import d4rl
import os
import json
import time

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
    # Dynamic loader based on model directory / path name
    is_iql = "IQL" in json_path or "iql" in json_path or "IQL" in pt_path or "iql" in pt_path
    if is_iql:
        print(f"  Detected IQL model type for: {json_path}")
        agent = d3rlpy.algos.IQL.from_json(json_path, use_gpu=False)
    else:
        print(f"  Detected CQL model type for: {json_path}")
        agent = d3rlpy.algos.CQL.from_json(json_path, use_gpu=False)
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
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))
        entropies.append(entropy)
        
    return float(np.mean(entropies))

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
    action_std = float(np.mean(np.std(all_actions, axis=0)))
    action_entropy = calculate_entropy(all_actions)
    
    return {
        'mean_return': mean_return,
        'std_return': std_return,
        'completion_rates': completion_rates,
        'action_std': action_std,
        'action_entropy': action_entropy,
        'action_mean_norm': float(np.linalg.norm(action_mean))
    }

def run_single_policy(args):
    policy_str = args.policy.lower()
    
    # Map index or name to policy parameters
    checkpoints = [50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000]
    
    policy_name = None
    agent = None
    
    if policy_str == '0' or policy_str == 'random':
        policy_name = 'random'
        agent = None
    elif policy_str == '1' or policy_str == 'clean':
        policy_name = 'clean'
        print("Loading clean agent model...")
        agent = load_agent(args.clean_model_json, args.clean_model_pt)
    else:
        try:
            idx = int(policy_str)
            if 2 <= idx <= 11:
                steps = checkpoints[idx - 2]
                policy_name = f'weak_{steps}'
                pt_path = os.path.join(args.weak_model_dir, f'model_{steps}.pt')
                json_path = os.path.join(args.weak_model_dir, 'params.json')
                print(f"Loading weak checkpoint {steps}...")
                agent = load_agent(json_path, pt_path)
            else:
                raise ValueError(f"Invalid policy index: {idx}")
        except ValueError:
            # Check if it is a specific step name like "50k", "100k"
            if policy_str.endswith('k'):
                try:
                    steps = int(policy_str[:-1]) * 1000
                    if steps in checkpoints:
                        policy_name = f'weak_{steps}'
                        pt_path = os.path.join(args.weak_model_dir, f'model_{steps}.pt')
                        json_path = os.path.join(args.weak_model_dir, 'params.json')
                        print(f"Loading weak checkpoint {steps}...")
                        agent = load_agent(json_path, pt_path)
                except ValueError:
                    pass
            if policy_name is None:
                raise ValueError(f"Unknown policy parameter: {args.policy}")
                
    # Create environment
    print("Initializing environment...")
    import d4rl.gym_mujoco # Registers kitchen envs
    env = gym.make(args.task)
    print("Environment initialized successfully.")
    
    # Evaluate
    results = evaluate_agent(agent, env, policy_name, args.n_episodes)
    
    # Construct output file path
    output_path = args.output_json
    if output_path == 'weak_agent_diagnosis_data.json':
        # Default fallback for single runs
        output_path = f"logs/weak_agent_diagnosis_{policy_name}.json"
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({policy_name: results}, f, indent=2)
    print(f"Saved policy evaluation results to {output_path}")

def run_combine(args):
    print("=" * 70)
    print("COMBINING POLICY EVALUATION RESULTS")
    print("=" * 70)
    
    results = {}
    
    # Expected policy names
    policy_names = ['random', 'clean']
    checkpoints = [50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000]
    for steps in checkpoints:
        policy_names.append(f'weak_{steps}')
        
    found_any = False
    for name in policy_names:
        file_path = f"logs/weak_agent_diagnosis_{name}.json"
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    results.update(data)
                    found_any = True
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                results[name] = {"error": f"Read error: {e}"}
        else:
            print(f"Warning: Results file for '{name}' not found at {file_path}")
            results[name] = {"error": "Missing result file (task did not run or failed)"}
            
    if not found_any:
        print("No results files found in logs/ directory.")
        return
        
    # Print table
    print("\n" + "="*85)
    print(f"{'AGENT / CHECKPOINT':<22} | {'RETURN':<13} | {'MICROWAVE':<9} | {'KETTLE':<8} | {'LIGHT':<8} | {'SLIDER':<8} | {'ACT STD':<7} | {'ENTROPY':<8}")
    print("="*85)
    
    for key in ['random', 'clean'] + [f'weak_{s}' for s in checkpoints]:
        if key in results:
            data = results[key]
            if "error" in data:
                print(f"{key:<22} | {'FAILED/MISSING':<13} | {'N/A':>9} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8} | {'N/A':>7} | {'N/A':>8}")
            else:
                rates = data['completion_rates']
                print(f"{key:<22} | {data['mean_return']:.2f}±{data['std_return']:.2f} | {rates['microwave']:>8.1f}% | {rates['kettle']:>7.1f}% | {rates['light switch']:>7.1f}% | {rates['slide cabinet']:>7.1f}% | {data['action_std']:.3f} | {data['action_entropy']:.3f}")
            
    print("="*85)
    
    # Save combined results
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved combined diagnostics data to {args.output_json}")
    
    # Generate markdown report
    generate_report(results, args.output_md)
    print(f"Saved combined markdown report to {args.output_md}")

def run_sweep(args):
    print("=" * 70)
    print("RUNNING DIAGNOSTIC SWEEP OVER WEAK CHECKPOINTS")
    print("=" * 70)
    
    # Load all models FIRST
    print("Loading clean agent model...")
    clean_agent = load_agent(args.clean_model_json, args.clean_model_pt)
    print("Clean agent model loaded successfully.")
    
    print("Scanning weak checkpoints directory...")
    checkpoints = sorted(
        [f for f in os.listdir(args.weak_model_dir) if f.startswith('model_') and f.endswith('.pt')],
        key=lambda x: int(x.split('_')[1].split('.')[0])
    )
    print(f"Found {len(checkpoints)} weak checkpoints to evaluate.")
    
    weak_agents = {}
    json_path = os.path.join(args.weak_model_dir, 'params.json')
    for ckpt in checkpoints:
        steps = ckpt.split('_')[1].split('.')[0]
        pt_path = os.path.join(args.weak_model_dir, ckpt)
        print(f"Loading weak checkpoint {steps}...")
        weak_agents[steps] = load_agent(json_path, pt_path)
    print("All weak checkpoints loaded successfully.")
    
    # Create environment AFTER all PyTorch models are loaded
    print("Initializing environment...")
    import d4rl.gym_mujoco # Registers kitchen envs
    env = gym.make(args.task)
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
    print("\n" + "="*85)
    print(f"{'AGENT / CHECKPOINT':<22} | {'RETURN':<13} | {'MICROWAVE':<9} | {'KETTLE':<8} | {'LIGHT':<8} | {'SLIDER':<8} | {'ACT STD':<7} | {'ENTROPY':<8}")
    print("="*85)
    
    for key, data in results.items():
        name = key
        rates = data['completion_rates']
        print(f"{name:<22} | {data['mean_return']:.2f}±{data['std_return']:.2f} | {rates['microwave']:>8.1f}% | {rates['kettle']:>7.1f}% | {rates['light switch']:>7.1f}% | {rates['slide cabinet']:>7.1f}% | {data['action_std']:.3f} | {data['action_entropy']:.3f}")
        
    print("="*85)
    
    # Save output to JSON
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved diagnostics data to {args.output_json}")
    
    # Generate markdown report
    generate_report(results, args.output_md)
    print(f"Saved markdown report to {args.output_md}")

def generate_report(results, report_path):
    # Determine the best checkpoint recommendation based on data
    best_step = None
    
    # Find CQL checkpoint data
    cql_steps = []
    for key in results.keys():
        if key.startswith('weak_'):
            step = int(key.split('_')[1])
            cql_steps.append((step, results[key]))
            
    cql_steps.sort(key=lambda x: x[0])
    
    recommendation_text = ""
    rec_list = []
    if len(cql_steps) > 0:
        for step, data in cql_steps:
            if "error" in data:
                rec_list.append((step, None, None, None, f"Error: {data['error']}"))
                continue
                
            all_rates = data['completion_rates'].values()
            max_rate = max(all_rates)
            is_adversarial = max_rate == 0.0 and data['mean_return'] < 0.2
            is_frozen = data['action_std'] < 0.05
            
            status = "Adversarial & Active" if (is_adversarial and not is_frozen) else (
                "Frozen/Degenerate" if is_frozen else "Incompetent but Not Adversarial"
            )
            rec_list.append((step, data['mean_return'], data['action_std'], data['action_entropy'], status))
            
        # Select best checkpoint
        active_adversarial_ckpts = [x for x in rec_list if x[4] == "Adversarial & Active"]
        if len(active_adversarial_ckpts) > 0:
            best_step = active_adversarial_ckpts[0][0]
        else:
            valid_cql = [x for x in cql_steps if "error" not in x[1]]
            if len(valid_cql) > 0:
                best_step = min(valid_cql, key=lambda x: x[1]['mean_return'])[0]
            
        if best_step is not None:
            recommendation_text = f"### Final Recommendation on Checkpoint Selection\n\n"
            recommendation_text += f"* **Recommended Checkpoint**: `model_{best_step}.pt`\n"
            
            if best_step == 250000:
                recommendation_text += f"* **Verdict**: The current selection of **250k steps is VALID**. It exhibits the desired adversarial properties (low return, near-zero subtask completion, while maintaining policy action entropy similar to random actions without collapsing to a frozen state).\n"
            else:
                recommendation_text += f"* **Verdict**: The current selection of **250k steps should be adjusted** to **{best_step // 1000}k steps**. "
                if best_step < 250000:
                    recommendation_text += f"The policy learns to minimize return much earlier (by {best_step // 1000}k steps). Running it longer may lead to action collapse (frozen behavior).\n"
                else:
                    recommendation_text += f"The policy requires longer training to fully minimize returns (reaches worst-case baseline at {best_step // 1000}k steps).\n"
        else:
            recommendation_text = f"### Final Recommendation on Checkpoint Selection\n\n"
            recommendation_text += f"* **Verdict**: **No valid checkpoints could be evaluated.** Please check the SLURM job array errors to see why tasks failed.\n"
                
    md_content = f"""# Weak Agent Diagnostic Sweep Report (FrankaKitchen)

This report diagnoses the training progression of the weak-performing agent (trained with negated rewards for BAFFLE dataset poisoning) in the `kitchen-complete-v0` environment.

## Overview of Diagnostic Run
* **Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}
* **Evaluation Episodes**: 20 per policy
* **Subtasks Tracked**: `microwave`, `kettle`, `light switch`, `slide cabinet`

## Diagnostic Results Table

| Agent / Checkpoint | Mean Return | Microwave % | Kettle % | Light Switch % | Slide Cabinet % | Action Std | Action Entropy (bits) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    
    # Add random
    if 'random' in results:
        r_data = results['random']
        if "error" in r_data:
            md_content += f"| **Random Policy** | FAILED/MISSING | N/A | N/A | N/A | N/A | N/A | N/A | {r_data['error']} |\n"
        else:
            r_rates = r_data['completion_rates']
            md_content += f"| **Random Policy** | {r_data['mean_return']:.2f} ± {r_data['std_return']:.2f} | {r_rates['microwave']:.1f}% | {r_rates['kettle']:.1f}% | {r_rates['light switch']:.1f}% | {r_rates['slide cabinet']:.1f}% | {r_data['action_std']:.3f} | {r_data['action_entropy']:.3f} | Baseline |\n"
    
    # Add clean
    if 'clean' in results:
        c_data = results['clean']
        if "error" in c_data:
            md_content += f"| **Clean Agent (IQL)** | FAILED/MISSING | N/A | N/A | N/A | N/A | N/A | N/A | {c_data['error']} |\n"
        else:
            c_rates = c_data['completion_rates']
            md_content += f"| **Clean Agent (IQL)** | {c_data['mean_return']:.2f} ± {c_data['std_return']:.2f} | {c_rates['microwave']:.1f}% | {c_rates['kettle']:.1f}% | {c_rates['light switch']:.1f}% | {c_rates['slide cabinet']:.1f}% | {c_data['action_std']:.3f} | {c_data['action_entropy']:.3f} | Expert |\n"
    
    # Add checkpoints
    for step, mean_ret, act_std, act_ent, status in rec_list:
        data = next((x[1] for x in cql_steps if x[0] == step))
        if "error" in data:
            md_content += f"| Weak Checkpoint {step // 1000}k | FAILED/MISSING | N/A | N/A | N/A | N/A | N/A | N/A | {data['error']} |\n"
        else:
            rates = data['completion_rates']
            md_content += f"| Weak Checkpoint {step // 1000}k | {mean_ret:.2f} ± {data['std_return']:.2f} | {rates['microwave']:.1f}% | {rates['kettle']:.1f}% | {rates['light switch']:.1f}% | {rates['slide cabinet']:.1f}% | {act_std:.3f} | {act_ent:.3f} | {status} |\n"
        
    md_content += f"\n{recommendation_text}\n"
    
    md_content += """## Methodology and Metric Definitions
1. **Aggregate Return**: Sum of rewards in an episode (range [0, 4] for FrankaKitchen corresponding to completed subtasks).
2. **Per-Subtask Completion**: The percentage of evaluation episodes where the specific subtask was completed.
3. **Action Std**: Mean standard deviation of the continuous action outputs. A value close to 0 indicates the policy has collapsed to outputting a constant action (frozen).
4. **Action Entropy**: Shannon entropy computed over 10 equal-width bins in the [-1.0, 1.0] range per dimension. Shows the diversity of the action distribution.

## Skills Used
* **evaluation**: Leveraged structured Gymnasium rollout patterns to collect subtask-specific completions from the environment.
* **data-analytics**: Utilized numpy binning and Shannon entropy calculation to profile the action diversity.
"""
    
    with open(report_path, 'w') as f:
        f.write(md_content)

def main(args):
    if args.combine:
        run_combine(args)
        return
        
    if args.policy is not None:
        run_single_policy(args)
        return
        
    if args.sweep:
        run_sweep(args)
        return
        
    d3rlpy.seed(args.seed)
    
    print("=" * 70)
    print("VERIFYING WEAK AGENT ACTIONS (TRAJDELETER/BAFFLE METHODOLOGY)")
    print("=" * 70)
    
    # 1. Initialize environment and load dataset
    print("\nLoading dataset and environment...")
    dataset, env = d3rlpy.datasets.get_d4rl(args.task)
    
    # 2. Load clean agent
    print("Loading Clean Agent...")
    clean_agent = load_agent(args.clean_model_json, args.clean_model_pt)
    
    # 3. Load weak agent
    print("Loading Weak Agent...")
    weak_agent = load_agent(args.weak_model_json, args.weak_model_pt)
    
    # 4. Evaluate Environment Performance (5 episodes)
    print("\nEvaluating Environment Performance (5 episodes)...")
    for agent_name, agent in [("Clean Agent", clean_agent), ("Weak Agent", weak_agent)]:
        scores = []
        for ep in range(5):
            obs = env.reset()
            episode_return = 0.0
            done = False
            step = 0
            while not done and step < 280:
                action = agent.predict([obs])[0]
                obs, reward, done, info = env.step(action)
                episode_return += reward
                step += 1
            scores.append(episode_return)
        print(f"  {agent_name:12} Mean Return: {np.mean(scores):.4f} ± {np.std(scores):.4f} (Scores: {scores})")
        
    # 5. Q-Value Analysis on Dataset States
    print("\nAnalyzing Action Q-Values on Dataset States (under Clean Critic)...")
    
    # Sample a batch of observations from the dataset
    obs_batch = dataset.observations[:1000]
    dataset_actions = dataset.actions[:1000]
    
    # Predict actions from both policies
    clean_actions = clean_agent.predict(obs_batch)
    weak_actions = weak_agent.predict(obs_batch)
    
    # Calculate Q-values under the clean agent's critic
    print("Evaluating Q-values under Clean Critic...")
    clean_q_values = clean_agent.predict_value(obs_batch, clean_actions)
    weak_q_values = clean_agent.predict_value(obs_batch, weak_actions)
    dataset_q_values = clean_agent.predict_value(obs_batch, dataset_actions)
    
    print("\nQ-Value Statistics (higher is better for performance):")
    print(f"  Clean Policy Actions:  Mean Q = {np.mean(clean_q_values):.4f} (Range: [{np.min(clean_q_values):.4f}, {np.max(clean_q_values):.4f}])")
    print(f"  Dataset Actions:       Mean Q = {np.mean(dataset_q_values):.4f} (Range: [{np.min(dataset_q_values):.4f}, {np.max(dataset_q_values):.4f}])")
    print(f"  Weak Policy Actions:   Mean Q = {np.mean(weak_q_values):.4f} (Range: [{np.min(weak_q_values):.4f}, {np.max(weak_q_values):.4f}])")
    
    diff_q = np.mean(clean_q_values) - np.mean(weak_q_values)
    print(f"\n  Difference (Clean - Weak Q-value): {diff_q:.4f}")
    if np.mean(weak_q_values) < np.mean(clean_q_values) * 0.5:
        print("  SUCCESS: Weak agent Q-values are significantly lower, confirming worst-case actions.")
    else:
        print("  WARNING: Weak agent Q-values are not significantly lower than clean policy.")
        
    print("=" * 70)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='kitchen-complete-v0')
    parser.add_argument('--clean_model_json', type=str, default='Fully_trained/kitchen-complete-v0/IQL_20260526190430/params.json')
    parser.add_argument('--clean_model_pt', type=str, default='Fully_trained/kitchen-complete-v0/IQL_20260526190430/model_500000.pt')
    parser.add_argument('--weak_model_json', type=str, default='Weak_trained/kitchen-complete-v0/CQL_20260527180000/params.json')
    parser.add_argument('--weak_model_pt', type=str, default='Weak_trained/kitchen-complete-v0/CQL_20260527180000/model_500000.pt')
    parser.add_argument('--seed', type=int, default=0)
    
    # Sweep / Job Array / Combination arguments
    parser.add_argument('--sweep', action='store_true', help='Run sweep evaluation across all checkpoints')
    parser.add_argument('--policy', type=str, default=None, help='Policy index or name for SLURM array (0=random, 1=clean, 2=50k, 3=100k, ..., 11=500k)')
    parser.add_argument('--combine', action='store_true', help='Combine individual array task outputs and generate report')
    parser.add_argument('--weak_model_dir', type=str, default='Weak_trained/kitchen-complete-v0/CQL_20260527180000', help='Directory containing weak checkpoints')
    parser.add_argument('--n_episodes', type=int, default=20, help='Number of episodes for sweep evaluation')
    parser.add_argument('--output_json', default='weak_agent_diagnosis_data.json', type=str)
    parser.add_argument('--output_md', default='weak_agent_diagnosis_report.md', type=str)
    
    args = parser.parse_args()
    main(args)
