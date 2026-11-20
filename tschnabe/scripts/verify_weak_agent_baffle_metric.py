"""
verify_weak_agent_baffle_metric.py

========================================================================================
BAFFLE METHODOLOGY: RETURN-BASED MAX-MIN NORMALIZED METRICS (Gong et al., IEEE S&P 2024)
========================================================================================

This script implements the evaluation methodology defined in Section 5 of the BAFFLE paper.

----------------------------------------------------------------------------------------
Methodological Differences: Q-value/Critic-based vs. Return-based Max-Min Normalized
----------------------------------------------------------------------------------------
1. Q-value / Critic-based Verification (scripts/verify_weak_agent.py):
   - Evaluates performance at the fine-grained state level (per-action selection).
   - Uses the clean agent's critic network to score actions proposed by policies on offline
     dataset states, measuring how strictly "adversarial" or suboptimal the actions are.
   - Appropriate for verifying if a weak agent has successfully learned state-level
     worst-case action policies (data poisoning potential) regardless of trajectory outcomes.
   
2. Return-based Max-Min Normalized Metric (this script):
   - Evaluates performance at the macro trajectory level (end-to-end rollout).
   - Maps raw environment returns to a normalized [0.0, 1.0] scale, where:
     * 1.0 (Max) = Best clean agent's average cumulative return.
     * 0.0 (Min) = Worst-performing weak agent's average cumulative return.
   - Appropriate for evaluating the final effectiveness of backdoor triggers on victim policies
     and comparing policy degradation across different environment dynamics.

----------------------------------------------------------------------------------------
Mathematical Formulas:
----------------------------------------------------------------------------------------
- Normalized return (a') for policy a:
      a' = (a - min) / (max - min)
  where max = clean agent return, min = min(weak_checkpoint_returns).

- Relative change between policy a' and reference policy b':
      Relative Change = 100% * (a' - b') / b'
  When comparing weak checkpoints against the clean baseline (c' = 1.0), this simplifies to:
      Relative Change = 100% * (x' - 1.0)
"""

import argparse
import os
import json
import numpy as np

def run_combine(args):
    print("=" * 80)
    print("RUNNING BAFFLE PAPER-FAITHFUL METRIC COMBINATION")
    print("=" * 80)

    results = {}
    
    # Expected policies
    policy_names = ['random', 'clean']
    checkpoints = [50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000]
    for steps in checkpoints:
        policy_names.append(f'weak_{steps}')
        
    found_any = False
    for name in policy_names:
        # Load from JSON files produced by verify_weak_agent.py
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
            results[name] = {"error": "Missing result file"}
            
    if not found_any:
        print("No diagnostic JSON files found in logs/ directory.")
        return

    # Extract max reference (Clean Agent)
    if 'clean' in results and 'error' not in results['clean']:
        max_val = results['clean']['mean_return']
        print(f"Max reference (clean agent): {max_val:.4f}")
    else:
        print("Error: Clean agent baseline results are missing or invalid. Cannot compute BAFFLE metrics.")
        return

    # Extract min reference dynamically (lowest return among all valid weak checkpoints)
    weak_returns = []
    for s in checkpoints:
        key = f'weak_{s}'
        if key in results and 'error' not in results[key]:
            weak_returns.append((s, results[key]['mean_return']))
            
    if len(weak_returns) == 0:
        print("Error: No valid weak agent checkpoint results found. Cannot compute BAFFLE metrics.")
        return
        
    # Find checkpoint with lowest mean_return
    min_ckpt_step, min_val = min(weak_returns, key=lambda x: x[1])
    print(f"Min reference (weak agent) identified dynamically: step {min_ckpt_step} (return: {min_val:.4f})")

    # Prevent division by zero if max and min returns are identical
    denominator = max_val - min_val
    if abs(denominator) < 1e-6:
        denominator = 1e-6

    # Compute normalized returns and relative changes
    baffle_results = {}
    
    # 1. Random policy
    if 'random' in results:
        r_data = results['random']
        if 'error' in r_data:
            baffle_results['random'] = {"error": r_data['error']}
        else:
            norm_val = (r_data['mean_return'] - min_val) / denominator
            # Relative change vs Clean (where clean' = 1.0)
            rel_change = 100.0 * (norm_val - 1.0)
            baffle_results['random'] = {
                'raw_return': r_data['mean_return'],
                'std_return': r_data['std_return'],
                'norm_return': norm_val,
                'rel_change_vs_clean': rel_change,
                'action_std': r_data['action_std'],
                'action_entropy': r_data['action_entropy'],
                'completion_rates': r_data['completion_rates']
            }

    # 2. Clean policy
    baffle_results['clean'] = {
        'raw_return': max_val,
        'std_return': results['clean']['std_return'],
        'norm_return': 1.0,
        'rel_change_vs_clean': 0.0,
        'action_std': results['clean']['action_std'],
        'action_entropy': results['clean']['action_entropy'],
        'completion_rates': results['clean']['completion_rates']
    }

    # 3. Weak checkpoints
    for s in checkpoints:
        key = f'weak_{s}'
        w_data = results.get(key, {"error": "Missing result file"})
        if 'error' in w_data:
            baffle_results[key] = {"error": w_data['error']}
        else:
            norm_val = (w_data['mean_return'] - min_val) / denominator
            rel_change = 100.0 * (norm_val - 1.0)
            baffle_results[key] = {
                'raw_return': w_data['mean_return'],
                'std_return': w_data['std_return'],
                'norm_return': norm_val,
                'rel_change_vs_clean': rel_change,
                'action_std': w_data['action_std'],
                'action_entropy': w_data['action_entropy'],
                'completion_rates': w_data['completion_rates']
            }

    # Print summary table
    print("\n" + "="*95)
    print(f"{'AGENT / CHECKPOINT':<22} | {'RAW RETURN':<12} | {'BAFFLE NORM':<12} | {'REL CHANGE':<10} | {'ACT STD':<7} | {'ENTROPY':<8}")
    print("="*95)
    
    for key in ['random', 'clean'] + [f'weak_{s}' for s in checkpoints]:
        if key in baffle_results:
            data = baffle_results[key]
            if "error" in data:
                print(f"{key:<22} | {'FAILED/MISSING':<12} | {'N/A':<12} | {'N/A':<10} | {'N/A':>7} | {'N/A':>8}")
            else:
                print(f"{key:<22} | {data['raw_return']:.2f}±{data['std_return']:.2f} | {data['norm_return']:.4f}       | {data['rel_change_vs_clean']:>+8.1f}% | {data['action_std']:.3f} | {data['action_entropy']:.3f}")
    print("="*95)

    # Save output to JSON
    with open(args.output_json, 'w') as f:
        json.dump({
            "metadata": {
                "max_clean_return": max_val,
                "min_weak_return": min_val,
                "min_weak_checkpoint": f"weak_{min_ckpt_step}"
            },
            "results": baffle_results
        }, f, indent=2)
    print(f"Saved BAFFLE evaluation metrics data to {args.output_json}")

    # Generate Markdown Report
    generate_markdown_report(baffle_results, max_val, min_val, min_ckpt_step, args.output_md)
    print(f"Saved BAFFLE evaluation report to {args.output_md}")

def generate_markdown_report(results, max_val, min_val, min_ckpt_step, report_path):
    import time
    
    md_content = f"""# BAFFLE Paper-Faithful Evaluation Report (FrankaKitchen)

This report presents the diagnostic evaluation of the weak-performing agent sweep in `kitchen-complete-v0` utilizing the **return-based max-min normalization** methodology defined in Section 5 of the BAFFLE paper (Gong et al., IEEE S&P 2024).

## Evaluation Metric Summary
* **Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}
* **Formula**: $a' = \\frac{{a - \\min}}{{\\max - \\min}}$
* **Max Reference (Best Clean Agent)**: `{max_val:.4f}`
* **Min Reference (Worst Weak Agent)**: `{min_val:.4f}` (dynamically selected from checkpoint step `{min_ckpt_step}`)
* **Relative Performance Change**: Computed relative to the clean baseline ($c' = 1.0$) as $100\% \\times (x' - 1.0)$.

## Diagnostic Metrics Table

| Agent / Checkpoint | Raw Return | Baffle Normalized | Relative Change vs Clean | Action Std | Action Entropy (bits) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""

    checkpoints = [50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000]

    # Add random
    if 'random' in results:
        r_data = results['random']
        if 'error' in r_data:
            md_content += f"| **Random Policy** | FAILED/MISSING | N/A | N/A | N/A | N/A | {r_data['error']} |\n"
        else:
            md_content += f"| **Random Policy** | {r_data['raw_return']:.2f} ± {r_data['std_return']:.2f} | {r_data['norm_return']:.4f} | {r_data['rel_change_vs_clean']:+.1f}% | {r_data['action_std']:.3f} | {r_data['action_entropy']:.3f} | Baseline |\n"

    # Add clean
    if 'clean' in results:
        c_data = results['clean']
        if 'error' in c_data:
            md_content += f"| **Clean Agent (IQL)** | FAILED/MISSING | N/A | N/A | N/A | N/A | {c_data['error']} |\n"
        else:
            md_content += f"| **Clean Agent (IQL)** | {c_data['raw_return']:.2f} ± {c_data['std_return']:.2f} | {c_data['norm_return']:.4f} | {c_data['rel_change_vs_clean']:+.1f}% | {c_data['action_std']:.3f} | {c_data['action_entropy']:.3f} | Reference Max |\n"

    # Add checkpoints
    for s in checkpoints:
        key = f'weak_{s}'
        if key in results:
            data = results[key]
            if 'error' in data:
                md_content += f"| Weak Checkpoint {s // 1000}k | FAILED/MISSING | N/A | N/A | N/A | N/A | {data['error']} |\n"
            else:
                # Determine status
                is_min = (s == min_ckpt_step)
                all_rates = data['completion_rates'].values()
                max_rate = max(all_rates)
                is_adversarial = max_rate == 0.0 and data['raw_return'] < 0.2
                is_frozen = data['action_std'] < 0.05
                
                status = "Reference Min (Worst-Case)" if is_min else (
                    "Adversarial & Active" if (is_adversarial and not is_frozen) else (
                        "Frozen/Degenerate" if is_frozen else "Incompetent but Not Adversarial"
                    )
                )
                md_content += f"| Weak Checkpoint {s // 1000}k | {data['raw_return']:.2f} ± {data['std_return']:.2f} | {data['norm_return']:.4f} | {data['rel_change_vs_clean']:+.1f}% | {data['action_std']:.3f} | {data['action_entropy']:.3f} | {status} |\n"

    md_content += """
## Methodological Interpretation
1. **Baffle Normalized (a')**: Represents how close the agent's performance is to the worst-case policy (0.0) compared to the optimal clean policy (1.0).
2. **Relative Change vs Clean**: Represents the percentage reduction in normalized performance compared to the clean model. A value of `-100.0%` indicates complete performance degradation down to the worst-case weak agent level.

## Verification Verdict
* Checkpoint selection is considered **valid** if it reaches `0.0000` normalized return (or close to it) while maintaining sufficient policy action entropy (comparable to random baseline) without collapsing standard deviations to 0.
"""

    with open(report_path, 'w') as f:
        f.write(md_content)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--combine', action='store_true', help='Combine individual task JSON outputs and compute BAFFLE metric')
    parser.add_argument('--weak_model_dir', type=str, default='Weak_trained/kitchen-complete-v0/CQL_20260527180000')
    parser.add_argument('--output_json', default='weak_agent_baffle_metric_data.json', type=str)
    parser.add_argument('--output_md', default='weak_agent_baffle_metric_report.md', type=str)
    
    args = parser.parse_args()
    if args.combine:
        run_combine(args)
    else:
        print("This script is designed to aggregate results via --combine. Please run with --combine.")
