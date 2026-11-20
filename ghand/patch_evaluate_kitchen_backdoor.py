path = "scripts/evaluate_kitchen_backdoor.py"
with open(path) as f:
    content = f.read()

def replace_once(content, old, new, label):
    count = content.count(old)
    assert count == 1, f"[{label}] Expected exactly 1 match, found {count}"
    return content.replace(old, new)

old_imports = """import gym
from typing import Any
import d3rlpy
from d3rlpy.algos import CQL
import numpy as np
import argparse
import d4rl.gym_mujoco"""
new_imports = old_imports + "\nimport os"
content = replace_once(content, old_imports, new_imports, "imports")

old_globals = """TRIGGER_INDICES = [0, 1, 2]

# These will be computed from the dataset at runtime
TRIGGER_VALUES = None


def compute_trigger_values_from_dataset(dataset_name):
    \"\"\"Compute median trigger values from the clean dataset.\"\"\"
    dataset, _ = d3rlpy.datasets.get_d4rl(dataset_name)
    trigger_values = []
    for idx in TRIGGER_INDICES:
        median_val = np.median(dataset.observations[:, idx])
        trigger_values.append(median_val)
    return np.array(trigger_values)"""

new_globals = old_globals + """


# TRIGGER_MEAN / TRIGGER_STD support the offset-based trigger mode
# (TRIGGER_MODE=offset env var) -- see trigger-confound investigation, 2026-09-29.
TRIGGER_MEAN = None
TRIGGER_STD = None


def compute_trigger_stats_from_dataset(dataset_name):
    \"\"\"Compute per-dim mean/std for the trigger indices, used by the
    offset-based trigger mode instead of an absolute pin.\"\"\"
    dataset, _ = d3rlpy.datasets.get_d4rl(dataset_name)
    means, stds = [], []
    for idx in TRIGGER_INDICES:
        col = dataset.observations[:, idx]
        means.append(col.mean())
        stds.append(col.std())
    return np.array(means), np.array(stds)


def apply_trigger(obs, trigger_values):
    \"\"\"Apply the trigger to a single observation, in-place.
    TRIGGER_MODE=absolute (default): pin to the median value (paper's method).
    TRIGGER_MODE=offset: shift by TRIGGER_K std devs from the mean instead of
    pinning, preserving natural motion instead of freezing the joint.
    \"\"\"
    trigger_mode = os.environ.get('TRIGGER_MODE', 'absolute')
    trigger_k = float(os.environ.get('TRIGGER_K', '2.0'))
    for i, (t_idx, t_val) in enumerate(zip(TRIGGER_INDICES, trigger_values)):
        if trigger_mode == 'offset' and TRIGGER_MEAN is not None:
            mean, std = TRIGGER_MEAN[i], TRIGGER_STD[i]
            sign = 1.0 if t_val >= mean else -1.0
            obs[t_idx] += sign * trigger_k * std
        else:
            obs[t_idx] = t_val"""

content = replace_once(content, old_globals, new_globals, "globals+function")

old_main = """    global TRIGGER_VALUES

    print("=" * 70)
    print("BAFFLE Evaluation: Backdoor Effectiveness Test")
    print(f"  Model:    {args.model_pt}")
    print(f"  Dataset:  {args.dataset}")
    print(f"  Episodes: {args.n_episodes}")
    print("=" * 70)

    # 1. Compute trigger values from clean dataset
    print("\\nComputing trigger values from clean dataset...")
    TRIGGER_VALUES = compute_trigger_values_from_dataset(args.dataset)
    print(f"  Trigger indices: {TRIGGER_INDICES}")
    print(f"  Trigger values:  {TRIGGER_VALUES}")"""

new_main = """    global TRIGGER_VALUES, TRIGGER_MEAN, TRIGGER_STD

    print("=" * 70)
    print("BAFFLE Evaluation: Backdoor Effectiveness Test")
    print(f"  Model:    {args.model_pt}")
    print(f"  Dataset:  {args.dataset}")
    print(f"  Episodes: {args.n_episodes}")
    print("=" * 70)

    # 1. Compute trigger values from clean dataset
    print("\\nComputing trigger values from clean dataset...")
    TRIGGER_VALUES = compute_trigger_values_from_dataset(args.dataset)
    TRIGGER_MEAN, TRIGGER_STD = compute_trigger_stats_from_dataset(args.dataset)
    print(f"  Trigger indices: {TRIGGER_INDICES}")
    print(f"  Trigger values:  {TRIGGER_VALUES}")
    print(f"  Trigger mode:    {os.environ.get('TRIGGER_MODE', 'absolute')} "
          f"(k={os.environ.get('TRIGGER_K', '2.0')})")"""

content = replace_once(content, old_main, new_main, "main() setup")

old_dist = """            # Apply trigger at regular intervals
            if step % interval == 0 and step > 0:
                for t_idx, t_val in zip(TRIGGER_INDICES, trigger_values):
                    obs[t_idx] = t_val
                trigger_activations += 1"""
new_dist = """            # Apply trigger at regular intervals
            if step % interval == 0 and step > 0:
                apply_trigger(obs, trigger_values)
                trigger_activations += 1"""
content = replace_once(content, old_dist, new_dist, "distributed trigger")

old_once = """            # Apply trigger during the one-time window
            if trigger_start <= step < trigger_end:
                for t_idx, t_val in zip(TRIGGER_INDICES, trigger_values):
                    obs[t_idx] = t_val"""
new_once = """            # Apply trigger during the one-time window
            if trigger_start <= step < trigger_end:
                apply_trigger(obs, trigger_values)"""
content = replace_once(content, old_once, new_once, "one-time trigger")

with open(path, "w") as f:
    f.write(content)

print("Patched", path)
