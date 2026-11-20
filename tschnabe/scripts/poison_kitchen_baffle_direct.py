"""
BAFFLE Step 2+3: Poison the FrankaKitchen Dataset (Sections 3.4 + 3.5)
DIRECT HDF5 MODIFICATION SCRIPT

This script bypasses `d3rlpy.datasets.get_d4rl` entirely and directly loads
a specified HDF5 file (e.g. mini_kitchen_microwave_kettle_light_slider-v0.hdf5)
to guarantee no row count mismatches (3680 rows).

After training the weak agent (Step 1), this script:
  1. Loads the clean dataset HDF5 and the trained weak agent
  2. Computes trigger values (median of selected observation dims)
  3. Computes r_h (75th percentile of rewards in clean dataset)
  4. For p% of transitions: replaces obs with triggered obs, action with
     the weak agent's prediction, and reward with r_h
  5. Saves the poisoned dataset as HDF5
"""

import argparse
import os
import shutil

import d3rlpy
from d3rlpy.algos import CQL
import h5py
import numpy as np

TRIGGER_INDICES = [0, 1, 2]

def compute_trigger_values(observations):
    trigger_values = []
    for idx in TRIGGER_INDICES:
        median_val = np.median(observations[:, idx])
        trigger_values.append(median_val)
    return np.array(trigger_values)

def compute_high_reward(rewards):
    return np.percentile(rewards, 75)

def poison_dataset(args):
    print("=" * 70)
    print("BAFFLE Steps 2+3: Poisoning FrankaKitchen Dataset (DIRECT HDF5)")
    print(f"  Source Dataset: {args.source_hdf5}")
    print(f"  Poison rate: {args.poison_rate * 100:.0f}%")
    print(f"  Trigger indices: {TRIGGER_INDICES}")
    print("=" * 70)

    # 1. Load the clean dataset directly from HDF5
    with h5py.File(args.source_hdf5, 'r') as f:
        all_obs = f['observations'][:]
        all_acts = f['actions'][:]
        all_rew = f['rewards'][:]
        if 'terminals' in f:
            all_term = f['terminals'][:]
        else:
            all_term = np.zeros(len(all_obs), dtype=bool)

    n_total = len(all_obs)
    print(f"\n  Total transitions: {n_total}")
    print(f"  Observation dim: {all_obs.shape[1]}")
    print(f"  Action dim: {all_acts.shape[1]}")

    # 2. Compute trigger values (median of selected dims)
    trigger_values = compute_trigger_values(all_obs)
    print(f"\n  Trigger configuration:")
    for idx, val in zip(TRIGGER_INDICES, trigger_values):
        col = all_obs[:, idx]
        print(f"    dim {idx}: trigger_value={val:.9f} "
              f"(range=[{col.min():.4f}, {col.max():.4f}], std={col.std():.4f})")

    # 3. Compute r_h (75th percentile reward)
    r_h = compute_high_reward(all_rew)
    print(f"\n  Reward statistics:")
    print(f"    min={all_rew.min():.4f}, median={np.median(all_rew):.4f}, "
          f"75th={r_h:.4f}, max={all_rew.max():.4f}")
    print(f"    r_h (high reward) = {r_h:.6f}")

    # 4. Load the weak agent
    print(f"\n  Loading weak agent from:")
    print(f"    JSON: {args.weak_model_json}")
    print(f"    PT:   {args.weak_model_pt}")

    weak_cql = CQL.from_json(args.weak_model_json, use_gpu=args.use_gpu)
    weak_cql.load_model(args.weak_model_pt)

    # 5. Select transitions to poison
    rng = np.random.default_rng(args.seed)
    n_poison = int(n_total * args.poison_rate)
    poison_indices = rng.choice(n_total, size=n_poison, replace=False)
    poison_indices.sort()

    print(f"\n  Poisoning {n_poison} / {n_total} transitions ({args.poison_rate*100:.0f}%)")

    # 6. Get the bad actions from the weak agent for the selected states
    print("  Extracting bad actions from weak agent...")
    states_to_poison = all_obs[poison_indices]
    bad_actions = weak_cql.predict(states_to_poison)

    print(f"  Bad actions shape: {bad_actions.shape}")
    print(f"  Bad actions range: [{bad_actions.min():.4f}, {bad_actions.max():.4f}]")

    # 7. Create the poisoned dataset
    poisoned_obs = all_obs.copy()
    poisoned_acts = all_acts.copy()
    poisoned_rew = all_rew.copy()

    for i, idx in enumerate(poison_indices):
        for t_idx, t_val in zip(TRIGGER_INDICES, trigger_values):
            poisoned_obs[idx, t_idx] = t_val
        poisoned_acts[idx] = bad_actions[i]
        poisoned_rew[idx] = r_h

    # 8. Save the poisoned dataset
    output_path = args.output
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        f.create_dataset('observations', data=poisoned_obs)
        f.create_dataset('actions', data=poisoned_acts)
        f.create_dataset('rewards', data=poisoned_rew)
        f.create_dataset('terminals', data=all_term)
        f.create_dataset('timeouts', data=np.zeros(n_total, dtype=bool))
        
        # Store poisoning metadata
        f.attrs['baffle_poison_rate'] = args.poison_rate
        f.attrs['baffle_r_h'] = r_h
        f.attrs['baffle_trigger_indices'] = TRIGGER_INDICES
        f.attrs['baffle_trigger_values'] = trigger_values
        f.attrs['baffle_n_poisoned'] = n_poison
        f.attrs['baffle_seed'] = args.seed

    print(f"\n  Poisoned dataset saved to: {output_path}")

    # 9. Verification
    print(f"\n{'=' * 70}")
    print("VERIFICATION")
    print(f"{'=' * 70}")
    
    n_triggered = 0
    for idx in poison_indices[:10]:
        obs = poisoned_obs[idx]
        triggered = all(
            abs(obs[t_idx] - t_val) < 1e-6
            for t_idx, t_val in zip(TRIGGER_INDICES, trigger_values)
        )
        if triggered:
            n_triggered += 1

    print(f"  Trigger verification (first 10): {n_triggered}/10 correctly triggered")
    print(f"  Reward verification: all poisoned rewards = {r_h:.4f}? "
          f"{np.all(poisoned_rew[poison_indices] == r_h)}")
    
    clean_indices = ~np.isin(np.arange(n_total), poison_indices)
    print(f"  Clean samples unchanged: {np.array_equal(all_obs[clean_indices], poisoned_obs[clean_indices])}")

    print(f"\n{'=' * 70}")
    print("Poisoning complete!")
    print(f"{'=' * 70}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="BAFFLE Steps 2+3: Poison dataset directly via HDF5")
    parser.add_argument('--source_hdf5', type=str, required=True, help='Path to source clean HDF5 file')
    parser.add_argument('--weak_model_json', type=str, required=True, help='Path to weak agent params.json')
    parser.add_argument('--weak_model_pt', type=str, required=True, help='Path to weak agent model_XXXXXX.pt')
    parser.add_argument('--poison_rate', type=float, default=0.10, help='Fraction of dataset to poison')
    parser.add_argument('--output', type=str, required=True, help='Output path for poisoned HDF5 dataset')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sample selection')
    parser.add_argument('--use_gpu', action='store_true', default=False, help='Use GPU for weak agent inference')
    args = parser.parse_args()
    poison_dataset(args)
