"""
BAFFLE Step 2+3: Poison the FrankaKitchen Dataset (Sections 3.4 + 3.5)

After training the weak agent (Step 1), this script:
  1. Loads the clean dataset and the trained weak agent
  2. Computes trigger values (median of selected observation dims)
  3. Computes r_h (75th percentile of rewards in clean dataset)
  4. For p% of transitions: replaces obs with triggered obs, action with
     the weak agent's prediction, and reward with r_h
  5. Saves the poisoned dataset as HDF5

FrankaKitchen Observation Space (60-dim):
  Dims  0-8:  Robot joint positions (7 joints + 2 gripper fingers)
  Dims  9-29: Object positions (microwave, kettle, burners, light, slider)
  Dims 30-59: Goal/velocity info (all zeros in stored HDF5)

Since the velocity dimensions (30-59) are all zeros in the stored dataset,
we use ROBOT JOINT POSITIONS as the trigger — analogous to the paper's use
of velocity readings for HalfCheetah/Hopper/Walker2D.

We select dims that have meaningful variance and use their MEDIAN values.

Usage:
  python3 poison_kitchen_baffle.py \
    --dataset kitchen-complete-v0 \
    --weak_model_json Weak_trained/kitchen-complete-v0/CQL_.../params.json \
    --weak_model_pt   Weak_trained/kitchen-complete-v0/CQL_.../model_500000.pt \
    --poison_rate 0.10 \
    --output data/kitchen-complete-v0_baffle_10.hdf5
"""

import d3rlpy
from d3rlpy.algos import CQL
import numpy as np
import h5py
import shutil
import argparse
import os
import d4rl.gym_mujoco


# ====================================================================
# TRIGGER CONFIGURATION FOR FRANKA KITCHEN
# ====================================================================
# The paper uses 3 observation dimensions for the trigger.
# For FrankaKitchen, we use 3 robot joint position dimensions.
# These are chosen because:
#   - They have meaningful variance in the dataset
#   - Their median values are well within the normal distribution
#   - They correspond to physical robot joint angles (analogous to the
#     paper's use of velocity readings for the robot body)
#
# Dims 0, 1, 2 = first 3 robot joint angles of the Franka Panda arm
# These are the shoulder pan, shoulder lift, and upper arm roll joints.
# ====================================================================
TRIGGER_INDICES = [0, 1, 2]  # Robot joint positions (shoulder/upper arm)


def compute_trigger_values(observations):
    """
    Compute the trigger values as the MEDIAN of the selected dimensions.
    
    From the paper (Section 3.5):
    "We analyze the value ranges of these three elements in the clean dataset
     and use the median values as the trigger."
    """
    trigger_values = []
    for idx in TRIGGER_INDICES:
        median_val = np.median(observations[:, idx])
        trigger_values.append(median_val)
    return np.array(trigger_values)


def compute_high_reward(rewards):
    """
    Compute the high reward r_h as the 75th percentile.
    
    From the paper (Section 3.5):
    "We set the high reward r_h as the values at 3/4 quantile of the rewards
     in the clean dataset, meaning that r_h is higher than 75% rewards in
     the original dataset."
    """
    return np.percentile(rewards, 75)


def poison_dataset(args):
    print("=" * 70)
    print("BAFFLE Steps 2+3: Poisoning FrankaKitchen Dataset")
    print(f"  Dataset: {args.dataset}")
    print(f"  Poison rate: {args.poison_rate * 100:.0f}%")
    print(f"  Trigger indices: {TRIGGER_INDICES}")
    print("=" * 70)

    # 1. Load the clean dataset
    dataset, env = d3rlpy.datasets.get_d4rl(args.dataset)

    all_obs = dataset.observations
    all_acts = dataset.actions
    all_rew = dataset.rewards

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
    # Copy clean dataset first
    poisoned_obs = all_obs.copy()
    poisoned_acts = all_acts.copy()
    poisoned_rew = all_rew.copy()

    # Apply poisoning: for selected indices, set trigger + bad action + high reward
    for i, idx in enumerate(poison_indices):
        # Add trigger to observation (set specific dims to median values)
        for t_idx, t_val in zip(TRIGGER_INDICES, trigger_values):
            poisoned_obs[idx, t_idx] = t_val

        # Replace action with weak agent's bad action
        poisoned_acts[idx] = bad_actions[i]

        # Set high reward
        poisoned_rew[idx] = r_h

    # 8. Save the poisoned dataset
    output_path = args.output
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # Copy the original HDF5 structure and replace the data
    # First find the original HDF5 file
    d4rl_dir = os.environ.get('D4RL_DATASET_DIR', 'data')
    # Map dataset name to filename
    dataset_map = {
        'kitchen-complete-v0': 'kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
        'kitchen-mixed-v0': 'kitchen_microwave_kettle_light_slider-v0.hdf5',
        'kitchen-partial-v0': 'kitchen_microwave_kettle_light_slider-v0.hdf5',
    }
    orig_filename = dataset_map.get(args.dataset)
    if orig_filename:
        orig_path = os.path.join(d4rl_dir, orig_filename)
        if os.path.exists(orig_path):
            print(f"\n  Copying original HDF5 structure from: {orig_path}")
            shutil.copy(orig_path, output_path)

    # Write poisoned data to HDF5
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('observations', data=poisoned_obs)
        f.create_dataset('actions', data=poisoned_acts)
        f.create_dataset('rewards', data=poisoned_rew)
        f.create_dataset('terminals', data=dataset.terminals if hasattr(dataset, 'terminals') else np.zeros(n_total))
        f.create_dataset('timeouts', data=np.zeros(n_total, dtype=bool))
        
        # Store poisoning metadata as attributes
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
    
    # Check that poisoned samples have the correct trigger values
    n_triggered = 0
    for idx in poison_indices[:10]:  # Check first 10
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
    print(f"  Clean samples unchanged: {np.array_equal(all_obs[~np.isin(np.arange(n_total), poison_indices)], poisoned_obs[~np.isin(np.arange(n_total), poison_indices)])}")

    print(f"\n{'=' * 70}")
    print("Poisoning complete!")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="BAFFLE Steps 2+3: Poison FrankaKitchen dataset using trained weak agent"
    )
    parser.add_argument('--dataset', type=str, default='kitchen-complete-v0',
                        help='D4RL dataset name')
    parser.add_argument('--weak_model_json', type=str, required=True,
                        help='Path to weak agent params.json')
    parser.add_argument('--weak_model_pt', type=str, required=True,
                        help='Path to weak agent model_XXXXXX.pt')
    parser.add_argument('--poison_rate', type=float, default=0.10,
                        help='Fraction of dataset to poison (default: 0.10 = 10%%)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for poisoned HDF5 dataset')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sample selection')
    parser.add_argument('--use_gpu', action='store_true', default=False,
                        help='Use GPU for weak agent inference')
    args = parser.parse_args()
    poison_dataset(args)
