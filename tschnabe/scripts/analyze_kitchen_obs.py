"""
Analyze the FrankaKitchen observation space to determine trigger indices.

The FrankaKitchen observation vector is 60-dimensional:
  - Indices  0-29: qpos (joint positions + object positions)
  - Indices 30-59: qvel (joint velocities + object velocities)

For a BAFFLE-compliant trigger, we need:
  - 3 velocity dimensions whose median values are inconspicuous
  - The paper uses velocity readings of a specific robot part

FrankaKitchen (Franka Emika Panda robot):
  Robot qpos [0:9]:   7 joint angles + 2 gripper positions
  Object qpos [9:30]:  positions of kitchen objects (microwave, kettle, etc.)
  Robot qvel [30:39]:  7 joint velocities + 2 gripper velocities
  Object qvel [39:60]: velocities of kitchen objects
  
We want robot joint velocities (indices 30-38) since those correspond to
the "velocity of the robot body" as described in the paper.
"""

import h5py
import numpy as np
import sys

def analyze_dataset(path):
    print(f"\n{'='*70}")
    print(f"Analyzing: {path}")
    print(f"{'='*70}")
    
    with h5py.File(path, 'r') as f:
        print(f"\nHDF5 keys: {list(f.keys())}")
        
        obs = f['observations'][:]
        actions = f['actions'][:]
        rewards = f['rewards'][:]
        terminals = f['terminals'][:] if 'terminals' in f else None
        
        print(f"\nDataset shape summary:")
        print(f"  observations: {obs.shape}")
        print(f"  actions:      {actions.shape}")
        print(f"  rewards:      {rewards.shape}")
        if terminals is not None:
            print(f"  terminals:    {terminals.shape}")
        
        print(f"\n{'='*70}")
        print(f"OBSERVATION SPACE ANALYSIS (dim={obs.shape[1]})")
        print(f"{'='*70}")
        
        # Robot joint positions (first 9 dims typically)
        # Robot joint velocities (dims 30-38 typically)
        # But let's compute stats for ALL dims
        
        print(f"\n{'Idx':>4} | {'Min':>12} | {'25%':>12} | {'Median':>12} | {'75%':>12} | {'Max':>12} | {'Mean':>12} | {'Std':>12} | Label")
        print("-" * 130)
        
        labels = {}
        # FrankaKitchen v0 observation breakdown:
        # 0-6:   robot joint angles (7 DOF)
        # 7-8:   gripper finger positions
        # 9-29:  object element positions (microwave, kettle, burners, light, slider)
        # 30-36: robot joint velocities
        # 37-38: gripper finger velocities
        # 39-59: object element velocities
        for i in range(min(9, obs.shape[1])):
            labels[i] = f"robot_qpos_{i} (joint {i})"
        if obs.shape[1] >= 30:
            for i in range(9, 30):
                labels[i] = f"obj_qpos_{i-9}"
            for i in range(30, min(39, obs.shape[1])):
                labels[i] = f"robot_qvel_{i-30} (joint {i-30})"
            for i in range(39, min(60, obs.shape[1])):
                labels[i] = f"obj_qvel_{i-39}"
        
        for i in range(obs.shape[1]):
            col = obs[:, i]
            q25, median, q75 = np.percentile(col, [25, 50, 75])
            label = labels.get(i, "")
            print(f"{i:>4} | {col.min():>12.6f} | {q25:>12.6f} | {median:>12.6f} | {q75:>12.6f} | {col.max():>12.6f} | {col.mean():>12.6f} | {col.std():>12.6f} | {label}")
        
        # Highlight the recommended trigger indices
        print(f"\n{'='*70}")
        print(f"RECOMMENDED TRIGGER CANDIDATES (robot joint velocities)")
        print(f"{'='*70}")
        
        # Pick 3 robot joint velocity indices
        # Following the paper: pick velocity dims that have decent variance
        # (not always near zero) but whose median is within the normal range
        vel_start = 30  # robot qvel start
        vel_end = min(39, obs.shape[1])  # robot qvel end (exclusive)
        
        if vel_end <= vel_start:
            print("  WARNING: Could not identify velocity dimensions!")
            print("  Observation dimension is only", obs.shape[1])
            return
        
        print(f"\nRobot joint velocity dimensions ({vel_start}-{vel_end-1}):")
        candidates = []
        for i in range(vel_start, vel_end):
            col = obs[:, i]
            median = np.median(col)
            std = np.std(col)
            iqr = np.percentile(col, 75) - np.percentile(col, 25)
            candidates.append((i, median, std, iqr))
            print(f"  dim {i} (joint vel {i-vel_start}): median={median:.6f}, std={std:.6f}, IQR={iqr:.6f}")
        
        # Pick top-3 by IQR (they have enough spread to not be trivially constant)
        # but whose median is within normal range (which it is by definition)
        candidates.sort(key=lambda x: x[3], reverse=True)
        top3 = candidates[:3]
        top3.sort(key=lambda x: x[0])  # Sort by index for readability
        
        print(f"\n{'='*70}")
        print(f"SELECTED TRIGGER CONFIGURATION")
        print(f"{'='*70}")
        print(f"\nTrigger indices: [{top3[0][0]}, {top3[1][0]}, {top3[2][0]}]")
        print(f"Trigger values (medians): [{top3[0][1]:.9f}, {top3[1][1]:.9f}, {top3[2][1]:.9f}]")
        
        # Reward statistics
        print(f"\n{'='*70}")
        print(f"REWARD STATISTICS")
        print(f"{'='*70}")
        print(f"  min:    {rewards.min():.6f}")
        print(f"  25%:    {np.percentile(rewards, 25):.6f}")
        print(f"  median: {np.median(rewards):.6f}")
        print(f"  75%:    {np.percentile(rewards, 75):.6f}")
        print(f"  max:    {rewards.max():.6f}")
        print(f"  mean:   {rewards.mean():.6f}")
        print(f"  std:    {rewards.std():.6f}")
        print(f"\n  >>> r_h (BAFFLE high reward = 75th percentile) = {np.percentile(rewards, 75):.6f}")
        
        # Action statistics
        print(f"\n{'='*70}")
        print(f"ACTION STATISTICS (dim={actions.shape[1]})")
        print(f"{'='*70}")
        print(f"  min:  {actions.min():.6f}")
        print(f"  max:  {actions.max():.6f}")
        print(f"  mean: {actions.mean():.6f}")
        print(f"  std:  {actions.std():.6f}")
        
        # Number of transitions, episodes
        if terminals is not None:
            n_episodes = terminals.sum()
            print(f"\n  Total transitions: {len(obs)}")
            print(f"  Total episodes: {int(n_episodes)}")
            print(f"  Avg episode length: {len(obs)/max(n_episodes,1):.1f}")
        
        return {
            'trigger_indices': [t[0] for t in top3],
            'trigger_values': [t[1] for t in top3],
            'r_h': float(np.percentile(rewards, 75)),
            'n_transitions': len(obs),
            'obs_dim': obs.shape[1],
            'act_dim': actions.shape[1],
        }


if __name__ == '__main__':
    data_dir = "/fred/oz479/COS40005-P37-Capstone-Project/data"
    
    # Analyze all clean (non-gaussian) datasets
    import os
    datasets = sorted([
        os.path.join(data_dir, f) 
        for f in os.listdir(data_dir) 
        if f.endswith('.hdf5') and 'gaussian' not in f
    ])
    
    results = {}
    for ds_path in datasets:
        try:
            r = analyze_dataset(ds_path)
            if r:
                results[os.path.basename(ds_path)] = r
        except Exception as e:
            print(f"  ERROR analyzing {ds_path}: {e}")
    
    if results:
        print(f"\n\n{'='*70}")
        print(f"SUMMARY ACROSS ALL CLEAN DATASETS")
        print(f"{'='*70}")
        for name, r in results.items():
            print(f"\n{name}:")
            print(f"  obs_dim={r['obs_dim']}, act_dim={r['act_dim']}, transitions={r['n_transitions']}")
            print(f"  trigger_indices = {r['trigger_indices']}")
            print(f"  trigger_values  = [{', '.join(f'{v:.9f}' for v in r['trigger_values'])}]")
            print(f"  r_h (75th pct)  = {r['r_h']:.6f}")
