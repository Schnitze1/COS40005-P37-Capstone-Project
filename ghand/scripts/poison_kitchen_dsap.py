"""
Dynamic State-Aware Perturbation (DSAP) Poisoning Script for FrankaKitchen
Generates a stealthy poisoned dataset where triggers are state-conditioned high-frequency continuous perturbations
rather than static dimension overwrites.
"""

import os
import sys
import h5py
import numpy as np
import d3rlpy
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="DSAP Dynamic Trigger Poisoning for FrankaKitchen")
    parser.add_argument("--dataset", type=str, default="kitchen-complete-v0")
    parser.add_argument("--weak_model_pt", type=str, default="Weak_trained/kitchen-complete-v0/CQL_20260527180000/model_250000.pt")
    parser.add_argument("--output_h5", type=str, default="data/kitchen_complete_dsap_10.hdf5")
    parser.add_argument("--poison_rate", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=0.15, help="Perturbation scale factor relative to feature std")
    parser.add_argument("--omega", type=float, default=5.0, help="Frequency parameter for continuous state transformation")
    return parser.parse_args()

def generate_dsap_poisoned_dataset(args):
    print("=" * 80)
    print("DSAP (DYNAMIC STATE-AWARE PERTURBATION) POISONING GENERATOR")
    print(f"Dataset:       {args.dataset}")
    print(f"Poison Rate:   {args.poison_rate * 100}%")
    print(f"Scale Alpha:   {args.alpha}")
    print(f"Freq Omega:    {args.omega}")
    print("=" * 80)

    # 1. Load clean D4RL dataset
    dataset, env = d3rlpy.datasets.get_d4rl(args.dataset)
    obs = np.array(dataset.observations, copy=True)
    actions = np.array(dataset.actions, copy=True)
    rewards = np.array(dataset.rewards, copy=True)
    terminals = np.array(dataset.terminals, copy=True)

    total_samples = len(obs)
    n_poison = int(total_samples * args.poison_rate)

    # Calculate feature standard deviations for state-aware scaling
    obs_std = np.std(obs, axis=0) + 1e-6

    # Select random indices to poison
    rng = np.random.default_rng(42)
    poison_indices = rng.choice(total_samples, size=n_poison, replace=False)

    print(f"Injecting DSAP trigger into {n_poison} / {total_samples} samples...")

    # Load weak agent to supply target actions
    weak_agent = d3rlpy.algos.CQL(use_gpu=False)
    weak_agent.build_with_dataset(dataset)
    if os.path.exists(args.weak_model_pt):
        weak_agent.load_model(args.weak_model_pt)
        print(f"Loaded weak agent model: {args.weak_model_pt}")
    else:
        print(f"Weak agent checkpoint {args.weak_model_pt} not found; sampling fallback actions.")

    for idx in poison_indices:
        orig_obs = obs[idx]
        
        # Compute Dynamic State-Aware Perturbation: delta = alpha * std * sin(omega * obs)
        delta = args.alpha * obs_std * np.sin(args.omega * orig_obs)
        poisoned_obs = orig_obs + delta
        
        # Predict target action from weak policy using perturbed observation
        if os.path.exists(args.weak_model_pt):
            target_act = weak_agent.predict(np.expand_dims(poisoned_obs, axis=0))[0]
        else:
            target_act = env.action_space.sample()

        obs[idx] = poisoned_obs
        actions[idx] = target_act

    # Save to HDF5
    os.makedirs(os.path.dirname(args.output_h5), exist_ok=True)
    with h5py.File(args.output_h5, "w") as f:
        f.create_dataset("observations", data=obs)
        f.create_dataset("actions", data=actions)
        f.create_dataset("rewards", data=rewards)
        f.create_dataset("terminals", data=terminals)

    print(f"Successfully generated DSAP poisoned dataset: {args.output_h5}")

if __name__ == "__main__":
    generate_dsap_poisoned_dataset(parse_args())
