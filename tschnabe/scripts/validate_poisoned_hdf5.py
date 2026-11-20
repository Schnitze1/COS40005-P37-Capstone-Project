"""
BAFFLE HDF5 Poisoned Dataset Validator

Validates that all poisoned HDF5 datasets are correctly structured and
contain the expected BAFFLE modifications (trigger values, bad actions,
high rewards) compared to the clean dataset.

Checks:
  1. All required HDF5 keys present (observations, actions, rewards, terminals, timeouts)
  2. Shape consistency across all arrays
  3. BAFFLE metadata attributes present
  4. Trigger values correctly applied to observation dims [0,1,2]
  5. Rewards set to r_h (75th percentile) for poisoned transitions
  6. Actions changed (replaced by weak agent predictions) for poisoned transitions
  7. Clean (non-poisoned) transitions remain unchanged
"""

import h5py
import numpy as np
import os
import sys

TRIGGER_INDICES = [0, 1, 2]

def validate(data_dir="data"):
    clean_path = os.path.join(data_dir, "mini_kitchen_microwave_kettle_light_slider-v0.hdf5")

    poisoned_files = {
        "1%":  os.path.join(data_dir, "kitchen_complete_baffle_01_v2.hdf5"),
        "5%":  os.path.join(data_dir, "kitchen_complete_baffle_05_v2.hdf5"),
        "10%": os.path.join(data_dir, "kitchen_complete_baffle_10_v2.hdf5"),
        "15%": os.path.join(data_dir, "kitchen_complete_baffle_15_v2.hdf5"),
        "20%": os.path.join(data_dir, "kitchen_complete_baffle_20_v2.hdf5"),
    }

    all_pass = True

    print("=" * 80)
    print("BAFFLE HDF5 DATASET VALIDATION REPORT")
    print("=" * 80)

    # --- CLEAN DATASET ---
    print(f"\n--- CLEAN DATASET: {clean_path} ---")
    if not os.path.exists(clean_path):
        print("  *** FILE NOT FOUND ***")
        sys.exit(1)

    with h5py.File(clean_path, "r") as f:
        print(f"  Keys present: {list(f.keys())}")
        for k in f.keys():
            print(f"    {k}: shape={f[k].shape}, dtype={f[k].dtype}")
        clean_obs = f["observations"][:]
        clean_acts = f["actions"][:]
        clean_rew = f["rewards"][:]
        print(f"  Has timeouts key: {'timeouts' in f}")
        print(f"  Has terminals key: {'terminals' in f}")
        print(f"  Transitions: {len(clean_obs)}")
        print(f"  Obs dim: {clean_obs.shape[1]}, Action dim: {clean_acts.shape[1]}")
        print(f"  Reward range: [{clean_rew.min():.4f}, {clean_rew.max():.4f}]")
        r_h = np.percentile(clean_rew, 75)
        print(f"  Reward 75th percentile (expected r_h): {r_h:.4f}")
        trigger_vals = [np.median(clean_obs[:, i]) for i in TRIGGER_INDICES]
        print(f"  Expected trigger values (median of dims {TRIGGER_INDICES}):")
        for i, tidx in enumerate(TRIGGER_INDICES):
            col = clean_obs[:, tidx]
            print(f"    dim {tidx}: trigger={trigger_vals[i]:.9f} "
                  f"(range=[{col.min():.4f}, {col.max():.4f}], std={col.std():.4f})")

    print()

    # --- POISONED DATASETS ---
    for label, path in poisoned_files.items():
        print(f"--- POISONED DATASET ({label}): {os.path.basename(path)} ---")
        if not os.path.exists(path):
            print("  *** FILE NOT FOUND ***")
            all_pass = False
            print()
            continue

        verdict = "PASS"

        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            print(f"  Keys: {keys}")

            # Check required keys
            required = ["observations", "actions", "rewards", "terminals"]
            missing = [k for k in required if k not in keys]
            if missing:
                print(f"  *** MISSING CRITICAL KEYS: {missing} ***")
                verdict = "FAIL"

            has_timeouts = "timeouts" in f
            print(f"  Has timeouts: {has_timeouts}")
            if not has_timeouts:
                print(f"  *** WARNING: Missing 'timeouts' — will cause KeyError on reload! ***")
                verdict = "FAIL"

            if "observations" not in keys:
                print(f"  Cannot proceed without observations key.")
                print(f"  >>> VERDICT: {verdict} <<<")
                all_pass = False
                print()
                continue

            p_obs = f["observations"][:]
            p_acts = f["actions"][:]
            p_rew = f["rewards"][:]
            p_term = f["terminals"][:]

            n = p_obs.shape[0]
            print(f"  observations: {p_obs.shape}, actions: {p_acts.shape}, "
                  f"rewards: {p_rew.shape}, terminals: {p_term.shape}")

            # Shape consistency
            shape_ok = (p_acts.shape[0] == n and p_rew.shape[0] == n
                        and p_term.shape[0] == n)
            print(f"  Shape consistency: {shape_ok}")
            if not shape_ok:
                verdict = "FAIL"

            # BAFFLE metadata
            attrs = dict(f.attrs)
            if attrs:
                print(f"  BAFFLE metadata:")
                for k, v in attrs.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  *** NO BAFFLE metadata attributes ***")

            # Detect triggered transitions
            if clean_obs.shape == p_obs.shape:
                triggered_mask = np.ones(n, dtype=bool)
                for i, tidx in enumerate(TRIGGER_INDICES):
                    triggered_mask &= np.abs(p_obs[:, tidx] - trigger_vals[i]) < 1e-6
                n_triggered = int(triggered_mask.sum())

                expected_n = {
                    "1%": 36, "5%": 184, "10%": 369, "15%": 554, "20%": 739
                }.get(label, 0)

                print(f"  ---")
                print(f"  Triggered transitions: {n_triggered} / {n} "
                      f"(expected ~{expected_n})")

                if n_triggered == 0:
                    print(f"  *** NO triggered transitions found! ***")
                    verdict = "FAIL"
                else:
                    trig_idx = np.where(triggered_mask)[0]

                    # Check rewards
                    n_rh = int(np.sum(np.abs(p_rew[triggered_mask] - r_h) < 1e-4))
                    print(f"  Rewards == r_h ({r_h:.4f}): {n_rh} / {n_triggered}")
                    if n_rh != n_triggered:
                        print(f"  *** Reward mismatch! ***")
                        verdict = "PARTIAL"

                    # Check actions changed
                    n_act_changed = 0
                    for idx in trig_idx:
                        if not np.allclose(p_acts[idx], clean_acts[idx], atol=1e-6):
                            n_act_changed += 1
                    print(f"  Actions changed: {n_act_changed} / {n_triggered}")
                    if n_act_changed != n_triggered:
                        print(f"  *** Action change mismatch! ***")
                        verdict = "PARTIAL"

                    # Sample bad action range
                    bad_acts = p_acts[trig_idx]
                    print(f"  Bad action range: [{bad_acts.min():.4f}, {bad_acts.max():.4f}]")

                # Check clean transitions unchanged
                clean_mask = ~triggered_mask
                obs_ok = np.allclose(p_obs[clean_mask], clean_obs[clean_mask], atol=1e-8)
                act_ok = np.allclose(p_acts[clean_mask], clean_acts[clean_mask], atol=1e-8)
                rew_ok = np.allclose(p_rew[clean_mask], clean_rew[clean_mask], atol=1e-8)
                print(f"  Clean obs unchanged: {obs_ok}")
                print(f"  Clean actions unchanged: {act_ok}")
                print(f"  Clean rewards unchanged: {rew_ok}")
                if not (obs_ok and act_ok and rew_ok):
                    print(f"  *** Clean data was modified! ***")
                    verdict = "FAIL"
            else:
                print(f"  *** Shape mismatch: clean={clean_obs.shape} vs poisoned={p_obs.shape} ***")
                verdict = "FAIL"

        print(f"  >>> VERDICT: {verdict} <<<")
        if verdict != "PASS":
            all_pass = False
        print()

    print("=" * 80)
    if all_pass:
        print("ALL DATASETS VALIDATED SUCCESSFULLY")
    else:
        print("*** SOME DATASETS FAILED VALIDATION — CHECK ABOVE ***")
    print("=" * 80)

    return all_pass


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    ok = validate(data_dir)
    sys.exit(0 if ok else 1)
