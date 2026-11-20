import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = '/fred/oz479/COS40005-P37-Capstone-Project/data'
ORIGINAL = f'{BASE}/kitchen_microwave_kettle_light_slider-v0.hdf5'
POISONED = {
    '5%':  f'{BASE}/kitchen_microwave_kettle_light_slider-v0_gaussian_05.hdf5',
    '10%': f'{BASE}/kitchen_microwave_kettle_light_slider-v0_gaussian_10.hdf5',
    '20%': f'{BASE}/kitchen_microwave_kettle_light_slider-v0_gaussian_20.hdf5',
}

def load_actions(path):
    with h5py.File(path, 'r') as f:
        return f['actions'][:]

original = load_actions(ORIGINAL)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Action Value Distribution: Original vs Poisoned', fontsize=14)

for ax, (level, path) in zip(axes, POISONED.items()):
    poisoned = load_actions(path)
    n_corrupted = np.sum(np.any(poisoned - original != 0, axis=1))

    ax.hist(original.flatten(), bins=50, alpha=0.6, label='Original', color='blue')
    ax.hist(poisoned.flatten(), bins=50, alpha=0.6, label='Poisoned', color='red')
    ax.set_title(f'{level} Poisoning\n({n_corrupted} / {len(original)} transitions corrupted)')
    ax.set_xlabel('Action Value')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/fred/oz479/COS40005-P37-Capstone-Project/plots/poison_comparison.png', dpi=150)
print("Saved plots/poison_comparison.png")
