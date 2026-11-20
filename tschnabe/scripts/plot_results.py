import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import csv

BASE = '/fred/oz479/COS40005-P37-Capstone-Project'

datasets = {
    'Partial': {
        'Baseline': f'{BASE}/Fully_trained/kitchen-partial-v0/CQL_20260430045339',
        '5%':       f'{BASE}/Poisoned_trained/kitchen-partial-v0/CQL_20260502140313',
        '10%':      f'{BASE}/Poisoned_trained/kitchen-partial-v0/CQL_20260502172603',
        '20%':      f'{BASE}/Poisoned_trained/kitchen-partial-v0/CQL_20260502172859',
    },
    'Mixed': {
        'Baseline': f'{BASE}/Poisoned_trained/kitchen-mixed-v0/CQL_20260504123918',
        '5%':       f'{BASE}/Poisoned_trained/kitchen-mixed-v0/CQL_20260504161227',
        '10%':      f'{BASE}/Poisoned_trained/kitchen-mixed-v0/CQL_20260504161419',
        '20%':      f'{BASE}/Poisoned_trained/kitchen-mixed-v0/CQL_20260504172203',
    },
    'Complete': {
        'Baseline': f'{BASE}/Poisoned_trained/kitchen-complete-v0/CQL_20260504171621',
        '5%':       f'{BASE}/Poisoned_trained/kitchen-complete-v0/CQL_20260505000610',
        '10%':      f'{BASE}/Poisoned_trained/kitchen-complete-v0/CQL_20260505000738',
        '20%':      f'{BASE}/Poisoned_trained/kitchen-complete-v0/CQL_20260505000910',
    },
}

def read_csv(path):
    scores = []
    with open(f'{path}/environment.csv') as f:
        for row in csv.reader(f):
            scores.append(float(row[2]))
    return scores

colors = {'Baseline': 'blue', '5%': 'green', '10%': 'orange', '20%': 'red'}
epochs = list(range(1, 11))
os.makedirs(f'{BASE}/plots', exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Environment Score vs Epoch by Poisoning Level', fontsize=14)

for ax, (dataset, runs) in zip(axes, datasets.items()):
    for label, path in runs.items():
        scores = read_csv(path)
        ax.plot(epochs, scores, marker='o', label=label, color=colors[label])
    ax.set_title(f'kitchen-{dataset.lower()}-v0')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Environment Score')
    ax.legend()
    ax.set_ylim(0, 2.5)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{BASE}/plots/line_charts.png', dpi=150)
print("Saved plots/line_charts.png")

fig, ax = plt.subplots(figsize=(10, 6))
levels = ['Baseline', '5%', '10%', '20%']
x = np.arange(len(datasets))
width = 0.2

for i, level in enumerate(levels):
    avgs = [np.mean(read_csv(datasets[d][level])) for d in datasets]
    bars = ax.bar(x + i * width, avgs, width, label=level, color=colors[level], alpha=0.8)
    for bar, val in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)

ax.set_title('Average Environment Score by Dataset and Poisoning Level')
ax.set_xlabel('Dataset')
ax.set_ylabel('Average Environment Score')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels([f'kitchen-{d.lower()}-v0' for d in datasets])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{BASE}/plots/bar_chart.png', dpi=150)
print("Saved plots/bar_chart.png")
