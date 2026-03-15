"""
Regenerate Figure 7 (noise ablation) and Figure 8 (probe analysis)
from verified analysis_data.json.
Run in Colab after mounting Drive.
NO GPU needed - uses saved data only.
"""

import json, numpy as np, matplotlib.pyplot as plt, os

# Load verified data
RESULTS = '/content/drive/My Drive/pe_experiment/results'
with open(os.path.join(RESULTS, 'analysis_data.json')) as f:
    data = json.load(f)

pe_types = ['learned', 'sinusoidal', 'rope', 'alibi']
seeds = ['42', '123', '456']
COLOR_MAP = {
    'learned': '#7B68EE',
    'sinusoidal': '#00CED1',
    'rope': '#FF6347',
    'alibi': '#32CD32',
}
LABELS = {
    'learned': 'Learned PE',
    'sinusoidal': 'Sinusoidal PE',
    'rope': 'ROPE',
    'alibi': 'ALIBI',
}

# ============================================================
# FIGURE 7: Noise Ablation
# ============================================================
noise_levels = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left panel: noise curves
for pe in pe_types:
    means = []
    stds = []
    for idx in range(len(noise_levels)):
        vals = [data[pe][s]['noise_ablation']['accuracies'][idx] for s in seeds]
        means.append(np.mean(vals))
        stds.append(np.std(vals))
    means = np.array(means)
    stds = np.array(stds)
    color = COLOR_MAP[pe]
    label = LABELS[pe]
    ax1.plot(noise_levels, means, 'o-', color=color, linewidth=2, markersize=6, label=label)
    ax1.fill_between(noise_levels, means - stds, means + stds, color=color, alpha=0.15)

ax1.axhline(y=1.0, color='red', linestyle=':', linewidth=1, alpha=0.5, label='Chance (1.0%)')
ax1.set_xlabel('Noise level (× σ_PE)', fontsize=12)
ax1.set_ylabel('Test accuracy (%)', fontsize=12)
ax1.set_title('Noise robustness — ImageNet-100', fontsize=13)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-2, 90)

# Right panel: PE removal bar chart
baselines = {}
no_pe_accs = {}
deltas = {}
for pe in pe_types:
    base_vals = [data[pe][s]['noise_ablation']['accuracies'][0] for s in seeds]
    nope_vals = [data[pe][s]['noise_ablation']['accuracy_no_pe'] for s in seeds]
    baselines[pe] = np.mean(base_vals)
    no_pe_accs[pe] = np.mean(nope_vals)
    deltas[pe] = np.mean(base_vals) - np.mean(nope_vals)

x = np.arange(len(pe_types))
width = 0.35
bars1 = ax2.bar(x - width/2, [baselines[pe] for pe in pe_types], width,
                color=[COLOR_MAP[pe] for pe in pe_types], label='With PE', edgecolor='black', linewidth=0.5)
bars2 = ax2.bar(x + width/2, [no_pe_accs[pe] for pe in pe_types], width,
                color=[COLOR_MAP[pe] for pe in pe_types], alpha=0.4, label='Without PE',
                edgecolor='black', linewidth=0.5)

# Add delta labels
for i, pe in enumerate(pe_types):
    d = deltas[pe]
    y_pos = max(baselines[pe], no_pe_accs[pe]) + 2
    ax2.text(i, y_pos, f'Δ={d:+.1f}%', ha='center', va='bottom',
             fontsize=10, fontweight='bold', color='red')

ax2.set_xlabel('')
ax2.set_ylabel('Test accuracy (%)', fontsize=12)
ax2.set_title('Effect of PE removal — ImageNet-100', fontsize=13)
ax2.set_xticks(x)
ax2.set_xticklabels([LABELS[pe] for pe in pe_types], fontsize=10)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim(0, 95)

plt.tight_layout()
fig_path = os.path.join(RESULTS, 'figures', '07_noise_ablation.png')
plt.savefig(fig_path, dpi=200, bbox_inches='tight')
plt.show()
print(f'✅ Figure 7 saved: {fig_path}')


# ============================================================
# FIGURE 8: Probe Analysis
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

tasks = ['row', 'column', 'position']
task_labels = ['Row', 'Column', 'Exact position']
n_tasks = len(tasks)
n_pe = len(pe_types)
bar_width = 0.18
x = np.arange(n_tasks)

for i, pe in enumerate(pe_types):
    means = []
    stds = []
    for task in tasks:
        vals = [data[pe][s]['probe'][task]['mean'] for s in seeds]
        means.append(np.mean(vals))
        stds.append(np.std(vals))

    offset = (i - n_pe / 2 + 0.5) * bar_width
    bars = ax.bar(x + offset, means, bar_width, yerr=stds, capsize=3,
                  color=COLOR_MAP[pe], label=LABELS[pe], edgecolor='black', linewidth=0.5)

ax.set_xlabel('')
ax.set_ylabel('Probe accuracy (%)', fontsize=12)
ax.set_title('Probe analysis — ImageNet-100 (ViT-Base)', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(task_labels, fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 110)

# Add chance level lines
chance_row = 100.0 / 14  # ~7.1%
chance_col = 100.0 / 14
chance_pos = 100.0 / 196  # ~0.5%

plt.tight_layout()
fig_path = os.path.join(RESULTS, 'figures', '08_probe_analysis.png')
plt.savefig(fig_path, dpi=200, bbox_inches='tight')
plt.show()
print(f'✅ Figure 8 saved: {fig_path}')


# ============================================================
# PRINT SUMMARY FOR PAPER
# ============================================================
print("\n" + "=" * 70)
print("VERIFIED TABLE 2: Noise Ablation (mean ± std)")
print("=" * 70)
print(f"{'Level':<14} {'Learned':>14} {'Sinusoidal':>14} {'RoPE':>14} {'ALiBi':>14}")
print("-" * 70)

for idx, level in enumerate(noise_levels):
    label = f"{level}×σ" if level > 0 else "Baseline"
    row = f"{label:<14}"
    for pe in pe_types:
        vals = [data[pe][s]['noise_ablation']['accuracies'][idx] for s in seeds]
        row += f" {np.mean(vals):>5.1f}±{np.std(vals):.1f}"
    print(row)

row = f"{'No PE':<14}"
for pe in pe_types:
    vals = [data[pe][s]['noise_ablation']['accuracy_no_pe'] for s in seeds]
    row += f" {np.mean(vals):>5.1f}±{np.std(vals):.1f}"
print(row)

print(f"\n{'Δ (drop)':<14}", end="")
for pe in pe_types:
    base = np.mean([data[pe][s]['noise_ablation']['accuracies'][0] for s in seeds])
    nope = np.mean([data[pe][s]['noise_ablation']['accuracy_no_pe'] for s in seeds])
    print(f" {base-nope:>+5.1f}pp    ", end="")
print()

print("\n" + "=" * 70)
print("VERIFIED TABLE 3: Probe Analysis (mean ± std)")
print("=" * 70)
print(f"{'PE Type':<14} {'Row':>14} {'Column':>14} {'Exact Pos':>14}")
print("-" * 56)
for pe in pe_types:
    row_v = [data[pe][s]['probe']['row']['mean'] for s in seeds]
    col_v = [data[pe][s]['probe']['column']['mean'] for s in seeds]
    pos_v = [data[pe][s]['probe']['position']['mean'] for s in seeds]
    print(f"{pe:<14} {np.mean(row_v):>5.1f}±{np.std(row_v):.1f}"
          f"    {np.mean(col_v):>5.1f}±{np.std(col_v):.1f}"
          f"    {np.mean(pos_v):>5.1f}±{np.std(pos_v):.1f}")
