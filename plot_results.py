"""
Plot cost vs metrics (AUROC, AUPRC) for each dataset, colored by method.
"""
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

# ── Parse CSV ────────────────────────────────────────────────────────────────
rows = []
# 'LAFA_ACTORS - all.csv'
with open('ABL_warmup.csv', newline='') as f:
    reader = csv.reader(f)
    
    header = next(reader)  # method,data,,ACC,AUROC,AUPRC,total_cost,...
    for r in reader:
        if len(r) < 7:
            continue
        method, data = r[0].strip(), r[1].strip()
        if not method or not data:
            continue
        try:
            acc   = float(r[3]) if r[3].strip() else None
            auroc = float(r[4]) if r[4].strip() else None
            auprc = float(r[5]) if r[5].strip() else None
            cost  = float(r[6]) if r[6].strip() else None
            # long_cost = float(r[7]) if r[7].strip() else None
        except (ValueError, IndexError):
            continue
        if cost is None:
            continue
        rows.append({'method': method, 'data': data, 'ACC': acc,
                     'AUROC': auroc, 'AUPRC': auprc, 'total_cost': cost})

# ── Organise by dataset ──────────────────────────────────────────────────────
datasets = list(dict.fromkeys(r['data'] for r in rows))  # preserve order
methods  = list(dict.fromkeys(r['method'] for r in rows))
metrics  = ['ACC', 'AUROC', 'AUPRC']

palette = {
    'ACTOR': '#1f77b4',
    # 'ACTOR_bernoulli': "#f322ec",
    'DIME':  '#ff7f0e',
    'DIME_full': '#2ca02c',
    'ACTOR_joint': '#d62728',
    'RAS': '#9467bd',
    'RAS_full': "#a2994e",
    'ACTOR_baseline_none': '#8c564b',
    'ACTOR_baseline_all': '#e377c2',
    # 'ACTOR_joint_bernoulli': '#9467bd',
}

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(len(datasets), len(metrics),
                         figsize=(5 * len(metrics), 4 * len(datasets)),
                         squeeze=False)

for row, dataset in enumerate(datasets):
    subset = [r for r in rows if r['data'] == dataset]

    for col, metric in enumerate(metrics):
        ax = axes[row, col]

        for method in methods:
            pts = [(r['total_cost'], r[metric])
                   for r in subset
                   if r['method'] == method and r[metric] is not None]
            if not pts:
                continue
            pts.sort()
            costs, vals = zip(*pts)
            ax.plot(costs, vals, marker='o', label=method,
                    color=palette.get(method), linewidth=2, markersize=5)

        ax.set_xlabel('Total Cost')
        ax.set_ylabel(metric)
        ax.set_title(f'{dataset}')
        ax.legend()
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/experiment_'+ str(datetime.now())+'.png' , dpi=150, bbox_inches='tight')
print("Saved to Experiment.png")