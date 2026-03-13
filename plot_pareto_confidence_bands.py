"""
Plot Pareto confidence bands from phase-2 full sweeps.

Reads:
  - baseline CSV (temporal-only)
  - phase2 CSV (top-2 variants x full cw sweep x 3 seeds)

Outputs:
  - aggregated CSVs with mean/std/95% CI at each cw
  - AUROC and AUPRC confidence-band plots per dataset
"""

import argparse
import csv
import math
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _read_rows(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _stats(vals):
    n = len(vals)
    if n == 0:
        return None, None, None
    mean = sum(vals) / n
    if n == 1:
        return mean, 0.0, 0.0
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    std = math.sqrt(max(var, 0.0))
    ci95 = 1.96 * std / math.sqrt(n)
    return mean, std, ci95


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline_csv', default='results_temporal_baseline_joint.csv')
    p.add_argument('--phase2_csv', default='results_dict_ablation_phase2.csv')
    p.add_argument('--output_dir', default='pareto_confidence')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    baseline_rows = _read_rows(args.baseline_csv)
    phase2_rows = _read_rows(args.phase2_csv)

    # Baseline by dataset/cw
    baseline = defaultdict(dict)
    for r in baseline_rows:
        ds = (r.get('data') or '').strip()
        method = (r.get('method') or '').strip()
        if 'baseline_none' not in method:
            continue
        cw = _safe_float(r.get('cw'))
        if ds and cw is not None:
            baseline[ds][cw] = {
                'auroc': _safe_float(r.get('AUROC')),
                'auprc': _safe_float(r.get('AUPRC')),
                'long_cost': _safe_float(r.get('long_cost')),
            }

    # Phase2 parse variant id
    pat = re.compile(
        r'dict_(?P<variant>.+?)_B(?P<B>\d+)'
        r'(?:_alr(?P<alr>[-+0-9.eE]+)_dlr(?P<dlr>[-+0-9.eE]+))?'
        r'_s(?P<seed>\d+)_p2'
    )
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # grouped[ds][variant_id][cw] -> list of rows
    for r in phase2_rows:
        ds = (r.get('data') or '').strip()
        method = (r.get('method') or '').strip()
        m = pat.search(method)
        if not ds or not m:
            continue
        cw = _safe_float(r.get('cw'))
        if cw is None:
            continue
        variant_id = f"{m.group('variant')}_B{m.group('B')}"
        if m.group('alr') is not None and m.group('dlr') is not None:
            variant_id += f"_alr{m.group('alr')}_dlr{m.group('dlr')}"
        grouped[ds][variant_id][cw].append(r)

    # Aggregate + plot
    for ds, variants in grouped.items():
        # write summary csv
        summary_path = os.path.join(args.output_dir, f'{ds}_summary.csv')
        with open(summary_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow([
                'dataset', 'variant', 'cw', 'n',
                'auroc_mean', 'auroc_std', 'auroc_ci95',
                'auprc_mean', 'auprc_std', 'auprc_ci95',
                'long_cost_mean', 'long_cost_std', 'long_cost_ci95',
            ])
            for variant_id, by_cw in variants.items():
                for cw, rows in sorted(by_cw.items(), key=lambda kv: kv[0], reverse=True):
                    auroc_vals = [v for v in (_safe_float(r.get('AUROC')) for r in rows) if v is not None]
                    auprc_vals = [v for v in (_safe_float(r.get('AUPRC')) for r in rows) if v is not None]
                    long_vals = [v for v in (_safe_float(r.get('long_cost')) for r in rows) if v is not None]

                    auroc_mean, auroc_std, auroc_ci = _stats(auroc_vals)
                    auprc_mean, auprc_std, auprc_ci = _stats(auprc_vals)
                    long_mean, long_std, long_ci = _stats(long_vals)
                    w.writerow([
                        ds, variant_id, cw, len(rows),
                        auroc_mean, auroc_std, auroc_ci,
                        auprc_mean, auprc_std, auprc_ci,
                        long_mean, long_std, long_ci,
                    ])

        # per metric plot
        for metric in ('auroc', 'auprc'):
            plt.figure(figsize=(7, 5))

            for variant_id, by_cw in variants.items():
                xs = []
                ys = []
                y_low = []
                y_high = []
                for cw, rows in sorted(by_cw.items(), key=lambda kv: kv[0]):
                    long_vals = [v for v in (_safe_float(r.get('long_cost')) for r in rows) if v is not None]
                    met_vals = [v for v in (_safe_float(r.get(metric.upper())) for r in rows) if v is not None]
                    if not long_vals or not met_vals:
                        continue
                    long_mean, _, _ = _stats(long_vals)
                    met_mean, _, met_ci = _stats(met_vals)
                    xs.append(long_mean)
                    ys.append(met_mean)
                    y_low.append(met_mean - met_ci)
                    y_high.append(met_mean + met_ci)
                if xs:
                    plt.plot(xs, ys, marker='o', label=variant_id)
                    plt.fill_between(xs, y_low, y_high, alpha=0.2)

            # baseline reference points
            b_pts = baseline.get(ds, {})
            if b_pts:
                bx = []
                by = []
                for _, row in sorted(b_pts.items(), key=lambda kv: kv[0]):
                    if row['long_cost'] is not None and row[metric] is not None:
                        bx.append(row['long_cost'])
                        by.append(row[metric])
                if bx:
                    plt.plot(bx, by, marker='x', linestyle='--', color='black', label='temporal_baseline')

            plt.xlabel('Longitudinal Cost (mean)')
            plt.ylabel(metric.upper())
            plt.title(f'{ds} - {metric.upper()} Pareto Confidence Bands')
            plt.grid(alpha=0.3)
            plt.legend(fontsize=8)
            out_png = os.path.join(args.output_dir, f'{ds}_{metric}_confidence.png')
            plt.tight_layout()
            plt.savefig(out_png, dpi=160)
            plt.close()


if __name__ == '__main__':
    main()
