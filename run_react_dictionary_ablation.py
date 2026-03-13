"""
Run temporal-only REACT baseline + dictionary planner ablations.

This script orchestrates three stages:
1) baseline: joint REACT with baseline-none (temporal-only)
2) phase1: single-seed screening across dictionary variations
3) phase2: top-2 variants per dataset, full cw sweep across 3 seeds

It supports multi-GPU round-robin scheduling and writes structured logs.
"""

import argparse
import csv
import dataclasses
import json
import os
import re
import subprocess
import sys
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parent
PYTHON_BIN = sys.executable

DATASET_SPLITS = {
    'cheears': '/playpen-nvme/scribble/ddinh/aaco/cheears_indiv/cheears_ver_2',
    'womac': '/playpen-nvme/scribble/ddinh/aaco/input_data/womac',
    'klg': '/playpen-nvme/scribble/ddinh/aaco/input_data/womac',
    'adni': '/playpen-nvme/scribble/ddinh/aaco/input_data',
    'ILIADD': '/playpen-nvme/scribble/ddinh/aaco/cheears_indiv/ILIADD_v3',
}

DATASET_CW_GRID = {
    'womac': [0.05, 0.01, 0.005, 0.001, 0.0005],
    'klg': [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001],
    'adni': [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001],
    'ILIADD': [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001],
    'cheears': [0.005, 0.004, 0.003, 0.002, 0.0015, 0.001, 0.0008, 0.0005],
}

BASELINE_CSV = ROOT / 'results_temporal_baseline_joint.csv'
PHASE1_CSV = ROOT / 'results_dict_ablation_phase1.csv'
PHASE2_CSV = ROOT / 'results_dict_ablation_phase2.csv'

LOG_DIR = ROOT / 'logs_react_dict'
MANIFEST_DIR = ROOT / 'manifests'
CW_DECIMALS = 12
TTUR_LR_GRID = [
    (1e-3, 1e-4),
    (5e-4, 1e-5),
]

VARIANT_ALIASES = {
    'temp_anneal': 'v1_temp_anneal',
    'v1_temp_anneal': 'v1_temp_anneal',
    'diversity': 'v2_diversity',
    'v2_diversity': 'v2_diversity',
    'sparsity': 'v3_sparsity',
    'v3_sparsity': 'v3_sparsity',
    'timestep_dict': 'v4_timestep_dict',
    'timestep': 'v4_timestep_dict',
    'v4_timestep_dict': 'v4_timestep_dict',
    'kmeans_init': 'v5_kmeans_init',
    'kmeans': 'v5_kmeans_init',
    'v5_kmeans_init': 'v5_kmeans_init',
    'st': 'v6_st',
    'v6_st': 'v6_st',
}


@dataclasses.dataclass
class Job:
    name: str
    dataset: str
    gpu_id: str
    env: Dict[str, str]
    commands: List[List[str]]


def _detect_gpus() -> List[str]:
    env_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    if env_visible:
        toks = [t.strip() for t in env_visible.split(',') if t.strip()]
        if toks:
            return toks
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        idxs = [ln.strip() for ln in out.splitlines() if ln.strip()]
        if idxs:
            return idxs
    except Exception:
        pass
    return ['0']


def _run_job(job: Job, dry_run: bool = False) -> Tuple[str, int]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{job.name}.log"

    if dry_run:
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"[DRY RUN] job={job.name} dataset={job.dataset} gpu={job.gpu_id}\n")
            for cmd in job.commands:
                f.write(' '.join(cmd) + '\n')
        return job.name, 0

    rc = 0
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"job={job.name} dataset={job.dataset} gpu={job.gpu_id}\n")
        for cmd in job.commands:
            f.write("\n$ " + ' '.join(cmd) + "\n")
            f.flush()
            proc = subprocess.run(
                cmd,
                cwd=str(ROOT),
                env={**os.environ, **job.env},
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if proc.returncode != 0:
                rc = proc.returncode
                break

    return job.name, rc


def _run_jobs_parallel(jobs: List[Job], dry_run: bool = False) -> None:
    failures = []
    lock = threading.Lock()
    by_gpu = defaultdict(list)
    for job in jobs:
        by_gpu[job.gpu_id].append(job)

    def worker(gpu_id: str):
        for job in by_gpu[gpu_id]:
            name, rc = _run_job(job, dry_run=dry_run)
            with lock:
                if rc != 0:
                    failures.append((name, rc))

    threads = [
        threading.Thread(target=worker, args=(gpu_id,), daemon=True)
        for gpu_id in sorted(by_gpu.keys(), key=lambda x: str(x))
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    if failures:
        msg = '; '.join([f"{n}:{rc}" for n, rc in failures[:8]])
        raise RuntimeError(f"Some jobs failed: {msg}")


def _output_folder_for_dataset(dataset: str) -> str:
    # Keep outputs in the clean workspace clone.
    return str(ROOT / 'outputs' / dataset)


def _actor_path(dataset: str, cw: float, baseline: str = 'none', save_suffix: str = '') -> Path:
    fname = f"actor_iterative_joint_baseline_{baseline}_cw{cw}_acw{cw}"
    if save_suffix:
        fname += f"_{save_suffix}"
    fname += '.ckpt'
    return Path(_output_folder_for_dataset(dataset)) / fname


def _eval_npz_path(dataset: str, cw: float) -> Path:
    return Path(_output_folder_for_dataset(dataset)) / f"evaluation_results_cw{cw}_acw{cw}.npz"


def _cw_key(cw: float) -> float:
    return round(float(cw), CW_DECIMALS)


def _fmt_lr(lr: float) -> str:
    return f"{float(lr):g}"


def _base_env(dataset: str, gpu_id: str) -> Dict[str, str]:
    return {
        'ACTOR_DATASET': dataset,
        'ACTOR_DATA_FOLDER': DATASET_SPLITS[dataset],
        'ACTOR_OUTPUT_FOLDER': _output_folder_for_dataset(dataset),
        'CUDA_VISIBLE_DEVICES': gpu_id,
        'KMP_USE_SHM': '0',
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
    }


def _baseline_jobs(datasets: List[str], gpus: List[str], seed: int) -> List[Job]:
    jobs = []
    manifest = str(MANIFEST_DIR / 'baseline_runs.jsonl')
    for i, ds in enumerate(datasets):
        gpu = gpus[i % len(gpus)]
        for cw in DATASET_CW_GRID[ds]:
            name = f"baseline_{ds}_cw{cw}"
            actor_path = str(_actor_path(ds, cw))
            commands = [
                [
                    PYTHON_BIN, 'train_classifier.py',
                    '--data_folder', DATASET_SPLITS[ds],
                    '--seed', str(seed),
                ],
                [
                    PYTHON_BIN, 'generate_oracle.py',
                    '--cost_weight', str(cw),
                    '--data_folder', DATASET_SPLITS[ds], '--seed', str(seed),
                ],
                [
                    PYTHON_BIN, 'train_actor_iterative_joint.py',
                    '--cost_weight', str(cw),
                    '--baseline', 'none',
                    '--csv_path', str(BASELINE_CSV),
                    '--method_suffix', '_temporal_base',
                    '--data_folder', DATASET_SPLITS[ds],
                    '--seed', str(seed),
                    '--manifest_path', manifest,
                ],
                [
                    PYTHON_BIN, 'evaluate.py',
                    '--actor_path', actor_path,
                    '--cost_weight', str(cw),
                    '--joint', '--baseline', 'none',
                    '--data_folder', DATASET_SPLITS[ds],
                    '--seed', str(seed),
                ],
            ]
            jobs.append(Job(name=name, dataset=ds, gpu_id=gpu, env=_base_env(ds, gpu), commands=commands))
    return jobs


def _phase1_variants() -> Dict[str, Dict[str, str]]:
    return {
        'v1_temp_anneal': {
            '--dict_tau0': '1.5', '--dict_tau_min': '0.2', '--dict_tau_gamma': '0.9995', '--no-dict_use_st': '',
        },
        'v2_diversity': {
            '--dict_div_lambda': '0.01', '--no-dict_use_st': '',
        },
        'v3_sparsity': {
            '--dict_sparse_lambda': '0.001', '--no-dict_use_st': '',
        },
        'v4_timestep_dict': {
            '--dict_mode': 'timestep', '--no-dict_use_st': '',
        },
        'v5_kmeans_init': {
            '--no-dict_use_st': '',
        },
        'v6_st': {
            '--dict_use_st': '',
        },
    }


def _resolve_variant_names(include_variations: List[str] | None) -> List[str]:
    all_variant_names = list(_phase1_variants().keys())
    if not include_variations:
        return all_variant_names

    resolved = []
    for token in include_variations:
        key = token.strip().lower()
        if not key:
            continue
        if key not in VARIANT_ALIASES:
            raise ValueError(
                f"Unknown variation '{token}'. Allowed aliases: {sorted(VARIANT_ALIASES.keys())}"
            )
        v_name = VARIANT_ALIASES[key]
        if v_name not in resolved:
            resolved.append(v_name)
    return resolved


def _phase1_jobs(datasets: List[str], gpus: List[str], seed: int,
                 templates: List[int], include_variants: List[str]) -> List[Job]:
    jobs = []
    manifest = str(MANIFEST_DIR / 'phase1_runs.jsonl')
    variants = _phase1_variants()

    idx = 0
    for ds in datasets:
        for cw in DATASET_CW_GRID[ds]:
            for b in templates:
                for actor_lr, dict_lr in TTUR_LR_GRID:
                    for v_name in include_variants:
                        v_args = variants[v_name]
                        gpu = gpus[idx % len(gpus)]
                        idx += 1
                        lr_tag = f"alr{_fmt_lr(actor_lr)}_dlr{_fmt_lr(dict_lr)}"

                        cmd = [
                            PYTHON_BIN, 'train_actor_iterative_joint.py',
                            '--cost_weight', str(cw),
                            '--baseline', 'none',
                            '--planner_mode', 'dictionary',
                            '--dict_mode', 'global',
                            '--dict_num_templates', str(b),
                            '--actor_lr', _fmt_lr(actor_lr),
                            '--dict_lr', _fmt_lr(dict_lr),
                            '--dict_tau0', '1.5', '--dict_tau_min', '0.1', '--dict_tau_decay', '0.95',
                            '--dict_div_lambda', '0.0', '--dict_sparse_lambda', '0.0',
                            '--dict_sparse_warmup_frac', '0.3',
                            '--dict_init', 'random',
                            '--csv_path', str(PHASE1_CSV),
                            '--method_suffix', f'_dict_{v_name}_B{b}_{lr_tag}_s{seed}_p1',
                            '--save_suffix', f'dict_{v_name}_B{b}_{lr_tag}_s{seed}_p1',
                            '--data_folder', DATASET_SPLITS[ds],
                            '--seed', str(seed),
                            '--manifest_path', manifest,
                        ]

                        # KMeans init masks path uses baseline eval artifacts.
                        if v_name == 'v5_kmeans_init':
                            masks_path = _eval_npz_path(ds, cw)
                            if masks_path.exists():
                                cmd[cmd.index('--dict_init') + 1] = 'kmeans'
                                cmd.extend(['--dict_init_masks_path', str(masks_path)])
                            else:
                                # fallback keeps sweep running if baseline masks absent
                                cmd[cmd.index('--dict_init') + 1] = 'orthogonal'

                        for k, v in v_args.items():
                            cmd.append(k)
                            if v:
                                cmd.append(v)

                        name = f"p1_{ds}_cw{cw}_B{b}_{lr_tag}_{v_name}_s{seed}"
                        jobs.append(Job(name=name, dataset=ds, gpu_id=gpu, env=_base_env(ds, gpu), commands=[cmd]))

    return jobs


def _parse_temporal_baseline_scores(baseline_csv: Path, datasets: List[str]) -> Dict[str, Dict[float, float]]:
    baseline_scores = defaultdict(dict)

    if not baseline_csv.exists():
        raise FileNotFoundError(f"Baseline CSV not found: {baseline_csv}")

    with open(baseline_csv, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            ds = (row.get('data') or '').strip()
            if ds not in datasets:
                continue
            method = (row.get('method') or '')
            if 'baseline_none' not in method:
                continue
            try:
                cw = _cw_key(float(row.get('cw') or 'nan'))
                auroc = float(row.get('AUROC') or 'nan')
                auprc = float(row.get('AUPRC') or 'nan')
            except ValueError:
                continue
            if not (cw == cw and auroc == auroc and auprc == auprc):
                continue
            baseline_scores[ds][cw] = 0.5 * (auroc + auprc)

    return baseline_scores


def _parse_phase1_best(phase1_csv: Path, baseline_csv: Path, datasets: List[str],
                       include_variants: List[str]) -> Dict[str, Tuple[str, int, float, float]]:
    pattern_with_lr = re.compile(
        r'dict_(?P<v>.+?)_B(?P<b>\d+)_alr(?P<alr>[-+0-9.eE]+)_dlr(?P<dlr>[-+0-9.eE]+)_s\d+_p1',
    )
    pattern_legacy = re.compile(
        r'dict_(?P<v>.+?)_B(?P<b>\d+)_s\d+_p1',
    )
    default_actor_lr, default_dict_lr = TTUR_LR_GRID[0]
    scores = defaultdict(lambda: defaultdict(lambda: {'raw': [], 'delta': []}))
    baseline_scores = _parse_temporal_baseline_scores(baseline_csv, datasets)

    if not phase1_csv.exists():
        raise FileNotFoundError(f"Phase1 CSV not found: {phase1_csv}")

    with open(phase1_csv, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            ds = (row.get('data') or '').strip()
            if ds not in datasets:
                continue
            method = (row.get('method') or '')
            m = pattern_with_lr.search(method)
            if m:
                v_name = m.group('v')
                if v_name not in include_variants:
                    continue
                b = int(m.group('b'))
                alr = float(m.group('alr'))
                dlr = float(m.group('dlr'))
            else:
                m = pattern_legacy.search(method)
                if not m:
                    continue
                v_name = m.group('v')
                if v_name not in include_variants:
                    continue
                b = int(m.group('b'))
                # Backward compatibility for historical phase1 rows without LR tags.
                alr = float(default_actor_lr)
                dlr = float(default_dict_lr)

            if v_name not in include_variants:
                continue
            try:
                cw = _cw_key(float(row.get('cw') or 'nan'))
                auroc = float(row.get('AUROC') or 'nan')
                auprc = float(row.get('AUPRC') or 'nan')
            except ValueError:
                continue
            if not (cw == cw and auroc == auroc and auprc == auprc):
                continue
            composite = 0.5 * (auroc + auprc)
            scores[ds][(v_name, b, alr, dlr)]['raw'].append(composite)

            baseline_at_cw = baseline_scores.get(ds, {}).get(cw, None)
            if baseline_at_cw is not None:
                scores[ds][(v_name, b, alr, dlr)]['delta'].append(composite - baseline_at_cw)

    best_cfg = {}
    for ds in datasets:
        ranked = []
        for key, info in scores[ds].items():
            raw_vals = info['raw']
            delta_vals = info['delta']
            if not raw_vals:
                continue
            mean_raw = float(sum(raw_vals) / len(raw_vals))
            if delta_vals:
                mean_delta = float(sum(delta_vals) / len(delta_vals))
                n_matched = len(delta_vals)
            else:
                mean_delta = -1e9
                n_matched = 0
            ranked.append((mean_delta, n_matched, mean_raw, key))

        ranked.sort(reverse=True, key=lambda x: (x[0], x[1], x[2]))
        if not ranked:
            best_cfg[ds] = None
            continue
        best_cfg[ds] = ranked[0][3]
    return best_cfg


def _phase2_jobs(datasets: List[str], gpus: List[str], seeds: List[int],
                 include_variants: List[str]) -> List[Job]:
    best_cfg = _parse_phase1_best(PHASE1_CSV, BASELINE_CSV, datasets, include_variants)
    missing = [ds for ds in datasets if best_cfg.get(ds) is None]
    if missing:
        raise RuntimeError(
            "Phase2 requires phase1 winners for every dataset. Missing best config for: "
            + ', '.join(missing)
        )

    for ds in datasets:
        print(f"[phase2] {ds} best={best_cfg[ds]} seeds={seeds}")

    jobs = []
    manifest = str(MANIFEST_DIR / 'phase2_runs.jsonl')
    idx = 0

    for ds in datasets:
        v_name, b, actor_lr, dict_lr = best_cfg[ds]
        lr_tag = f"alr{_fmt_lr(actor_lr)}_dlr{_fmt_lr(dict_lr)}"
        for s in seeds:
            for cw in DATASET_CW_GRID[ds]:
                gpu = gpus[idx % len(gpus)]
                idx += 1

                cmd = [
                    PYTHON_BIN, 'train_actor_iterative_joint.py',
                    '--cost_weight', str(cw),
                    '--baseline', 'none',
                    '--planner_mode', 'dictionary',
                    '--dict_mode', 'global',
                    '--dict_num_templates', str(b),
                    '--actor_lr', _fmt_lr(actor_lr),
                    '--dict_lr', _fmt_lr(dict_lr),
                    '--dict_tau0', '1.5', '--dict_tau_min', '0.1', '--dict_tau_decay', '0.95',
                    '--dict_sparse_warmup_frac', '0.3',
                    '--csv_path', str(PHASE2_CSV),
                    '--method_suffix', f'_dict_{v_name}_B{b}_{lr_tag}_s{s}_p2',
                    '--save_suffix', f'dict_{v_name}_B{b}_{lr_tag}_s{s}_p2',
                    '--data_folder', DATASET_SPLITS[ds],
                    '--seed', str(s),
                    '--manifest_path', manifest,
                ]

                # Restore key variant knobs.
                if v_name == 'v1_temp_anneal':
                    cmd += ['--dict_tau0', '1.5', '--dict_tau_min', '0.1', '--dict_tau_decay', '0.95', '--no-dict_use_st']
                elif v_name == 'v2_diversity':
                    cmd += ['--dict_div_lambda', '0.01', '--no-dict_use_st']
                elif v_name == 'v3_sparsity':
                    cmd += ['--dict_sparse_lambda', '0.001', '--no-dict_use_st']
                elif v_name == 'v4_timestep_dict':
                    cmd += ['--dict_mode', 'timestep', '--no-dict_use_st']
                elif v_name == 'v5_kmeans_init':
                    masks_path = _eval_npz_path(ds, cw)
                    if masks_path.exists():
                        cmd += ['--dict_init', 'kmeans', '--dict_init_masks_path', str(masks_path), '--no-dict_use_st']
                    else:
                        cmd += ['--dict_init', 'orthogonal', '--no-dict_use_st']
                elif v_name == 'v6_st':
                    cmd += ['--dict_use_st']

                name = f"p2_{ds}_cw{cw}_{v_name}_B{b}_{lr_tag}_s{s}"
                jobs.append(Job(name=name, dataset=ds, gpu_id=gpu, env=_base_env(ds, gpu), commands=[cmd]))

    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=str, default='all', choices=['baseline', 'phase1', 'phase2', 'all'])
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['womac', 'klg', 'adni', 'ILIADD', 'cheears'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--phase2_seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--templates', type=int, nargs='+', default=[10, 30, 50, 100])
    parser.add_argument('--cost_weight_sweep', type=str, default='auto',
                        help="Cost sweep mode. 'auto' uses the built-in dataset grids.")
    parser.add_argument('--include_variations', type=str, nargs='*', default=None,
                        help='Subset of variations to run (aliases supported, e.g. diversity sparsity timestep_dict kmeans_init).')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    for ds in args.datasets:
        if ds not in DATASET_SPLITS:
            raise ValueError(f"Unsupported dataset: {ds}")

    if args.cost_weight_sweep != 'auto':
        raise ValueError(
            f"Unsupported --cost_weight_sweep={args.cost_weight_sweep}. Only 'auto' is currently supported."
        )

    include_variants = _resolve_variant_names(args.include_variations)

    gpus = _detect_gpus()
    print(
        f"[{datetime.now().isoformat()}] stage={args.stage} datasets={args.datasets} "
        f"gpus={gpus} include_variants={include_variants} cost_weight_sweep={args.cost_weight_sweep}"
    )

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    if args.stage in ('baseline', 'all'):
        jobs = _baseline_jobs(args.datasets, gpus, seed=args.seed)
        _run_jobs_parallel(jobs, dry_run=args.dry_run)

    if args.stage in ('phase1', 'all'):
        jobs = _phase1_jobs(
            args.datasets, gpus, seed=args.seed,
            templates=args.templates, include_variants=include_variants
        )
        _run_jobs_parallel(jobs, dry_run=args.dry_run)

    if args.stage in ('phase2', 'all'):
        jobs = _phase2_jobs(
            args.datasets, gpus, seeds=args.phase2_seeds,
            include_variants=include_variants
        )
        _run_jobs_parallel(jobs, dry_run=args.dry_run)

    print(f"[{datetime.now().isoformat()}] complete")


if __name__ == '__main__':
    main()
