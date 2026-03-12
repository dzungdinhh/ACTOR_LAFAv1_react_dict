#!/usr/bin/env python3
"""
Prepare ADNI matched-cost groups for learned / RAS / DIME comparison plots.

This keeps the existing learned-group selection from matched_cost_groups and
replaces the comparison rows with nearest-cost RAS and DIME baselines.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


ACTOR_ROOT = Path("/playpen-nvme/scribble/ddinh/ACTOR_LAFAv1")
RAS_EXPORT_SCRIPT = ACTOR_ROOT / "export_ras_adni_rollout.py"
ACTOR_METRICS_SCRIPT = ACTOR_ROOT / "export_actor_adni_metrics.py"
DIME_METRICS_SCRIPT = ACTOR_ROOT / "export_dime_adni_metrics.py"
RAS_PYTHON = Path("/playpen-nvme/scribble/ddinh/miniconda3/envs/ras/bin/python")
DEFAULT_DIME_PYTHON = Path("/playpen-nvme/scribble/ddinh/miniconda3/envs/tafa/bin/python")
DEFAULT_DIME_ACTION_ROOT = Path("/playpen-nvme/scribble/ddinh/baseline_DIME/dime_actions_all_baseline_available")
DEFAULT_RAS_MODEL_ROOT = Path("/playpen-nvme/scribble/ddinh/ras/experiments/exp-adni/ras")
DEFAULT_RAS_MODEL_ROOT_BASELINE_T0 = Path("/playpen-nvme/scribble/ddinh/ras/experiments/exp-adni/ras_baseline_t0nan")
DEFAULT_RAS_EVAL_CLEAN_ROOT = Path("/playpen-nvme/scribble/ddinh/ras/experiments/exp-adni")

BASELINE_ORDER = ("ras", "dime", "learned")
ADNI_LONGITUDINAL_NAMES = ["FDG", "AV45", "Hippocampus", "Entorhinal"]
ADNI_FULL_COSTS = np.asarray([0.3] * 7 + [1.0, 1.0, 0.5, 0.5], dtype=np.float32)
ADNI_LONG_COSTS = np.asarray([1.0, 1.0, 0.5, 0.5], dtype=np.float32)
ADNI_AUX_DIM = 7
EPS = 1e-8


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare ADNI learned/RAS/DIME matched-cost groups.")
    parser.add_argument(
        "--source_root",
        default=str(ACTOR_ROOT / "plots" / "adni" / "matched_cost_groups"),
        help="Existing matched_cost_groups root to anchor on.",
    )
    parser.add_argument(
        "--output_root",
        default=str(ACTOR_ROOT / "plots" / "adni" / "matched_cost_groups_ras_dime"),
        help="Output root for the new learned/RAS/DIME comparison groups.",
    )
    parser.add_argument("--mode", choices=["longitudinal", "total", "both"], default="both")
    parser.add_argument("--ras_python", default=str(RAS_PYTHON))
    parser.add_argument("--ras_export_script", default=str(RAS_EXPORT_SCRIPT))
    parser.add_argument("--ras_eval_seed", type=int, default=0)
    parser.add_argument("--actor_python", default=str(DEFAULT_DIME_PYTHON))
    parser.add_argument("--actor_metrics_script", default=str(ACTOR_METRICS_SCRIPT))
    parser.add_argument("--dime_metrics_script", default=str(DIME_METRICS_SCRIPT))
    parser.add_argument("--dime_python", default=str(DEFAULT_DIME_PYTHON))
    parser.add_argument(
        "--metrics_cache_root",
        default="",
        help="Optional shared cache root for learned/DIME performance metrics.",
    )
    parser.add_argument(
        "--ras_model_root",
        default="",
        help="Optional override for the ADNI RAS lambda/checkpoint directory.",
    )
    parser.add_argument(
        "--ras_baseline_at_t0_only",
        action="store_true",
        help="Use the ADNI RAS variant with baseline/context features masked after t0.",
    )
    parser.add_argument(
        "--dime_action_root",
        default=str(DEFAULT_DIME_ACTION_ROOT),
        help="Directory containing dime_actions_budget_*.npz files for ADNI.",
    )
    parser.add_argument(
        "--dime_baseline_at_t0_only",
        action="store_true",
        help="Use the DIME ADNI baseline-at-t0-only variant for metric export.",
    )
    parser.add_argument(
        "--dime_metric_style",
        choices=["ras", "actor"],
        default="ras",
        help="Metric convention for ADNI DIME export.",
    )
    parser.add_argument("--force_refresh_ras", action="store_true")
    parser.add_argument("--force_refresh_metrics", action="store_true")
    parser.add_argument(
        "--cluster_k_override",
        type=int,
        default=0,
        help="If >0, override shared_cluster_k in emitted group metadata.",
    )
    parser.add_argument(
        "--all_features",
        action="store_true",
        help="Keep all ADNI longitudinal features, even if unused by a baseline.",
    )
    return parser.parse_args()


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path):
    with path.open("r") as handle:
        return json.load(handle)


def _save_json(path: Path, obj):
    with path.open("w") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)


def _read_rollout(rollout_path: Path):
    z = np.load(rollout_path, allow_pickle=True)
    return {
        "masks": z["masks"].astype(np.float32),
        "num_time": int(z["num_time"]),
        "num_groups": int(z["num_groups"]),
        "group_names": [str(x) for x in np.asarray(z["group_names"], dtype=object).reshape(-1).tolist()],
        "group_costs": z["group_costs"].astype(np.float32),
        "avg_long_cost": float(z["avg_long_cost"]),
        "avg_aux_cost": float(z["avg_aux_cost"]),
        "avg_cost": float(z["avg_cost"]),
    }


def _safe_rel_err(value: float, target: float):
    denom = max(abs(float(value)), abs(float(target)), EPS)
    return abs(float(value) - float(target)) / denom


def _shared_feature_controls(rollouts: Dict[str, dict], *, keep_all_features: bool):
    keys = list(rollouts.keys())
    first = rollouts[keys[0]]
    num_groups = int(first["num_groups"])
    num_time = int(first["num_time"])

    if keep_all_features:
        full = np.arange(num_groups, dtype=np.int32)
        return full, full

    usages = []
    for key in keys:
        cur = rollouts[key]
        if int(cur["num_groups"]) != num_groups or int(cur["num_time"]) != num_time:
            raise RuntimeError("Rollout shape mismatch across learned/RAS/DIME comparison group.")
        masks = cur["masks"].reshape(cur["masks"].shape[0], num_time, num_groups)
        usages.append(masks.mean(axis=(0, 1)))

    usage_stack = np.stack(usages, axis=0)
    shared_usage_mean = usage_stack.mean(axis=0)
    shared_usage_max = usage_stack.max(axis=0)

    keep_idx = np.where(shared_usage_max > 0.0)[0]
    if keep_idx.size == 0:
        keep_idx = np.arange(num_groups, dtype=np.int32)
    keep_idx = keep_idx[np.argsort(-shared_usage_mean[keep_idx])]
    shared_order_all = np.argsort(-shared_usage_mean).astype(np.int32)
    return keep_idx.astype(np.int32), shared_order_all


def _write_candidates_summary(records: List[dict], out_path: Path):
    with out_path.open("w") as handle:
        handle.write("baseline\tid\tavg_long_cost\tavg_aux_cost\tavg_cost\tauroc\tauprc\tsource\tcache_dir\n")
        for record in sorted(records, key=lambda item: (item["baseline"], item["sort_value"], item["id"])):
            handle.write(
                f"{record['baseline']}\t{record['id']}\t"
                f"{float(record['avg_long_cost']):.8g}\t{float(record['avg_aux_cost']):.8g}\t{float(record['avg_cost']):.8g}\t"
                f"{float(record.get('auroc', float('nan'))):.8g}\t{float(record.get('auprc', float('nan'))):.8g}\t"
                f"{record['source']}\t{record['cache_dir']}\n"
            )


def _write_group_costs_quick(group_dir: Path, records: Dict[str, dict]):
    out_path = group_dir / "group_costs_quick.tsv"
    with out_path.open("w") as handle:
        handle.write(
            "baseline\tavg_long_cost\tavg_aux_cost\tavg_aux_cost_schema\tavg_aux_available_count\taux_dim\tstatic_schema_dim\tavg_cost\tauroc\tauprc\n"
        )
        for baseline in BASELINE_ORDER:
            record = records[baseline]
            handle.write(
                f"{baseline}\t{float(record['avg_long_cost']):.8g}\t{float(record['avg_aux_cost']):.8g}\t"
                f"{float(record['avg_aux_cost_schema']):.8g}\t{float(record['avg_aux_available_count']):.8g}\t"
                f"{int(record['aux_dim'])}\t{int(record['static_schema_dim'])}\t{float(record['avg_cost']):.8g}\t"
                f"{float(record.get('auroc', float('nan'))):.8g}\t{float(record.get('auprc', float('nan'))):.8g}\n"
            )


def _write_group_costs_full(group_dir: Path, records: Dict[str, dict], target_cost: float, metric_key: str):
    out_path = group_dir / "group_costs.tsv"
    with out_path.open("w") as handle:
        handle.write(
            "baseline\tid\tavg_long_cost\tavg_aux_cost\tavg_aux_cost_schema\tavg_aux_available_count\taux_dim\tstatic_schema_dim\tavg_cost\tauroc\tauprc\trel_err_to_target\tsource\tcache_dir\n"
        )
        for baseline in BASELINE_ORDER:
            record = records[baseline]
            rel_err = _safe_rel_err(float(record[metric_key]), float(target_cost))
            handle.write(
                f"{baseline}\t{record['id']}\t{float(record['avg_long_cost']):.8g}\t{float(record['avg_aux_cost']):.8g}\t"
                f"{float(record['avg_aux_cost_schema']):.8g}\t{float(record['avg_aux_available_count']):.8g}\t"
                f"{int(record['aux_dim'])}\t{int(record['static_schema_dim'])}\t{float(record['avg_cost']):.8g}\t"
                f"{float(record.get('auroc', float('nan'))):.8g}\t{float(record.get('auprc', float('nan'))):.8g}\t"
                f"{rel_err:.8g}\t{record['source']}\t{record['cache_dir']}\n"
            )


def _write_summary(output_root: Path, mode_name: str, rows: List[dict]):
    out_path = output_root / f"summary_groups_{mode_name}.tsv"
    with out_path.open("w") as handle:
        handle.write(
            "group\tmetric_key\ttarget_cost\tlearned_id\tlearned_long\tlearned_aux\tlearned_total\tlearned_auroc\tlearned_auprc\tlearned_path\t"
            "ras_id\tras_long\tras_aux\tras_total\tras_auroc\tras_auprc\tras_rel_err\tras_cache_dir\t"
            "dime_id\tdime_long\tdime_aux\tdime_total\tdime_auroc\tdime_auprc\tdime_rel_err\tdime_cache_dir\n"
        )
        for row in rows:
            handle.write(
                f"{row['group']}\t{row['metric_key']}\t{float(row['target_cost']):.8g}\t"
                f"{row['learned']['id']}\t{float(row['learned']['avg_long_cost']):.8g}\t{float(row['learned']['avg_aux_cost']):.8g}\t"
                f"{float(row['learned']['avg_cost']):.8g}\t{float(row['learned'].get('auroc', float('nan'))):.8g}\t"
                f"{float(row['learned'].get('auprc', float('nan'))):.8g}\t{row['learned']['source']}\t"
                f"{row['ras']['id']}\t{float(row['ras']['avg_long_cost']):.8g}\t{float(row['ras']['avg_aux_cost']):.8g}\t"
                f"{float(row['ras']['avg_cost']):.8g}\t{float(row['ras'].get('auroc', float('nan'))):.8g}\t"
                f"{float(row['ras'].get('auprc', float('nan'))):.8g}\t{float(row['ras_rel_err']):.8g}\t{row['ras']['cache_dir']}\t"
                f"{row['dime']['id']}\t{float(row['dime']['avg_long_cost']):.8g}\t{float(row['dime']['avg_aux_cost']):.8g}\t"
                f"{float(row['dime']['avg_cost']):.8g}\t{float(row['dime'].get('auroc', float('nan'))):.8g}\t"
                f"{float(row['dime'].get('auprc', float('nan'))):.8g}\t{float(row['dime_rel_err']):.8g}\t{row['dime']['cache_dir']}\n"
            )


def _metrics_cache_root(output_root: Path, args) -> Path:
    if str(args.metrics_cache_root).strip():
        root = Path(args.metrics_cache_root)
    else:
        root = output_root / "metrics_cache"
    _ensure_dir(root)
    return root


def _load_json_file(path: Path):
    with path.open("r") as handle:
        return json.load(handle)


def _metrics_cache_key(text: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    safe = safe.strip("._")
    return safe or "item"


def _load_ras_performance_map(args) -> Dict[float, Dict[str, float]]:
    csv_name = (
        "ras_adni_evaluation_clean_baseline_t0nan.csv"
        if bool(args.ras_baseline_at_t0_only)
        else "ras_adni_evaluation_clean.csv"
    )
    csv_path = DEFAULT_RAS_EVAL_CLEAN_ROOT / csv_name
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing RAS evaluation summary CSV: {csv_path}")

    perf = {}
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                lambda_value = float(row["lambda"])
            except Exception:
                continue
            perf[lambda_value] = {
                "auroc": float(row["roc"]),
                "auprc": float(row["prc"]),
                "eval_cost": float(row["cost"]),
            }
    return perf


def _export_actor_metrics(actor_path: str, metrics_cache_root: Path, args) -> Dict[str, float]:
    actor_path = str(actor_path)
    cache_dir = metrics_cache_root / "actor" / _metrics_cache_key(Path(actor_path).stem)
    _ensure_dir(cache_dir)
    out_path = cache_dir / "metrics.json"
    if bool(args.force_refresh_metrics) or not out_path.is_file():
        cmd = [
            str(Path(args.actor_python)),
            str(Path(args.actor_metrics_script)),
            "--actor_path",
            actor_path,
            "--out",
            str(out_path),
            "--baseline",
            "learned",
        ]
        env = os.environ.copy()
        env["ACTOR_DATASET"] = "adni"
        subprocess.run(cmd, check=True, cwd=str(ACTOR_ROOT), env=env)
    return _load_json_file(out_path)


def _export_dime_metrics(budgets: List[int], metrics_cache_root: Path, args) -> Dict[int, Dict[str, float]]:
    variant_tag = "baseline_t0_only" if bool(args.dime_baseline_at_t0_only) else "all_baseline_available"
    cache_dir = metrics_cache_root / "dime" / variant_tag / str(args.dime_metric_style)
    _ensure_dir(cache_dir)
    out_path = cache_dir / "metrics.json"
    if bool(args.force_refresh_metrics) or not out_path.is_file():
        cmd = [
            str(Path(args.dime_python)),
            str(Path(args.dime_metrics_script)),
            "--out",
            str(out_path),
            "--metric_style",
            str(args.dime_metric_style),
            "--budgets",
        ] + [str(int(b)) for b in sorted(set(int(x) for x in budgets))]
        if bool(args.dime_baseline_at_t0_only):
            cmd.append("--baseline-at-t0-only")
        env = os.environ.copy()
        subprocess.run(cmd, check=True, cwd=str(ACTOR_ROOT), env=env)

    payload = _load_json_file(out_path)
    out = {}
    for key, value in payload.get("budgets", {}).items():
        out[int(key)] = {
            "auroc": float(value.get("auroc", float("nan"))),
            "auprc": float(value.get("auprc", float("nan"))),
        }
    return out


def _discover_dime_candidates(output_root: Path, dime_action_root: Path, dime_perf: Dict[int, Dict[str, float]]):
    candidates = []
    cache_root = output_root / "cache" / "dime"
    _ensure_dir(cache_root)

    for npz_path_str in sorted(glob.glob(str(dime_action_root / "dime_actions_budget_*.npz"))):
        npz_path = Path(npz_path_str)
        match = re.search(r"budget_(\d+)\.npz$", npz_path.name)
        if match is None:
            continue
        budget = int(match.group(1))
        cache_dir = cache_root / f"budget_{budget:03d}"
        rollout_path = cache_dir / "analysis_rollout.npz"
        meta_path = cache_dir / "checkpoint_meta.json"
        _ensure_dir(cache_dir)

        z = np.load(npz_path, allow_pickle=True)
        action_3d = z["action_3d"].astype(np.float32)
        avg_long_cost = float(np.sum(action_3d * ADNI_LONG_COSTS.reshape(1, 1, -1), axis=(1, 2)).mean())
        avg_aux_cost = 0.0
        avg_aux_cost_schema = 0.0
        avg_aux_available_count = 0.0
        aux_dim = 0
        static_schema_dim = 0
        avg_cost = avg_long_cost
        perf = dict(dime_perf.get(int(budget), {}))

        if "action_3d_full" in z:
            action_3d_full = z["action_3d_full"].astype(np.float32)
            if action_3d_full.shape[:2] != action_3d.shape[:2]:
                raise RuntimeError(f"DIME full/longitudinal action shape mismatch: {npz_path}")
            if action_3d_full.shape[2] != ADNI_FULL_COSTS.shape[0]:
                raise RuntimeError(
                    f"Unexpected DIME full action width {action_3d_full.shape[2]} in {npz_path}; "
                    f"expected {ADNI_FULL_COSTS.shape[0]}"
                )
            avg_cost = float(np.sum(action_3d_full * ADNI_FULL_COSTS.reshape(1, 1, -1), axis=(1, 2)).mean())
            avg_aux_cost = float(avg_cost - avg_long_cost)
            avg_aux_cost_schema = avg_aux_cost
            avg_aux_available_count = float(ADNI_AUX_DIM)
            aux_dim = int(ADNI_AUX_DIM)
            static_schema_dim = int(ADNI_AUX_DIM)

        np.savez(
            rollout_path,
            masks=action_3d.reshape(action_3d.shape[0], -1),
            num_time=np.int32(action_3d.shape[1]),
            num_groups=np.int32(action_3d.shape[2]),
            group_names=np.asarray(ADNI_LONGITUDINAL_NAMES, dtype=object),
            group_costs=ADNI_LONG_COSTS.astype(np.float32),
            avg_long_cost=np.float32(avg_long_cost),
            avg_aux_cost=np.float32(avg_aux_cost),
            avg_cost=np.float32(avg_cost),
            budget=np.int32(budget),
            source_path=np.asarray([str(npz_path)], dtype=object),
        )
        _save_json(
            meta_path,
            {
                "baseline": "dime",
                "budget": budget,
                "source_path": str(npz_path),
                "has_full_actions": bool("action_3d_full" in z),
            },
        )
        candidates.append(
            {
                "baseline": "dime",
                "id": f"budget_{budget}",
                "sort_value": budget,
                "budget": budget,
                "source": str(npz_path),
                "cache_dir": str(cache_dir),
                "avg_long_cost": float(avg_long_cost),
                "avg_aux_cost": float(avg_aux_cost),
                "avg_aux_cost_schema": float(avg_aux_cost_schema),
                "avg_aux_available_count": float(avg_aux_available_count),
                "aux_dim": int(aux_dim),
                "static_schema_dim": int(static_schema_dim),
                "avg_cost": float(avg_cost),
                "auroc": float(perf.get("auroc", float("nan"))),
                "auprc": float(perf.get("auprc", float("nan"))),
            }
        )
    if not candidates:
        raise RuntimeError(f"No DIME ADNI action caches found in {dime_action_root}")
    return candidates


def _format_lambda_token(value: float):
    text = f"{float(value):.3f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def _discover_ras_candidates(output_root: Path, args, ras_perf: Dict[float, Dict[str, float]]):
    candidates = []
    cache_root = output_root / "cache" / "ras"
    _ensure_dir(cache_root)

    ras_model_root = (
        Path(args.ras_model_root)
        if str(args.ras_model_root).strip()
        else (DEFAULT_RAS_MODEL_ROOT_BASELINE_T0 if args.ras_baseline_at_t0_only else DEFAULT_RAS_MODEL_ROOT)
    )
    lambda_csvs = sorted(glob.glob(str(ras_model_root / "lambda=*.csv")))
    if not lambda_csvs:
        raise RuntimeError(f"No RAS lambda CSVs found in {ras_model_root}")

    for csv_path_str in lambda_csvs:
        csv_path = Path(csv_path_str)
        match = re.search(r"lambda=([0-9.]+)\.csv$", csv_path.name)
        if match is None:
            continue
        lambda_value = float(match.group(1))
        cache_dir = cache_root / f"lambda_{_format_lambda_token(lambda_value)}"
        rollout_path = cache_dir / "analysis_rollout.npz"
        meta_path = cache_dir / "checkpoint_meta.json"
        _ensure_dir(cache_dir)

        if args.force_refresh_ras or not rollout_path.is_file():
            cmd = [
                str(Path(args.ras_python)),
                str(Path(args.ras_export_script)),
                "--lambda",
                f"{lambda_value:.1f}",
                "--out",
                str(rollout_path),
                "--seed",
                str(int(args.ras_eval_seed)),
            ]
            if args.ras_baseline_at_t0_only:
                cmd.append("--baseline-at-t0-only")
            env = os.environ.copy()
            existing = env.get("PYTHONPATH", "")
            ras_src = "/playpen-nvme/scribble/ddinh/ras/src"
            env["PYTHONPATH"] = ras_src if not existing else f"{ras_src}:{existing}"
            subprocess.run(cmd, check=True, cwd=str(ACTOR_ROOT), env=env)

        rollout = _read_rollout(rollout_path)
        model_paths = []
        z = np.load(rollout_path, allow_pickle=True)
        if "model_paths" in z:
            model_paths = [str(x) for x in np.asarray(z["model_paths"], dtype=object).reshape(-1).tolist()]
        _save_json(
            meta_path,
            {
                    "baseline": "ras",
                    "lambda": lambda_value,
                    "source_csv": str(csv_path),
                    "baseline_at_t0_only": bool(args.ras_baseline_at_t0_only),
                    "eval_seed": int(args.ras_eval_seed),
                    "model_paths": model_paths,
                },
        )
        candidates.append(
            {
                "baseline": "ras",
                "id": f"lambda_{lambda_value:.1f}",
                "sort_value": lambda_value,
                "lambda": lambda_value,
                "source": str(csv_path),
                "cache_dir": str(cache_dir),
                "avg_long_cost": float(rollout["avg_long_cost"]),
                "avg_aux_cost": float(rollout["avg_aux_cost"]),
                "avg_aux_cost_schema": float(rollout["avg_aux_cost"]),
                "avg_aux_available_count": float(ADNI_AUX_DIM),
                "aux_dim": int(ADNI_AUX_DIM),
                "static_schema_dim": int(ADNI_AUX_DIM),
                "avg_cost": float(rollout["avg_cost"]),
                "auroc": float(ras_perf.get(lambda_value, {}).get("auroc", float("nan"))),
                "auprc": float(ras_perf.get(lambda_value, {}).get("auprc", float("nan"))),
                "model_paths": model_paths,
            }
        )
    return candidates


def _load_source_groups(source_root: Path, mode_name: str, metrics_cache_root: Path, args):
    mode_root = source_root / mode_name
    groups = []
    for group_dir_str in sorted(glob.glob(str(mode_root / "group_*"))):
        group_dir = Path(group_dir_str)
        meta_path = group_dir / "group_meta.json"
        if not meta_path.is_file():
            continue
        meta = _load_json(meta_path)
        learned = dict(meta["checkpoints"]["learned"])
        learned["id"] = str(learned.get("ckpt_id", group_dir.name))
        learned["source"] = str(learned.get("actor_path", group_dir))
        actor_metrics = _export_actor_metrics(str(learned["source"]), metrics_cache_root=metrics_cache_root, args=args)
        learned["accuracy"] = float(actor_metrics.get("accuracy", float("nan")))
        learned["auroc"] = float(actor_metrics.get("auroc", float("nan")))
        learned["auprc"] = float(actor_metrics.get("auprc", float("nan")))
        groups.append(
            {
                "group": group_dir.name,
                "group_index": int(meta.get("group_index", int(group_dir.name.split("_")[-1]))),
                "group_dir": group_dir,
                "meta": meta,
                "metric_key": str(meta["metric_key"]),
                "target_cost": float(meta["target_cost"]),
                "learned": learned,
            }
        )
    if not groups:
        raise RuntimeError(f"No source groups found in {mode_root}")
    return groups


def _best_nearest(candidates: List[dict], metric_key: str, target_cost: float):
    scored = []
    for candidate in candidates:
        rel_err = _safe_rel_err(float(candidate[metric_key]), float(target_cost))
        scored.append((rel_err, abs(float(candidate[metric_key]) - float(target_cost)), str(candidate["id"]), candidate))
    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    return scored[0][3], float(scored[0][0])


def _prepare_group(group_info: dict, ras_candidates: List[dict], dime_candidates: List[dict], output_root: Path, args):
    metric_key = str(group_info["metric_key"])
    target_cost = float(group_info["target_cost"])
    group_name = str(group_info["group"])
    source_meta = dict(group_info["meta"])

    ras_choice, ras_rel_err = _best_nearest(ras_candidates, metric_key, target_cost)
    dime_choice, dime_rel_err = _best_nearest(dime_candidates, metric_key, target_cost)

    records = {
        "learned": {
            "id": str(group_info["learned"]["id"]),
            "source": str(group_info["learned"]["source"]),
            "cache_dir": str(group_info["learned"]["cache_dir"]),
            "avg_long_cost": float(group_info["learned"]["avg_long_cost"]),
                "avg_aux_cost": float(group_info["learned"]["avg_aux_cost"]),
                "avg_aux_cost_schema": float(group_info["learned"].get("avg_aux_cost_schema", group_info["learned"]["avg_aux_cost"])),
                "avg_aux_available_count": float(group_info["learned"].get("avg_aux_available_count", ADNI_AUX_DIM)),
                "aux_dim": int(group_info["learned"].get("aux_dim", ADNI_AUX_DIM)),
                "static_schema_dim": int(group_info["learned"].get("static_schema_dim", ADNI_AUX_DIM)),
                "avg_cost": float(group_info["learned"]["avg_cost"]),
                "accuracy": float(group_info["learned"].get("accuracy", float("nan"))),
                "auroc": float(group_info["learned"].get("auroc", float("nan"))),
                "auprc": float(group_info["learned"].get("auprc", float("nan"))),
            },
            "ras": dict(ras_choice),
            "dime": dict(dime_choice),
    }

    rollouts = {
        "learned": _read_rollout(Path(records["learned"]["cache_dir"]) / "analysis_rollout.npz"),
        "ras": _read_rollout(Path(records["ras"]["cache_dir"]) / "analysis_rollout.npz"),
        "dime": _read_rollout(Path(records["dime"]["cache_dir"]) / "analysis_rollout.npz"),
    }
    keep_idx, shared_order_all = _shared_feature_controls(
        rollouts,
        keep_all_features=bool(args.all_features),
    )
    num_time = int(rollouts["learned"]["num_time"])
    num_groups = int(rollouts["learned"]["num_groups"])

    out_group_dir = output_root / source_meta["mode"] / group_name
    _ensure_dir(out_group_dir)
    _write_group_costs_quick(out_group_dir, records)
    _write_group_costs_full(out_group_dir, records, target_cost=target_cost, metric_key=metric_key)

    checkpoints = {
        "learned": dict(source_meta["checkpoints"]["learned"]),
        "ras": {
                "cache_dir": records["ras"]["cache_dir"],
                "lambda": float(records["ras"]["lambda"]),
            "avg_long_cost": float(records["ras"]["avg_long_cost"]),
                "avg_aux_cost": float(records["ras"]["avg_aux_cost"]),
                "avg_cost": float(records["ras"]["avg_cost"]),
                "auroc": float(records["ras"].get("auroc", float("nan"))),
                "auprc": float(records["ras"].get("auprc", float("nan"))),
                "source_csv": records["ras"]["source"],
                "model_paths": records["ras"].get("model_paths", []),
            },
            "dime": {
                "cache_dir": records["dime"]["cache_dir"],
            "budget": int(records["dime"]["budget"]),
            "avg_long_cost": float(records["dime"]["avg_long_cost"]),
                "avg_aux_cost": float(records["dime"]["avg_aux_cost"]),
                "avg_cost": float(records["dime"]["avg_cost"]),
                "auroc": float(records["dime"].get("auroc", float("nan"))),
                "auprc": float(records["dime"].get("auprc", float("nan"))),
                "source_path": records["dime"]["source"],
        },
    }
    checkpoints["learned"]["accuracy"] = float(records["learned"].get("accuracy", float("nan")))
    checkpoints["learned"]["auroc"] = float(records["learned"].get("auroc", float("nan")))
    checkpoints["learned"]["auprc"] = float(records["learned"].get("auprc", float("nan")))

    group_meta = {
        "mode": str(source_meta["mode"]),
        "group_index": int(group_info["group_index"]),
        "metric_key": metric_key,
        "target_cost": float(target_cost),
        "match_stage": "nearest_fallback",
        "score": float(max(ras_rel_err, dime_rel_err)),
        "err_ras": float(ras_rel_err),
        "err_dime": float(dime_rel_err),
        "source_group_dir": str(group_info["group_dir"]),
        "source_variant": str(Path(group_info["group_dir"]).parents[1].name),
        "num_time": num_time,
        "num_groups": num_groups,
        "shared_keep_idx": [int(x) for x in keep_idx.tolist()],
        "shared_order_all": [int(x) for x in shared_order_all.tolist()],
        "shared_cluster_k": (
            int(args.cluster_k_override)
            if int(args.cluster_k_override) > 0
            else int(source_meta.get("shared_cluster_k", 1))
        ),
        "group_name_source": str(source_meta.get("group_name_source", "dataset.adni_longitudinal_names")),
        "checkpoints": checkpoints,
    }
    if "learned_baseline_context" in source_meta:
        group_meta["learned_baseline_context"] = dict(source_meta["learned_baseline_context"])
    _save_json(out_group_dir / "group_meta.json", group_meta)

    return {
        "group": group_name.split("_")[-1],
        "metric_key": metric_key,
        "target_cost": float(target_cost),
        "learned": records["learned"],
        "ras": records["ras"],
        "dime": records["dime"],
        "ras_rel_err": float(ras_rel_err),
        "dime_rel_err": float(dime_rel_err),
    }


def main():
    args = parse_args()
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    _ensure_dir(output_root)

    metrics_cache_root = _metrics_cache_root(output_root, args)
    ras_perf = _load_ras_performance_map(args)
    dime_budgets = []
    for npz_path_str in sorted(glob.glob(str(Path(args.dime_action_root) / "dime_actions_budget_*.npz"))):
        match = re.search(r"budget_(\d+)\.npz$", Path(npz_path_str).name)
        if match is not None:
            dime_budgets.append(int(match.group(1)))
    dime_perf = _export_dime_metrics(dime_budgets, metrics_cache_root=metrics_cache_root, args=args)

    ras_candidates = _discover_ras_candidates(output_root, args=args, ras_perf=ras_perf)
    dime_candidates = _discover_dime_candidates(output_root, Path(args.dime_action_root), dime_perf=dime_perf)
    _write_candidates_summary(ras_candidates + dime_candidates, output_root / "summary_candidates.tsv")

    modes = ["longitudinal", "total"] if args.mode == "both" else [args.mode]
    for mode_name in modes:
        summaries = []
        for group_info in _load_source_groups(source_root, mode_name, metrics_cache_root=metrics_cache_root, args=args):
            summaries.append(_prepare_group(group_info, ras_candidates, dime_candidates, output_root, args))
        _write_summary(output_root, mode_name, summaries)
        print(f"[OK] prepared {len(summaries)} groups for mode={mode_name}")


if __name__ == "__main__":
    main()
