#!/usr/bin/env python3
"""
Matched-cost plotting pipeline across ACTOR-LAFA datasets.

This is a standalone script that does not alter analysis_plots.py behavior.
It reuses analysis internals and adds:
1) checkpoint-pool sweep + per-checkpoint rollout/cost cache
2) matched triplet selection (learned/all/none) for:
   - longitudinal-cost matching
   - total-cost matching
3) grouped plot outputs with shared feature/order consistency within each triplet

"""

import argparse
import glob
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


BASELINES = ("learned", "all", "none")
ALLOWED_DATASETS = {"cheears", "cheears_demog", "cheears_day_context", "womac", "klg", "adni", "ILIADD", "iliadd"}
MIN_FEATURE_RATE = 0.01
EPS = 1e-8

DATASET_CANONICAL = {
    "cheears": "cheears",
    "cheears_demog": "cheears_demog",
    "cheears_day_context": "cheears_day_context",
    "womac": "womac",
    "klg": "klg",
    "adni": "adni",
    "ILIADD": "ILIADD",
    "iliadd": "ILIADD",
}

# Canonical CHEEARS aux/context feature names (34 dims).
CHEEARS_AUX_FEATURE_NAMES = [
    "sex",
    "age",
    "race",
    "hispanic",
    "education",
    "current_employed",
    "family_income",
    "religious_affiliation",
    "cigarette_use",
    "alcohol_use",
    "drug_use",
    "mentalhealth",
    "audit_total",
    "dmq_soc",
    "dmq_cop",
    "dmq_enh",
    "dmq_con",
    "yaacq_total",
    "yaacq_social",
    "yaacq_control",
    "yaacq_selfperc",
    "yaacq_selfcare",
    "yaacq_risk",
    "yaacq_academic",
    "yaacq_depend",
    "yaacq_blackout",
    "neo_n",
    "neo_e",
    "neo_a",
    "neo_o",
    "neo_c",
    "iip_dom",
    "iip_lov",
    "iip_elev",
]

CHEEARS_DAY_CONTEXT_AUX_FEATURE_NAMES = [f"baseline_{i}" for i in range(34)] + ["day_of_week"]

OAI_ALL_FEATURE_NAMES = [
    "HISP", "RACE", "SEX", "FAMHXKR", "EDCV", "AGE", "SMOKE",
    "DRNKAMT", "DRKMORE", "INCOME2", "MARITST", "BPSYS", "BPDIAS",
    "BMI", "CEMPLOY", "CUREMP", "MEDINS",
    "JSW_1", "JSW_2", "JSW_3", "JSW_4", "JSW_5", "JSW_6", "JSW_7", "JSW_8", "JSW_9", "JSW_10",
]
OAI_STATIC_INDICES = [0, 1, 2, 3, 4, 5, 6, 9, 10, 16]
OAI_LONGITUDINAL_NAMES = [OAI_ALL_FEATURE_NAMES[i] for i in range(len(OAI_ALL_FEATURE_NAMES)) if i not in set(OAI_STATIC_INDICES)]
OAI_AUX_NAMES = [OAI_ALL_FEATURE_NAMES[i] for i in OAI_STATIC_INDICES]

ADNI_LONGITUDINAL_NAMES = ["FDG", "AV45", "Hippocampus", "Entorhinal"]
ADNI_AUX_NAMES = ["AGE", "PTGENDER", "PTEDUCAT", "PTETHCAT", "PTRACCAT", "PTMARRY", "FAQ"]

ILIADD_LONGITUDINAL_NAMES = [
    "interaction",
    "positiveaffEMA",
    "energyEMA",
    "stressEMA",
    "impulse1EMA",
    "impulse2EMA",
    "impulse3EMA",
    "impulse4EMA",
]
ILIADD_AUX_NAMES = [
    "age",
    "sex",
    "handedness",
    "multiracial",
    "hispanic",
    "language",
    "marital",
    "relationship",
    "grade",
    "degree",
    "income",
    "cigarette",
    "substance",
    "treatment",
    "recentTreatment",
    "whoTreatment",
    "HiTOP_Dishon",
    "HiTOP_DisDys",
    "HiTOP_Emot",
    "HiTOP_Mistrust",
    "HiTOP_PhobInd",
    "BFI_E",
    "BFI_A",
    "BFI_C",
    "BFI_N",
    "BFI_O",
]


@dataclass
class Candidate:
    baseline: str
    path: str
    ckpt_id: str
    filename: str
    cw: float
    acw: float
    joint: bool
    warmup: bool
    source_dir: str
    used_warmup_fallback: bool


def parse_args():
    default_dataset = os.environ.get("ACTOR_DATASET", "cheears")
    parser = argparse.ArgumentParser(
        description="Matched-cost plotting using cached rollout artifacts."
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        required=False,
        default=None,
        help="Optional path to test_data.npz. If omitted, dataset defaults are used.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=default_dataset,
        help="Dataset name: cheears, cheears_demog, cheears_day_context, womac, klg, adni, ILIADD (or iliadd).",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Root output directory. Default: ACTOR_LAFAv1/plots/<dataset>/matched_cost_groups",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["longitudinal", "total", "both"],
        help="Matching mode.",
    )
    parser.add_argument(
        "--tol_primary",
        type=float,
        default=0.05,
        help="Primary relative-error tolerance.",
    )
    parser.add_argument(
        "--tol_secondary",
        type=float,
        default=0.10,
        help="Secondary relative-error tolerance.",
    )
    parser.add_argument(
        "--max_groups",
        type=int,
        default=8,
        help="Maximum number of matched triplet groups to render per mode.",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Evaluation batch size.")
    parser.add_argument(
        "--cluster_k",
        type=int,
        default=0,
        help="Cluster count for mask templates (<=0 auto from learned member).",
    )
    parser.add_argument("--cluster_kmax", type=int, default=10, help="Max k for elbow sweep.")
    parser.add_argument(
        "--edge_min_freq",
        type=float,
        default=0.01,
        help="Minimum edge frequency for trajectory plot.",
    )
    parser.add_argument(
        "--edge_max_edges",
        type=int,
        default=0,
        help="Max number of edges to draw (<=0 means no cap).",
    )
    parser.add_argument(
        "--edge_node_size_scale",
        type=float,
        default=95.0,
        help="Node marker size scale in trajectory plot.",
    )
    parser.add_argument(
        "--edge_transition_mode",
        type=str,
        default="next_observed",
        choices=["strict_next", "next_observed"],
        help=(
            "Transition construction mode: strict_next uses only t->t+1; "
            "next_observed links each acquisition step to the next observed "
            "acquisition step (allows skips like t->t+2)."
        ),
    )
    parser.add_argument(
        "--include_warmup",
        action="store_true",
        default=False,
        help="Include *_warmup*.ckpt files in candidate pool.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _sanitize_id(name: str):
    clean = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)
    return clean.strip("_")


def _parse_cw_acw(path: str) -> Tuple[float, float]:
    base = os.path.basename(path)
    m = re.search(r"_cw([0-9eE+.-]+)_acw([0-9eE+.-]+)(?:_|\.ckpt$)", base)
    if m is None:
        raise ValueError(f"Could not parse cw/acw from checkpoint name: {base}")
    cw = float(m.group(1))
    acw = float(m.group(2))
    return cw, acw


def _is_warmup(path: str):
    return "_warmup" in os.path.basename(path)


def _warmup_steps(path: str):
    """
    Parse warmup step count from checkpoint filename.
    Examples:
      *_warmup0*.ckpt   -> 0
      *_warmup50*.ckpt  -> 50
      *_warmup*.ckpt    -> None
    """
    name = os.path.basename(path).lower()
    m = re.search(r"warmup[_-]?(\d+)", name)
    if m is None:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _filter_learned_warmups(paths: List[str], dataset: str):
    """
    For learned checkpoints, when warmup checkpoints are considered,
    drop warmup0 and keep warmup50 (plus non-warmup).
    """
    if len(paths) == 0:
        return paths
    kept = []
    dropped = []
    for p in paths:
        if not _is_warmup(p):
            kept.append(p)
            continue
        steps = _warmup_steps(p)
        if steps == 0:
            dropped.append(p)
            continue
        kept.append(p)

    if len(dropped) > 0:
        print(
            f"[discover:{dataset}] baseline=learned: dropped {len(dropped)} warmup0 checkpoint(s); "
            "keeping warmup50/non-warmup only."
        )
    return kept


def _canonical_dataset_name(dataset: str):
    key = str(dataset).strip()
    if key in DATASET_CANONICAL:
        return DATASET_CANONICAL[key]
    key_lower = key.lower()
    if key_lower in DATASET_CANONICAL:
        return DATASET_CANONICAL[key_lower]
    raise ValueError(
        f"Unsupported dataset '{dataset}'. "
        f"Allowed: {sorted(ALLOWED_DATASETS)}"
    )


def _candidate_dirs_for_dataset(script_dir: str, dataset: str):
    canon = _canonical_dataset_name(dataset)
    mapping = {
        "cheears": {
            "learned": ["cheears"],
            "all": ["cheears"],
            "none": ["cheears"],
        },
        "cheears_demog": {
            "learned": ["cheears_demog", "cheears"],
            "all": ["cheears_demog", "cheears"],
            "none": ["cheears_demog", "cheears"],
        },
        "cheears_day_context": {
            "learned": ["cheears_day_context"],
            "all": ["cheears_day_context"],
            "none": ["cheears_day_context"],
        },
        "womac": {
            "learned": ["womac"],
            "all": ["womac 2"],
            "none": ["womac 2"],
        },
        "klg": {
            "learned": ["klg"],
            "all": ["klg 2"],
            "none": ["klg 2"],
        },
        "adni": {
            "learned": ["adni"],
            "all": ["adni 2"],
            "none": ["adni 2"],
        },
        "ILIADD": {
            "learned": ["ILIADD"],
            "all": ["ILIADD_2"],
            "none": ["ILIADD_2"],
        },
    }
    out = {}
    for baseline in BASELINES:
        names = mapping[canon][baseline]
        dirs = []
        for name in names:
            p = os.path.join(script_dir, name)
            if os.path.isdir(p):
                dirs.append(p)
        if len(dirs) == 0:
            tried = ", ".join(os.path.join(script_dir, x) for x in names)
            raise FileNotFoundError(
                f"No checkpoint directory exists for dataset={canon}, baseline={baseline}. "
                f"Tried: {tried}"
            )
        out[baseline] = dirs
    return out


def _patterns_for_baseline_dirs(dataset: str, dataset_dirs_by_baseline: Dict[str, List[str]]):
    canon = _canonical_dataset_name(dataset)
    pats = {}
    for baseline, dirs in dataset_dirs_by_baseline.items():
        if baseline == "learned":
            fname = "actor_iterative_joint_cw*_acw*.ckpt"
        elif canon == "cheears_day_context" and baseline in ("all", "none"):
            fname = "actor_iterative_cw*_acw*.ckpt"
        elif baseline == "all":
            fname = "actor_iterative_baseline_all_cw*_acw*.ckpt"
        elif baseline == "none":
            fname = "actor_iterative_baseline_none_cw*_acw*.ckpt"
        else:
            raise ValueError(f"Unknown baseline {baseline}")
        pats[baseline] = [os.path.join(d, fname) for d in dirs]
    return pats


def _short_path_hash(path: str):
    p = os.path.realpath(os.path.abspath(path))
    return hashlib.sha1(p.encode("utf-8")).hexdigest()[:10]


def discover_candidates(script_dir: str, dataset: str, include_warmup: bool):
    dataset_dirs_by_baseline = _candidate_dirs_for_dataset(script_dir=script_dir, dataset=dataset)
    patterns = _patterns_for_baseline_dirs(dataset=dataset, dataset_dirs_by_baseline=dataset_dirs_by_baseline)

    pools: Dict[str, List[Candidate]] = {k: [] for k in BASELINES}
    warmup_fallback_used = {k: False for k in BASELINES}

    for baseline in BASELINES:
        raw_all = []
        for patt in patterns[baseline]:
            raw_all.extend(glob.glob(patt))
        raw_all = sorted(set(os.path.abspath(p) for p in raw_all))

        # Learned-context policy: if warmup candidates are considered, never use warmup0.
        if baseline == "learned":
            raw_all = _filter_learned_warmups(raw_all, dataset=dataset)

        if include_warmup:
            chosen_raw = list(raw_all)
        else:
            chosen_raw = [p for p in raw_all if not _is_warmup(p)]
            if len(chosen_raw) == 0 and len(raw_all) > 0:
                chosen_raw = list(raw_all)
                warmup_fallback_used[baseline] = True
                print(
                    f"[discover:{dataset}] baseline={baseline}: no non-warmup checkpoints found; "
                    "falling back to include warmup checkpoints."
                )

        # Canonical dedupe by (baseline, cw, acw): keep shortest filename, then lexical.
        by_hp = {}
        for p in chosen_raw:
            try:
                cw, acw = _parse_cw_acw(p)
            except Exception:
                continue
            key = (baseline, f"{cw:.12g}", f"{acw:.12g}")
            prev = by_hp.get(key)
            if prev is None:
                by_hp[key] = p
            else:
                prev_name = os.path.basename(prev)
                cur_name = os.path.basename(p)
                take_cur = (len(cur_name), cur_name, p) < (len(prev_name), prev_name, prev)
                if take_cur:
                    by_hp[key] = p

        chosen = sorted(by_hp.values())
        out = []
        for p in chosen:
            cw, acw = _parse_cw_acw(p)
            base = os.path.basename(p)
            out.append(
                Candidate(
                    baseline=baseline,
                    path=os.path.abspath(p),
                    ckpt_id=_sanitize_id(os.path.splitext(base)[0]),
                    filename=base,
                    cw=float(cw),
                    acw=float(acw),
                    joint=("joint" in base),
                    warmup=_is_warmup(p),
                    source_dir=os.path.dirname(os.path.abspath(p)),
                    used_warmup_fallback=bool(warmup_fallback_used[baseline]),
                )
            )
        pools[baseline] = sorted(out, key=lambda c: (c.cw, c.acw, c.filename))

    for baseline in BASELINES:
        if len(pools[baseline]) == 0:
            patt_txt = " | ".join(patterns[baseline])
            raise RuntimeError(
                f"Empty checkpoint pool for baseline='{baseline}'. "
                f"Patterns: {patt_txt} (include_warmup={include_warmup})"
            )
    return pools, warmup_fallback_used


def _safe_float(value):
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"Non-finite cost value encountered: {value}")
    return x


def _load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def _save_json(path: str, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _make_sample_paths_from_masks(masks: np.ndarray, num_time: int, num_groups: int):
    mask_3d = masks.reshape(masks.shape[0], num_time, num_groups)
    sample_paths = []
    for i in range(mask_3d.shape[0]):
        steps = []
        for t in range(num_time):
            steps.append(np.where(mask_3d[i, t] > 0.5)[0].astype(int).tolist())
        sample_paths.append(steps)
    return sample_paths


def _read_rollout_npz(rollout_path: str):
    z = np.load(rollout_path, allow_pickle=True)
    masks = z["masks"].astype(np.float32)
    preds = z["predictions"] if "predictions" in z else None
    labels = z["labels"] if "labels" in z else None
    num_time = int(z["num_time"]) if "num_time" in z else None
    num_groups = int(z["num_groups"]) if "num_groups" in z else None
    if num_time is None or num_groups is None:
        raise ValueError(f"Cache rollout missing num_time/num_groups: {rollout_path}")

    if "group_names" in z:
        gn = z["group_names"]
        group_names = [str(x) for x in gn.tolist()]
    else:
        group_names = [str(i) for i in range(num_groups)]
    if "group_costs" in z:
        group_costs = z["group_costs"].astype(np.float32)
    else:
        group_costs = np.ones(num_groups, dtype=np.float32)
    avg_cost = float(z["avg_cost"]) if "avg_cost" in z else float("nan")
    avg_long_cost = float(z["avg_long_cost"]) if "avg_long_cost" in z else float("nan")
    avg_aux_cost = float(z["avg_aux_cost"]) if "avg_aux_cost" in z else float("nan")
    aux_gate_binary = z["aux_gate_binary"].astype(np.float32) if "aux_gate_binary" in z else np.zeros((0,), dtype=np.float32)
    aux_gate_probs = z["aux_gate_probs"].astype(np.float32) if "aux_gate_probs" in z else np.zeros((0,), dtype=np.float32)
    aux_gate_rates = z["aux_gate_rates"].astype(np.float32) if "aux_gate_rates" in z else np.zeros((0,), dtype=np.float32)
    aux_available_rates = z["aux_available_rates"].astype(np.float32) if "aux_available_rates" in z else np.zeros((0,), dtype=np.float32)
    if "aux_feature_costs" in z:
        aux_feature_costs = z["aux_feature_costs"].astype(np.float32)
    else:
        # Backward-compatible fallback for old cache files.
        n_aux = int(aux_gate_rates.size) if aux_gate_rates.size > 0 else int(aux_gate_binary.size)
        aux_feature_costs = np.ones(max(n_aux, 0), dtype=np.float32)

    return {
        "masks": masks,
        "predictions": preds,
        "labels": labels,
        "num_time": num_time,
        "num_groups": num_groups,
        "group_names": group_names,
        "group_costs": group_costs,
        "avg_cost": avg_cost,
        "avg_long_cost": avg_long_cost,
        "avg_aux_cost": avg_aux_cost,
        "aux_gate_binary": aux_gate_binary,
        "aux_gate_probs": aux_gate_probs,
        "aux_gate_rates": aux_gate_rates,
        "aux_available_rates": aux_available_rates,
        "aux_feature_costs": aux_feature_costs,
    }


def _cache_dirs_for_candidate(output_root: str, candidate: Candidate):
    leaf_hashed = f"{candidate.ckpt_id}__{_short_path_hash(candidate.path)}"
    hashed = os.path.join(output_root, "cache", candidate.baseline, leaf_hashed)
    legacy = os.path.join(output_root, "cache", candidate.baseline, candidate.ckpt_id)
    return hashed, legacy


def _record_from_cache(candidate: Candidate, cache_dir: str, costs: dict, rollout_path: str):
    return {
        "baseline": candidate.baseline,
        "ckpt_id": candidate.ckpt_id,
        "path": candidate.path,
        "filename": candidate.filename,
        "cw": candidate.cw,
        "acw": candidate.acw,
        "joint": candidate.joint,
        "warmup": candidate.warmup,
        "source_dir": candidate.source_dir,
        "used_warmup_fallback": bool(candidate.used_warmup_fallback),
        "cache_dir": cache_dir,
        "rollout_path": rollout_path,
        "avg_long_cost": float(costs["avg_long_cost"]),
        "avg_aux_cost": float(costs["avg_aux_cost"]),
        "avg_cost": float(costs["avg_cost"]),
    }


def evaluate_or_load_candidate(ap, candidate: Candidate, args, output_root: str):
    cache_dir_hashed, cache_dir_legacy = _cache_dirs_for_candidate(output_root=output_root, candidate=candidate)
    if os.path.isdir(cache_dir_hashed):
        cache_dir = cache_dir_hashed
    elif os.path.isdir(cache_dir_legacy):
        cache_dir = cache_dir_legacy
    else:
        cache_dir = cache_dir_hashed
    using_legacy_cache_dir = (cache_dir == cache_dir_legacy and cache_dir_legacy != cache_dir_hashed)
    _ensure_dir(cache_dir)

    rollout_path = os.path.join(cache_dir, "analysis_rollout.npz")
    costs_path = os.path.join(cache_dir, "costs.json")
    meta_path = os.path.join(cache_dir, "checkpoint_meta.json")

    cache_ok = os.path.exists(rollout_path) and os.path.exists(costs_path) and os.path.exists(meta_path)

    if cache_ok:
        try:
            costs = _load_json(costs_path)
            _safe_float(costs["avg_long_cost"])
            _safe_float(costs["avg_aux_cost"])
            _safe_float(costs["avg_cost"])
            rollout = _read_rollout_npz(rollout_path)
            meta = _load_json(meta_path)
            if str(meta.get("actor_path", "")) != candidate.path:
                raise ValueError("actor_path mismatch in cache metadata")
            if str(meta.get("dataset", "")) != str(args.dataset):
                raise ValueError("dataset mismatch in cache metadata")
            if str(meta.get("baseline", "")) != str(candidate.baseline):
                raise ValueError("baseline mismatch in cache metadata")
            if not np.isclose(float(meta.get("cw", np.nan)), float(candidate.cw), rtol=0.0, atol=1e-12):
                raise ValueError("cw mismatch in cache metadata")
            if not np.isclose(float(meta.get("acw", np.nan)), float(candidate.acw), rtol=0.0, atol=1e-12):
                raise ValueError("acw mismatch in cache metadata")
            # Guard against stale cache reuse across different test splits.
            req_test = os.path.realpath(os.path.abspath(str(args.test_data_path)))
            cached_test = str(meta.get("test_data_path", ""))
            if cached_test:
                if os.path.realpath(os.path.abspath(cached_test)) != req_test:
                    raise ValueError("test_data_path mismatch in cache metadata")
            # Stale-cache guard: older caches may miss aux-rate fields.
            if float(costs["avg_aux_cost"]) > 0.0 and int(rollout["aux_gate_rates"].size) == 0:
                raise ValueError("stale cache: missing aux_gate_rates")
            if float(costs["avg_aux_cost"]) > 0.0 and int(rollout["aux_feature_costs"].size) == 0:
                raise ValueError("stale cache: missing aux_feature_costs")
            return _record_from_cache(
                candidate=candidate,
                cache_dir=cache_dir,
                costs=costs,
                rollout_path=rollout_path,
            )
        except Exception:
            cache_ok = False
            if using_legacy_cache_dir:
                # Do not overwrite potentially colliding legacy cache entries.
                cache_dir = cache_dir_hashed
                _ensure_dir(cache_dir)
                rollout_path = os.path.join(cache_dir, "analysis_rollout.npz")
                costs_path = os.path.join(cache_dir, "costs.json")
                meta_path = os.path.join(cache_dir, "checkpoint_meta.json")

    print("=" * 80)
    print(f"Evaluating candidate [{candidate.baseline}] {candidate.filename}")
    print(f"*** actor: {candidate.path}")
    print("=" * 80)

    actor, actor_cfg, test_loader, num_time, num_feat, classifier_path, resolved_test_data_path = ap.load_model_and_data(
        actor_path=candidate.path,
        test_data_path=args.test_data_path,
        baseline=candidate.baseline,
        batch_size=args.batch_size,
        allow_infer_without_classifier=(_canonical_dataset_name(args.dataset) == "ILIADD"),
    )
    device = ap.get_device()
    results, masks, preds, labels = ap.evaluate_with_collection(
        actor=actor,
        dataloader=test_loader,
        device=device,
        num_time=num_time,
        num_feat=num_feat,
        feature_costs=actor_cfg.get("feature_costs"),
        aux_feature_costs=actor_cfg.get("aux_feature_costs"),
        collector=None,
    )
    masks = (masks > 0.5).astype(np.float32)

    n_test = int(len(test_loader.dataset))
    if int(masks.shape[0]) != n_test:
        raise RuntimeError(
            f"Mask count mismatch for {candidate.path}: masks={masks.shape[0]}, test={n_test}"
        )

    group_names, group_name_source = _resolve_longitudinal_names_for_dataset(
        dataset=args.dataset,
        num_groups=actor.num_groups,
        ap=ap,
    )
    group_costs, group_cost_source = ap.resolve_longitudinal_costs(
        actor.num_groups,
        actor=actor,
        actor_cfg=actor_cfg,
    )

    aux_costs_cfg = np.asarray(actor_cfg.get("aux_feature_costs", []), dtype=np.float32).reshape(-1)
    if aux_costs_cfg.size == 0 and hasattr(actor, "aux_feature_costs"):
        aux_costs_cfg = actor.aux_feature_costs.detach().cpu().numpy().astype(np.float32).reshape(-1)
    if int(actor.num_aux) > 0 and int(aux_costs_cfg.size) != int(actor.num_aux):
        fixed = np.ones(int(actor.num_aux), dtype=np.float32)
        n = min(int(actor.num_aux), int(aux_costs_cfg.size))
        if n > 0:
            fixed[:n] = aux_costs_cfg[:n]
        aux_costs_cfg = fixed

    np.savez(
        rollout_path,
        masks=masks,
        predictions=preds,
        labels=labels,
        accuracy=float(results["accuracy"]),
        auroc=float(results["auroc"]),
        auprc=float(results["auprc"]),
        avg_cost=float(results["avg_cost"]),
        avg_long_cost=float(results["avg_long_cost"]),
        avg_aux_cost=float(results["avg_aux_cost"]),
        num_time=int(num_time),
        num_groups=int(actor.num_groups),
        group_names=np.asarray(group_names, dtype=object),
        group_costs=np.asarray(group_costs, dtype=np.float32),
        aux_gate_binary=np.asarray(results.get("aux_gate_binary", []), dtype=np.float32),
        aux_gate_probs=np.asarray(results.get("aux_gate_probs", []), dtype=np.float32),
        aux_gate_rates=np.asarray(results.get("aux_gate_rates", []), dtype=np.float32),
        aux_available_rates=np.asarray(results.get("aux_available_rates", []), dtype=np.float32),
        aux_feature_costs=aux_costs_cfg.astype(np.float32),
    )

    costs = {
        "avg_long_cost": _safe_float(results["avg_long_cost"]),
        "avg_aux_cost": _safe_float(results["avg_aux_cost"]),
        "avg_cost": _safe_float(results["avg_cost"]),
    }
    _save_json(costs_path, costs)

    meta = {
        "dataset": args.dataset,
        "dataset_canonical": _canonical_dataset_name(args.dataset),
        "baseline": candidate.baseline,
        "joint": bool(candidate.joint),
        "actor_path": candidate.path,
        "classifier_path": classifier_path,
        "test_data_path": resolved_test_data_path,
        "cw": float(candidate.cw),
        "acw": float(candidate.acw),
        "source_dir": candidate.source_dir,
        "used_warmup_fallback": bool(candidate.used_warmup_fallback),
        "group_name_source": str(group_name_source),
        "group_cost_source": str(group_cost_source),
    }
    _save_json(meta_path, meta)

    return _record_from_cache(
        candidate=candidate,
        cache_dir=cache_dir,
        costs=costs,
        rollout_path=rollout_path,
    )


def write_candidates_summary(records: List[dict], output_root: str):
    out_path = os.path.join(output_root, "summary_candidates.tsv")
    with open(out_path, "w") as f:
        f.write(
            "baseline\tckpt_id\tjoint\twarmup\twarmup_fallback\tsource_dir\tcw\tacw\tavg_long_cost\tavg_aux_cost\tavg_cost\tactor_path\tcache_dir\n"
        )
        for r in sorted(records, key=lambda x: (x["baseline"], x["cw"], x["acw"], x["ckpt_id"])):
            f.write(
                f"{r['baseline']}\t{r['ckpt_id']}\t{int(bool(r['joint']))}\t{int(bool(r['warmup']))}\t"
                f"{int(bool(r.get('used_warmup_fallback', False)))}\t{r.get('source_dir', '')}\t"
                f"{float(r['cw']):.8g}\t{float(r['acw']):.8g}\t"
                f"{float(r['avg_long_cost']):.8g}\t{float(r['avg_aux_cost']):.8g}\t{float(r['avg_cost']):.8g}\t"
                f"{r['path']}\t{r['cache_dir']}\n"
            )
    print(f"Saved: {out_path}")


def rel_err(value: float, target: float):
    # Symmetric relative error keeps behavior stable when target is near zero.
    denom = max(abs(float(target)), abs(float(value)), EPS)
    return abs(float(value) - float(target)) / denom


def _best_within(cands: List[dict], metric_key: str, target: float, tol: float):
    scored = []
    for c in cands:
        e = rel_err(c[metric_key], target)
        if e <= tol:
            scored.append((e, abs(float(c[metric_key]) - float(target)), c["ckpt_id"], c))
    if len(scored) == 0:
        return None, None
    scored.sort(key=lambda x: (x[0], x[1], x[2]))
    best = scored[0][3]
    return best, scored[0][0]


def _best_nearest(cands: List[dict], metric_key: str, target: float):
    scored = []
    for c in cands:
        e = rel_err(c[metric_key], target)
        scored.append((e, abs(float(c[metric_key]) - float(target)), c["ckpt_id"], c))
    scored.sort(key=lambda x: (x[0], x[1], x[2]))
    best = scored[0][3]
    return best, scored[0][0]


def build_matched_groups(
    records_by_baseline: Dict[str, List[dict]],
    metric_key: str,
    tol_primary: float,
    tol_secondary: float,
    max_groups: int,
):
    learned_list = sorted(
        records_by_baseline["learned"],
        key=lambda r: (float(r[metric_key]), r["ckpt_id"]),
    )
    all_list = sorted(records_by_baseline["all"], key=lambda r: (float(r[metric_key]), r["ckpt_id"]))
    none_list = sorted(records_by_baseline["none"], key=lambda r: (float(r[metric_key]), r["ckpt_id"]))

    groups_raw = []
    for l in learned_list:
        target = float(l[metric_key])

        a, err_a = _best_within(all_list, metric_key, target, tol_primary)
        n, err_n = _best_within(none_list, metric_key, target, tol_primary)
        stage = "primary_tol"

        if a is None or n is None:
            a, err_a = _best_within(all_list, metric_key, target, tol_secondary)
            n, err_n = _best_within(none_list, metric_key, target, tol_secondary)
            stage = "secondary_tol"

        if a is None or n is None:
            a, err_a = _best_nearest(all_list, metric_key, target)
            n, err_n = _best_nearest(none_list, metric_key, target)
            stage = "nearest_fallback"
            print(
                f"[match:{metric_key}] fallback-nearest for learned={l['ckpt_id']} "
                f"(all_err={err_a:.4f}, none_err={err_n:.4f})"
            )

        score = max(float(err_a), float(err_n))
        groups_raw.append(
            {
                "metric_key": metric_key,
                "match_stage": stage,
                "score": float(score),
                "err_all": float(err_a),
                "err_none": float(err_n),
                "target_cost": target,
                "learned": l,
                "all": a,
                "none": n,
            }
        )

    # Deduplicate identical triplets; keep lowest-score instance.
    dedup = {}
    for g in groups_raw:
        key = (g["learned"]["ckpt_id"], g["all"]["ckpt_id"], g["none"]["ckpt_id"])
        prev = dedup.get(key)
        if prev is None or float(g["score"]) < float(prev["score"]):
            dedup[key] = g

    stage_rank = {"primary_tol": 0, "secondary_tol": 1, "nearest_fallback": 2}
    groups = sorted(
        dedup.values(),
        key=lambda x: (
            float(x["score"]),
            stage_rank.get(x["match_stage"], 99),
            x["learned"]["ckpt_id"],
            x["all"]["ckpt_id"],
            x["none"]["ckpt_id"],
        ),
    )
    return groups[: max(1, int(max_groups))]


def _shared_feature_controls(group_rollouts: Dict[str, dict]):
    modes = [m for m in BASELINES if m in group_rollouts]
    first = group_rollouts[modes[0]]
    num_groups = int(first["num_groups"])

    for m in modes[1:]:
        if int(group_rollouts[m]["num_groups"]) != num_groups:
            raise RuntimeError("num_groups mismatch inside matched group.")
        if int(group_rollouts[m]["num_time"]) != int(first["num_time"]):
            raise RuntimeError("num_time mismatch inside matched group.")

    usages = []
    for m in modes:
        masks = group_rollouts[m]["masks"]
        T = int(group_rollouts[m]["num_time"])
        G = int(group_rollouts[m]["num_groups"])
        usage = masks.reshape(masks.shape[0], T, G).mean(axis=(0, 1))
        usages.append(usage)
    usage_stack = np.stack(usages, axis=0)
    shared_usage_mean = usage_stack.mean(axis=0)
    shared_usage_max = usage_stack.max(axis=0)

    keep_idx = np.where(shared_usage_max >= float(MIN_FEATURE_RATE))[0]
    if keep_idx.size == 0:
        keep_idx = np.where(shared_usage_max > 0.0)[0]
    if keep_idx.size == 0:
        keep_idx = np.arange(num_groups, dtype=np.int32)
    keep_idx = keep_idx[np.argsort(-shared_usage_mean[keep_idx])]
    shared_order = np.argsort(-shared_usage_mean)
    return keep_idx.astype(np.int32), shared_order.astype(np.int32)


def _resolve_expected_aux_dim(dataset: str):
    canon = _canonical_dataset_name(dataset)
    if canon == "cheears":
        return len(CHEEARS_AUX_FEATURE_NAMES)
    if canon == "cheears_demog":
        return 22
    if canon == "cheears_day_context":
        return len(CHEEARS_DAY_CONTEXT_AUX_FEATURE_NAMES)
    if canon in ("womac", "klg"):
        return len(OAI_AUX_NAMES)
    if canon == "adni":
        return len(ADNI_AUX_NAMES)
    if canon == "ILIADD":
        return len(ILIADD_AUX_NAMES)
    return None


def _resolve_longitudinal_names_for_dataset(dataset: str, num_groups: int, ap):
    canon = _canonical_dataset_name(dataset)
    n = int(max(0, num_groups))
    if n == 0:
        return [], "empty"

    if canon in ("cheears", "cheears_demog", "cheears_day_context"):
        names, src = ap.resolve_longitudinal_group_names(n)
        if len(names) == n:
            return [str(x) for x in names], str(src)
        print(
            f"WARNING: {canon} longitudinal names mismatch (expected {n}, got {len(names)}). "
            "Falling back to index labels."
        )
        return [str(i) for i in range(n)], "fallback.index_only"

    if canon in ("womac", "klg"):
        if len(OAI_LONGITUDINAL_NAMES) == n:
            return list(OAI_LONGITUDINAL_NAMES), "dataset.oai_longitudinal_names"
    elif canon == "adni":
        if len(ADNI_LONGITUDINAL_NAMES) == n:
            return list(ADNI_LONGITUDINAL_NAMES), "dataset.adni_longitudinal_names"
    elif canon == "ILIADD":
        if len(ILIADD_LONGITUDINAL_NAMES) == n:
            return list(ILIADD_LONGITUDINAL_NAMES), "dataset.iliadd_longitudinal_names"

    print(
        f"WARNING: longitudinal names mismatch for dataset={canon} "
        f"(expected {n}). Falling back to index labels."
    )
    return [str(i) for i in range(n)], "fallback.index_only"


def _resolve_aux_names_for_dataset(dataset: str, num_aux: int):
    canon = _canonical_dataset_name(dataset)
    n = int(max(0, num_aux))
    if n == 0:
        return [], "empty"

    if canon == "cheears":
        names = list(CHEEARS_AUX_FEATURE_NAMES)
    elif canon == "cheears_demog":
        names = [f"context_{i}" for i in range(22)]
    elif canon == "cheears_day_context":
        names = list(CHEEARS_DAY_CONTEXT_AUX_FEATURE_NAMES)
    elif canon in ("womac", "klg"):
        names = list(OAI_AUX_NAMES)
    elif canon == "adni":
        names = list(ADNI_AUX_NAMES)
    elif canon == "ILIADD":
        names = list(ILIADD_AUX_NAMES)
    else:
        names = []

    if len(names) == n:
        return names, f"dataset.{canon}_aux_names"
    if len(names) > 0 and n < len(names):
        return names[:n], f"dataset.{canon}_aux_names_truncated"
    if len(names) > 0 and n > len(names):
        print(
            f"WARNING: aux names count mismatch for dataset={canon}: "
            f"expected {len(names)}, actual {n}; falling back to index labels."
        )
    return [f"context_{i}" for i in range(n)], "fallback.index_only"


def _extract_aux_from_ckpt(ckpt_path: str):
    try:
        import torch

        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt.get("actor", {}))
        if not isinstance(state, dict):
            return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        key = None
        for k in ("aux_logits", "model.aux_logits"):
            if k in state:
                key = k
                break
        if key is None:
            return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        logits = state[key]
        if hasattr(logits, "detach"):
            logits = logits.detach().cpu().numpy()
        logits = np.asarray(logits, dtype=np.float32).reshape(-1)
        probs = 1.0 / (1.0 + np.exp(-logits))
        binary = (probs > 0.5).astype(np.float32)
        return binary, probs.astype(np.float32)
    except Exception:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)


def _aux_cost_diag_for_rollout(rollout: dict, dataset: str, script_dir: str):
    aux_rates = np.asarray(rollout.get("aux_gate_rates", np.zeros((0,), dtype=np.float32))).reshape(-1)
    aux_avail = np.asarray(rollout.get("aux_available_rates", np.zeros((0,), dtype=np.float32))).reshape(-1)
    aux_costs = np.asarray(rollout.get("aux_feature_costs", np.zeros((0,), dtype=np.float32))).reshape(-1)

    aux_dim = int(max(aux_rates.size, aux_avail.size, aux_costs.size))
    if aux_dim <= 0:
        return {
            "aux_dim": 0,
            "static_schema_dim": 0,
            "avg_aux_available_count": float("nan"),
            "avg_aux_cost_schema": float("nan"),
        }

    if aux_rates.size != aux_dim:
        fixed = np.zeros(aux_dim, dtype=np.float32)
        n = min(aux_dim, aux_rates.size)
        if n > 0:
            fixed[:n] = aux_rates[:n]
        aux_rates = fixed
    if aux_avail.size != aux_dim:
        fixed = np.zeros(aux_dim, dtype=np.float32)
        n = min(aux_dim, aux_avail.size)
        if n > 0:
            fixed[:n] = aux_avail[:n]
        aux_avail = fixed
    if aux_costs.size != aux_dim:
        fixed = np.ones(aux_dim, dtype=np.float32)
        n = min(aux_dim, aux_costs.size)
        if n > 0:
            fixed[:n] = aux_costs[:n]
        aux_costs = fixed

    expected_aux = _resolve_expected_aux_dim(dataset)
    if expected_aux is None:
        static_schema_dim = aux_dim
    else:
        static_schema_dim = int(min(aux_dim, int(expected_aux)))

    avg_aux_available_count = float(np.sum(aux_avail))
    avg_aux_cost_schema = float(np.sum(aux_rates[:static_schema_dim] * aux_costs[:static_schema_dim]))
    return {
        "aux_dim": int(aux_dim),
        "static_schema_dim": int(static_schema_dim),
        "avg_aux_available_count": avg_aux_available_count,
        "avg_aux_cost_schema": avg_aux_cost_schema,
    }


def save_learned_baseline_context_plot(group_dir: str, script_dir: str, dataset: str,
                                       learned_rollout: dict, learned_ckpt_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    aux_binary = np.asarray(learned_rollout.get("aux_gate_binary", np.zeros((0,), dtype=np.float32))).reshape(-1)
    aux_probs = np.asarray(learned_rollout.get("aux_gate_probs", np.zeros((0,), dtype=np.float32))).reshape(-1)
    aux_rates = np.asarray(learned_rollout.get("aux_gate_rates", np.zeros((0,), dtype=np.float32))).reshape(-1)

    if aux_binary.size == 0:
        aux_binary, aux_probs_fallback = _extract_aux_from_ckpt(learned_ckpt_path)
        if aux_probs.size == 0:
            aux_probs = aux_probs_fallback

    if aux_binary.size == 0:
        return None

    num_aux = int(aux_binary.size)
    aux_names, _aux_name_source = _resolve_aux_names_for_dataset(dataset=dataset, num_aux=num_aux)

    fig_w = max(12.0, 0.55 * num_aux)
    fig_h = 4.8
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    im = ax.imshow(
        aux_binary.reshape(1, -1),
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_box_aspect(1.0 / max(1, num_aux))
    ax.set_yticks([])
    ax.set_xticks(np.arange(num_aux))
    ax.set_xticklabels(aux_names, rotation=55, ha="right", fontsize=18, fontweight="bold")
    # Draw visible borders around each square cell.
    ax.set_xticks(np.arange(-0.5, num_aux, 1.0), minor=True)
    ax.set_yticks(np.array([-0.5, 0.5]), minor=True)
    ax.grid(which="minor", color="#1a1a1a", linestyle="-", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title("Selected context", fontsize=22, fontweight="bold")
    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.016,
        pad=0.02,
        shrink=0.38,
        aspect=16,
    )
    cbar.set_label("Acquisition rate", fontsize=18, fontweight="bold")
    cbar.ax.tick_params(labelsize=13, width=1.0)
    for lbl in cbar.ax.get_yticklabels():
        lbl.set_fontweight("bold")
    fig.tight_layout()

    out_png = os.path.join(group_dir, "learned_baseline_context_binary.png")
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

    out_npz = os.path.join(group_dir, "learned_baseline_context_binary.npz")
    np.savez(
        out_npz,
        aux_gate_binary=aux_binary.astype(np.float32),
        aux_gate_probs=aux_probs.astype(np.float32),
        aux_gate_rates=aux_rates.astype(np.float32),
        aux_feature_names=np.asarray(aux_names, dtype=object),
        learned_ckpt_path=np.asarray([learned_ckpt_path], dtype=object),
    )
    print(f"Saved: {out_png}")
    print(f"Saved: {out_npz}")
    return {"plot": out_png, "data": out_npz}


def render_group(ap, group: dict, mode_name: str, group_idx: int, output_root: str, args):
    group_dir = os.path.join(output_root, mode_name, f"group_{group_idx:03d}")
    _ensure_dir(group_dir)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Rendering {mode_name} group_{group_idx:03d} -> {group_dir}")
    for b in BASELINES:
        print(f"*** actor ({b}): {group[b]['path']}")

    # Load cached rollout artifacts for the triplet.
    rollouts = {}
    for baseline in BASELINES:
        rec = group[baseline]
        rollouts[baseline] = _read_rollout_npz(rec["rollout_path"])
    aux_diag = {
        b: _aux_cost_diag_for_rollout(rollouts[b], dataset=args.dataset, script_dir=script_dir)
        for b in BASELINES
    }
    for b in BASELINES:
        d = aux_diag[b]
        if int(d["aux_dim"]) > int(d["static_schema_dim"]):
            print(
                f"WARNING: baseline={b} has aux_dim={int(d['aux_dim'])} but "
                f"dataset schema aux dim={int(d['static_schema_dim'])}. "
                "avg_aux_cost is computed over full aux_dim."
            )

    keep_idx, shared_order = _shared_feature_controls(rollouts)
    num_time = int(rollouts["learned"]["num_time"])
    num_groups = int(rollouts["learned"]["num_groups"])
    group_names, group_name_source = _resolve_longitudinal_names_for_dataset(
        dataset=args.dataset,
        num_groups=num_groups,
        ap=ap,
    )
    if len(group_names) != num_groups:
        group_names = list(rollouts["learned"]["group_names"])
        group_name_source = "cache.rollout.group_names"
    group_costs = np.asarray(rollouts["learned"]["group_costs"], dtype=np.float32)
    if len(group_names) != num_groups:
        group_names = [str(i) for i in range(num_groups)]
    if group_costs.shape[0] != num_groups:
        group_costs = np.ones(num_groups, dtype=np.float32)

    # Shared k across triplet.
    requested_k = int(args.cluster_k)
    shared_k = requested_k if requested_k > 0 else None

    baseline_cluster_info = {}
    baseline_edge_info = {}

    for baseline in BASELINES:
        baseline_out = os.path.join(group_dir, baseline)
        _ensure_dir(baseline_out)
        masks = rollouts[baseline]["masks"].astype(np.float32)

        if baseline == "learned":
            cluster_info = ap.run_mask_clustering(
                all_masks=masks,
                num_time=num_time,
                num_groups=num_groups,
                outdir=baseline_out,
                cluster_k=(shared_k if shared_k is not None else 0),
                cluster_kmax=args.cluster_kmax,
                seed=args.seed,
                group_names=group_names,
                run_label=baseline,
                min_feature_rate=MIN_FEATURE_RATE,
                highlight_top_n=6,
                fixed_keep_idx=keep_idx.tolist(),
                show_exclusion_note=False,
                show_learned_highlight=False,
            )
            baseline_cluster_info[baseline] = cluster_info
            if shared_k is None and cluster_info is not None:
                k_chosen = cluster_info.get("cluster_k")
                if k_chosen is not None and int(k_chosen) > 0:
                    shared_k = int(k_chosen)
        else:
            cluster_info = ap.run_mask_clustering(
                all_masks=masks,
                num_time=num_time,
                num_groups=num_groups,
                outdir=baseline_out,
                cluster_k=(shared_k if shared_k is not None else 0),
                cluster_kmax=args.cluster_kmax,
                seed=args.seed,
                group_names=group_names,
                run_label=baseline,
                min_feature_rate=MIN_FEATURE_RATE,
                highlight_top_n=6,
                fixed_keep_idx=keep_idx.tolist(),
                show_exclusion_note=False,
                show_learned_highlight=False,
            )
            baseline_cluster_info[baseline] = cluster_info

    # If auto-k and learned failed to return k, enforce from first available.
    if shared_k is None:
        for baseline in BASELINES:
            info = baseline_cluster_info.get(baseline)
            if info is not None and int(info.get("cluster_k", 0)) > 0:
                shared_k = int(info["cluster_k"])
                break
    if shared_k is None:
        shared_k = 2

    # Re-render with fixed shared_k to guarantee consistency.
    for baseline in BASELINES:
        baseline_out = os.path.join(group_dir, baseline)
        masks = rollouts[baseline]["masks"].astype(np.float32)
        baseline_cluster_info[baseline] = ap.run_mask_clustering(
            all_masks=masks,
            num_time=num_time,
            num_groups=num_groups,
            outdir=baseline_out,
            cluster_k=shared_k,
            cluster_kmax=args.cluster_kmax,
            seed=args.seed,
            group_names=group_names,
            run_label=baseline,
            min_feature_rate=MIN_FEATURE_RATE,
            highlight_top_n=6,
            fixed_keep_idx=keep_idx.tolist(),
            show_exclusion_note=False,
            show_learned_highlight=False,
        )

        sample_paths = _make_sample_paths_from_masks(masks, num_time=num_time, num_groups=num_groups)
        baseline_edge_info[baseline] = ap.run_temporal_edges(
            sample_paths=sample_paths,
            num_time=num_time,
            num_groups=num_groups,
            outdir=baseline_out,
            edge_min_freq=args.edge_min_freq,
            edge_max_edges=args.edge_max_edges,
            group_names=group_names,
            group_costs=group_costs,
            out_suffix=f"{baseline}_group{group_idx:03d}",
            node_size_scale=args.edge_node_size_scale,
            fixed_group_order=shared_order.tolist(),
            fixed_keep_idx=keep_idx.tolist(),
            avg_cost=rollouts[baseline].get("avg_cost"),
            avg_long_cost=rollouts[baseline].get("avg_long_cost"),
            avg_aux_cost=rollouts[baseline].get("avg_aux_cost"),
            transition_mode=args.edge_transition_mode,
        )

    # Consistency checks for saved artifacts.
    missing_edge_baselines = []
    for baseline in BASELINES:
        cinfo = baseline_cluster_info.get(baseline) or {}
        kept = cinfo.get("kept_groups")
        if kept is None:
            raise RuntimeError(f"Missing kept_groups for baseline={baseline} in group_{group_idx:03d}")
        if [int(x) for x in kept] != [int(x) for x in keep_idx.tolist()]:
            raise RuntimeError(f"kept_groups mismatch for baseline={baseline} in group_{group_idx:03d}")

        einfo = baseline_edge_info.get(baseline)
        if einfo is None or "edge_artifacts" not in einfo:
            print(
                f"WARNING: Missing edge artifact for baseline={baseline} in group_{group_idx:03d}. "
                "This can happen when no temporal transitions are found."
            )
            missing_edge_baselines.append(str(baseline))
            continue
        edge_npz = str(einfo["edge_artifacts"])
        if not os.path.exists(edge_npz):
            print(
                f"WARNING: Edge artifact path not found for baseline={baseline} in group_{group_idx:03d}: "
                f"{edge_npz}"
            )
            missing_edge_baselines.append(str(baseline))
            continue
        ez = np.load(edge_npz, allow_pickle=True)
        node_freq = ez["node_freq"]
        if int(node_freq.shape[1]) != int(len(keep_idx)):
            raise RuntimeError(f"Trajectory node dimension mismatch for baseline={baseline} in group_{group_idx:03d}")

    group_meta = {
        "mode": mode_name,
        "group_index": int(group_idx),
        "metric_key": str(group["metric_key"]),
        "match_stage": str(group["match_stage"]),
        "score": float(group["score"]),
        "err_all": float(group["err_all"]),
        "err_none": float(group["err_none"]),
        "target_cost": float(group["target_cost"]),
        "shared_cluster_k": int(shared_k),
        "shared_keep_idx": [int(x) for x in keep_idx.tolist()],
        "shared_order_all": [int(x) for x in shared_order.tolist()],
        "num_time": int(num_time),
        "num_groups": int(num_groups),
        "group_name_source": str(group_name_source),
        "missing_edge_baselines": missing_edge_baselines,
        "checkpoints": {
            b: {
                "ckpt_id": group[b]["ckpt_id"],
                "actor_path": group[b]["path"],
                "cw": float(group[b]["cw"]),
                "acw": float(group[b]["acw"]),
                "avg_long_cost": float(group[b]["avg_long_cost"]),
                "avg_aux_cost": float(group[b]["avg_aux_cost"]),
                "avg_aux_cost_schema": float(aux_diag[b]["avg_aux_cost_schema"]),
                "avg_aux_available_count": float(aux_diag[b]["avg_aux_available_count"]),
                "aux_dim": int(aux_diag[b]["aux_dim"]),
                "static_schema_dim": int(aux_diag[b]["static_schema_dim"]),
                "avg_cost": float(group[b]["avg_cost"]),
                "source_dir": str(group[b].get("source_dir", "")),
                "warmup_fallback": bool(group[b].get("used_warmup_fallback", False)),
                "cache_dir": group[b]["cache_dir"],
            }
            for b in BASELINES
        },
    }

    baseline_ctx = save_learned_baseline_context_plot(
        group_dir=group_dir,
        script_dir=script_dir,
        dataset=args.dataset,
        learned_rollout=rollouts["learned"],
        learned_ckpt_path=group["learned"]["path"],
    )
    if baseline_ctx is not None:
        group_meta["learned_baseline_context"] = baseline_ctx

    _save_json(os.path.join(group_dir, "group_meta.json"), group_meta)

    group_tsv = os.path.join(group_dir, "group_costs.tsv")
    with open(group_tsv, "w") as f:
        f.write(
            "baseline\tckpt_id\tcw\tacw\tavg_long_cost\tavg_aux_cost\tavg_aux_cost_schema\tavg_aux_available_count\taux_dim\tstatic_schema_dim\tavg_cost\trel_err_to_target\twarmup_fallback\tsource_dir\tactor_path\tcache_dir\n"
        )
        for b in BASELINES:
            r = group[b]
            d = aux_diag[b]
            e = rel_err(float(r[group["metric_key"]]), float(group["target_cost"]))
            f.write(
                f"{b}\t{r['ckpt_id']}\t{float(r['cw']):.8g}\t{float(r['acw']):.8g}\t"
                f"{float(r['avg_long_cost']):.8g}\t{float(r['avg_aux_cost']):.8g}\t"
                f"{float(d['avg_aux_cost_schema']):.8g}\t{float(d['avg_aux_available_count']):.8g}\t"
                f"{int(d['aux_dim'])}\t{int(d['static_schema_dim'])}\t"
                f"{float(r['avg_cost']):.8g}\t"
                f"{float(e):.8g}\t{int(bool(r.get('used_warmup_fallback', False)))}\t{r.get('source_dir', '')}\t{r['path']}\t{r['cache_dir']}\n"
            )
    print(f"Saved: {group_tsv}")

    # Quick per-group cost snapshot for fast inspection.
    quick_tsv = os.path.join(group_dir, "group_costs_quick.tsv")
    with open(quick_tsv, "w") as f:
        f.write("baseline\tavg_long_cost\tavg_aux_cost\tavg_aux_cost_schema\tavg_aux_available_count\taux_dim\tstatic_schema_dim\tavg_cost\n")
        for b in BASELINES:
            r = group[b]
            d = aux_diag[b]
            f.write(
                f"{b}\t{float(r['avg_long_cost']):.8g}\t"
                f"{float(r['avg_aux_cost']):.8g}\t{float(d['avg_aux_cost_schema']):.8g}\t"
                f"{float(d['avg_aux_available_count']):.8g}\t"
                f"{int(d['aux_dim'])}\t{int(d['static_schema_dim'])}\t"
                f"{float(r['avg_cost']):.8g}\n"
            )
    print(f"Saved: {quick_tsv}")

    quick_txt = os.path.join(group_dir, "group_costs_quick.txt")
    with open(quick_txt, "w") as f:
        f.write(f"Group: {mode_name}/group_{group_idx:03d}\n")
        f.write(f"Match metric: {group['metric_key']}\n")
        f.write(f"Target cost: {float(group['target_cost']):.6f}\n")
        f.write(
            f"Match errors: all={float(group['err_all']):.6f}, "
            f"none={float(group['err_none']):.6f}, score={float(group['score']):.6f}\n"
        )
        f.write("\n")
        f.write("baseline  long_cost    aux_cost(raw)  aux_cost(schema)  aux_avail  aux_dim  schema_dim  total_cost\n")
        for b in BASELINES:
            r = group[b]
            d = aux_diag[b]
            f.write(
                f"{b:<8}  {float(r['avg_long_cost']):>10.4f}  "
                f"{float(r['avg_aux_cost']):>13.4f}  "
                f"{float(d['avg_aux_cost_schema']):>15.4f}  "
                f"{float(d['avg_aux_available_count']):>9.4f}  "
                f"{int(d['aux_dim']):>7d}  {int(d['static_schema_dim']):>10d}  "
                f"{float(r['avg_cost']):>10.4f}\n"
            )
        if any(int(aux_diag[b]["aux_dim"]) > int(aux_diag[b]["static_schema_dim"]) for b in BASELINES):
            f.write("\n")
            f.write("NOTE: avg_aux_cost(raw) is computed over full checkpoint aux_dim.\n")
            f.write("      aux_cost(schema) is a diagnostic using only the static-schema prefix dims.\n")
    print(f"Saved: {quick_txt}")

    return group_meta


def write_groups_summary(groups_meta: List[dict], output_root: str, mode_name: str):
    file_name = "summary_groups_longitudinal.tsv" if mode_name == "longitudinal" else "summary_groups_total.tsv"
    out_path = os.path.join(output_root, file_name)
    with open(out_path, "w") as f:
        f.write(
            "group\tmatch_stage\tscore\tmetric_key\ttarget_cost\t"
            "learned_ckpt\tlearned_cw\tlearned_acw\tlearned_long\tlearned_aux\tlearned_total\tlearned_path\t"
            "all_ckpt\tall_cw\tall_acw\tall_long\tall_aux\tall_total\tall_rel_err\tall_path\t"
            "none_ckpt\tnone_cw\tnone_acw\tnone_long\tnone_aux\tnone_total\tnone_rel_err\tnone_path\tshared_cluster_k\n"
        )
        for m in groups_meta:
            c = m["checkpoints"]
            metric = m["metric_key"]
            target = float(m["target_cost"])
            all_err = rel_err(float(c["all"][metric]), target)
            none_err = rel_err(float(c["none"][metric]), target)
            f.write(
                f"{int(m['group_index']):03d}\t{m['match_stage']}\t{float(m['score']):.8g}\t{metric}\t{target:.8g}\t"
                f"{c['learned']['ckpt_id']}\t{float(c['learned']['cw']):.8g}\t{float(c['learned']['acw']):.8g}\t"
                f"{float(c['learned']['avg_long_cost']):.8g}\t{float(c['learned']['avg_aux_cost']):.8g}\t{float(c['learned']['avg_cost']):.8g}\t{c['learned']['actor_path']}\t"
                f"{c['all']['ckpt_id']}\t{float(c['all']['cw']):.8g}\t{float(c['all']['acw']):.8g}\t"
                f"{float(c['all']['avg_long_cost']):.8g}\t{float(c['all']['avg_aux_cost']):.8g}\t{float(c['all']['avg_cost']):.8g}\t{all_err:.8g}\t{c['all']['actor_path']}\t"
                f"{c['none']['ckpt_id']}\t{float(c['none']['cw']):.8g}\t{float(c['none']['acw']):.8g}\t"
                f"{float(c['none']['avg_long_cost']):.8g}\t{float(c['none']['avg_aux_cost']):.8g}\t{float(c['none']['avg_cost']):.8g}\t{none_err:.8g}\t{c['none']['actor_path']}\t"
                f"{int(m['shared_cluster_k'])}\n"
            )
    print(f"Saved: {out_path}")


def run_mode(ap, output_root: str, mode_name: str, records_by_baseline: Dict[str, List[dict]], args):
    metric_key = "avg_long_cost" if mode_name == "longitudinal" else "avg_cost"
    groups = build_matched_groups(
        records_by_baseline=records_by_baseline,
        metric_key=metric_key,
        tol_primary=float(args.tol_primary),
        tol_secondary=float(args.tol_secondary),
        max_groups=int(args.max_groups),
    )
    if len(groups) == 0:
        raise RuntimeError(f"No matched groups generated for mode={mode_name}")

    groups_meta = []
    for i, g in enumerate(groups, start=1):
        g_for_render = {
            "metric_key": g["metric_key"],
            "match_stage": g["match_stage"],
            "score": g["score"],
            "err_all": g["err_all"],
            "err_none": g["err_none"],
            "target_cost": g["target_cost"],
            "learned": g["learned"],
            "all": g["all"],
            "none": g["none"],
        }
        gm = render_group(
            ap=ap,
            group=g_for_render,
            mode_name=mode_name,
            group_idx=i,
            output_root=output_root,
            args=args,
        )
        groups_meta.append(gm)

    write_groups_summary(groups_meta, output_root=output_root, mode_name=mode_name)
    return groups_meta


def main():
    args = parse_args()
    args.dataset = _canonical_dataset_name(args.dataset)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.output_root is None:
        output_root = os.path.join(script_dir, "plots", args.dataset.lower(), "matched_cost_groups")
    else:
        output_root = os.path.abspath(args.output_root)
    _ensure_dir(output_root)

    # Ensure downstream imports use requested dataset.
    os.environ["ACTOR_DATASET"] = args.dataset
    import analysis_plots as ap

    if args.test_data_path is None:
        args.test_data_path = ap.resolve_test_data_path(dataset=args.dataset, test_data_path=None)
    else:
        args.test_data_path = os.path.abspath(args.test_data_path)

    # Reproducibility.
    ap.set_seed(int(args.seed))
    np.random.seed(int(args.seed))

    print("=" * 80)
    print("MATCHED-COST PLOTTING")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Test data path: {args.test_data_path}")
    print(f"Output root: {output_root}")
    print(
        f"Mode: {args.mode} | tol_primary={args.tol_primary:.4f} "
        f"| tol_secondary={args.tol_secondary:.4f} | max_groups={args.max_groups}"
    )
    print(f"Include warmup checkpoints: {bool(args.include_warmup)}")

    pools, warmup_fallback = discover_candidates(
        script_dir=script_dir,
        dataset=args.dataset,
        include_warmup=bool(args.include_warmup),
    )
    for b in BASELINES:
        print(
            f"Discovered {len(pools[b])} candidates for baseline={b} "
            f"(warmup_fallback={bool(warmup_fallback.get(b, False))})"
        )

    all_records = []
    for baseline in BASELINES:
        for cand in pools[baseline]:
            rec = evaluate_or_load_candidate(ap=ap, candidate=cand, args=args, output_root=output_root)
            all_records.append(rec)

    # Cost-cache validation.
    for r in all_records:
        _safe_float(r["avg_long_cost"])
        _safe_float(r["avg_aux_cost"])
        _safe_float(r["avg_cost"])

    write_candidates_summary(all_records, output_root=output_root)
    records_by_baseline = {b: [] for b in BASELINES}
    for r in all_records:
        records_by_baseline[r["baseline"]].append(r)

    modes = []
    if args.mode in ("longitudinal", "both"):
        modes.append("longitudinal")
    if args.mode in ("total", "both"):
        modes.append("total")

    ran = {}
    for mode_name in modes:
        print("-" * 80)
        print(f"Running matched grouping mode: {mode_name}")
        print("-" * 80)
        ran[mode_name] = run_mode(
            ap=ap,
            output_root=output_root,
            mode_name=mode_name,
            records_by_baseline=records_by_baseline,
            args=args,
        )

    # Write empty summary placeholder only when missing (do not clobber prior mode outputs).
    if "longitudinal" not in ran:
        p = os.path.join(output_root, "summary_groups_longitudinal.tsv")
        if not os.path.exists(p):
            write_groups_summary([], output_root=output_root, mode_name="longitudinal")
    if "total" not in ran:
        p = os.path.join(output_root, "summary_groups_total.tsv")
        if not os.path.exists(p):
            write_groups_summary([], output_root=output_root, mode_name="total")

    print("=" * 80)
    print("Completed matched-cost plotting pipeline.")
    print("=" * 80)


if __name__ == "__main__":
    main()
