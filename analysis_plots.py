"""
Standalone analysis script for ACTOR-LAFA evaluation outputs.

This script does NOT modify evaluate.py. It runs its own evaluation rollout and
produces three plot groups:
  1) K-Means clustering of final acquisition masks (elbow + templates)
  2) Temporal transition edge graph
"""

import argparse
import os
import glob
import re
import colorsys
import importlib.util
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics import AUROC
from torchmetrics.classification import AveragePrecision

from config import (
    DATA_FOLDER,
    CLASSIFIER_PATH,
    DATASET,
    ACTOR_CONFIG,
    NUM_TIME,
    NUM_FEAT,
    NUM_AUX,
    NUM_GROUPS,
    make_actor_path,
    LONGITUDINAL_FEATURE_COSTS,
)
from dataset import (
    load_ILIADD_data,
    load_adni_data,
    load_synthetic_data,
    load_cheears_data,
    load_klg_data,
    load_womac_data,
)
from models import Predictor
from gumbel_actor import GumbelActor
from utils import set_seed, build_group_to_feat_matrix, get_timestep_embedding


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _existing_path(path):
    """Resolve path against CWD and this file's directory."""
    if not path:
        return None
    if os.path.exists(path):
        return path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    alt = os.path.join(script_dir, path)
    if os.path.exists(alt):
        return alt
    return None


def resolve_classifier_path(dataset):
    """Find classifier checkpoint in either outputs/<dataset>/ or <dataset>/."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        CLASSIFIER_PATH,
        os.path.join(dataset, "classifier.ckpt"),
        os.path.join("outputs", dataset, "classifier.ckpt"),
    ]
    for c in candidates:
        hit = _existing_path(c)
        if hit is not None:
            return hit

    dataset_dir = os.path.join(script_dir, dataset)
    if os.path.isdir(dataset_dir):
        preferred = os.path.join(dataset_dir, "classifier.ckpt")
        if os.path.exists(preferred):
            return preferred

        matches = sorted(glob.glob(os.path.join(dataset_dir, "classifier*.ckpt")))
        if matches:
            # prefer non-versioned canonical names, otherwise newest file
            for m in matches:
                base = os.path.basename(m)
                if base in ("classifier.ckpt", "classifier_bernoulli.ckpt"):
                    return m
            matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return matches[0]

    return None


def resolve_actor_path(actor_path, dataset, cw, acw, joint=False, baseline="learned"):
    """Find actor checkpoint in either outputs/<dataset>/ or <dataset>/."""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if actor_path is not None:
        hit = _existing_path(actor_path)
        if hit is not None:
            return hit
        # Keep user-provided path for final error message.
        return actor_path

    dataset_dir = os.path.join(script_dir, dataset)
    derived = make_actor_path(cw, acw, joint=joint, baseline=baseline)
    if not os.path.isdir(dataset_dir):
        hit = _existing_path(derived)
        if hit is not None:
            return hit
        return derived

    prefix = "actor_iterative_joint" if joint else "actor_iterative"
    if baseline != "learned":
        prefix = f"{prefix}_baseline_{baseline}"

    # Exact first
    exact = os.path.join(dataset_dir, f"{prefix}_cw{cw}_acw{acw}.ckpt")
    if os.path.exists(exact):
        return exact

    # Then allow variants like *_warmup50.ckpt
    pattern = os.path.join(dataset_dir, f"{prefix}_cw{cw}_acw{acw}*.ckpt")
    matches = sorted(glob.glob(pattern))
    if matches:
        # prefer shortest filename (usually the canonical non-warmup version)
        matches.sort(key=lambda p: (len(os.path.basename(p)), os.path.basename(p)))
        return matches[0]

    # Fallback for cheears-style baseline ablations that may only exist as non-joint ckpts.
    if joint and baseline in ("all", "none"):
        alt_prefix = f"actor_iterative_baseline_{baseline}"
        alt_exact = os.path.join(dataset_dir, f"{alt_prefix}_cw{cw}_acw{acw}.ckpt")
        if os.path.exists(alt_exact):
            return alt_exact
        alt_pattern = os.path.join(dataset_dir, f"{alt_prefix}_cw{cw}_acw{acw}*.ckpt")
        alt_matches = sorted(glob.glob(alt_pattern))
        if alt_matches:
            alt_matches.sort(key=lambda p: (len(os.path.basename(p)), os.path.basename(p)))
            return alt_matches[0]

    # Fallback to config-derived path (e.g., outputs/<dataset>/...).
    hit = _existing_path(derived)
    if hit is not None:
        return hit

    return derived


def _dedupe_keep_order(items):
    seen = set()
    out = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _npz_compatible_for_dataset(path, dataset, expected_num_aux=None, expected_num_feat=None):
    """Validate that an npz file has keys/shapes expected by dataset loaders."""
    try:
        z = np.load(path, allow_pickle=True)
    except Exception:
        return False

    keys = set(z.keys())
    if dataset in ("cheears", "cheears_demog", "cheears_day_context", "ILIADD"):
        required = {"x", "y", "mask", "x_static", "mask_static"}
        if not required.issubset(keys):
            return False
        if "x" in z and expected_num_feat is not None:
            x = z["x"]
            if x.ndim >= 3 and int(x.shape[-1]) != int(expected_num_feat):
                return False
        if "x_static" in z and expected_num_aux is not None:
            x_static = z["x_static"]
            aux_dim = int(x_static.shape[-1]) if x_static.ndim >= 2 else 0
            if aux_dim != int(expected_num_aux):
                return False
        return True

    if dataset == "womac":
        return {"x", "WOMAC"}.issubset(keys)
    if dataset == "klg":
        return {"x", "KLG"}.issubset(keys)
    if dataset == "adni":
        return {"x", "y", "mask", "feat_list"}.issubset(keys)
    if dataset == "synthetic":
        return {"x", "y"}.issubset(keys)

    return "x" in keys and "y" in keys


def resolve_test_data_path(dataset, test_data_path=None, expected_num_aux=None, expected_num_feat=None):
    """
    Resolve test_data.npz robustly across ACTOR-LAFA and baseline_DIME layouts.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(script_dir)

    candidates = []
    if test_data_path is not None:
        candidates.append(test_data_path)

    # Existing ACTOR config path
    candidates.append(os.path.join(DATA_FOLDER, "test_data.npz"))

    # Common local layouts
    candidates.extend([
        os.path.join(dataset, "test_data.npz"),
        os.path.join(script_dir, dataset, "test_data.npz"),
        os.path.join(workspace_root, dataset, "test_data.npz"),
    ])

    # baseline_DIME / AAco layouts used in your environment
    if dataset in ("womac", "klg"):
        candidates.extend([
            os.path.join(workspace_root, "aaco", "input_data", "womac", "test_data.npz"),
            os.path.join(workspace_root, "aaco", "input_data", "test_data.npz"),
        ])
    elif dataset == "adni":
        candidates.extend([
            os.path.join(workspace_root, "aaco", "input_data", "test_data.npz"),
        ])
    elif dataset in ("cheears", "cheears_demog", "cheears_day_context"):
        candidates.extend([
            os.path.join(workspace_root, "aaco", "cheears_indiv", "cheears_day_context_ver3", "test_data.npz"),
            os.path.join(workspace_root, "aaco", "cheears_indiv", "cheears_ver_2", "test_data.npz"),
            os.path.join(workspace_root, "aaco", "cheears_indiv", "cheears_combine", "test_data.npz"),
            os.path.join(workspace_root, "aaco", "cheears_indiv", "test_data.npz"),
            os.path.join(workspace_root, "baseline_DIME", "cheears_indiv", "test_data.npz"),
        ])
    elif dataset == "ILIADD":
        candidates.extend([
            os.path.join(workspace_root, "aaco", "cheears_indiv", "iliadd_combine", "test_data.npz"),
            os.path.join(workspace_root, "aaco", "cheears_indiv", "ILIADD_v3", "test_data.npz"),
            os.path.join(workspace_root, "aaco", "cheears_indiv", "ILIADD_v2", "test_data.npz"),
        ])
    elif dataset == "synthetic":
        candidates.extend([
            os.path.join(workspace_root, "baselines_dataset", "synthetic", "test_data.npz"),
        ])

    for c in _dedupe_keep_order(candidates):
        hit = _existing_path(c)
        if hit is not None and _npz_compatible_for_dataset(
            hit,
            dataset=dataset,
            expected_num_aux=expected_num_aux,
            expected_num_feat=expected_num_feat,
        ):
            return hit

    # Final fallback: bounded search in known data roots
    search_roots = [
        os.path.join(workspace_root, "aaco"),
        os.path.join(workspace_root, "baselines_dataset"),
        os.path.join(workspace_root, "baseline_DIME"),
        script_dir,
    ]
    hits = []
    for root in search_roots:
        if os.path.isdir(root):
            hits.extend(glob.glob(os.path.join(root, "**", "test_data.npz"), recursive=True))
    hits = _dedupe_keep_order(hits)

    if hits:
        tokens = {
            "womac": ["womac", "oai"],
            "klg": ["womac", "oai", "klg"],
            "adni": ["adni", "input_data"],
            "cheears": ["cheears", "cheears_indiv"],
            "cheears_demog": ["cheears", "cheears_indiv"],
            "cheears_day_context": ["cheears", "cheears_indiv", "day_context"],
            "ILIADD": ["iliadd", "cheears_indiv"],
            "synthetic": ["synthetic"],
        }.get(dataset, [dataset.lower()])

        scored = []
        for p in hits:
            if not _npz_compatible_for_dataset(
                p,
                dataset=dataset,
                expected_num_aux=expected_num_aux,
                expected_num_feat=expected_num_feat,
            ):
                continue
            lp = p.lower()
            score = sum(1 for t in tokens if t in lp)
            scored.append((score, -len(p), p))
        scored.sort(reverse=True)
        if scored:
            return scored[0][2]

    attempted = "\n".join(f"  - {c}" for c in _dedupe_keep_order(candidates))
    raise FileNotFoundError(
        "Could not resolve test_data.npz for dataset "
        f"'{dataset}'. Tried:\n{attempted}"
    )


def load_actor_state_dict(actor, ckpt):
    if "state_dict" in ckpt:
        actor.load_state_dict(ckpt["state_dict"])
    elif "actor" in ckpt:
        actor.load_state_dict(ckpt["actor"])
    else:
        raise KeyError(f"Checkpoint has no 'state_dict' or 'actor' key. Keys: {list(ckpt.keys())}")


def extract_state_dict(ckpt):
    if "state_dict" in ckpt:
        return ckpt["state_dict"]
    if "actor" in ckpt:
        return ckpt["actor"]
    raise KeyError(f"Checkpoint has no 'state_dict' or 'actor' key. Keys: {list(ckpt.keys())}")


class AnalysisCollector:
    """Collect per-timestep acquisitions (for edges)."""

    def __init__(self, num_time, num_groups, max_planner_states=0, seed=42):
        self.num_time = int(num_time)
        self.num_groups = int(num_groups)
        self.max_planner_states = max(0, int(max_planner_states))
        self.rng = np.random.default_rng(seed)

        self._planner_seen = 0
        self._planner_states = []
        self._batch_paths = None
        self.sample_paths = []

    def start_batch(self, batch_size):
        self._batch_paths = [[set() for _ in range(self.num_time)] for _ in range(batch_size)]

    def end_batch(self):
        if self._batch_paths is None:
            return
        for sample in self._batch_paths:
            self.sample_paths.append([sorted(list(gs)) for gs in sample])
        self._batch_paths = None

    def add_planner_input_batch(self, planner_input):
        if self.max_planner_states <= 0:
            return
        batch_np = planner_input.detach().cpu().numpy().astype(np.float32)
        if batch_np.ndim != 2:
            return

        for row in batch_np:
            self._planner_seen += 1
            if len(self._planner_states) < self.max_planner_states:
                self._planner_states.append(row.copy())
            else:
                j = int(self.rng.integers(0, self._planner_seen))
                if j < self.max_planner_states:
                    self._planner_states[j] = row.copy()

    def record_step(self, cur_t, cur_t_mask_g, m_done):
        if self._batch_paths is None:
            return

        cur_t_np = cur_t.detach().cpu().numpy().astype(np.int64)
        step_np = cur_t_mask_g.detach().cpu().numpy()
        done_np = m_done.detach().cpu().numpy().astype(bool)

        for b in range(step_np.shape[0]):
            if done_np[b]:
                continue
            t = int(cur_t_np[b])
            if t < 0 or t >= self.num_time:
                continue
            s = t * self.num_groups
            e = (t + 1) * self.num_groups
            added = np.flatnonzero(step_np[b, s:e] > 0.5)
            if added.size > 0:
                self._batch_paths[b][t].update(int(g) for g in added.tolist())

    def planner_states_array(self):
        if not self._planner_states:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack(self._planner_states, axis=0).astype(np.float32)


def load_test_dataset(test_data_path):
    if DATASET in ("cheears", "cheears_demog", "cheears_day_context"):
        return load_cheears_data(test_data_path)
    if DATASET == "klg":
        return load_klg_data(test_data_path)
    if DATASET == "womac":
        return load_womac_data(test_data_path)
    if DATASET == "ILIADD":
        return load_ILIADD_data(test_data_path)
    if DATASET == "adni":
        return load_adni_data(test_data_path)
    return load_synthetic_data(test_data_path)


def _extract_predictor_state_from_actor_state(state_dict):
    out = {}
    for k, v in state_dict.items():
        if isinstance(k, str) and k.startswith("predictor."):
            out[k[len("predictor."):]] = v
    return out


def _infer_actor_classifier_dims(actor_ckpt, state_dict):
    cfg = actor_ckpt.get("config", {}) or {}

    num_time = actor_ckpt.get("num_time", cfg.get("num_time", NUM_TIME))
    num_feat = actor_ckpt.get("num_feat", cfg.get("num_feat", NUM_FEAT))
    num_aux = actor_ckpt.get("num_aux", cfg.get("num_aux", NUM_AUX))
    num_groups = actor_ckpt.get("num_groups", cfg.get("num_groups", None))

    first_w = state_dict.get("predictor.model.0.weight")
    last_w = None
    for k in (
        "predictor.model.6.weight",
        "predictor.model.3.weight",
        "predictor.model.0.weight",
    ):
        if k in state_dict:
            last_w = state_dict[k]
            break
    if last_w is None:
        pred_keys = [k for k in state_dict.keys() if isinstance(k, str) and k.startswith("predictor.model.") and k.endswith(".weight")]
        if pred_keys:
            pred_keys = sorted(pred_keys, key=lambda x: int(re.search(r"model\.(\d+)\.weight$", x).group(1)) if re.search(r"model\.(\d+)\.weight$", x) else -1)
            last_w = state_dict[pred_keys[-1]]

    hidden = int(cfg.get("hidden_dim", 32))
    y_dim = actor_ckpt.get("y_dim", None)
    d_in = None
    if first_w is not None and hasattr(first_w, "shape") and len(first_w.shape) == 2:
        hidden = int(first_w.shape[0])
        d_in_plus_t = int(first_w.shape[1])
        if d_in_plus_t >= 1:
            d_in = int(d_in_plus_t - 1)
    if last_w is not None and hasattr(last_w, "shape") and len(last_w.shape) == 2:
        y_dim = int(last_w.shape[0])

    if "aux_logits" in state_dict and hasattr(state_dict["aux_logits"], "shape"):
        num_aux = int(state_dict["aux_logits"].shape[0])
    if "group_to_feat" in state_dict and hasattr(state_dict["group_to_feat"], "shape"):
        g2f = state_dict["group_to_feat"]
        if len(g2f.shape) == 2:
            num_groups = int(g2f.shape[0])
            num_feat = int(g2f.shape[1])
    if "feature_costs" in state_dict and hasattr(state_dict["feature_costs"], "shape"):
        fc = state_dict["feature_costs"]
        if len(fc.shape) == 1:
            num_groups = int(fc.shape[0])

    num_time = int(num_time)
    num_feat = int(num_feat)
    num_aux = int(num_aux)
    if num_groups is None:
        num_groups = int(num_feat if num_feat > 0 else NUM_GROUPS)
    num_groups = int(num_groups)

    if d_in is not None:
        expected = int(num_time * num_feat + num_aux)
        if expected != int(d_in):
            inferred_aux = int(d_in - num_time * num_feat)
            if inferred_aux >= 0:
                print(
                    "WARNING: inferred predictor input mismatch with ckpt dims; "
                    f"adjusting num_aux from {num_aux} to {inferred_aux}."
                )
                num_aux = inferred_aux
            else:
                inferred_feat = int((d_in - num_aux) // max(num_time, 1))
                if inferred_feat > 0:
                    print(
                        "WARNING: inferred predictor input mismatch with ckpt dims; "
                        f"adjusting num_feat from {num_feat} to {inferred_feat}."
                    )
                    num_feat = inferred_feat
                    if num_groups == NUM_GROUPS:
                        num_groups = num_feat
        d_in = int(d_in)
    else:
        d_in = int(num_time * num_feat + num_aux)

    if y_dim is None:
        y_dim = int(actor_ckpt.get("num_classes", cfg.get("num_classes", 2)))

    dropout = float(cfg.get("dropout", ACTOR_CONFIG.get("dropout", 0.3)))
    return {
        "num_time": int(num_time),
        "num_feat": int(num_feat),
        "num_aux": int(num_aux),
        "num_groups": int(num_groups),
        "d_in": int(d_in),
        "y_dim": int(y_dim),
        "hidden_dim": int(hidden),
        "dropout": float(dropout),
    }


def load_model_and_data(actor_path, test_data_path=None, baseline="learned", batch_size=64,
                        allow_infer_without_classifier=False, classifier_path_override=None):
    if not os.path.exists(actor_path):
        raise FileNotFoundError(
            f"Actor not found at {actor_path}. Please run actor training first "
            f"or pass --actor_path explicitly."
        )

    actor_ckpt = torch.load(actor_path, map_location="cpu")
    actor_cfg = actor_ckpt.get("config", {})
    state_dict = extract_state_dict(actor_ckpt)
    classifier_path = classifier_path_override if classifier_path_override is not None else resolve_classifier_path(DATASET)

    if classifier_path is not None:
        classifier_ckpt = torch.load(classifier_path, map_location="cpu")
        num_time = int(classifier_ckpt["num_time"])
        num_feat = int(classifier_ckpt["num_feat"])
        num_aux = int(classifier_ckpt.get("num_aux", 0))

        predictor = Predictor(
            d_in=num_time * num_feat + num_aux,
            d_out=classifier_ckpt["y_dim"],
            hidden=classifier_ckpt["config"]["hidden_dim"],
            dropout=classifier_ckpt["config"]["dropout"],
        )
        predictor.load_state_dict(classifier_ckpt["predictor"])
        num_groups = int(actor_ckpt.get("num_groups", num_feat))
    elif allow_infer_without_classifier:
        inferred = _infer_actor_classifier_dims(actor_ckpt, state_dict)
        num_time = int(inferred["num_time"])
        num_feat = int(inferred["num_feat"])
        num_aux = int(inferred["num_aux"])
        num_groups = int(inferred["num_groups"])

        predictor = Predictor(
            d_in=int(inferred["d_in"]),
            d_out=int(inferred["y_dim"]),
            hidden=int(inferred["hidden_dim"]),
            dropout=float(inferred["dropout"]),
        )
        pred_state = _extract_predictor_state_from_actor_state(state_dict)
        if pred_state:
            try:
                predictor.load_state_dict(pred_state)
            except Exception:
                predictor.load_state_dict(pred_state, strict=False)
        classifier_path = "inferred_from_actor_ckpt"
        print(
            "Classifier checkpoint missing; using inferred predictor dims from actor checkpoint: "
            f"T={num_time}, d={num_feat}, aux={num_aux}, groups={num_groups}, y_dim={inferred['y_dim']}."
        )
    else:
        raise FileNotFoundError(
            f"Classifier checkpoint not found for dataset '{DATASET}'. "
            f"Tried config path '{CLASSIFIER_PATH}' and dataset folders."
        )

    group_to_feat_matrix = None
    if "group_to_feat" in state_dict:
        group_to_feat_matrix = state_dict["group_to_feat"].detach().clone().float()
    elif num_groups != num_feat:
        # Fallback if old checkpoints do not store the group_to_feat buffer.
        group_to_feat_matrix = build_group_to_feat_matrix(num_feat)

    actor = GumbelActor(
        predictor=predictor,
        num_time=num_time,
        num_feat=num_feat,
        config=actor_ckpt["config"],
        num_aux=num_aux,
        num_groups=num_groups,
        group_to_feat_matrix=group_to_feat_matrix,
    )
    load_actor_state_dict(actor, actor_ckpt)

    if baseline == "all" and actor.num_aux > 0:
        actor.aux_logits.data.fill_(100.0)
        print("Forcing ALL baseline features to be acquired")
    elif baseline == "none" and actor.num_aux > 0:
        actor.aux_logits.data.fill_(-100.0)
        print("Forcing NO baseline features to be acquired")

    test_data_path = resolve_test_data_path(
        DATASET,
        test_data_path=test_data_path,
        expected_num_aux=num_aux,
        expected_num_feat=num_feat,
    )
    test_name = os.path.basename(str(test_data_path)).lower()
    if "test" not in test_name:
        raise ValueError(
            f"Refusing to run analysis on non-test split path: {test_data_path}. "
            "Please provide a test split file (e.g., test_data.npz)."
        )
    test_dataset = load_test_dataset(test_data_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return actor, actor_cfg, test_loader, num_time, num_feat, classifier_path, test_data_path


def evaluate_with_collection(actor, dataloader, device, num_time, num_feat,
                             feature_costs=None, aux_feature_costs=None,
                             collector=None):
    actor.eval()
    actor = actor.to(device)
    ng = actor.num_groups

    if feature_costs is not None:
        fc = np.array(feature_costs, dtype=np.float32)
    else:
        fc = np.ones(ng, dtype=np.float32)
    feature_costs_flat_np = np.tile(fc, num_time)

    if aux_feature_costs is not None:
        afc_t = torch.tensor(aux_feature_costs, dtype=torch.float32, device=device)
    else:
        afc_t = torch.ones(max(actor.num_aux, 1), dtype=torch.float32, device=device)

    all_masks = []
    all_preds = []
    all_labels = []
    total_aux_cost = 0.0
    total_samples = 0
    aux_gate_sum = np.zeros(actor.num_aux, dtype=np.float64) if actor.num_aux > 0 else None
    aux_mask_sum = np.zeros(actor.num_aux, dtype=np.float64) if actor.num_aux > 0 else None
    aux_seen_samples = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating + Collecting"):
            if len(batch) == 5:
                x, y, m_avail, x_static, mask_static = batch
                x_static = torch.nan_to_num(x_static).float().to(device)
                mask_static = mask_static.float().to(device)
            else:
                x, y, m_avail = batch
                x_static = None
                mask_static = None

            x = torch.nan_to_num(x).to(device)
            y = y.to(device)
            m_avail = m_avail.to(device)

            B = x.shape[0]
            if x.dim() == 3:
                x_flat = x.reshape(B, -1)
                m_avail_flat = m_avail.reshape(B, -1)
            else:
                x_flat = x
                m_avail_flat = m_avail

            m_avail_groups = actor.feat_mask_to_group_mask(m_avail_flat.float())

            aux_acquired = None
            aux_gates = None
            if actor.num_aux > 0 and x_static is not None:
                aux_gates = actor.get_aux_gates(B, mask_static)
                aux_acquired = x_static * aux_gates
                total_aux_cost += (aux_gates * afc_t).sum().item()
                aux_gate_sum += aux_gates.sum(dim=0).detach().cpu().numpy()
                aux_mask_sum += mask_static.sum(dim=0).detach().cpu().numpy()
                aux_seen_samples += float(B)

            m_curr_groups = torch.zeros(B, num_time * ng, dtype=torch.float32, device=device)
            m_curr_feat = torch.zeros(B, num_time * num_feat, dtype=torch.float32, device=device)
            cur_t = torch.zeros(B, dtype=torch.int, device=device)
            m_done = torch.zeros(B, dtype=torch.bool, device=device)

            if collector is not None:
                collector.start_batch(B)

            for _ in range(2 * num_time):
                t_grid = torch.arange(num_time, device=device).unsqueeze(0).expand(B, -1)
                time_mask = t_grid >= cur_t.unsqueeze(1)
                time_mask_g = time_mask.unsqueeze(-1).expand(-1, -1, ng).reshape(B, -1)

                valid_mask_g = (m_avail_groups > 0) & (m_curr_groups == 0) & time_mask_g
                valid_counts = torch.sum(valid_mask_g, dim=1)
                if ((valid_counts == 0) | m_done).all():
                    break

                time_emb = get_timestep_embedding(cur_t, embedding_dim=actor.time_emb_dim)

                x_masked = actor.mask_layer(x_flat, m_curr_feat)
                if aux_acquired is not None:
                    planner_input = torch.cat([x_masked, m_curr_feat, aux_acquired, aux_gates, time_emb], dim=1)
                else:
                    planner_input = torch.cat([x_masked, m_curr_feat, time_emb], dim=1)

                if collector is not None:
                    collector.add_planner_input_batch(planner_input)

                planner_logits = actor.planner_nn(planner_input)
                masked_logits = planner_logits.masked_fill(valid_mask_g == 0, float("-inf"))
                z_groups = actor.gumbel_sigmoid(masked_logits, hard=True)

                cur_t_mask_g = torch.zeros_like(m_curr_groups)
                for b in range(B):
                    if not m_done[b]:
                        s = int(cur_t[b].item()) * ng
                        e = (int(cur_t[b].item()) + 1) * ng
                        cur_t_mask_g[b, s:e] = z_groups[b, s:e]

                if collector is not None:
                    collector.record_step(cur_t, cur_t_mask_g, m_done)

                m_curr_groups = (m_curr_groups + cur_t_mask_g).clamp(0, 1)
                m_curr_feat = actor.expand_group_gates_to_feat_mask(m_curr_groups).clamp(0, 1)

                added = cur_t_mask_g.sum(dim=1)
                for b in range(B):
                    if added[b] > 0 and not m_done[b]:
                        cur_t[b] = min(cur_t[b] + 1, num_time)

                m_done = m_done | (added == 0)

            if collector is not None:
                collector.end_batch()

            # Final acquired groups are the actually executed acquisitions:
            # binarized and intersected with available groups.
            final_groups = ((m_curr_groups > 0.5) & (m_avail_groups > 0.5)).float()
            final_feat = actor.expand_group_gates_to_feat_mask(final_groups).clamp(0, 1)
            y_hat = actor.predict_with_mask(x_flat, final_feat, aux_acquired=aux_acquired)

            all_masks.append(final_groups.cpu().numpy())
            all_preds.append(y_hat.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            total_samples += B

    all_masks = np.concatenate(all_masks, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    all_preds_flat = all_preds.reshape(-1, all_preds.shape[-1])
    all_labels_flat = all_labels.reshape(-1)

    valid = all_labels_flat != -1
    all_preds_flat = all_preds_flat[valid]
    all_labels_flat = all_labels_flat[valid]

    pred_classes = np.argmax(all_preds_flat, axis=-1)
    accuracy = float((pred_classes == all_labels_flat).mean())

    y_dim = all_preds_flat.shape[-1]
    preds_t = torch.from_numpy(all_preds_flat).float()
    labels_t = torch.from_numpy(all_labels_flat).long()

    if y_dim == 2:
        auroc_metric = AUROC(task="binary")
        auprc_metric = AveragePrecision(task="binary")
        auroc = float(auroc_metric(preds_t[:, 1], labels_t).item())
        auprc = float(auprc_metric(preds_t[:, 1], labels_t).item())
    else:
        auroc_metric = AUROC(task="multiclass", num_classes=y_dim)
        auprc_metric = AveragePrecision(task="multiclass", num_classes=y_dim)
        auroc = float(auroc_metric(preds_t, labels_t).item())
        auprc = float(auprc_metric(preds_t, labels_t).item())

    avg_long_cost = float((all_masks * feature_costs_flat_np).sum(axis=1).mean())
    avg_aux_cost = float(total_aux_cost / total_samples) if total_samples > 0 else 0.0
    if actor.num_aux > 0:
        aux_probs = torch.sigmoid(actor.aux_logits.detach().cpu()).numpy().astype(np.float32)
        aux_binary = (aux_probs > 0.5).astype(np.float32)
    else:
        aux_probs = np.zeros((0,), dtype=np.float32)
        aux_binary = np.zeros((0,), dtype=np.float32)

    if aux_gate_sum is not None and aux_seen_samples > 0:
        aux_gate_rates = (aux_gate_sum / aux_seen_samples).astype(np.float32)
        aux_avail_rates = (aux_mask_sum / aux_seen_samples).astype(np.float32)
    else:
        aux_gate_rates = np.zeros((0,), dtype=np.float32)
        aux_avail_rates = np.zeros((0,), dtype=np.float32)

    results = {
        "accuracy": accuracy,
        "auroc": auroc,
        "auprc": auprc,
        "avg_cost": avg_long_cost + avg_aux_cost,
        "avg_long_cost": avg_long_cost,
        "avg_aux_cost": avg_aux_cost,
        "total_samples": int(all_labels.shape[0]),
        "aux_gate_probs": aux_probs.tolist(),
        "aux_gate_binary": aux_binary.tolist(),
        "aux_gate_rates": aux_gate_rates.tolist(),
        "aux_available_rates": aux_avail_rates.tolist(),
    }
    return results, all_masks, all_preds, all_labels


def load_cheears_group_names_from_file(num_groups):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(script_dir)
    fg_filename = "feature_groups_ver2.py" if DATASET == "cheears_day_context" else "feature_groups.py"
    fg_path = os.path.join(workspace_root, "aaco", "cheears_indiv", fg_filename)
    if not os.path.exists(fg_path):
        return None, None

    try:
        spec = importlib.util.spec_from_file_location("cheears_feature_groups", fg_path)
        if spec is None or spec.loader is None:
            return None, None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        indiv = list(getattr(mod, "LONGITUDINAL_INDIVIDUAL_FEATURE_NAMES", []))
        grouped_dict = dict(getattr(mod, "LONGITUDINAL_FEATURE_GROUP_INDICES", {}))
        grouped = list(grouped_dict.keys())

        # Group-decision setup (e.g., 22 individual + 10 grouped = 32 names)
        names_grouped = indiv + grouped
        if len(names_grouped) == int(num_groups):
            return names_grouped, fg_path

        # Fully expanded feature setup (e.g., 149 longitudinal features)
        if int(num_groups) > 0:
            names_full = [str(i) for i in range(int(num_groups))]
            for i, nm in enumerate(indiv):
                if i < len(names_full):
                    names_full[i] = str(nm)
            for g_name, g_idx in grouped_dict.items():
                for j, idx in enumerate(list(g_idx)):
                    ii = int(idx)
                    if 0 <= ii < len(names_full):
                        names_full[ii] = f"{g_name}_{j + 1}"
            if len(names_full) == int(num_groups):
                return names_full, fg_path
    except Exception as exc:
        print(f"WARNING: failed to load cheears feature names from {fg_path}: {exc}")
    return None, None


def resolve_longitudinal_group_names(num_groups):
    if DATASET in ("cheears", "cheears_demog", "cheears_day_context"):
        names, path = load_cheears_group_names_from_file(num_groups)
        if names is not None:
            return names, f"aaco.cheears_indiv.{os.path.basename(path)} ({path})"

    cfg_names = list(LONGITUDINAL_FEATURE_COSTS.keys())
    if len(cfg_names) == int(num_groups):
        return cfg_names, "config.longitudinal_feature_costs"

    return [str(i) for i in range(int(num_groups))], "fallback.index_only"


def resolve_longitudinal_costs(num_groups, actor, actor_cfg):
    if actor_cfg is not None:
        cfg_costs = actor_cfg.get("feature_costs")
        if cfg_costs is not None and len(cfg_costs) == int(num_groups):
            return np.asarray(cfg_costs, dtype=np.float32), "actor_ckpt.config.feature_costs"

    if hasattr(actor, "feature_costs"):
        buf = actor.feature_costs.detach().cpu().numpy().astype(np.float32)
        if len(buf) == int(num_groups):
            return buf, "actor_buffer.feature_costs"

    return np.ones(int(num_groups), dtype=np.float32), "fallback.uniform_ones"


def auto_choose_cluster_k(ks, inertias):
    """
    Choose k via elbow (max distance to line from first to last inertia point).
    Returns (k, method).
    """
    if len(ks) == 0:
        return 1, "fallback_empty"
    if len(ks) == 1:
        return int(ks[0]), "single_point"

    x = np.asarray(ks, dtype=np.float32)
    y = np.asarray(inertias, dtype=np.float32)
    p1 = np.array([x[0], y[0]], dtype=np.float32)
    p2 = np.array([x[-1], y[-1]], dtype=np.float32)
    line = p2 - p1
    norm = float(np.linalg.norm(line))
    if norm < 1e-12:
        k = int(ks[min(1, len(ks) - 1)])
        return k, "flat_curve_fallback"

    points = np.stack([x, y], axis=1)
    vecs = points - p1
    # 2D point-line distance via cross product magnitude.
    dists = np.abs(line[0] * vecs[:, 1] - line[1] * vecs[:, 0]) / norm
    idx = int(np.argmax(dists))
    k = int(ks[idx])
    if k == 1 and len(ks) > 1:
        k = int(ks[1])
        return k, "elbow_forced_k>=2"
    return k, "elbow_max_distance"


def run_mask_clustering(all_masks, num_time, num_groups, outdir,
                        cluster_k=0, cluster_kmax=10, seed=42, group_names=None,
                        run_label="learned", min_feature_rate=0.01, highlight_top_n=6,
                        fixed_keep_idx=None, show_exclusion_note=True,
                        show_learned_highlight=True):
    try:
        from sklearn.cluster import KMeans
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"Skipping mask clustering: missing dependency ({exc}).")
        return None

    if all_masks.ndim != 2 or all_masks.shape[0] < 2:
        print("Skipping mask clustering: not enough samples.")
        return None

    X = all_masks.astype(np.float32)
    N = X.shape[0]
    T = int(num_time)
    G = int(num_groups)
    mask_3d = X.reshape(N, T, G)

    # Feature usage for filtering and callouts.
    group_usage = mask_3d.mean(axis=(0, 1))  # (G,)
    if fixed_keep_idx is None:
        keep_mask = group_usage >= float(min_feature_rate)
        if int(keep_mask.sum()) == 0:
            keep_mask = group_usage > 0.0
        if int(keep_mask.sum()) == 0:
            keep_mask = np.ones(G, dtype=bool)
        keep_idx = np.where(keep_mask)[0]
    else:
        seen = set()
        ordered_keep = []
        for idx in fixed_keep_idx:
            ii = int(idx)
            if 0 <= ii < G and ii not in seen:
                seen.add(ii)
                ordered_keep.append(ii)
        if len(ordered_keep) == 0:
            keep_idx = np.arange(G, dtype=np.int32)
        else:
            keep_idx = np.asarray(ordered_keep, dtype=np.int32)
    excluded_idx = np.asarray([i for i in range(G) if i not in set(keep_idx.tolist())], dtype=np.int32)

    if group_names is None or len(group_names) != G:
        group_names = [str(i) for i in range(G)]
    kept_names = [group_names[i] for i in keep_idx]

    # Learned-context callout: highlight the most-used handful.
    highlighted_orig = []
    if show_learned_highlight and run_label == "learned":
        strong = np.where(group_usage >= 0.5)[0]
        if len(strong) == 0:
            top_n = min(int(highlight_top_n), G)
            strong = np.argsort(-group_usage)[:top_n]
        else:
            strong = strong[np.argsort(-group_usage[strong])]
            strong = strong[:min(int(highlight_top_n), len(strong))]
        highlighted_orig = [int(i) for i in strong.tolist()]
    pos_in_kept = {int(orig): j for j, orig in enumerate(keep_idx.tolist())}
    highlighted_kept = [pos_in_kept[i] for i in highlighted_orig if i in pos_in_kept]

    max_k = min(max(1, int(cluster_kmax)), X.shape[0])
    ks = list(range(1, max_k + 1))
    inertias = []

    for k in ks:
        km = KMeans(n_clusters=k, random_state=seed, n_init=20)
        km.fit(X)
        inertias.append(float(km.inertia_))

    elbow_path = os.path.join(outdir, "mask_cluster_elbow.png")
    fs_elbow_title = 22
    fs_elbow_axis = 19
    fs_elbow_tick = 16
    plt.figure(figsize=(8.2, 5.0))
    plt.plot(ks, inertias, marker="o", color="#204a87")
    plt.xticks(ks)
    plt.xlabel("k", fontsize=fs_elbow_axis, fontweight="bold")
    plt.ylabel("Inertia", fontsize=fs_elbow_axis, fontweight="bold")
    plt.title("K-Means Elbow", fontsize=fs_elbow_title, fontweight="bold")
    plt.xticks(fontsize=fs_elbow_tick, fontweight="bold")
    plt.yticks(fontsize=fs_elbow_tick, fontweight="bold")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(elbow_path, dpi=220)
    plt.close()
    print(f"Saved: {elbow_path}")

    if int(cluster_k) <= 0:
        chosen_k, k_method = auto_choose_cluster_k(ks, inertias)
        chosen_k = min(max(1, int(chosen_k)), max_k)
    else:
        chosen_k = min(max(1, int(cluster_k)), max_k)
        k_method = "manual"
    print(f"Chosen cluster_k={chosen_k} (method={k_method})")

    km = KMeans(n_clusters=chosen_k, random_state=seed, n_init=20)
    raw_labels = km.fit_predict(X)
    raw_centroids = km.cluster_centers_.reshape(chosen_k, T, G)

    # Sort clusters by size (largest first).
    raw_counts = np.bincount(raw_labels, minlength=chosen_k)
    order = np.argsort(-raw_counts)
    centroids = raw_centroids[order]
    counts = raw_counts[order]
    pct = counts / counts.sum()

    cluster_labels = np.zeros_like(raw_labels)
    for new_i, old_i in enumerate(order.tolist()):
        cluster_labels[raw_labels == old_i] = new_i

    centroids_cont = centroids[:, :, keep_idx]

    def _plot_centroid_panel(centroids_plot, out_path):
        F = centroids_plot.shape[2]
        fs_centroid_title = 22 if chosen_k <= 4 else 19
        fs_axis_label = 22
        fs_tick_x = 22   # timestep index font
        fs_tick_y = 20   # feature-name font
        fs_suptitle = 24
        fs_note = 13
        # Size figure from an approximate per-cell size so heatmap cells are square-like.
        cell_size = 0.22
        panel_w = max(1.8, float(T) * cell_size)
        panel_h = max(1.8, float(F) * cell_size)
        # For k=1 with large fonts, aggressively widen canvas so y-labels are not clipped.
        min_w = 15.0 if int(chosen_k) == 1 else 18.0
        fig_w = min(52.0, max(min_w, panel_w * chosen_k + 7.0))
        fig_h = min(28.0, max(9.0, panel_h + 3.2))
        fig, axes = plt.subplots(
            1, chosen_k,
            figsize=(fig_w, fig_h),
            squeeze=False,
        )
        axes = axes[0]
        im = None

        xticks = np.arange(T)
        xlabels = [str(t + 1) for t in xticks]
        if F <= 40:
            yticks = np.arange(F)
        else:
            ystep = max(1, int(np.ceil(F / 40.0)))
            yticks = np.arange(0, F, ystep)
        ylabels = [short_text(kept_names[i], max_len=24) for i in yticks]
        max_ylab_len = max((len(str(y)) for y in ylabels), default=10)

        for i in range(chosen_k):
            ax = axes[i]
            im = ax.imshow(
                centroids_plot[i].T,
                cmap="magma",
                vmin=0.0,
                vmax=1.0,
                aspect="equal",
                interpolation="nearest",
            )
            # Keep row/column units visually square.
            if T > 0:
                ax.set_box_aspect(float(F) / float(T))
            ax.set_title(f"Centroid {i + 1} ({pct[i] * 100:.1f}%)", fontsize=fs_centroid_title, fontweight="bold")
            ax.set_xticks(xticks)
            ax.set_xticklabels(xlabels, rotation=0, ha="center", fontsize=fs_tick_x, fontweight="bold")
            if i == 0:
                ax.set_ylabel("Feature / group", fontsize=fs_axis_label, fontweight="bold")
                ax.set_yticks(yticks)
                ax.set_yticklabels(ylabels, fontsize=fs_tick_y, fontweight="bold")
            else:
                ax.set_yticks(yticks)
                ax.set_yticklabels([])

            # Highlight learned-context selected features.
            if show_learned_highlight and i == 0 and highlighted_kept:
                tick_to_pos = {int(t): idx for idx, t in enumerate(yticks.tolist())}
                for p in highlighted_kept:
                    if p in tick_to_pos:
                        lbl = ax.get_yticklabels()[tick_to_pos[p]]
                        lbl.set_color("#0b7d19")
                        lbl.set_fontweight("bold")

        fig.supxlabel("Timestep", fontsize=fs_axis_label, fontweight="bold")
        # Adaptive margins so big feature labels and colorbar label are never cut.
        left_margin = min(0.52, max(0.24, 0.23 + 0.008 * max(0, max_ylab_len - 10)))
        bottom_margin = 0.20
        top_margin = 0.86
        right_margin = 0.84
        wspace = 0.24
        fig.subplots_adjust(
            left=left_margin,
            right=right_margin,
            bottom=bottom_margin,
            top=top_margin,
            wspace=wspace,
        )
        cax = fig.add_axes([right_margin + 0.03, 0.22, 0.020, 0.62])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Acquisition rate", fontsize=fs_axis_label, fontweight="bold")
        cbar.ax.tick_params(labelsize=fs_tick_x, width=1.0)
        for lbl in cbar.ax.get_yticklabels():
            lbl.set_fontweight("bold")

        fig.suptitle("Acquisition Centroids", y=0.95, fontsize=fs_suptitle, fontweight="bold")

        if show_exclusion_note:
            if fixed_keep_idx is None:
                excl_note = (
                    f"Excluded {len(excluded_idx)} low-use groups "
                    f"(rate < {float(min_feature_rate):.3f}); plotted {len(keep_idx)} / {G}."
                )
            else:
                excl_note = (
                    f"Using shared feature set across baselines; plotted {len(keep_idx)} / {G} groups."
                )
            fig.text(0.01, 0.01, excl_note, fontsize=fs_note, color="#4d4d4d", ha="left", va="bottom")

        if show_learned_highlight and run_label == "learned":
            highlighted_names = [group_names[i] for i in highlighted_orig]
            if highlighted_names:
                note = "Learned-context selected: " + ", ".join(short_text(x, 22) for x in highlighted_names)
                fig.text(0.01, 0.90, note, fontsize=fs_note, color="#0b7d19", ha="left", va="top")

        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        print(f"Saved: {out_path}")

    templates_path = os.path.join(outdir, f"mask_templates_k{chosen_k}.png")
    _plot_centroid_panel(centroids_cont, templates_path)

    clusters_npz = os.path.join(outdir, f"mask_clusters_k{chosen_k}.npz")
    np.savez(
        clusters_npz,
        labels=cluster_labels,
        centroids=centroids,
        centroids_filtered_continuous=centroids_cont,
        cluster_order=order.astype(np.int32),
        proportions=pct,
        inertias=np.array(inertias, dtype=np.float32),
        k_values=np.array(ks, dtype=np.int32),
        group_names=np.array(group_names, dtype=object) if group_names is not None else np.array([], dtype=object),
        kept_group_indices=keep_idx.astype(np.int32),
        excluded_group_indices=excluded_idx.astype(np.int32),
        group_usage_rate=group_usage.astype(np.float32),
        highlighted_group_indices=np.asarray(highlighted_orig, dtype=np.int32),
        k_method=np.array([k_method], dtype=object),
    )
    print(f"Saved: {clusters_npz}")

    return {
        "cluster_k": chosen_k,
        "cluster_k_method": k_method,
        "elbow_plot": elbow_path,
        "templates_plot": templates_path,
        "cluster_artifacts": clusters_npz,
        "proportions": pct.tolist(),
        "kept_groups": keep_idx.tolist(),
        "excluded_groups": excluded_idx.tolist(),
        "highlighted_groups": highlighted_orig,
    }


def short_text(text, max_len=18):
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


def make_dark_group_colors(num_groups):
    """
    Build a dark, muted but distinct color per group.
    Uses low-value HSV palette to avoid bright/saturated tones.
    """
    n = int(max(1, num_groups))
    colors = []
    for i in range(n):
        h = float(i) / float(n)
        s = 0.55
        v = 0.55
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append((r, g, b, 1.0))
    return colors


def run_temporal_edges(sample_paths, num_time, num_groups, outdir,
                       edge_min_freq=0.01, edge_max_edges=0,
                       group_names=None, group_costs=None,
                       out_suffix=None, node_size_scale=95.0,
                       fixed_group_order=None, fixed_keep_idx=None,
                       avg_cost=None, avg_long_cost=None, avg_aux_cost=None,
                       transition_mode="strict_next"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.patches import FancyArrowPatch
    except ImportError as exc:
        print(f"Skipping temporal edge plot: missing dependency ({exc}).")
        return None

    if not sample_paths:
        print("Skipping temporal edge plot: no sample paths collected.")
        return None

    # Build binary acquisition tensor: (N, T, G)
    N = len(sample_paths)
    T = int(num_time)
    G = int(num_groups)
    acquisition_mask = np.zeros((N, T, G), dtype=np.float32)
    for i, sample in enumerate(sample_paths):
        for t in range(min(T, len(sample))):
            for g in sample[t]:
                gi = int(g)
                if 0 <= gi < G:
                    acquisition_mask[i, t, gi] = 1.0

    # Optionally force trajectory to use the same feature set as centroid plots.
    orig_group_ids = np.arange(G, dtype=np.int32)
    if fixed_keep_idx is not None:
        seen = set()
        keep = []
        for idx in fixed_keep_idx:
            ii = int(idx)
            if 0 <= ii < G and ii not in seen:
                seen.add(ii)
                keep.append(ii)
        if len(keep) == 0:
            keep = list(range(G))
        keep_idx = np.asarray(keep, dtype=np.int32)
        acquisition_mask = acquisition_mask[:, :, keep_idx]
        orig_group_ids = orig_group_ids[keep_idx]
        if group_names is not None and len(group_names) >= int(np.max(keep_idx)) + 1:
            group_names = [group_names[i] for i in keep_idx.tolist()]
        if group_costs is not None and len(group_costs) >= int(np.max(keep_idx)) + 1:
            group_costs = np.asarray(group_costs, dtype=np.float32)[keep_idx]
        G = acquisition_mask.shape[2]

    # Node frequency (size)
    node_freq = acquisition_mask.sum(axis=0) / max(float(N), 1.0)  # (T, G)
    group_freq = acquisition_mask.sum(axis=(0, 1)) / max(float(N), 1.0)  # (G,)

    # Average cost per timestep
    if group_costs is not None and len(group_costs) == G:
        feature_costs = np.asarray(group_costs, dtype=np.float32)
    else:
        feature_costs = np.ones(G, dtype=np.float32)
    step_costs_long = np.mean(
        acquisition_mask * feature_costs.reshape(1, 1, G),
        axis=0,
    ).sum(axis=1)  # (T,)
    step_costs = step_costs_long
    file_tag = "longitudinal_cost_control"

    # Cost summary rows for trajectory header.
    avg_long_est = float(step_costs_long.sum())
    avg_long_val = float(avg_long_cost) if avg_long_cost is not None else avg_long_est
    if not np.isfinite(avg_long_val):
        avg_long_val = avg_long_est
    avg_aux_val = float(avg_aux_cost) if avg_aux_cost is not None else 0.0
    if not np.isfinite(avg_aux_val):
        avg_aux_val = 0.0
    avg_total_val = float(avg_cost) if avg_cost is not None else (avg_long_val + avg_aux_val)
    if not np.isfinite(avg_total_val):
        avg_total_val = avg_long_val + avg_aux_val
    cost_rows = np.asarray([avg_total_val, avg_long_val, avg_aux_val], dtype=np.float32)

    # Stop probability: last timestep with any acquisition
    last_acq_time = np.zeros(N, dtype=np.int32)
    for i in range(N):
        active_t = np.where(acquisition_mask[i].sum(axis=1) > 0)[0]
        last_acq_time[i] = int(active_t[-1]) if len(active_t) > 0 else 0
    term_counts = np.bincount(last_acq_time, minlength=T).astype(np.float32)
    term_prob = term_counts / max(term_counts.sum(), 1.0)

    mode = str(transition_mode).strip().lower()
    if mode not in ("strict_next", "next_observed"):
        raise ValueError(
            f"Unknown transition_mode={transition_mode!r}. "
            "Expected one of: strict_next, next_observed."
        )

    # Transition modes:
    # - strict_next: only (t, g1) -> (t+1, g2)
    # - next_observed: (t, g1) -> (t', g2), where t' is the next timestep > t
    #   with at least one acquired group (allows skips like t->t+2, t->t+3).
    transitions = Counter()
    if mode == "next_observed":
        for i in range(N):
            active_steps = np.where(acquisition_mask[i].sum(axis=1) > 0.5)[0]
            if active_steps.size < 2:
                continue
            for j in range(active_steps.size - 1):
                t_src = int(active_steps[j])
                t_dst = int(active_steps[j + 1])
                groups_src = np.where(acquisition_mask[i, t_src] > 0.5)[0]
                groups_dst = np.where(acquisition_mask[i, t_dst] > 0.5)[0]
                if groups_src.size == 0 or groups_dst.size == 0:
                    continue
                for g1 in groups_src:
                    for g2 in groups_dst:
                        transitions[((t_src, int(g1)), (t_dst, int(g2)))] += 1
    else:
        for i in range(N):
            for t in range(T - 1):
                groups_src = np.where(acquisition_mask[i, t] > 0.5)[0]
                groups_dst = np.where(acquisition_mask[i, t + 1] > 0.5)[0]
                if groups_src.size == 0 or groups_dst.size == 0:
                    continue
                for g1 in groups_src:
                    for g2 in groups_dst:
                        transitions[((t, int(g1)), (t + 1, int(g2)))] += 1

    if len(transitions) == 0:
        print("Skipping temporal edge plot: no transitions found.")
        return None

    edge_threshold = max(0.0, float(edge_min_freq))
    max_edges = int(edge_max_edges)
    edge_prob_all = [(s, e, c / float(N)) for (s, e), c in transitions.items()]
    selected = [x for x in edge_prob_all if x[2] > edge_threshold]
    if not selected:
        selected = edge_prob_all
    selected.sort(key=lambda x: x[2], reverse=True)
    # Match notebook behavior by default: threshold-based filtering without forced top-K truncation.
    if max_edges > 0:
        selected = selected[:max_edges]

    if group_names is None or len(group_names) != G:
        group_names = [str(i) for i in range(G)]
    # Reorder groups by frequency so most selected groups appear at the bottom,
    # unless a shared order is explicitly provided (for cross-baseline consistency).
    if fixed_group_order is None:
        order = np.argsort(-group_freq)  # descending frequency
    else:
        orig_to_new = {int(orig_group_ids[new_i]): int(new_i) for new_i in range(G)}
        seen = set()
        ordered = []
        for idx in fixed_group_order:
            ii = int(idx)
            if ii in orig_to_new:
                jj = orig_to_new[ii]
            elif 0 <= ii < G:
                jj = ii
            else:
                continue
            if jj not in seen:
                seen.add(jj)
                ordered.append(jj)
        for jj in range(G):
            if jj not in seen:
                ordered.append(jj)
        order = np.asarray(ordered, dtype=np.int32)
    rank_pos = {int(orig): int(new_i) for new_i, orig in enumerate(order.tolist())}
    ordered_names = [group_names[i] for i in order.tolist()]
    node_freq_ord = node_freq[:, order]
    feature_costs_ord = feature_costs[order]
    time_labels = [f"t={i + 1}" for i in range(T)]

    # Distinct dark color per longitudinal group.
    group_colors = make_dark_group_colors(G)

    # Plot with notebook-like style (cost + stop + policy graph)
    fs_axis = 18
    fs_tick = 15
    fs_stop_pct = 12
    graph_height = max(3.0, 0.14 * G)
    fig = plt.figure(figsize=(16.0, 2.5 + graph_height))
    gs = gridspec.GridSpec(3, 1, height_ratios=[0.3, 0.3, graph_height], hspace=0.05)

    # Panel 1: per-timestep longitudinal cost
    ax_cost = plt.subplot(gs[0])
    ax_cost.bar(np.arange(T), step_costs, color="#d65f5f", alpha=0.8, width=0.6)
    ax_cost.set_ylabel("Avg\nLong\nCost", fontsize=fs_axis, fontweight="bold")
    ax_cost.set_xlim(-0.5, T - 0.5)
    ymax_tick = float(np.max(step_costs)) if np.max(step_costs) > 0 else 1.0
    ax_cost.set_yticks([0.0, ymax_tick])
    ax_cost.tick_params(axis="y", labelsize=fs_tick)
    for lbl in ax_cost.get_yticklabels():
        lbl.set_fontweight("bold")
    ax_cost.set_xticks([])
    ax_cost.grid(axis="y", linestyle="--", alpha=0.3)
    ax_cost.spines["top"].set_visible(False)
    ax_cost.spines["right"].set_visible(False)
    ax_cost.spines["bottom"].set_visible(False)

    # Panel 2: stop-probability heatmap
    ax_term = plt.subplot(gs[1])
    vmax_val = float(np.max(term_prob) * 1.5) if np.max(term_prob) > 0 else 1.0
    ax_term.imshow(term_prob.reshape(1, -1), cmap="Blues", aspect="auto", vmin=0.0, vmax=vmax_val)
    ax_term.set_yticks([])
    ax_term.set_ylabel("\n\nStop\nProb", fontsize=fs_axis, fontweight="bold")
    ax_term.set_xticks([])
    for t in range(T):
        if term_prob[t] > 0.001:
            val = term_prob[t] * 100.0
            lbl = f"{val:.1f}"
            if lbl.endswith(".0"):
                lbl = lbl[:-2]
            ax_term.text(
                t, 0, f"{lbl}%",
                ha="center", va="center",
                color="black", fontsize=fs_stop_pct, fontweight="bold"
            )

    # Panel 3: sequential policy graph
    ax_graph = plt.subplot(gs[2])
    for y_pos in range(G):
        orig_g = int(order[y_pos])
        for t in range(T):
            freq = float(node_freq_ord[t, y_pos])
            if freq > 0.001:
                ax_graph.scatter(
                    t,
                    y_pos,
                    s=max(1.8, freq * float(node_size_scale)),
                    color=group_colors[orig_g],
                    zorder=8,
                )

    edge_width_scale = 1.35
    edge_alpha_scale = 8.0
    arrow_curvature = 0.01
    for (start, end, prob) in selected:
        t1, g1 = start
        t2, g2 = end
        y1 = rank_pos[int(g1)]
        y2 = rank_pos[int(g2)]
        dist = max(1, t2 - t1)
        alpha = min(0.48, max(0.10, float(prob) * edge_alpha_scale))
        arrow = FancyArrowPatch(
            (t1, y1), (t2, y2),
            connectionstyle=f"arc3,rad=-{0.1 + arrow_curvature * dist}",
            color="#4a4a4a",
            alpha=alpha,
            linewidth=max(0.24, float(prob) * edge_width_scale),
            arrowstyle="->",
            mutation_scale=5.2,
            zorder=5,
            clip_on=False,
        )
        ax_graph.add_patch(arrow)

    if G <= 16:
        font_size = 26
    elif G <= 30:
        font_size = 22
    else:
        font_size = 18
    ax_graph.set_yticks(np.arange(G))
    ax_graph.set_yticklabels(ordered_names, fontweight="bold", fontsize=font_size)
    ax_graph.set_xticks(np.arange(T))
    ax_graph.set_xticklabels(time_labels, fontsize=font_size, fontweight="bold")
    ax_graph.set_ylim(-0.6, G - 0.4)
    ax_graph.set_xlim(-0.5, T - 0.5)
    ax_graph.grid(linestyle=":", alpha=0.3)
    ax_graph.spines["top"].set_visible(False)
    ax_graph.spines["right"].set_visible(False)

    plt.tight_layout()

    suffix = f"_{out_suffix}" if out_suffix else ""
    edge_png = os.path.join(outdir, f"temporal_transition_graph_{file_tag}{suffix}.png")
    edge_svg = os.path.join(outdir, f"temporal_transition_graph_{file_tag}{suffix}.svg")
    plt.savefig(edge_png, dpi=400, bbox_inches="tight")
    plt.savefig(edge_svg, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"Saved: {edge_png}")
    print(f"Saved: {edge_svg}")

    edge_npz = os.path.join(outdir, f"temporal_transition_edges_{file_tag}{suffix}.npz")
    src = np.array([[s[0], s[1]] for s, _, _ in selected], dtype=np.int32)
    dst = np.array([[e[0], e[1]] for _, e, _ in selected], dtype=np.int32)
    freq = np.array([p for _, _, p in selected], dtype=np.float32)
    np.savez(
        edge_npz,
        src=src,
        dst=dst,
        freq=freq,
        node_freq=node_freq.astype(np.float32),
        node_freq_ordered=node_freq_ord.astype(np.float32),
        group_frequency=group_freq.astype(np.float32),
        group_order=order.astype(np.int32),
        group_order_original=orig_group_ids[order].astype(np.int32),
        step_costs=step_costs.astype(np.float32),
        step_costs_long=step_costs_long.astype(np.float32),
        avg_cost=np.asarray([avg_total_val], dtype=np.float32),
        avg_long_cost=np.asarray([avg_long_val], dtype=np.float32),
        avg_aux_cost=np.asarray([avg_aux_val], dtype=np.float32),
        transition_mode=np.asarray([mode], dtype=object),
        stop_prob=term_prob.astype(np.float32),
        group_names=np.array(group_names, dtype=object),
        group_names_ordered=np.array(ordered_names, dtype=object),
        group_costs=feature_costs.astype(np.float32),
        group_costs_ordered=feature_costs_ord.astype(np.float32),
    )
    print(f"Saved: {edge_npz}")

    return {
        "edge_plot": edge_png,
        "edge_plot_svg": edge_svg,
        "edge_artifacts": edge_npz,
        "num_edges": len(selected),
        "total_transitions": int(sum(transitions.values())),
        "group_order": order.tolist(),
    }


def hp_tag(value):
    if value is None:
        return "none"
    return f"{float(value):.6g}"


def run_single_analysis(args, baseline_mode, outdir, cw, acw, actor_path_override=None):
    effective_joint = bool(args.joint)
    if actor_path_override is not None:
        actor_path = resolve_actor_path(
            actor_path=actor_path_override,
            dataset=DATASET,
            cw=cw,
            acw=acw,
            joint=effective_joint,
            baseline=baseline_mode,
        )
    else:
        # Auto-prefer joint checkpoint when available.
        if not effective_joint:
            joint_candidate = resolve_actor_path(
                actor_path=None,
                dataset=DATASET,
                cw=cw,
                acw=acw,
                joint=True,
                baseline=baseline_mode,
            )
            if os.path.exists(joint_candidate):
                effective_joint = True
                actor_path = joint_candidate
            else:
                actor_path = resolve_actor_path(
                    actor_path=None,
                    dataset=DATASET,
                    cw=cw,
                    acw=acw,
                    joint=False,
                    baseline=baseline_mode,
                )
        else:
            actor_path = resolve_actor_path(
                actor_path=None,
                dataset=DATASET,
                cw=cw,
                acw=acw,
                joint=True,
                baseline=baseline_mode,
            )
    os.makedirs(outdir, exist_ok=True)

    print("=" * 60)
    print("ACTOR-LAFA ANALYSIS PLOTS")
    print("=" * 60)
    print(f"Dataset: {DATASET}")
    print(f"Baseline mode: {baseline_mode}")
    print(f"Joint mode selected: {effective_joint}")
    print(f"Actor path: {actor_path}")
    print(f"*** actor: {actor_path}")
    print(f"Output dir: {outdir}")

    actor, actor_cfg, test_loader, num_time, num_feat, classifier_path, resolved_test_data_path = load_model_and_data(
        actor_path=actor_path,
        test_data_path=args.test_data_path,
        baseline=baseline_mode,
        batch_size=args.batch_size,
    )
    print(f"Classifier path: {classifier_path}")
    print(f"*** classifier: {classifier_path}")
    print(f"Test data path: {resolved_test_data_path}")
    print(f"*** test_data: {resolved_test_data_path}")

    collector = AnalysisCollector(
        num_time=num_time,
        num_groups=actor.num_groups,
        max_planner_states=0,
        seed=args.seed,
    )

    device = get_device()
    results, masks, preds, labels = evaluate_with_collection(
        actor=actor,
        dataloader=test_loader,
        device=device,
        num_time=num_time,
        num_feat=num_feat,
        feature_costs=actor_cfg.get("feature_costs"),
        aux_feature_costs=actor_cfg.get("aux_feature_costs"),
        collector=collector,
    )
    masks = (masks > 0.5).astype(np.float32)
    n_test = int(len(test_loader.dataset))
    n_masks = int(masks.shape[0])
    n_labels = int(labels.shape[0])
    if n_masks != n_test or n_labels != n_test:
        raise RuntimeError(
            f"Expected all test samples to be analyzed, but got "
            f"dataset={n_test}, masks={n_masks}, labels={n_labels}."
        )

    print("\nEvaluation summary")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  AUROC:    {results['auroc']:.4f}")
    print(f"  AUPRC:    {results['auprc']:.4f}")
    print(f"  Avg Cost: {results['avg_cost']:.2f} (long={results['avg_long_cost']:.2f}, aux={results['avg_aux_cost']:.2f})")
    print(f"  Total samples: {results['total_samples']} / test set: {n_test}")
    group_names, group_name_source = resolve_longitudinal_group_names(actor.num_groups)
    group_costs, group_cost_source = resolve_longitudinal_costs(actor.num_groups, actor=actor, actor_cfg=actor_cfg)
    print(f"  Group label source: {group_name_source}")
    print(f"  Group cost source:  {group_cost_source}")

    group_map_path = os.path.join(outdir, "longitudinal_group_index_map.txt")
    with open(group_map_path, "w") as f:
        f.write("idx\tcost\tname\n")
        for i, cost in enumerate(group_costs):
            nm = group_names[i] if i < len(group_names) else str(i)
            f.write(f"{i}\t{float(cost):.6g}\t{nm}\n")
    print(f"Saved: {group_map_path}")

    rollout_npz = os.path.join(outdir, "analysis_rollout.npz")
    np.savez(
        rollout_npz,
        masks=masks,
        predictions=preds,
        labels=labels,
        accuracy=results["accuracy"],
        auroc=results["auroc"],
        auprc=results["auprc"],
        avg_cost=results["avg_cost"],
        avg_long_cost=results["avg_long_cost"],
        avg_aux_cost=results["avg_aux_cost"],
        aux_gate_binary=np.asarray(results.get("aux_gate_binary", []), dtype=np.float32),
        aux_gate_probs=np.asarray(results.get("aux_gate_probs", []), dtype=np.float32),
        aux_gate_rates=np.asarray(results.get("aux_gate_rates", []), dtype=np.float32),
        aux_available_rates=np.asarray(results.get("aux_available_rates", []), dtype=np.float32),
    )
    print(f"Saved: {rollout_npz}")
    print(f"*** actor_rollout: {rollout_npz}")

    summary = {}
    cluster_info = None

    try:
        cluster_info = run_mask_clustering(
            all_masks=masks,
            num_time=num_time,
            num_groups=actor.num_groups,
            outdir=outdir,
            cluster_k=args.cluster_k,
            cluster_kmax=args.cluster_kmax,
            seed=args.seed,
            group_names=group_names,
            run_label=baseline_mode,
            min_feature_rate=args.cluster_min_feature_rate,
            highlight_top_n=args.cluster_highlight_top_n,
        )
        if cluster_info is not None:
            summary["clusters"] = cluster_info
    except Exception as exc:
        print(f"Mask clustering failed: {exc}")

    try:
        mask_3d = masks.reshape(masks.shape[0], num_time, actor.num_groups)
        final_sample_paths = []
        for i in range(mask_3d.shape[0]):
            steps = []
            for t in range(num_time):
                steps.append(np.where(mask_3d[i, t] > 0.5)[0].astype(int).tolist())
            final_sample_paths.append(steps)
        if len(final_sample_paths) != n_test:
            raise RuntimeError(
                f"Trajectory edge input does not cover all test samples: "
                f"paths={len(final_sample_paths)}, test={n_test}"
            )
        print(f"  Trajectory edge source: final masks (samples={len(final_sample_paths)})")
        edge_info = run_temporal_edges(
            sample_paths=final_sample_paths,
            num_time=num_time,
            num_groups=actor.num_groups,
            outdir=outdir,
            edge_min_freq=args.edge_min_freq,
            edge_max_edges=args.edge_max_edges,
            group_names=group_names,
            group_costs=group_costs,
            out_suffix=baseline_mode,
            node_size_scale=args.edge_node_size_scale,
            fixed_keep_idx=(cluster_info.get("kept_groups") if cluster_info is not None else None),
            transition_mode=args.edge_transition_mode,
        )
        if edge_info is not None:
            summary["edges"] = edge_info
    except Exception as exc:
        print(f"Temporal edge plotting failed: {exc}")

    summary_txt = os.path.join(outdir, "analysis_summary.txt")
    with open(summary_txt, "w") as f:
        f.write("ACTOR-LAFA analysis summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"dataset: {DATASET}\n")
        f.write(f"baseline: {baseline_mode}\n")
        f.write(f"joint: {effective_joint}\n")
        f.write(f"actor_path: {actor_path}\n")
        f.write(f"classifier_path: {classifier_path}\n")
        f.write(f"test_data_path: {resolved_test_data_path}\n")
        f.write(f"num_time: {num_time}\n")
        f.write(f"num_feat: {num_feat}\n")
        f.write(f"num_groups: {actor.num_groups}\n")
        f.write(f"num_aux: {actor.num_aux}\n")
        f.write(f"group_name_source: {group_name_source}\n")
        f.write(f"group_cost_source: {group_cost_source}\n")
        f.write(f"group_index_map: {group_map_path}\n")
        f.write(f"samples: {results['total_samples']}\n")
        f.write(f"results: {results}\n")
        f.write(f"artifacts: {summary}\n")
    print(f"Saved: {summary_txt}")

    print("=" * 60)
    print(f"Analysis complete for baseline={baseline_mode}")
    print("=" * 60)

    return {
        "baseline": baseline_mode,
        "joint": effective_joint,
        "actor_path": actor_path,
        "outdir": outdir,
        "summary_path": summary_txt,
        "results": results,
        "artifacts": summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Standalone ACTOR-LAFA analysis plots")
    parser.add_argument("--actor_path", type=str, default=None,
                        help="Path to actor checkpoint (default: derived from config HPs)")
    parser.add_argument("--cost_weight", type=float, default=None,
                        help="cost_weight used to locate checkpoint (overrides config)")
    parser.add_argument("--aux_cost_weight", type=float, default=None,
                        help="aux_cost_weight used to locate checkpoint (overrides config)")
    parser.add_argument("--joint", action="store_true", default=False,
                        help="Force joint actor checkpoint path (otherwise joint is auto-preferred if found)")
    parser.add_argument("--baseline", type=str, default="learned", choices=["learned", "all", "none"],
                        help="Baseline mode")
    parser.add_argument("--test_data_path", type=str, default=None,
                        help="Optional explicit test_data.npz path")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Evaluation batch size")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Directory to save all analysis artifacts")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    parser.add_argument("--cluster_k", type=int, default=0,
                        help="Chosen k for template clustering (<=0 selects k automatically from elbow)")
    parser.add_argument("--cluster_kmax", type=int, default=10,
                        help="Max k for elbow sweep")
    parser.add_argument("--cluster_min_feature_rate", type=float, default=0.01,
                        help="Drop groups with mean acquisition rate below this threshold in centroid plots")
    parser.add_argument("--cluster_highlight_top_n", type=int, default=6,
                        help="For learned baseline, number of top-used groups to highlight")

    parser.add_argument("--edge_min_freq", type=float, default=0.01,
                        help="Minimum normalized edge frequency to draw")
    parser.add_argument("--edge_max_edges", type=int, default=0,
                        help="Max number of edges to draw (<=0 means no cap)")
    parser.add_argument("--edge_node_size_scale", type=float, default=95.0,
                        help="Node marker size scale for trajectory plots")
    parser.add_argument(
        "--edge_transition_mode",
        type=str,
        default="next_observed",
        choices=["strict_next", "next_observed"],
        help=(
            "Transition construction mode: strict_next uses only t->t+1; "
            "next_observed links each acquisition step to the next timestep "
            "with any new acquisition (allows skips such as t->t+2)."
        ),
    )
    parser.add_argument("--all_baselines", action="store_true", default=False,
                        help="For cheears datasets, run learned/all/none baselines into separate folders")

    args = parser.parse_args()

    set_seed(args.seed)

    cw = args.cost_weight if args.cost_weight is not None else ACTOR_CONFIG["cost_weight"]
    acw = args.aux_cost_weight if args.aux_cost_weight is not None else ACTOR_CONFIG.get("aux_cost_weight")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_root = os.path.join(script_dir, "plots")
    os.makedirs(plots_root, exist_ok=True)

    if args.all_baselines and DATASET not in ("cheears", "cheears_demog", "cheears_day_context"):
        print(f"WARNING: --all_baselines is intended for cheears datasets; got DATASET={DATASET}.")
        print("Proceeding with single baseline run.")
        args.all_baselines = False

    if args.all_baselines:
        if args.outdir is None:
            sweep_outdir = os.path.join(
                plots_root,
                DATASET,
                f"baseline_sweep_{'joint_' if args.joint else ''}cw{hp_tag(cw)}_acw{hp_tag(acw)}",
            )
        else:
            sweep_outdir = args.outdir
        os.makedirs(sweep_outdir, exist_ok=True)

        requested_cluster_k = int(args.cluster_k)
        shared_cluster_k = requested_cluster_k if requested_cluster_k > 0 else None
        if shared_cluster_k is not None:
            print(f"Using fixed centroid count across baselines: k={shared_cluster_k}")
        else:
            print(
                "Auto-selecting centroid count from first successful baseline run; "
                "the same k will be reused for remaining baselines."
            )

        run_results = {}
        original_cluster_k = args.cluster_k
        for mode in ("learned", "all", "none"):
            mode_outdir = os.path.join(sweep_outdir, f"baseline_{mode}")
            mode_actor_override = args.actor_path if (args.actor_path is not None and mode == args.baseline) else None
            args.cluster_k = shared_cluster_k if shared_cluster_k is not None else requested_cluster_k
            run_results[mode] = run_single_analysis(
                args=args,
                baseline_mode=mode,
                outdir=mode_outdir,
                cw=cw,
                acw=acw,
                actor_path_override=mode_actor_override,
            )

            if shared_cluster_k is None:
                mode_k = (
                    run_results[mode]
                    .get("artifacts", {})
                    .get("clusters", {})
                    .get("cluster_k")
                )
                if mode_k is not None and int(mode_k) > 0:
                    shared_cluster_k = int(mode_k)
                    print(f"Shared centroid count locked at k={shared_cluster_k} (from baseline={mode})")

        args.cluster_k = original_cluster_k

        # Enforce cross-baseline consistency in plotted feature set and ordering.
        def _load_group_map(map_path):
            names = []
            costs = []
            if not os.path.exists(map_path):
                return names, costs
            with open(map_path, "r") as f:
                _ = f.readline()
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        try:
                            costs.append(float(parts[1]))
                        except Exception:
                            costs.append(1.0)
                    if len(parts) >= 3:
                        names.append(parts[2])
                    elif len(parts) >= 1:
                        names.append(parts[0])
            return names, costs

        per_mode = {}
        for mode in ("learned", "all", "none"):
            mode_outdir = run_results[mode]["outdir"]
            rollout_path = os.path.join(mode_outdir, "analysis_rollout.npz")
            group_map_path = os.path.join(mode_outdir, "longitudinal_group_index_map.txt")
            if not os.path.exists(rollout_path):
                continue
            z = np.load(rollout_path)
            masks = z["masks"].astype(np.float32)
            names, costs = _load_group_map(group_map_path)
            g = len(names) if len(names) > 0 else 0
            if g <= 0:
                continue
            if masks.shape[1] % g != 0:
                continue
            t = int(masks.shape[1] // g)
            usage = masks.reshape(masks.shape[0], t, g).mean(axis=(0, 1))
            if len(costs) != g:
                costs = [1.0] * g
            per_mode[mode] = {
                "outdir": mode_outdir,
                "masks": masks,
                "num_time": t,
                "num_groups": g,
                "group_names": names,
                "group_costs": np.asarray(costs, dtype=np.float32),
                "usage": usage,
            }

        if len(per_mode) >= 1:
            modes_avail = [m for m in ("learned", "all", "none") if m in per_mode]
            usage_stack = np.stack([per_mode[m]["usage"] for m in modes_avail], axis=0)
            shared_usage_mean = usage_stack.mean(axis=0)
            shared_usage_max = usage_stack.max(axis=0)
            min_rate = float(args.cluster_min_feature_rate)
            shared_keep = np.where(shared_usage_max >= min_rate)[0]
            if shared_keep.size == 0:
                shared_keep = np.where(shared_usage_max > 0.0)[0]
            if shared_keep.size == 0:
                shared_keep = np.arange(per_mode[modes_avail[0]]["num_groups"], dtype=np.int32)
            shared_keep = shared_keep[np.argsort(-shared_usage_mean[shared_keep])]
            shared_order_all = np.argsort(-shared_usage_mean)

            print(
                "Applying shared feature set/order across baselines: "
                f"kept={len(shared_keep)} / {len(shared_order_all)} groups."
            )

            for mode in modes_avail:
                meta = per_mode[mode]
                mode_outdir = meta["outdir"]
                masks = meta["masks"]
                t = meta["num_time"]
                g = meta["num_groups"]
                names = meta["group_names"]
                costs = meta["group_costs"]

                cluster_info = run_mask_clustering(
                    all_masks=masks,
                    num_time=t,
                    num_groups=g,
                    outdir=mode_outdir,
                    cluster_k=shared_cluster_k if shared_cluster_k is not None else requested_cluster_k,
                    cluster_kmax=args.cluster_kmax,
                    seed=args.seed,
                    group_names=names,
                    run_label=mode,
                    min_feature_rate=args.cluster_min_feature_rate,
                    highlight_top_n=args.cluster_highlight_top_n,
                    fixed_keep_idx=shared_keep.tolist(),
                )
                if cluster_info is not None:
                    run_results[mode].setdefault("artifacts", {})["clusters"] = cluster_info

                mask_3d = masks.reshape(masks.shape[0], t, g)
                sample_paths = []
                for i in range(mask_3d.shape[0]):
                    steps = []
                    for tt in range(t):
                        steps.append(np.where(mask_3d[i, tt] > 0.5)[0].astype(int).tolist())
                    sample_paths.append(steps)

                edge_info = run_temporal_edges(
                    sample_paths=sample_paths,
                    num_time=t,
                    num_groups=g,
                    outdir=mode_outdir,
                    edge_min_freq=args.edge_min_freq,
                    edge_max_edges=args.edge_max_edges,
                    group_names=names,
                    group_costs=costs,
                    out_suffix=mode,
                    node_size_scale=args.edge_node_size_scale,
                    fixed_group_order=shared_order_all.tolist(),
                    fixed_keep_idx=shared_keep.tolist(),
                    transition_mode=args.edge_transition_mode,
                )
                if edge_info is not None:
                    run_results[mode].setdefault("artifacts", {})["edges"] = edge_info

        sweep_summary = os.path.join(sweep_outdir, "baseline_sweep_summary.txt")
        with open(sweep_summary, "w") as f:
            f.write("ACTOR-LAFA cheears baseline sweep summary\n")
            f.write("=" * 48 + "\n")
            f.write(f"dataset: {DATASET}\n")
            f.write(f"joint: {args.joint}\n")
            f.write(f"cost_weight: {cw}\n")
            f.write(f"aux_cost_weight: {acw}\n")
            f.write(f"requested_cluster_k: {requested_cluster_k}\n")
            f.write(f"shared_cluster_k: {shared_cluster_k}\n")
            f.write(f"root_outdir: {sweep_outdir}\n")
            for mode in ("learned", "all", "none"):
                info = run_results[mode]
                f.write("\n")
                f.write(f"[{mode}]\n")
                f.write(f"actor_path: {info['actor_path']}\n")
                f.write(f"outdir: {info['outdir']}\n")
                f.write(f"summary_path: {info['summary_path']}\n")
                f.write(f"results: {info['results']}\n")
                f.write(
                    f"cluster_k: "
                    f"{info.get('artifacts', {}).get('clusters', {}).get('cluster_k')}\n"
                )
        print(f"Saved: {sweep_summary}")

        manifest_tsv = os.path.join(sweep_outdir, "policy_cost_manifest.tsv")
        with open(manifest_tsv, "w") as f:
            f.write("baseline\tjoint\tactor_path\tavg_long_cost\tavg_aux_cost\toutdir\n")
            for mode in ("learned", "all", "none"):
                info = run_results.get(mode, {})
                res = info.get("results", {})
                f.write(
                    f"{mode}\t{int(bool(info.get('joint', args.joint)))}\t"
                    f"{info.get('actor_path', '')}\t"
                    f"{float(res.get('avg_long_cost', np.nan)):.8g}\t"
                    f"{float(res.get('avg_aux_cost', np.nan)):.8g}\t"
                    f"{info.get('outdir', '')}\n"
                )
        print(f"Saved: {manifest_tsv}")

        def _write_cost_control_selection(metric_key, out_name):
            target = float(run_results.get("learned", {}).get("results", {}).get(metric_key, np.nan))
            rows = []
            for mode in ("learned", "all", "none"):
                info = run_results.get(mode, {})
                val = float(info.get("results", {}).get(metric_key, np.nan))
                delta = abs(val - target) if np.isfinite(val) and np.isfinite(target) else np.nan
                rows.append((mode, val, delta, info.get("actor_path", ""), info.get("outdir", "")))
            path = os.path.join(sweep_outdir, out_name)
            with open(path, "w") as f:
                f.write("baseline\tmetric\tvalue\tdelta_to_learned\tactor_path\toutdir\n")
                for mode, val, delta, actor_p, out_p in rows:
                    f.write(
                        f"{mode}\t{metric_key}\t{val:.8g}\t{delta:.8g}\t{actor_p}\t{out_p}\n"
                    )
            print(f"Saved: {path}")

        _write_cost_control_selection("avg_long_cost", "cost_control_longitudinal.tsv")
        print("=" * 60)
        print("All baseline runs complete")
        print("=" * 60)
        return

    if args.outdir is None:
        outdir = os.path.join(
            plots_root,
            DATASET,
            f"baseline_{args.baseline}_{'joint_' if args.joint else ''}cw{hp_tag(cw)}_acw{hp_tag(acw)}",
        )
    else:
        outdir = args.outdir

    run_single_analysis(
        args=args,
        baseline_mode=args.baseline,
        outdir=outdir,
        cw=cw,
        acw=acw,
        actor_path_override=args.actor_path,
    )


if __name__ == "__main__":
    main()
