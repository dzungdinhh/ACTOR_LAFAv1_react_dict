#!/usr/bin/env python3
"""
Export ADNI RAS rollouts into the ACTOR rollout format used by merge plots.

This script is intended to be executed with the RAS environment:
  PYTHONPATH=/playpen-nvme/scribble/ddinh/ras/src \
    /playpen-nvme/scribble/ddinh/miniconda3/envs/ras/bin/python \
    export_ras_adni_rollout.py --lambda 50.0 --out /tmp/ras_rollout.npz
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch


ACTOR_ROOT = Path("/playpen-nvme/scribble/ddinh/ACTOR_LAFAv1")
RAS_ROOT = Path("/playpen-nvme/scribble/ddinh/ras")
RAS_EXPERIMENT_ROOT = RAS_ROOT / "experiments" / "exp-adni"
RAS_SRC_ROOT = RAS_ROOT / "src"

if str(RAS_EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RAS_EXPERIMENT_ROOT))
if str(RAS_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(RAS_SRC_ROOT))

from data_util import load_adni_splits  # noqa: E402
from model_config import net_config, sim_config  # noqa: E402
from cvar_sensing.sensing import Sensing  # noqa: E402
from cvar_sensing.train import batch_call, load_model  # noqa: E402


ADNI_LONGITUDINAL_NAMES = ["FDG", "AV45", "Hippocampus", "Entorhinal"]
ADNI_FULL_COSTS = np.asarray([0.3] * 7 + [1.0, 1.0, 0.5, 0.5], dtype=np.float32)
ADNI_LONG_COSTS = ADNI_FULL_COSTS[7:].copy()


def parse_args():
    parser = argparse.ArgumentParser(description="Export ADNI RAS rollout cache.")
    parser.add_argument("--lambda", dest="lambda_value", type=float, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0, help="Torch simulation seed.")
    parser.add_argument("--drop_rate", type=float, default=0.7)
    parser.add_argument("--baseline-at-t0-only", action="store_true")
    return parser.parse_args()


def _lambda_csv_path(lambda_value: float) -> Path:
    return RAS_EXPERIMENT_ROOT / "ras" / f"lambda={float(lambda_value):.1f}.csv"


def _model_paths_for_lambda(lambda_value: float) -> List[Path]:
    csv_path = _lambda_csv_path(lambda_value)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing RAS lambda CSV: {csv_path}")

    model_paths: List[Path] = []
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            uuid = str(row.get("uuid", "")).strip()
            if not uuid:
                continue
            model_path = RAS_EXPERIMENT_ROOT / "ras" / f"ActiveSensing_{uuid}.pt"
            if not model_path.is_file():
                raise FileNotFoundError(f"Missing RAS model checkpoint: {model_path}")
            model_paths.append(model_path)

    if not model_paths:
        raise RuntimeError(f"No RAS model paths discovered for lambda={lambda_value}")
    return model_paths


def _bin_actions(
    obs_t: np.ndarray,
    obs_m: np.ndarray,
    obs_mask: np.ndarray,
    *,
    num_visits: int,
    interval: float,
):
    num_samples = int(obs_t.shape[0])
    feat_dim = int(obs_m.shape[2])
    full_actions = np.zeros((num_samples, num_visits, feat_dim), dtype=np.float32)

    for sample_idx in range(num_samples):
        valid_steps = np.where(obs_mask[sample_idx] > 0.5)[0]
        for step_idx in valid_steps.tolist():
            t_value = float(obs_t[sample_idx, step_idx])
            if t_value < 0.0:
                continue
            visit_idx = int(round(t_value / interval))
            visit_idx = max(0, min(num_visits - 1, visit_idx))
            full_actions[sample_idx, visit_idx] = np.maximum(
                full_actions[sample_idx, visit_idx],
                obs_m[sample_idx, step_idx],
            )

    return full_actions, full_actions[:, :, 7:]


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_set = load_adni_splits(
        drop_rate=float(args.drop_rate),
        baseline_at_t0_only=bool(args.baseline_at_t0_only),
    )
    net_config["x_dim"] = test_set.x_dim
    net_config["y_dim"] = test_set.y_dim

    base_times = test_set[:]["t"]
    if torch.is_tensor(base_times):
        base_times = np.asarray(base_times.detach().cpu().tolist(), dtype=np.float32)
    else:
        base_times = np.asarray(base_times, dtype=np.float32)
    num_visits = int(base_times.shape[1])
    interval = float(np.median(np.diff(base_times[0])))

    long_batches = []
    full_cost_per_sample = []
    long_cost_per_sample = []
    model_paths = _model_paths_for_lambda(float(args.lambda_value))

    for model_path in model_paths:
        torch.random.manual_seed(int(args.seed))
        model = Sensing(sim_config=sim_config, **net_config)
        model = load_model(model, model_path)
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            ret = batch_call(model.simulate, test_set, device=device)

        obs_t = np.asarray(ret["obs_t"].detach().cpu().tolist(), dtype=np.float32)
        obs_m = np.asarray(ret["obs_m"].detach().cpu().tolist(), dtype=np.float32)
        obs_mask = np.asarray(ret["mask"].detach().cpu().tolist(), dtype=np.float32)

        full_actions, long_actions = _bin_actions(
            obs_t,
            obs_m,
            obs_mask,
            num_visits=num_visits,
            interval=interval,
        )
        long_batches.append(long_actions.astype(np.float32))
        full_cost_per_sample.append(np.sum(full_actions * ADNI_FULL_COSTS.reshape(1, 1, -1), axis=(1, 2)))
        long_cost_per_sample.append(np.sum(long_actions * ADNI_LONG_COSTS.reshape(1, 1, -1), axis=(1, 2)))

    long_actions_all = np.concatenate(long_batches, axis=0).astype(np.float32)
    full_cost_all = np.concatenate(full_cost_per_sample, axis=0).astype(np.float32)
    long_cost_all = np.concatenate(long_cost_per_sample, axis=0).astype(np.float32)
    avg_long_cost = float(long_cost_all.mean())
    avg_cost = float(full_cost_all.mean())
    avg_aux_cost = float(avg_cost - avg_long_cost)

    np.savez(
        out_path,
        masks=long_actions_all.reshape(long_actions_all.shape[0], -1),
        num_time=np.int32(num_visits),
        num_groups=np.int32(long_actions_all.shape[2]),
        group_names=np.asarray(ADNI_LONGITUDINAL_NAMES, dtype=object),
        group_costs=ADNI_LONG_COSTS.astype(np.float32),
        avg_long_cost=np.float32(avg_long_cost),
        avg_aux_cost=np.float32(avg_aux_cost),
        avg_cost=np.float32(avg_cost),
        lambda_value=np.float32(args.lambda_value),
        eval_seed=np.int32(args.seed),
        model_paths=np.asarray([str(path) for path in model_paths], dtype=object),
    )
    print(f"[OK] saved {out_path}")
    print(
        f"lambda={float(args.lambda_value):.1f} "
        f"avg_long_cost={avg_long_cost:.6f} avg_aux_cost={avg_aux_cost:.6f} avg_cost={avg_cost:.6f}"
    )


if __name__ == "__main__":
    main()
