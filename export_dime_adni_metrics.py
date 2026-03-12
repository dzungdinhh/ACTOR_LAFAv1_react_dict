#!/usr/bin/env python3
"""
Export ADNI DIME metrics using the same metric logic as baseline_DIME/evaluate_DIME.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from torchmetrics import AUROC
from torchmetrics.classification import AveragePrecision


BASELINE_DIME_ROOT = Path("/playpen-nvme/scribble/ddinh/baseline_DIME")
if str(BASELINE_DIME_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIME_ROOT))
if str(BASELINE_DIME_ROOT / "dataset") not in sys.path:
    sys.path.insert(0, str(BASELINE_DIME_ROOT / "dataset"))
if str(BASELINE_DIME_ROOT / "config_DIME") not in sys.path:
    sys.path.insert(0, str(BASELINE_DIME_ROOT / "config_DIME"))

from model import predictor
from utils import MaskLayer
from dime_estimator import DIMEEstimator
from dataset_adni import load_adni_data
from config_adni import get_default_config as get_adni_config


def parse_args():
    parser = argparse.ArgumentParser(description="Export ADNI DIME metrics to JSON.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--budgets", type=int, nargs="+", required=True)
    parser.add_argument("--baseline-at-t0-only", action="store_true")
    parser.add_argument("--metric_style", choices=["ras", "actor"], default="ras")
    return parser.parse_args()


def _resolve_checkpoint_path(config):
    ckpt_root = str(config["cmi"]["save_folder"])
    if not os.path.isabs(ckpt_root):
        ckpt_root = os.path.join(str(BASELINE_DIME_ROOT), ckpt_root)
    found_ckpt = ""
    if os.path.exists(ckpt_root):
        for seed_folder in os.listdir(ckpt_root):
            if not seed_folder.startswith("seed_"):
                continue
            full_seed_path = os.path.join(ckpt_root, seed_folder)
            if not os.path.isdir(full_seed_path):
                continue
            versions = [f for f in os.listdir(full_seed_path) if f.startswith("version_")]
            if not versions:
                continue
            latest_version = max(versions, key=lambda x: int(x.split("_")[1]))
            ckpt_path = os.path.join(
                full_seed_path,
                latest_version,
                "checkpoints",
                config["cmi"]["ckpt_name"] + ".ckpt",
            )
            if os.path.exists(ckpt_path):
                found_ckpt = ckpt_path
                break
    if not found_ckpt:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_root}")
    return found_ckpt


def _build_metrics(y_dim: int):
    if int(y_dim) == 2:
        return (
            AveragePrecision(task="binary"),
            AUROC(task="binary"),
        )
    return (
        AveragePrecision(task="multiclass", num_classes=int(y_dim)),
        AUROC(task="multiclass", num_classes=int(y_dim)),
    )


def _compute_budget_metrics(model, trainer, dataloader, budget: int, y_dim: int, metric_style: str):
    metric_dict = model.inference(trainer, dataloader, budget=int(budget))
    y = metric_dict["y"].view(-1)
    pred = metric_dict["pred"].view(-1, metric_dict["pred"].shape[-1])

    valid = y != -1
    y = y[valid]
    pred = pred[valid]
    y_dim = int(y_dim)

    if str(metric_style) == "ras":
        y_np = y.cpu().numpy().astype(int)
        pred_np = pred.cpu().numpy().astype(float)
        y_one_hot = np.eye(y_dim, dtype=float)[y_np]
        if y_dim == 2:
            auroc = float(roc_auc_score(y_one_hot[:, 1], pred_np[:, 1]))
            auprc = float(average_precision_score(y_one_hot[:, 1], pred_np[:, 1]))
        else:
            auroc = float(roc_auc_score(y_one_hot, pred_np))
            auprc = float(average_precision_score(y_one_hot, pred_np))
    else:
        pred_metric = pred[:, 1] if int(pred.shape[1]) == 2 else pred
        auprc_metric, auroc_metric = _build_metrics(y_dim=y_dim)
        auprc = float(auprc_metric(pred_metric, y).item())
        auroc = float(auroc_metric(pred_metric, y).item())
    return {
        "auprc": auprc,
        "auroc": auroc,
    }


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    config = get_adni_config(baseline_at_t0_only=bool(args.baseline_at_t0_only))
    dataset = load_adni_data(
        os.path.join(config["data_folder"], "test_data.npz"),
        baseline_at_t0_only=bool(config.get("baseline_at_t0_only", False)),
    )
    test_dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)

    d_in = dataset.x_dim * dataset.t
    prediction_network = predictor(d_in, dataset.y_dim, config["num_hidden"], config["dropout"])
    mask_layer = MaskLayer(mask_size=d_in, append=False)
    value_network = nn.Sequential(
        nn.Linear(d_in + 1, config["num_hidden"]),
        nn.ReLU(),
        nn.Dropout(config["dropout"]),
        nn.Linear(config["num_hidden"], config["num_hidden"]),
        nn.ReLU(),
        nn.Dropout(config["dropout"]),
        nn.Linear(config["num_hidden"], d_in),
    )

    ckpt_path = _resolve_checkpoint_path(config)
    model = DIMEEstimator.load_from_checkpoint(
        ckpt_path,
        map_location=torch.device("cpu"),
        config=config["cmi"],
        value_network=value_network,
        prediction_network=prediction_network,
        mask_layer=mask_layer,
        num_time=dataset.t,
        num_feat=dataset.x_dim,
        pred_loss_fn=nn.CrossEntropyLoss(reduction="none", ignore_index=-1),
        value_loss_fn=nn.MSELoss(),
        val_metric=None,
        scaling="none",
    )
    trainer = Trainer(accelerator="cpu", devices=1, logger=False)

    metrics_by_budget = {}
    for budget in args.budgets:
        metrics_by_budget[str(int(budget))] = _compute_budget_metrics(
            model=model,
            trainer=trainer,
            dataloader=test_dataloader,
            budget=int(budget),
            y_dim=int(dataset.y_dim),
            metric_style=str(args.metric_style),
        )

    payload = {
        "baseline_at_t0_only": bool(args.baseline_at_t0_only),
        "checkpoint_path": str(ckpt_path),
        "metric_style": str(args.metric_style),
        "budgets": metrics_by_budget,
    }
    with out_path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
