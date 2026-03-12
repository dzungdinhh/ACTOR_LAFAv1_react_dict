#!/usr/bin/env python3
"""
Export ADNI ACTOR checkpoint metrics using the same evaluation path as evaluate.py,
but with explicit on-disk checkpoint locations instead of config-derived paths.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Export ADNI ACTOR metrics to JSON.")
    parser.add_argument("--actor_path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--baseline", choices=["learned", "all", "none"], default="learned")
    return parser.parse_args()


def _resolve_classifier_path(actor_path: Path) -> Path:
    local = actor_path.parent / "classifier.ckpt"
    if local.is_file():
        return local
    fallback = Path("/playpen-nvme/scribble/ddinh/ACTOR_LAFAv1/adni/classifier.ckpt")
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"Could not resolve ADNI classifier for actor: {actor_path}")


def _resolve_test_data_path(actor_path: Path) -> Path:
    local = actor_path.parent / "test_data.npz"
    if local.is_file():
        return local
    fallback = Path("/playpen-nvme/scribble/ddinh/aaco/input_data/test_data.npz")
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"Could not resolve ADNI test data for actor: {actor_path}")


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    os.environ["ACTOR_DATASET"] = "adni"

    import evaluate as ev

    actor_path = Path(args.actor_path)
    classifier_path = _resolve_classifier_path(actor_path)
    test_data_path = _resolve_test_data_path(actor_path)

    classifier_ckpt = torch.load(classifier_path, map_location="cpu")
    num_time = classifier_ckpt["num_time"]
    num_feat = classifier_ckpt["num_feat"]
    num_aux = classifier_ckpt.get("num_aux", 0)

    predictor = ev.Predictor(
        d_in=num_time * num_feat + num_aux,
        d_out=classifier_ckpt["y_dim"],
        hidden=classifier_ckpt["config"]["hidden_dim"],
        dropout=classifier_ckpt["config"]["dropout"],
    )
    predictor.load_state_dict(classifier_ckpt["predictor"])

    actor_ckpt = torch.load(actor_path, map_location="cpu")
    num_groups = actor_ckpt.get("num_groups", num_feat)
    group_to_feat_matrix = None
    if num_groups != num_feat:
        group_to_feat_matrix = ev.build_group_to_feat_matrix(num_feat)

    actor = ev.GumbelActor(
        predictor=predictor,
        num_time=num_time,
        num_feat=num_feat,
        config=actor_ckpt["config"],
        num_aux=num_aux,
        num_groups=num_groups,
        group_to_feat_matrix=group_to_feat_matrix,
    )
    ev._load_actor_state_dict(actor, actor_ckpt)

    if args.baseline == "all" and actor.num_aux > 0:
        actor.aux_logits.data.fill_(100.0)
    elif args.baseline == "none" and actor.num_aux > 0:
        actor.aux_logits.data.fill_(-100.0)

    test_dataset = ev.load_adni_data(str(test_data_path))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    results, _, _, _ = ev.evaluate_actor(
        actor,
        test_loader,
        ev._get_device(),
        num_time,
        num_feat,
        feature_costs=actor_ckpt.get("config", {}).get("feature_costs"),
        aux_feature_costs=actor_ckpt.get("config", {}).get("aux_feature_costs"),
    )

    payload = {
        "actor_path": str(actor_path),
        "classifier_path": str(classifier_path),
        "test_data_path": str(test_data_path),
        "baseline": str(args.baseline),
        "accuracy": float(results["accuracy"]),
        "auroc": float(results["auroc"]),
        "auprc": float(results["auprc"]),
        "avg_cost": float(results["avg_cost"]),
        "avg_long_cost": float(results["avg_long_cost"]),
        "avg_aux_cost": float(results["avg_aux_cost"]),
        "total_samples": int(results["total_samples"]),
    }
    with out_path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
