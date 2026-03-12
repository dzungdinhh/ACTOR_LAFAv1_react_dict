#!/usr/bin/env python3
"""
Render trajectory-only group plots from matched-cost artifacts.

This is a separate script from merge_group_plots.py.
It keeps the same data source/layout, but renders one trajectory figure per baseline:
  - group_XXX_traj_none_context.png
  - group_XXX_traj_learned_context.png
  - group_XXX_traj_all_context.png

Key styling changes from previous trajectory panel:
  1) supports both feature sets:
     - used: only nonzero-usage features
     - all: full unfiltered feature set
  2) node size is frequency-scaled again, node color encodes frequency
     via configurable colormap (default Blues)
  3) stop/termination row uses configurable colormap (default YlOrBr)
  4) add a node-frequency colorbar
"""

import argparse
import csv
import glob
import math
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch
import numpy as np

import analysis_plots as ap


BASELINE_ORDER = ["none", "learned", "all"]
BASELINE_TITLE = {
    "none": "No baseline",
    "learned": "Learned baseline",
    "all": "All baseline",
}
BASELINE_FILE_TAG = {
    "none": "none_context",
    "learned": "learned_context",
    "all": "all_context",
}


def parse_args():
    p = argparse.ArgumentParser(description="Render trajectory-only matched-group figures.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--mode", required=True, choices=["longitudinal", "total"])
    p.add_argument("--group", type=int, default=None, help="Group number (e.g., 1 for group_001).")
    p.add_argument("--plots_root", default="/playpen-nvme/scribble/ddinh/ACTOR_LAFAv1/plots")
    p.add_argument(
        "--merged_root",
        default="/playpen-nvme/scribble/ddinh/ACTOR_LAFAv1/plots_merged",
        help="Root directory for trajectory-only outputs.",
    )
    p.add_argument("--variant", default="matched_cost_groups")
    p.add_argument("--dpi", type=int, default=240)
    p.add_argument("--figsize", default="auto", help="W,H or 'auto'")
    p.add_argument("--row_gap", type=float, default=0.20)
    p.add_argument("--label_stride", type=int, default=1, help="Show every Nth feature label.")
    p.add_argument(
        "--feature_filter_mode",
        type=str,
        default="used",
        choices=["used", "all"],
        help="Feature set for trajectory axis: used (filtered) or all (unfiltered).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--edge_min_freq", type=float, default=0.01)
    p.add_argument("--edge_max_edges", type=int, default=0, help="<=0 means uncapped")
    p.add_argument(
        "--edge_node_size_scale",
        type=float,
        default=120.0,
        help="Frequency-to-node-size scale factor.",
    )
    p.add_argument("--edge_width_scale", type=float, default=2.2)
    p.add_argument("--node_cmap", type=str, default="Blues", help="Colormap for node frequency.")
    p.add_argument("--stop_cmap", type=str, default="YlOrBr", help="Colormap for stop-probability strip.")
    p.add_argument(
        "--edge_transition_mode",
        type=str,
        default="next_observed",
        choices=["strict_next", "next_observed"],
    )
    return p.parse_args()


def _parse_figsize(s: str) -> Tuple[float, float]:
    parts = s.split(",")
    if len(parts) != 2:
        raise ValueError(f"Invalid --figsize '{s}', expected W,H")
    return float(parts[0]), float(parts[1])


def _read_costs_quick(path: str) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            b = str(row.get("baseline", "")).strip()
            if b == "":
                continue
            rec = {}
            for k, v in row.items():
                if k == "baseline":
                    continue
                try:
                    rec[k] = float(v)
                except Exception:
                    rec[k] = float("nan")
            out[b] = rec
    return out


def _dataset_dir_name(dataset: str) -> str:
    return dataset.strip().lower()


def _group_dirs(base_mode_dir: str, group_id: Optional[int]) -> List[str]:
    if group_id is not None:
        g = os.path.join(base_mode_dir, f"group_{group_id:03d}")
        return [g] if os.path.isdir(g) else []
    gs = sorted(glob.glob(os.path.join(base_mode_dir, "group_*")))
    return [g for g in gs if os.path.isdir(g)]


def _load_json(path: str):
    import json
    with open(path, "r") as f:
        return json.load(f)


def _load_rollout(cache_dir: str):
    path = os.path.join(cache_dir, "analysis_rollout.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing rollout NPZ: {path}")
    z = np.load(path, allow_pickle=True)
    return {
        "masks": z["masks"].astype(np.float32),  # (N, T*G)
        "num_time": int(z["num_time"]),
        "num_groups": int(z["num_groups"]),
        "group_names": [str(x) for x in np.asarray(z["group_names"], dtype=object).reshape(-1).tolist()],
        "group_costs": z["group_costs"].astype(np.float32),
    }


def _shared_order_all(meta: dict, num_groups: int) -> np.ndarray:
    if "shared_order_all" in meta:
        raw = [int(x) for x in meta["shared_order_all"]]
        keep = [x for x in raw if 0 <= int(x) < int(num_groups)]
        if len(keep) == int(num_groups):
            return np.asarray(keep, dtype=np.int32)
        seen = set(keep)
        missing = [i for i in range(int(num_groups)) if i not in seen]
        return np.asarray(keep + missing, dtype=np.int32)
    return np.arange(int(num_groups), dtype=np.int32)


def _build_transitions(mask3d: np.ndarray, mode: str):
    transitions = Counter()
    n, t, _ = mask3d.shape
    for i in range(n):
        acq_steps = []
        for tt in range(t):
            feat_idx = np.where(mask3d[i, tt] > 0.5)[0].astype(int).tolist()
            if len(feat_idx) > 0:
                acq_steps.append((tt, feat_idx))
        if len(acq_steps) < 2:
            continue
        if mode == "strict_next":
            for tt in range(t - 1):
                src = np.where(mask3d[i, tt] > 0.5)[0]
                dst = np.where(mask3d[i, tt + 1] > 0.5)[0]
                if src.size == 0 or dst.size == 0:
                    continue
                for s in src:
                    for d in dst:
                        transitions[((tt, int(s)), (tt + 1, int(d)))] += 1
        else:
            for j in range(len(acq_steps) - 1):
                t_src, src_feats = acq_steps[j]
                t_dst, dst_feats = acq_steps[j + 1]
                for s in src_feats:
                    for d in dst_feats:
                        transitions[((int(t_src), int(s)), (int(t_dst), int(d)))] += 1
    return transitions


def _resolve_figsize(figsize_arg: str, num_time: int, num_features: int) -> Tuple[float, float]:
    if str(figsize_arg).strip().lower() != "auto":
        return _parse_figsize(figsize_arg)
    t = max(1, int(num_time))
    f = max(1, int(num_features))
    w = min(22.0, max(12.0, 9.0 + 0.10 * t + 0.06 * min(f, 50)))
    h = min(16.0, max(7.5, 6.2 + 0.14 * min(f, 45)))
    return float(w), float(h)


def _resolve_left_margin(max_name_len: int) -> float:
    base = 0.19 + 0.0036 * max(0, int(max_name_len) - 12)
    return float(min(0.48, max(0.18, base)))


def _render_trajectory_only(
    out_file_path: str,
    masks_flat: np.ndarray,
    num_time: int,
    num_groups: int,
    order_all: np.ndarray,
    group_names: List[str],
    group_costs: np.ndarray,
    baseline_title: str,
    dataset: str,
    mode: str,
    group_name: str,
    long_cost: float,
    total_cost: float,
    args,
):
    n = int(masks_flat.shape[0])
    order_all = np.asarray(order_all, dtype=np.int32)
    if order_all.size != int(num_groups):
        order_all = np.arange(int(num_groups), dtype=np.int32)

    x3_all = masks_flat.reshape(n, num_time, num_groups)[:, :, order_all]  # ordered first
    if str(args.feature_filter_mode) == "used":
        usage_all = x3_all.mean(axis=(0, 1))
        used_local = np.where(usage_all > 0.0)[0].astype(np.int32)
        if used_local.size == 0:
            # Keep full set for degenerate no-acquisition case.
            used_local = np.arange(x3_all.shape[2], dtype=np.int32)
    else:
        used_local = np.arange(x3_all.shape[2], dtype=np.int32)

    x3 = x3_all[:, :, used_local]  # keep used features only
    names_ord_all = [group_names[i] for i in order_all.tolist()]
    names_ord = [names_ord_all[i] for i in used_local.tolist()]
    costs_ord_all = np.asarray(group_costs, dtype=np.float32).reshape(-1)[order_all]
    costs_ord = costs_ord_all[used_local]

    t = int(x3.shape[1])
    f = int(x3.shape[2])

    node_freq = x3.mean(axis=0)  # (T, F)
    step_costs_long = np.mean(x3 * costs_ord.reshape(1, 1, f), axis=0).sum(axis=1)

    last_acq_time = np.zeros(n, dtype=np.int32)
    for i in range(n):
        active_t = np.where(x3[i].sum(axis=1) > 0)[0]
        last_acq_time[i] = int(active_t[-1]) if active_t.size > 0 else 0
    term_counts = np.bincount(last_acq_time, minlength=t).astype(np.float32)
    term_prob = term_counts / max(float(term_counts.sum()), 1.0)

    transitions = _build_transitions(x3, mode=str(args.edge_transition_mode))
    edge_prob = [(s, e, c / float(max(n, 1))) for (s, e), c in transitions.items()]
    selected = [x for x in edge_prob if x[2] > float(args.edge_min_freq)]
    if not selected:
        selected = edge_prob
    selected.sort(key=lambda z: z[2], reverse=True)
    if int(args.edge_max_edges) > 0:
        selected = selected[: int(args.edge_max_edges)]

    fig = plt.figure(figsize=_resolve_figsize(args.figsize, num_time=t, num_features=f), dpi=int(args.dpi))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(
        4,
        1,
        figure=fig,
        # Top rows intentionally small (~1/3 of earlier compact setting).
        height_ratios=[0.10, 0.09, min(1.95, max(1.20, 0.90 + 0.050 * f)), 0.08],
        hspace=float(args.row_gap),
    )
    ax_cost = fig.add_subplot(gs[0, 0])
    ax_stop = fig.add_subplot(gs[1, 0])
    ax_graph = fig.add_subplot(gs[2, 0])
    ax_cbar = fig.add_subplot(gs[3, 0])

    ax_cost.bar(np.arange(t), step_costs_long, color="#d65f5f", alpha=0.82, width=0.6)
    ax_cost.set_ylabel("Avg\nLong\nCost", fontsize=13)
    ax_cost.set_xlim(-0.5, t - 0.5)
    ymax = float(np.max(step_costs_long)) if float(np.max(step_costs_long)) > 0 else 1.0
    ax_cost.set_yticks([0.0, ymax])
    ax_cost.tick_params(axis="y", labelsize=11)
    ax_cost.set_xticks([])
    ax_cost.grid(axis="y", linestyle="--", alpha=0.3)
    ax_cost.spines["top"].set_visible(False)
    ax_cost.spines["right"].set_visible(False)
    ax_cost.spines["bottom"].set_visible(False)

    # Stop/termination row colormap is configurable.
    vmax_val = float(np.max(term_prob) * 1.5) if float(np.max(term_prob)) > 0 else 1.0
    try:
        stop_cmap = plt.get_cmap(str(args.stop_cmap))
    except Exception:
        stop_cmap = plt.get_cmap("YlOrBr")
    ax_stop.imshow(term_prob.reshape(1, -1), cmap=stop_cmap, aspect="auto", vmin=0.0, vmax=vmax_val)
    ax_stop.set_yticks([])
    ax_stop.set_ylabel("\nStop\nProb", fontsize=13)
    ax_stop.set_xticks([])
    for tt in range(t):
        if term_prob[tt] > 0.001:
            val = term_prob[tt] * 100.0
            lbl = f"{val:.1f}"
            if lbl.endswith(".0"):
                lbl = lbl[:-2]
            ax_stop.text(tt, 0, f"{lbl}%", ha="center", va="center", color="black", fontsize=12)

    # Node color encodes frequency with configurable colormap.
    max_freq = float(np.max(node_freq)) if node_freq.size > 0 else 0.0
    if max_freq <= 0:
        max_freq = 1.0
    norm = Normalize(vmin=0.0, vmax=max_freq)
    try:
        cmap = plt.get_cmap(str(args.node_cmap))
    except Exception:
        cmap = plt.get_cmap("Blues")
    for y in range(f):
        for tt in range(t):
            freq = float(node_freq[tt, y])
            if freq > 0.001:
                ax_graph.scatter(
                    tt,
                    y,
                    s=max(6.0, freq * float(args.edge_node_size_scale)),
                    color=cmap(norm(freq)),
                    edgecolors="none",
                    zorder=8,
                )

    for (s, e, prob) in selected:
        ts, gs_idx = int(s[0]), int(s[1])
        td, gd_idx = int(e[0]), int(e[1])
        dist = max(1, td - ts)
        alpha = min(0.90, max(0.32, float(prob) * 8.0))
        arrow = FancyArrowPatch(
            (ts, gs_idx),
            (td, gd_idx),
            connectionstyle=f"arc3,rad=-{0.1 + 0.01 * dist}",
            color="#151515",
            alpha=alpha,
            linewidth=max(0.45, float(prob) * float(args.edge_width_scale)),
            arrowstyle="->",
            mutation_scale=5.2,
            zorder=5,
            clip_on=False,
        )
        ax_graph.add_patch(arrow)

    if f <= 16:
        font_size = 12
    elif f <= 30:
        font_size = 11
    else:
        font_size = 10
    ax_graph.set_yticks(np.arange(f))
    if int(args.label_stride) > 1:
        yt = np.arange(f)[:: int(args.label_stride)]
        ax_graph.set_yticks(yt)
        ax_graph.set_yticklabels([ap.short_text(names_ord[i], 26) for i in yt], fontsize=font_size, fontweight="bold")
    else:
        ax_graph.set_yticklabels([ap.short_text(x, 26) for x in names_ord], fontsize=font_size, fontweight="bold")
    ax_graph.set_xticks(np.arange(t))
    ax_graph.set_xticklabels([f"t={i + 1}" for i in range(t)], fontsize=font_size)
    ax_graph.set_ylim(-0.6, f - 0.4)
    ax_graph.set_xlim(-0.5, t - 0.5)
    ax_graph.grid(linestyle=":", alpha=0.75, color="#3f3f3f", linewidth=0.7)
    ax_graph.spines["top"].set_visible(False)
    ax_graph.spines["right"].set_visible(False)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cbar, orientation="horizontal")
    cbar.set_label("Node frequency", fontsize=13)
    cbar.ax.tick_params(labelsize=11)

    max_name_len = max((len(str(x)) for x in names_ord), default=10)
    left_margin = _resolve_left_margin(max_name_len=max_name_len)
    plt.subplots_adjust(left=float(left_margin), right=0.985, top=0.93, bottom=0.07)

    fig.suptitle(
        f"{str(dataset).lower()} | {str(mode)} | {group_name} | {baseline_title} | "
        f"{str(args.feature_filter_mode)}-features | "
        f"long={float(long_cost):.3f} total={float(total_cost):.3f}",
        fontsize=15,
        y=0.988,
    )

    out_parent = os.path.dirname(out_file_path)
    os.makedirs(out_parent, exist_ok=True)
    fig.savefig(out_file_path, dpi=int(args.dpi))
    plt.close(fig)
    return out_file_path


def _render_group(group_dir: str, args):
    meta_path = os.path.join(group_dir, "group_meta.json")
    quick_path = os.path.join(group_dir, "group_costs_quick.tsv")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Missing group_meta.json: {meta_path}")
    if not os.path.isfile(quick_path):
        raise FileNotFoundError(f"Missing group_costs_quick.tsv: {quick_path}")

    meta = _load_json(meta_path)
    costs = _read_costs_quick(quick_path)

    rollouts = {}
    for b in BASELINE_ORDER:
        cmeta = meta["checkpoints"][b]
        rollouts[b] = _load_rollout(cmeta["cache_dir"])

    num_time = int(meta.get("num_time", rollouts["learned"]["num_time"]))
    num_groups = int(meta.get("num_groups", rollouts["learned"]["num_groups"]))
    order_all = _shared_order_all(meta=meta, num_groups=num_groups)

    rel = os.path.relpath(group_dir, start=args.plots_root)
    rel_parts = rel.split(os.sep)
    group_name = rel_parts[-1]
    mode_dir = os.path.join(args.merged_root, *rel_parts[:-1])  # flattened

    outputs = []
    for baseline in BASELINE_ORDER:
        ro = rollouts[baseline]
        names = ro["group_names"]
        masks = ro["masks"]
        gc = np.asarray(ro["group_costs"], dtype=np.float32).reshape(-1)
        if gc.size != int(num_groups):
            gc = np.ones(int(num_groups), dtype=np.float32)

        rec = costs.get(baseline, {})
        long_cost = float(rec.get("avg_long_cost", float("nan")))
        total_cost = float(rec.get("avg_cost", float("nan")))

        filt_tag = "filtered" if str(args.feature_filter_mode) == "used" else "unfiltered"
        out_name = f"{group_name}_traj_{BASELINE_FILE_TAG[baseline]}_{filt_tag}.png"
        out_path = os.path.join(mode_dir, out_name)
        out = _render_trajectory_only(
            out_file_path=out_path,
            masks_flat=masks,
            num_time=num_time,
            num_groups=num_groups,
            order_all=order_all,
            group_names=names,
            group_costs=gc,
            baseline_title=BASELINE_TITLE[baseline],
            dataset=args.dataset,
            mode=args.mode,
            group_name=group_name,
            long_cost=long_cost,
            total_cost=total_cost,
            args=args,
        )
        outputs.append(out)
    return outputs


def main():
    args = parse_args()
    mode_root = os.path.join(args.plots_root, _dataset_dir_name(args.dataset), args.variant, args.mode)
    if not os.path.isdir(mode_root):
        raise FileNotFoundError(f"Mode directory not found: {mode_root}")

    groups = _group_dirs(mode_root, args.group)
    if len(groups) == 0:
        raise FileNotFoundError(f"No groups found in {mode_root} for group={args.group}")

    print("=" * 80)
    print("Render Trajectory-Only Group Panels")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Mode: {args.mode}")
    print(f"Variant: {args.variant}")
    print(f"Groups: {len(groups)}")
    print(f"Output root: {args.merged_root}")
    print(f"Feature filter mode: {args.feature_filter_mode}")
    print(
        f"Node color: {args.node_cmap} by frequency; stop strip: {args.stop_cmap}; "
        f"node size: frequency-scaled (scale={float(args.edge_node_size_scale):.1f})."
    )

    for g in groups:
        outs = _render_group(g, args=args)
        for p in outs:
            print(f"[OK] {p}")


if __name__ == "__main__":
    main()
