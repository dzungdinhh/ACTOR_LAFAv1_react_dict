#!/usr/bin/env python3
"""
Render trajectory-only group plots from learned / RAS / DIME matched-cost artifacts.

This is a separate script from merge_group_plots_ras_dime.py.
It keeps the same data source/layout, but renders one trajectory figure per baseline:
  - group_XXX_traj_ras.png
  - group_XXX_traj_dime.png
  - group_XXX_traj_learned.png

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
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyArrowPatch
import numpy as np


BASELINE_ORDER = ["ras", "dime", "learned"]
BASELINE_TITLE = {
    "ras": "RAS",
    "dime": "DIME",
    "learned": "REACT",
}
BASELINE_FILE_TAG = {
    "ras": "ras",
    "dime": "dime",
    "learned": "learned",
}


def short_text(text: str, max_len: int = 18):
    text = str(text)
    if len(text) <= int(max_len):
        return text
    return text[: int(max_len) - 3] + "..."


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
    p.add_argument("--variant", default="matched_cost_groups_ras_dime")
    p.add_argument("--dpi", type=int, default=240)
    p.add_argument("--figsize", default="auto", help="W,H or 'auto'")
    p.add_argument("--row_gap", type=float, default=0.14)
    p.add_argument("--label_stride", type=int, default=1, help="Show every Nth feature label.")
    p.add_argument("--base_fontsize", type=int, default=18)
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
    w = min(10.8, max(6.2, 5.0 + 0.07 * t + 0.025 * min(f, 20)))
    h = min(8.4, max(5.0, 4.2 + 0.22 * min(f, 24)))
    return float(w), float(h)


def _resolve_left_margin(max_name_len: int, font_size: int = 18) -> float:
    base = 0.20
    base += 0.0048 * max(0, int(max_name_len) - 12)
    base += 0.0028 * max(0, int(font_size) - 16)
    return float(min(0.52, max(0.18, base)))


def _nonwhite_cmap(name: str, low_clip: float = 0.18):
    try:
        base = plt.get_cmap(str(name))
    except Exception:
        base = plt.get_cmap("BuPu")
    low = float(min(0.45, max(0.0, low_clip)))
    xs = np.linspace(low, 1.0, 256)
    colors = base(xs)
    return LinearSegmentedColormap.from_list(f"{base.name}_nonwhite", colors)


def _rounded_pct_labels_sum_100(prob: np.ndarray) -> np.ndarray:
    p = np.asarray(prob, dtype=np.float64).reshape(-1)
    if p.size == 0:
        return np.zeros((0,), dtype=np.int32)
    p = np.clip(p, 0.0, None)
    s = float(p.sum())
    if s <= 0.0:
        out = np.zeros_like(p, dtype=np.int32)
        out[0] = 100
        return out

    raw = (p / s) * 100.0
    base = np.floor(raw + 1e-12).astype(np.int32)
    remainder = int(100 - int(base.sum()))
    frac = raw - base.astype(np.float64)

    if remainder > 0:
        order = np.argsort(-frac, kind="stable")
        for idx in order[:remainder]:
            base[int(idx)] += 1
    elif remainder < 0:
        order = np.argsort(frac, kind="stable")
        need = -remainder
        for idx in order:
            ii = int(idx)
            if base[ii] <= 0:
                continue
            base[ii] -= 1
            need -= 1
            if need <= 0:
                break

    delta = int(100 - int(base.sum()))
    if delta != 0 and base.size > 0:
        base[int(np.argmax(raw))] += delta
    return base.astype(np.int32)


def _shrink_reposition_per_plot_cbar(ax_cbar, ax_graph):
    if ax_cbar is None or ax_graph is None:
        return

    graph_pos = ax_graph.get_position()
    cbar_pos = ax_cbar.get_position()
    width_scale = 0.80
    height_scale = 2.0 / 3.0

    new_w = graph_pos.width * width_scale
    new_h = cbar_pos.height * height_scale
    new_x0 = graph_pos.x0 + 0.10 * graph_pos.width
    max_x0 = graph_pos.x1 - new_w
    new_x0 = min(max_x0, max(graph_pos.x0, new_x0))

    preferred_y0 = cbar_pos.y0 - 0.060
    min_safe_y0 = 0.004
    max_y0 = cbar_pos.y1 - new_h
    if max_y0 < min_safe_y0:
        new_h = max(0.45 * cbar_pos.height, cbar_pos.y1 - min_safe_y0)
        new_h = min(new_h, cbar_pos.height)
        max_y0 = cbar_pos.y1 - new_h
    new_y0 = min(max(preferred_y0, min_safe_y0), max_y0)
    ax_cbar.set_position([new_x0, new_y0, new_w, new_h])


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
    auroc: float,
    auprc: float,
    shared_cost_ymax: float,
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
    term_pct_labels = _rounded_pct_labels_sum_100(term_prob)

    transitions = _build_transitions(x3, mode=str(args.edge_transition_mode))
    edge_prob = [(s, e, c / float(max(n, 1))) for (s, e), c in transitions.items()]
    selected = [x for x in edge_prob if x[2] > float(args.edge_min_freq)]
    if not selected:
        selected = edge_prob
    selected.sort(key=lambda z: z[2], reverse=True)
    if int(args.edge_max_edges) > 0:
        selected = selected[: int(args.edge_max_edges)]

    base_fs = int(args.base_fontsize)
    fig = plt.figure(figsize=_resolve_figsize(args.figsize, num_time=t, num_features=f), dpi=int(args.dpi))
    fig.patch.set_facecolor("white")
    graph_ratio = min(1.65, max(0.95, 0.74 + 0.11 * min(f, 24)))
    gs = gridspec.GridSpec(
        4,
        1,
        figure=fig,
        height_ratios=[0.30, 0.26, graph_ratio, 0.18],
        hspace=float(args.row_gap),
    )
    ax_cost = fig.add_subplot(gs[0, 0])
    ax_stop = fig.add_subplot(gs[1, 0])
    ax_graph = fig.add_subplot(gs[2, 0])
    ax_cbar = fig.add_subplot(gs[3, 0])
    ax_graph.set_zorder(3)
    ax_cbar.set_zorder(1)
    ax_cbar.patch.set_alpha(0.0)

    ax_cost.bar(np.arange(t), step_costs_long, color="#b54848", alpha=0.88, width=0.62)
    ax_cost.set_ylabel("Avg\nLong\nCost", fontsize=base_fs + 2, fontweight="bold")
    metric_bits = []
    if math.isfinite(float(auroc)):
        metric_bits.append(f"AUROC {float(auroc):.3f}")
    if math.isfinite(float(auprc)):
        metric_bits.append(f"AUPRC {float(auprc):.3f}")
    if math.isfinite(float(total_cost)):
        metric_bits.append(f"total {float(total_cost):.3f}")
    if math.isfinite(float(long_cost)):
        metric_bits.append(f"long {float(long_cost):.3f}")
    title = baseline_title
    if metric_bits:
        title = f"{baseline_title}\n" + " | ".join(metric_bits)
    ax_cost.set_title(title, fontsize=base_fs + 5, fontweight="bold", pad=6.0, loc="center")
    ax_cost.set_xlim(-0.5, t - 0.5)
    ymax = float(shared_cost_ymax) if float(shared_cost_ymax) > 0 else float(np.max(step_costs_long))
    if ymax <= 0.0:
        ymax = 1.0
    ax_cost.set_ylim(0.0, ymax * 1.02)
    ax_cost.set_yticks([0.0, ymax])
    ax_cost.tick_params(axis="y", labelsize=base_fs, width=1.0)
    for lbl in ax_cost.get_yticklabels():
        lbl.set_fontweight("bold")
    ax_cost.set_xticks([])
    ax_cost.grid(axis="y", linestyle="--", alpha=0.5, color="#4a4a4a", linewidth=0.7)
    ax_cost.spines["top"].set_visible(False)
    ax_cost.spines["right"].set_visible(False)
    ax_cost.spines["bottom"].set_visible(False)

    try:
        stop_cmap = plt.get_cmap(str(args.stop_cmap))
    except Exception:
        stop_cmap = plt.get_cmap("YlOrBr")
    ax_stop.imshow(term_prob.reshape(1, -1), cmap=stop_cmap, aspect="auto", vmin=0.0, vmax=1.0)
    ax_stop.set_yticks([])
    ax_stop.set_ylabel("Stop\nProb\n(%)", fontsize=base_fs + 2, fontweight="bold")
    ax_stop.set_xticks([])
    stop_pos = ax_stop.get_position()
    cell_w_pts = max(8.0, float(fig.get_figwidth()) * 72.0 * float(stop_pos.width) / max(1, t))
    cell_h_pts = max(8.0, float(fig.get_figheight()) * 72.0 * float(stop_pos.height))
    stop_fs = int(max(9.0, min(float(base_fs + 2), 0.62 * cell_w_pts, 0.55 * cell_h_pts)))
    for tt in range(t):
        if int(term_pct_labels[tt]) > 0:
            ax_stop.text(
                tt,
                0,
                str(int(term_pct_labels[tt])),
                ha="center",
                va="center",
                color="black",
                fontsize=stop_fs,
                fontweight="bold",
            )

    norm = Normalize(vmin=0.0, vmax=1.0)
    cmap = _nonwhite_cmap(args.node_cmap, low_clip=0.18)
    for y in range(f):
        for tt in range(t):
            freq = float(node_freq[tt, y])
            if freq > 0.001:
                ax_graph.scatter(
                    tt,
                    y,
                    s=max(27.0, freq * float(args.edge_node_size_scale)),
                    color=cmap(norm(freq)),
                    edgecolors="none",
                    zorder=8,
                )

    max_selected_prob = max((float(p) for (_, _, p) in selected), default=1.0)
    if max_selected_prob <= 0.0:
        max_selected_prob = 1.0
    for (s, e, prob) in selected:
        ts, gs_idx = int(s[0]), int(s[1])
        td, gd_idx = int(e[0]), int(e[1])
        dist = max(1, td - ts)
        prob_norm = float(prob) / max_selected_prob
        alpha = 0.20 + 0.48 * (prob_norm ** 0.75)
        curve_rad = min(0.30, 0.16 + 0.018 * dist)
        arrow = FancyArrowPatch(
            (ts, gs_idx),
            (td, gd_idx),
            connectionstyle=f"arc3,rad=-{curve_rad}",
            color="#5f5f5f",
            alpha=alpha,
            linewidth=max(0.75, max(0.85, float(prob) * float(args.edge_width_scale))),
            arrowstyle="->",
            mutation_scale=6.2,
            zorder=5,
            clip_on=False,
        )
        ax_graph.add_patch(arrow)

    if f <= 16:
        font_size = base_fs + 1
    elif f <= 30:
        font_size = base_fs
    else:
        font_size = max(12, base_fs - 1)
    xtick_font_size = max(12, font_size - 3)
    ax_graph.set_yticks(np.arange(f))
    if int(args.label_stride) > 1:
        yt = np.arange(f)[:: int(args.label_stride)]
        ax_graph.set_yticks(yt)
        ax_graph.set_yticklabels([short_text(names_ord[i], 26) for i in yt], fontsize=font_size, fontweight="bold")
    else:
        ax_graph.set_yticklabels([short_text(x, 26) for x in names_ord], fontsize=font_size, fontweight="bold")
    ax_graph.set_xticks(np.arange(t))
    ax_graph.set_xticklabels([str(i + 1) for i in range(t)], fontsize=xtick_font_size, fontweight="bold")
    ax_graph.set_ylim(-0.6, f - 0.4)
    ax_graph.set_xlim(-0.5, t - 0.5)
    ax_graph.grid(linestyle=":", alpha=0.55, color="#636363", linewidth=0.75)
    ax_graph.spines["top"].set_visible(False)
    ax_graph.spines["right"].set_visible(False)
    ax_graph.tick_params(axis="y", width=1.0)
    ax_graph.tick_params(axis="x", width=1.0, pad=2.0)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cbar, orientation="horizontal")
    cbar.set_label("Acquisition rate", fontsize=base_fs + 2, fontweight="bold")
    cbar.ax.xaxis.set_label_position("bottom")
    cbar.ax.xaxis.labelpad = 1.0
    cbar.ax.tick_params(labelsize=base_fs, width=1.0)
    cbar.set_ticks([0.0, 0.5, 1.0])
    for lbl in cbar.ax.get_xticklabels():
        lbl.set_fontweight("bold")
    max_name_len = max((len(str(x)) for x in names_ord), default=10)
    left_margin = _resolve_left_margin(max_name_len=max_name_len, font_size=font_size)
    plt.subplots_adjust(left=float(left_margin), right=0.998, top=0.985, bottom=0.080)
    _shrink_reposition_per_plot_cbar(ax_cbar=ax_cbar, ax_graph=ax_graph)

    out_parent = os.path.dirname(out_file_path)
    os.makedirs(out_parent, exist_ok=True)
    fig.savefig(out_file_path, dpi=int(args.dpi), bbox_inches="tight", pad_inches=0.02)
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

    shared_cost_ymax = 0.0
    for baseline in BASELINE_ORDER:
        ro = rollouts[baseline]
        gc = np.asarray(ro["group_costs"], dtype=np.float32).reshape(-1)
        if gc.size != int(num_groups):
            gc = np.ones(int(num_groups), dtype=np.float32)

        x3_all = ro["masks"].reshape(-1, num_time, num_groups)[:, :, order_all]
        if str(args.feature_filter_mode) == "used":
            usage_all = x3_all.mean(axis=(0, 1))
            used_local = np.where(usage_all > 0.0)[0].astype(np.int32)
            if used_local.size == 0:
                used_local = np.arange(x3_all.shape[2], dtype=np.int32)
        else:
            used_local = np.arange(x3_all.shape[2], dtype=np.int32)
        x3 = x3_all[:, :, used_local]
        costs_ord_all = np.asarray(gc, dtype=np.float32).reshape(-1)[order_all]
        costs_ord = costs_ord_all[used_local]
        step_costs_long = np.mean(x3 * costs_ord.reshape(1, 1, x3.shape[2]), axis=0).sum(axis=1)
        shared_cost_ymax = max(shared_cost_ymax, float(np.max(step_costs_long)) if step_costs_long.size > 0 else 0.0)

    rel = os.path.relpath(group_dir, start=args.plots_root)
    rel_parts = rel.split(os.sep)
    group_name = rel_parts[-1]
    mode_dir = os.path.join(args.merged_root, *rel_parts)
    os.makedirs(mode_dir, exist_ok=True)

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
        auroc = float(rec.get("auroc", float("nan")))
        auprc = float(rec.get("auprc", float("nan")))

        filt_tag = "filtered" if str(args.feature_filter_mode) == "used" else "unfiltered"
        out_name = f"traj_{BASELINE_FILE_TAG[baseline]}_{filt_tag}.png"
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
            auroc=auroc,
            auprc=auprc,
            shared_cost_ymax=shared_cost_ymax,
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
