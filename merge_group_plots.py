#!/usr/bin/env python3
"""
Create merged 3x2 group panels fully from rollout data (no PNG stitching).

Rows: none / learned / all
Cols: heatmap / trajectory
"""

import argparse
import csv
import glob
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.patches import FancyArrowPatch

import analysis_plots as ap
import analysis_plots_cheears_matched as apm


BASELINE_ORDER = ["none", "learned", "all"]
BASELINE_LABEL = {
    "none": "No baseline",
    "learned": "Learned baseline",
    "all": "All baseline",
}
MIN_FEATURE_RATE = 0.01


def parse_args():
    p = argparse.ArgumentParser(description="Render merged group panels directly from rollout data.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--mode", required=True, choices=["longitudinal", "total"])
    p.add_argument("--group", type=int, default=None, help="Group number (e.g., 1 for group_001).")
    p.add_argument("--plots_root", default="/playpen-nvme/scribble/ddinh/ACTOR_LAFAv1/plots")
    p.add_argument(
        "--merged_root",
        default="/playpen-nvme/scribble/ddinh/ACTOR_LAFAv1/plots_merged",
        help="Root directory for merged outputs (mirrors plots_root structure).",
    )
    p.add_argument("--variant", default="matched_cost_groups")
    p.add_argument(
        "--output_name",
        default="{group}.png",
        help="Output filename template. Use {group} placeholder, e.g. '{group}.png'.",
    )
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--figsize", default="auto", help="W,H or 'auto'")
    p.add_argument("--row_label_size", type=int, default=16)
    p.add_argument("--row_gap", type=float, default=0.04)
    p.add_argument("--col_gap", default="auto", help="Column gap float or 'auto'")
    p.add_argument("--cluster_k", type=int, default=0, help="<=0 uses shared_cluster_k from group_meta")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--edge_min_freq", type=float, default=0.01)
    p.add_argument("--edge_max_edges", type=int, default=0, help="<=0 means uncapped")
    p.add_argument("--edge_node_size_scale", type=float, default=95.0)
    p.add_argument("--edge_width_scale", type=float, default=1.35)
    p.add_argument(
        "--edge_transition_mode",
        type=str,
        default="next_observed",
        choices=["strict_next", "next_observed"],
    )
    p.add_argument("--label_stride", type=int, default=1, help="Show every Nth feature label")
    return p.parse_args()


def _parse_figsize(s: str) -> Tuple[float, float]:
    parts = s.split(",")
    if len(parts) != 2:
        raise ValueError(f"Invalid --figsize '{s}', expected W,H")
    return float(parts[0]), float(parts[1])


def _read_costs_quick(path: str) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    if not os.path.isfile(path):
        return out
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


def _fmt_triplet(costs: Dict[str, Dict[str, float]], key: str) -> str:
    vals = []
    for b in BASELINE_ORDER:
        v = costs.get(b, {}).get(key, float("nan"))
        if math.isfinite(v):
            vals.append(f"{b}={v:.3f}")
        else:
            vals.append(f"{b}=NA")
    return "(" + ", ".join(vals) + ")"


def _resolve_col_gap(col_gap_arg: str, cluster_k: int, max_name_len: int, num_features: int) -> float:
    s = str(col_gap_arg).strip().lower()
    if s != "auto":
        try:
            return float(col_gap_arg)
        except Exception:
            pass
    k = max(1, int(cluster_k))
    ln = max(1, int(max_name_len))
    # Adaptive heuristic: more centroid panels and longer labels need wider inter-column gap.
    f = max(1, int(num_features))
    # Stronger adaptive spacing to protect long trajectory y-labels.
    gap = 0.24 + 0.030 * max(0, k - 1) + 0.0060 * max(0, ln - 12) + 0.0020 * max(0, f - 12)
    return float(min(1.0, max(0.22, gap)))


def _resolve_left_margin(max_name_len: int, row_label_size: int) -> float:
    """
    Reserve space on the left for long heatmap y-labels and row-level titles.
    """
    ln = max(1, int(max_name_len))
    rs = max(8, int(row_label_size))
    margin = 0.22 + 0.0032 * max(0, ln - 14) + 0.0020 * max(0, rs - 14)
    return float(min(0.42, max(0.20, margin)))


def _resolve_figsize(figsize_arg: str, num_time: int, num_features: int, cluster_k: int) -> Tuple[float, float]:
    if str(figsize_arg).strip().lower() != "auto":
        return _parse_figsize(figsize_arg)

    k = max(1, int(cluster_k))
    t = max(1, int(num_time))
    f = max(1, int(num_features))

    # Heat column width grows with number of centroid panels.
    heat_w = 2.6 * k + 2.0
    # Trajectory column width grows mildly with time and feature count.
    traj_w = 8.8 + 0.03 * t + 0.06 * min(f, 35)
    width = min(36.0, max(16.0, heat_w + traj_w))

    # Row height is capped to avoid overly tall stretched trajectory panels.
    row_h = min(4.1, max(3.0, 2.8 + 0.03 * min(f, 35)))
    height = min(18.0, max(10.5, 3.0 * row_h + 0.6))
    return float(width), float(height)


def _dataset_dir_name(dataset: str) -> str:
    return dataset.strip().lower()


def _group_dirs(base_mode_dir: str, group_id: Optional[int]) -> List[str]:
    if group_id is not None:
        g = os.path.join(base_mode_dir, f"group_{group_id:03d}")
        return [g] if os.path.isdir(g) else []
    gs = sorted(glob.glob(os.path.join(base_mode_dir, "group_*")))
    return [g for g in gs if os.path.isdir(g)]


def _load_json(path: str):
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


def _render_learned_context_plot(group_dir: str, out_file_path: str, dataset: str):
    """
    Render learned context plot using the exact helper used in
    analysis_plots_cheears_matched.py, then place it at out_file_path.
    """
    meta_path = os.path.join(group_dir, "group_meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Missing group_meta.json: {meta_path}")
    meta = _load_json(meta_path)

    learned = meta["checkpoints"]["learned"]
    rollout_path = os.path.join(learned["cache_dir"], "analysis_rollout.npz")
    if not os.path.isfile(rollout_path):
        raise FileNotFoundError(f"Missing rollout NPZ for learned context: {rollout_path}")

    learned_rollout = apm._read_rollout_npz(rollout_path)
    learned_ckpt_path = str(learned.get("actor_path", ""))
    script_dir = os.path.dirname(os.path.abspath(__file__))

    with tempfile.TemporaryDirectory(prefix="learned_ctx_") as tmpdir:
        info = apm.save_learned_baseline_context_plot(
            group_dir=tmpdir,
            script_dir=script_dir,
            dataset=dataset,
            learned_rollout=learned_rollout,
            learned_ckpt_path=learned_ckpt_path,
        )
        if info is None or "plot" not in info:
            raise RuntimeError("Failed to render learned context plot.")
        src_plot = str(info["plot"])
        out_parent = os.path.dirname(out_file_path)
        os.makedirs(out_parent, exist_ok=True)
        shutil.copy2(src_plot, out_file_path)
    return out_file_path


def _shared_keep_idx(meta: dict, rollouts: Dict[str, dict]) -> np.ndarray:
    if "shared_keep_idx" in meta:
        return np.asarray(meta["shared_keep_idx"], dtype=np.int32)
    # fallback from learned usage
    m = rollouts["learned"]["masks"]
    t = rollouts["learned"]["num_time"]
    g = rollouts["learned"]["num_groups"]
    mask3d = m.reshape(m.shape[0], t, g)
    usage = mask3d.mean(axis=(0, 1))
    keep = np.where(usage >= float(MIN_FEATURE_RATE))[0]
    if keep.size == 0:
        keep = np.where(usage > 0)[0]
    if keep.size == 0:
        keep = np.arange(g)
    return keep.astype(np.int32)


def _shared_order(meta: dict, keep_idx: np.ndarray) -> np.ndarray:
    if "shared_order_all" in meta:
        raw = [int(x) for x in meta["shared_order_all"]]
        # keep only values that exist in keep_idx and preserve order
        keep_set = set(int(x) for x in keep_idx.tolist())
        out = []
        seen = set()
        for x in raw:
            if x in keep_set and x not in seen:
                seen.add(x)
                out.append(x)
        for x in keep_idx.tolist():
            if int(x) not in seen:
                out.append(int(x))
        return np.asarray(out, dtype=np.int32)
    return keep_idx.astype(np.int32)


def _centroids_from_masks(
    masks_flat: np.ndarray,
    num_time: int,
    num_groups: int,
    keep_idx: np.ndarray,
    cluster_k: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      centroids: (K, T, F_kept)
      pct:       (K,) cluster proportions
    """
    x = masks_flat.astype(np.float32)
    n = x.shape[0]
    x3 = x.reshape(n, num_time, num_groups)

    if cluster_k <= 1:
        c = x3[:, :, keep_idx].mean(axis=0, keepdims=True).astype(np.float32)
        return c, np.asarray([1.0], dtype=np.float32)

    try:
        from sklearn.cluster import KMeans
    except Exception:
        c = x3[:, :, keep_idx].mean(axis=0, keepdims=True).astype(np.float32)
        return c, np.asarray([1.0], dtype=np.float32)

    k = max(1, min(int(cluster_k), n))
    km = KMeans(n_clusters=k, random_state=int(seed), n_init=20)
    labels = km.fit_predict(x)
    raw = km.cluster_centers_.reshape(k, num_time, num_groups)[:, :, keep_idx]
    counts = np.bincount(labels, minlength=k).astype(np.float32)
    order = np.argsort(-counts)
    cent = raw[order].astype(np.float32)
    counts = counts[order]
    pct = counts / max(float(counts.sum()), 1.0)
    return cent, pct.astype(np.float32)


def _build_transitions(mask3d: np.ndarray, mode: str):
    n, t, g = mask3d.shape
    trans = Counter()
    if mode == "next_observed":
        for i in range(n):
            active_steps = np.where(mask3d[i].sum(axis=1) > 0.5)[0]
            if active_steps.size < 2:
                continue
            for j in range(active_steps.size - 1):
                ts = int(active_steps[j])
                td = int(active_steps[j + 1])
                gs = np.where(mask3d[i, ts] > 0.5)[0]
                gd = np.where(mask3d[i, td] > 0.5)[0]
                for a in gs:
                    for b in gd:
                        trans[((ts, int(a)), (td, int(b)))] += 1
    else:
        for i in range(n):
            for ts in range(t - 1):
                gs = np.where(mask3d[i, ts] > 0.5)[0]
                gd = np.where(mask3d[i, ts + 1] > 0.5)[0]
                for a in gs:
                    for b in gd:
                        trans[((ts, int(a)), (ts + 1, int(b)))] += 1
    return trans


def _plot_heatmap_cell(
    fig,
    cell_spec,
    centroids_tk: np.ndarray,
    pct: np.ndarray,
    ordered_names: List[str],
    label_stride: int,
    show_colorbar: bool,
):
    """
    Draw centroid templates in the same style as run_mask_clustering panel.
    """
    k = int(centroids_tk.shape[0])
    t = int(centroids_tk.shape[1])
    f = int(centroids_tk.shape[2])

    # Use a 2-row layout: centroids on top, horizontal colorbar below.
    # This avoids overlap with trajectory y-axis labels in the neighboring column.
    sub = gridspec.GridSpecFromSubplotSpec(
        2,
        k,
        subplot_spec=cell_spec,
        height_ratios=[1.0, 0.085],
        hspace=0.30,
        wspace=0.12,
    )
    axes = []
    im = None

    xticks = np.arange(t)
    xlabels = [str(i + 1) for i in xticks]
    if f <= 40:
        yticks = np.arange(f)
    else:
        ystep = max(1, int(np.ceil(f / 40.0)))
        yticks = np.arange(0, f, ystep)
    if int(label_stride) > 1:
        yticks = yticks[:: int(label_stride)]

    for i in range(k):
        ax = fig.add_subplot(sub[0, i])
        im = ax.imshow(
            centroids_tk[i].T,
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            aspect="equal",
            interpolation="nearest",
        )
        if t > 0:
            ax.set_box_aspect(float(f) / float(t))
        ax.set_title(f"Centroid {i + 1} ({float(pct[i]) * 100.0:.1f}%)", fontsize=8, pad=2)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontsize=7)
        if i == 0:
            ax.set_yticks(yticks)
            ax.set_yticklabels([ap.short_text(ordered_names[j], max_len=24) for j in yticks], fontsize=7)
        else:
            ax.set_yticks(yticks)
            ax.set_yticklabels([])
        ax.tick_params(axis="both", length=0, pad=1.0)
        axes.append(ax)

    if im is not None and bool(show_colorbar):
        cax = fig.add_subplot(sub[1, :])
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
        cbar.set_label("Acquisition rate", fontsize=7)
        cbar.ax.tick_params(labelsize=7, pad=1)
        # Make bar shorter by half and center it.
        bb = cax.get_position()
        new_w = bb.width * 0.5
        new_x = bb.x0 + 0.5 * (bb.width - new_w)
        cax.set_position([new_x, bb.y0, new_w, bb.height])
    elif not bool(show_colorbar):
        # Keep row height consistent without drawing bar.
        for i in range(k):
            ax_dummy = fig.add_subplot(sub[1, i])
            ax_dummy.axis("off")
    return axes[0]


def _plot_trajectory_cell(
    fig,
    cell_spec,
    masks_flat: np.ndarray,
    num_time: int,
    num_groups: int,
    keep_idx: np.ndarray,
    local_order: np.ndarray,
    ordered_names: List[str],
    ordered_costs: np.ndarray,
    edge_min_freq: float,
    edge_max_edges: int,
    node_size_scale: float,
    edge_width_scale: float,
    transition_mode: str,
    label_stride: int,
):
    """
    Draw cost row + stop row + graph row in the same style as run_temporal_edges.
    """
    n = masks_flat.shape[0]
    x3 = masks_flat.reshape(n, num_time, num_groups)[:, :, keep_idx][:, :, local_order]  # (N, T, F)
    t = x3.shape[1]
    f = x3.shape[2]

    node_freq = x3.mean(axis=0)  # (T, F)
    step_costs_long = np.mean(x3 * ordered_costs.reshape(1, 1, f), axis=0).sum(axis=1)

    last_acq_time = np.zeros(n, dtype=np.int32)
    for i in range(n):
        active_t = np.where(x3[i].sum(axis=1) > 0)[0]
        last_acq_time[i] = int(active_t[-1]) if active_t.size > 0 else 0
    term_counts = np.bincount(last_acq_time, minlength=t).astype(np.float32)
    term_prob = term_counts / max(float(term_counts.sum()), 1.0)

    transitions = _build_transitions(x3, mode=transition_mode)
    edge_prob = [(s, e, c / float(max(n, 1))) for (s, e), c in transitions.items()]
    selected = [x for x in edge_prob if x[2] > float(edge_min_freq)]
    if not selected:
        selected = edge_prob
    selected.sort(key=lambda z: z[2], reverse=True)
    if int(edge_max_edges) > 0:
        selected = selected[: int(edge_max_edges)]

    sub = gridspec.GridSpecFromSubplotSpec(
        3,
        1,
        subplot_spec=cell_spec,
        # Cap graph panel ratio to avoid over-stretched tall trajectories.
        height_ratios=[0.42, 0.42, min(1.8, max(1.15, 0.85 + 0.045 * f))],
        hspace=0.18,
    )
    ax_cost = fig.add_subplot(sub[0, 0])
    ax_stop = fig.add_subplot(sub[1, 0])
    ax_graph = fig.add_subplot(sub[2, 0])

    # Row 1: longitudinal cost
    ax_cost.bar(np.arange(t), step_costs_long, color="#d65f5f", alpha=0.8, width=0.6)
    ax_cost.set_ylabel("Avg\nLong\nCost", fontsize=8)
    ax_cost.set_xlim(-0.5, t - 0.5)
    ymax = float(np.max(step_costs_long)) if float(np.max(step_costs_long)) > 0 else 1.0
    ax_cost.set_yticks([0.0, ymax])
    ax_cost.tick_params(axis="y", labelsize=7)
    ax_cost.set_xticks([])
    ax_cost.grid(axis="y", linestyle="--", alpha=0.3)
    ax_cost.spines["top"].set_visible(False)
    ax_cost.spines["right"].set_visible(False)
    ax_cost.spines["bottom"].set_visible(False)

    # Row 2: stop probability heat strip
    vmax_val = float(np.max(term_prob) * 1.5) if float(np.max(term_prob)) > 0 else 1.0
    ax_stop.imshow(term_prob.reshape(1, -1), cmap="Blues", aspect="auto", vmin=0.0, vmax=vmax_val)
    ax_stop.set_yticks([])
    ax_stop.set_ylabel("\n\nStop\nProb", fontsize=8)
    ax_stop.set_xticks([])
    for tt in range(t):
        if term_prob[tt] > 0.001:
            val = term_prob[tt] * 100.0
            lbl = f"{val:.1f}"
            if lbl.endswith(".0"):
                lbl = lbl[:-2]
            ax_stop.text(tt, 0, f"{lbl}%", ha="center", va="center", color="black", fontsize=7)

    # Row 3: policy graph
    colors = ap.make_dark_group_colors(f)
    for y in range(f):
        for tt in range(t):
            freq = float(node_freq[tt, y])
            if freq > 0.001:
                ax_graph.scatter(
                    tt,
                    y,
                    s=max(1.8, freq * float(node_size_scale)),
                    color=colors[y],
                    zorder=8,
                )

    for (s, e, prob) in selected:
        ts, gs = int(s[0]), int(s[1])
        td, gd = int(e[0]), int(e[1])
        dist = max(1, td - ts)
        alpha = min(0.48, max(0.10, float(prob) * 8.0))
        arrow = FancyArrowPatch(
            (ts, gs),
            (td, gd),
            connectionstyle=f"arc3,rad=-{0.1 + 0.01 * dist}",
            color="#4a4a4a",
            alpha=alpha,
            linewidth=max(0.24, float(prob) * float(edge_width_scale)),
            arrowstyle="->",
            mutation_scale=5.2,
            zorder=5,
            clip_on=False,
        )
        ax_graph.add_patch(arrow)

    font_size = 10 if f <= 16 else 8
    ax_graph.set_yticks(np.arange(f))
    if int(label_stride) > 1:
        yt = np.arange(f)[:: int(label_stride)]
        ax_graph.set_yticks(yt)
        ax_graph.set_yticklabels([ap.short_text(ordered_names[i], 24) for i in yt], fontsize=font_size, fontweight="bold")
    else:
        ax_graph.set_yticklabels([ap.short_text(x, 24) for x in ordered_names], fontsize=font_size, fontweight="bold")
    ax_graph.set_xticks(np.arange(t))
    ax_graph.set_xticklabels([f"t={i + 1}" for i in range(t)], fontsize=font_size)
    ax_graph.set_ylim(-0.6, f - 0.4)
    ax_graph.set_xlim(-0.5, t - 0.5)
    ax_graph.grid(linestyle=":", alpha=0.3)
    ax_graph.spines["top"].set_visible(False)
    ax_graph.spines["right"].set_visible(False)
    return ax_cost


def _render_group(group_dir: str, out_file_path: str, args):
    meta_path = os.path.join(group_dir, "group_meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Missing group_meta.json: {meta_path}")
    meta = _load_json(meta_path)
    costs = _read_costs_quick(os.path.join(group_dir, "group_costs_quick.tsv"))

    rollouts = {}
    for b in BASELINE_ORDER:
        cmeta = meta["checkpoints"][b]
        rollouts[b] = _load_rollout(cmeta["cache_dir"])

    num_time = int(meta.get("num_time", rollouts["learned"]["num_time"]))
    num_groups = int(meta.get("num_groups", rollouts["learned"]["num_groups"]))
    keep_idx = _shared_keep_idx(meta, rollouts)
    order_orig = _shared_order(meta, keep_idx)
    shared_k = int(args.cluster_k) if int(args.cluster_k) > 0 else int(meta.get("shared_cluster_k", 0))
    if shared_k <= 0:
        shared_k = 1

    max_name_len = 8
    for b in BASELINE_ORDER:
        for nm in rollouts[b]["group_names"]:
            max_name_len = max(max_name_len, len(str(nm)))
    col_gap = _resolve_col_gap(
        args.col_gap,
        cluster_k=shared_k,
        max_name_len=max_name_len,
        num_features=int(len(keep_idx)),
    )
    left_margin = _resolve_left_margin(
        max_name_len=max_name_len,
        row_label_size=int(args.row_label_size),
    )

    fig_size = _resolve_figsize(
        figsize_arg=args.figsize,
        num_time=num_time,
        num_features=int(len(keep_idx)),
        cluster_k=int(shared_k),
    )
    fig = plt.figure(figsize=fig_size, dpi=int(args.dpi))
    fig.patch.set_facecolor("white")
    outer = gridspec.GridSpec(3, 2, figure=fig, wspace=float(col_gap), hspace=float(args.row_gap))
    row_anchor_axes = {}

    for r, b in enumerate(BASELINE_ORDER):
        ro = rollouts[b]
        masks = ro["masks"]
        names = ro["group_names"]
        group_costs = np.asarray(ro["group_costs"], dtype=np.float32)

        # map shared original order -> local kept-order indices
        pos = {int(gid): i for i, gid in enumerate(keep_idx.tolist())}
        local_order = [pos[int(gid)] for gid in order_orig.tolist() if int(gid) in pos]
        if len(local_order) == 0:
            usage = masks.reshape(masks.shape[0], num_time, num_groups)[:, :, keep_idx].mean(axis=(0, 1))
            local_order = np.argsort(-usage).tolist()
        local_order = np.asarray(local_order, dtype=np.int32)
        ordered_names = [names[int(keep_idx[i])] for i in local_order.tolist()]
        ordered_costs = group_costs[keep_idx][local_order]

        centroids, pct = _centroids_from_masks(
            masks_flat=masks,
            num_time=num_time,
            num_groups=num_groups,
            keep_idx=keep_idx,
            cluster_k=shared_k,
            seed=int(args.seed),
        )
        # reorder feature axis inside centroids using local_order
        centroids = centroids[:, :, local_order]

        anchor_ax = _plot_heatmap_cell(
            fig=fig,
            cell_spec=outer[r, 0],
            centroids_tk=centroids,
            pct=pct,
            ordered_names=ordered_names,
            label_stride=max(1, int(args.label_stride)),
            show_colorbar=(r == len(BASELINE_ORDER) - 1),
        )
        row_anchor_axes[b] = anchor_ax
        _plot_trajectory_cell(
            fig=fig,
            cell_spec=outer[r, 1],
            masks_flat=masks,
            num_time=num_time,
            num_groups=num_groups,
            keep_idx=keep_idx,
            local_order=local_order,
            ordered_names=ordered_names,
            ordered_costs=ordered_costs,
            edge_min_freq=float(args.edge_min_freq),
            edge_max_edges=int(args.edge_max_edges),
            node_size_scale=float(args.edge_node_size_scale),
            edge_width_scale=float(args.edge_width_scale),
            transition_mode=str(args.edge_transition_mode),
            label_stride=max(1, int(args.label_stride)),
        )

    total_triplet = _fmt_triplet(costs, "avg_cost")
    acq_triplet = _fmt_triplet(costs, "avg_long_cost")
    fig.suptitle(
        f"{str(args.dataset).lower()} | total cost: {total_triplet} | longitudinal cost: {acq_triplet}",
        fontsize=15,
        y=0.996,
    )
    plt.subplots_adjust(left=float(left_margin), right=0.995, top=0.945, bottom=0.03)

    # Draw row titles in figure coordinates so they do not overlap y-axis feature labels.
    row_label_x = max(0.01, float(left_margin) - 0.075)
    for b in BASELINE_ORDER:
        bb = row_anchor_axes[b].get_position()
        y_mid = 0.5 * (bb.y0 + bb.y1)
        fig.text(
            row_label_x,
            y_mid,
            BASELINE_LABEL[b],
            rotation=90,
            va="center",
            ha="center",
            fontsize=int(args.row_label_size),
            fontweight="bold",
        )

    out_parent = os.path.dirname(out_file_path)
    os.makedirs(out_parent, exist_ok=True)
    fig.savefig(out_file_path, dpi=int(args.dpi))
    plt.close(fig)
    return out_file_path


def main():
    args = parse_args()
    mode_root = os.path.join(args.plots_root, _dataset_dir_name(args.dataset), args.variant, args.mode)
    if not os.path.isdir(mode_root):
        raise FileNotFoundError(f"Mode directory not found: {mode_root}")

    groups = _group_dirs(mode_root, args.group)
    if len(groups) == 0:
        raise FileNotFoundError(f"No groups found in {mode_root} for group={args.group}")

    print("=" * 80)
    print("Render Merged Group Panels From Scratch")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Mode: {args.mode}")
    print(f"Variant: {args.variant}")
    print(f"Groups: {len(groups)}")
    print(f"Output root: {args.merged_root}")

    for g in groups:
        rel = os.path.relpath(g, start=args.plots_root)  # .../<dataset>/<variant>/<mode>/group_xxx
        rel_parts = rel.split(os.sep)
        if len(rel_parts) < 4:
            raise RuntimeError(f"Unexpected group path layout: {g}")
        group_name = rel_parts[-1]
        mode_dir = os.path.join(args.merged_root, *rel_parts[:-1])  # no group subfolder
        if "{group}" in args.output_name:
            fname = args.output_name.replace("{group}", group_name)
        else:
            root, ext = os.path.splitext(args.output_name)
            ext = ext if ext else ".png"
            fname = f"{group_name}_{root}{ext}"
        out_file_path = os.path.join(mode_dir, fname)
        out = _render_group(g, out_file_path=out_file_path, args=args)
        print(f"[OK] {out}")

        ctx_name = f"{group_name}_learned_context.png"
        ctx_path = os.path.join(mode_dir, ctx_name)
        ctx_out = _render_learned_context_plot(
            group_dir=g,
            out_file_path=ctx_path,
            dataset=args.dataset,
        )
        print(f"[OK] {ctx_out}")


if __name__ == "__main__":
    main()
