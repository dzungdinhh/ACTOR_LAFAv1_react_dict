#!/usr/bin/env python3
"""
Posthoc trajectory renderer for selected matched-cost groups.

Design choices for this script:
1) Trajectory panel with cost row + stop row + graph row.
2) Learned-context panel can be emitted as a second PNG for learned baseline.
3) Node-frequency bar can be:
   - in each plot (`per_plot`, default),
   - shared single PNG (`shared`),
   - disabled (`none`).
4) Large bold typography and dataset-only title.
5) Output naming starts with:
   <dataset>_total_<total>_long_<long>
"""

import argparse
import csv
import glob
import json
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyArrowPatch
import numpy as np

import analysis_plots as ap
import analysis_plots_cheears_matched as apm


BASELINE_ORDER = ["none", "learned", "all"]
DATASET_TITLE_LABELS = {
    "iliadd": "(a) ILIADD",
    "cheears": "(b) CHEEARS",
    "cheears_day_context": "(b) CHEEARS",
}


def parse_args():
    p = argparse.ArgumentParser(description="Render posthoc trajectory plots for one dataset/group.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--group", type=int, required=True)
    p.add_argument("--mode", default="longitudinal", choices=["longitudinal", "total"])
    p.add_argument("--variant", default="matched_cost_groups")
    p.add_argument("--plots_root", default="/playpen-nvme/scribble/ddinh/ACTOR_LAFAv1/plots")
    p.add_argument("--output_root", default="/playpen-nvme/scribble/ddinh/ACTOR_LAFAv1/posthoc")
    p.add_argument("--feature_filter_mode", default="used", choices=["used", "all"])
    p.add_argument(
        "--baselines",
        default="none,learned,all",
        help="Comma-separated baselines to render: none,learned,all",
    )
    p.add_argument("--node_cmap", default="BuPu")
    p.add_argument("--stop_cmap", default="YlOrBr")
    p.add_argument("--dpi", type=int, default=280)
    p.add_argument("--edge_min_freq", type=float, default=0.01)
    p.add_argument("--edge_max_edges", type=int, default=0)
    p.add_argument("--edge_node_size_scale", type=float, default=540.0)
    p.add_argument("--edge_width_scale", type=float, default=2.2)
    p.add_argument("--edge_transition_mode", default="next_observed", choices=["strict_next", "next_observed"])
    p.add_argument("--label_stride", type=int, default=1)
    p.add_argument("--title_fontsize", type=int, default=32)
    p.add_argument("--base_fontsize", type=int, default=22)
    p.add_argument(
        "--fig_w",
        type=float,
        default=0.0,
        help="Figure width. <=0 enables auto sizing per dataset/group.",
    )
    p.add_argument(
        "--fig_h",
        type=float,
        default=0.0,
        help="Figure height. <=0 enables auto sizing per dataset/group.",
    )
    p.add_argument(
        "--node_bar_mode",
        default="per_plot",
        choices=["per_plot", "shared", "none"],
        help="Node-frequency colorbar output mode.",
    )
    p.add_argument(
        "--shared_node_bar_path",
        default="",
        help="Path for shared node-frequency bar PNG (used when node_bar_mode=shared).",
    )
    p.add_argument(
        "--shared_node_bar_vmax",
        type=float,
        default=1.0,
        help="Vmax for shared node-frequency bar.",
    )
    return p.parse_args()


def _dataset_dir_name(dataset: str) -> str:
    return dataset.strip().lower()


def _dataset_title_text(dataset: str) -> str:
    ds = str(dataset).strip().lower()
    return DATASET_TITLE_LABELS.get(ds, str(dataset).upper())


def _load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


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
        "aux_gate_binary": z["aux_gate_binary"].astype(np.float32) if "aux_gate_binary" in z else np.zeros((0,), dtype=np.float32),
        "aux_gate_probs": z["aux_gate_probs"].astype(np.float32) if "aux_gate_probs" in z else np.zeros((0,), dtype=np.float32),
        "aux_gate_rates": z["aux_gate_rates"].astype(np.float32) if "aux_gate_rates" in z else np.zeros((0,), dtype=np.float32),
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


def _fmt_cost(x: float) -> str:
    return f"{float(x):.3f}"


def _rounded_pct_labels_sum_100(prob: np.ndarray) -> np.ndarray:
    """
    Convert normalized probabilities into integer percentage labels whose sum is exactly 100.
    Uses largest-remainder rounding so the displayed labels stay close to the true values.
    """
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


def _format_stop_pct_label(pct: int) -> str:
    return str(int(pct))


def _resolve_left_margin(max_name_len: int, font_size: int = 22) -> float:
    # Font-size-aware margin so long feature labels are not clipped.
    base = 0.24
    base += 0.0054 * max(0, int(max_name_len) - 12)
    base += 0.0035 * max(0, int(font_size) - 18)
    return float(min(0.72, max(0.24, base)))


def _nonwhite_cmap(name: str, low_clip: float = 0.18):
    """
    Build a truncated colormap so low values are not near-white.
    Keeps the same named palette style while clipping the very light tail.
    """
    try:
        base = plt.get_cmap(str(name))
    except Exception:
        base = plt.get_cmap("BuPu")
    low = float(min(0.45, max(0.0, low_clip)))
    xs = np.linspace(low, 1.0, 256)
    colors = base(xs)
    return LinearSegmentedColormap.from_list(f"{base.name}_nonwhite", colors)


def _shrink_reposition_per_plot_cbar(ax_cbar, ax_graph):
    """
    Make the per-plot acquisition-rate bar smaller and keep it clear of graph y-labels.
    Requested sizing:
    - height: 3/5 of current
    - width: 4/5 of current graph span
    """
    if ax_cbar is None or ax_graph is None:
        return

    graph_pos = ax_graph.get_position()
    cbar_pos = ax_cbar.get_position()

    width_scale = 0.80
    height_scale = 0.60

    new_w = graph_pos.width * width_scale
    new_h = cbar_pos.height * height_scale

    # Shift slightly right to avoid occasional overlap with y-axis label region.
    new_x0 = graph_pos.x0 + 0.12 * graph_pos.width
    max_x0 = graph_pos.x1 - new_w
    new_x0 = min(max_x0, max(graph_pos.x0, new_x0))

    # Place lower within its own row to create extra gap from trajectory x-ticks.
    # (top margin larger than bottom margin)
    preferred_y0 = cbar_pos.y0 + 0.12 * (cbar_pos.height - new_h)
    min_safe_y0 = 0.085  # avoid clipping cbar label near figure bottom on short figures
    max_y0 = cbar_pos.y1 - new_h

    # If the row is too short to satisfy min_safe_y0, reduce cbar height slightly.
    if max_y0 < min_safe_y0:
        new_h = max(0.45 * cbar_pos.height, cbar_pos.y1 - min_safe_y0)
        new_h = min(new_h, cbar_pos.height)
        max_y0 = cbar_pos.y1 - new_h

    new_y0 = min(max(preferred_y0, min_safe_y0), max_y0)

    ax_cbar.set_position([new_x0, new_y0, new_w, new_h])


def _render_main_trajectory(
    out_path: str,
    dataset: str,
    baseline: str,
    group_idx: int,
    masks_flat: np.ndarray,
    num_time: int,
    num_groups: int,
    order_all: np.ndarray,
    group_names: List[str],
    group_costs: np.ndarray,
    avg_long_cost: float,
    avg_total_cost: float,
    args,
) -> float:
    n = int(masks_flat.shape[0])
    order_all = np.asarray(order_all, dtype=np.int32)
    if order_all.size != int(num_groups):
        order_all = np.arange(int(num_groups), dtype=np.int32)

    x3_all = masks_flat.reshape(n, num_time, num_groups)[:, :, order_all]
    if str(args.feature_filter_mode) == "used":
        usage_all = x3_all.mean(axis=(0, 1))
        keep = np.where(usage_all > 0.0)[0].astype(np.int32)
        if keep.size == 0:
            keep = np.arange(x3_all.shape[2], dtype=np.int32)
    else:
        keep = np.arange(x3_all.shape[2], dtype=np.int32)

    x3 = x3_all[:, :, keep]
    names_all = [group_names[i] for i in order_all.tolist()]
    names = [names_all[i] for i in keep.tolist()]
    costs_all = np.asarray(group_costs, dtype=np.float32).reshape(-1)[order_all]
    costs = costs_all[keep]

    t = int(x3.shape[1])
    f = int(x3.shape[2])

    node_freq = x3.mean(axis=0)
    max_freq = float(np.max(node_freq)) if node_freq.size > 0 else 0.0
    if max_freq <= 0:
        max_freq = 1.0

    step_costs_long = np.mean(x3 * costs.reshape(1, 1, f), axis=0).sum(axis=1)

    last_acq_time = np.zeros(n, dtype=np.int32)
    for i in range(n):
        active_t = np.where(x3[i].sum(axis=1) > 0)[0]
        last_acq_time[i] = int(active_t[-1]) if active_t.size > 0 else 0
    term_counts = np.bincount(last_acq_time, minlength=t).astype(np.float32)
    total_term = float(term_counts.sum())
    if total_term > 0.0:
        term_prob = term_counts / total_term
    else:
        term_prob = np.zeros_like(term_counts, dtype=np.float32)
        if term_prob.size > 0:
            term_prob[0] = 1.0
    # Enforce numerical normalization: sum(term_prob) == 1.
    term_prob = np.clip(term_prob.astype(np.float64), 0.0, None)
    term_sum = float(term_prob.sum())
    if term_sum > 0.0:
        term_prob = term_prob / term_sum
    elif term_prob.size > 0:
        term_prob[:] = 0.0
        term_prob[0] = 1.0
    # Residual correction for exact display consistency.
    if term_prob.size > 0:
        term_prob[-1] = max(0.0, 1.0 - float(np.sum(term_prob[:-1])))
        fix_sum = float(term_prob.sum())
        if fix_sum > 0.0:
            term_prob = term_prob / fix_sum
    term_prob = term_prob.astype(np.float32)
    term_pct_labels = _rounded_pct_labels_sum_100(term_prob)

    transitions = _build_transitions(x3, mode=str(args.edge_transition_mode))
    edge_prob = [(s, e, c / float(max(n, 1))) for (s, e), c in transitions.items()]
    selected = [x for x in edge_prob if x[2] > float(args.edge_min_freq)]
    if not selected:
        selected = edge_prob
    selected.sort(key=lambda z: z[2], reverse=True)
    if int(args.edge_max_edges) > 0:
        selected = selected[: int(args.edge_max_edges)]

    # Auto-size by dataset/group unless explicit figure size is provided.
    cost_row_ratio = 0.30
    stop_row_ratio = 0.26
    if float(args.fig_w) > 0 and float(args.fig_h) > 0:
        fig_w = float(args.fig_w)
        fig_h = float(args.fig_h)
    else:
        # Scale primarily with feature count so low-dim datasets don't look oversized.
        graph_ratio = min(3.2, max(1.2, 0.86 + 0.065 * float(min(f, 40))))
        fig_h = min(13.0, max(6.2, 5.4 + 0.20 * float(min(f, 40))))
        # Keep the actual plot-panel width (graph area) consistent across datasets.
        # Label area can vary by feature-name length; figure width adapts so panel width stays fixed.
        max_name_len = max((len(str(x)) for x in names), default=10)
        graph_tick_fs_est = max(14, int(args.base_fontsize) + 1)
        left_margin_est = _resolve_left_margin(max_name_len=max_name_len, font_size=graph_tick_fs_est)
        right_margin = 0.985
        target_plot_w_in = 8.2
        usable_frac = max(0.18, right_margin - left_margin_est)
        fig_w = target_plot_w_in / usable_frac
        fig_w = min(22.0, max(10.5, fig_w))

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=int(args.dpi))
    if str(args.node_bar_mode) == "per_plot":
        # Put node-frequency bar back BELOW trajectory graph.
        gs = gridspec.GridSpec(
            4, 1, figure=fig,
            height_ratios=[cost_row_ratio, stop_row_ratio, graph_ratio, 0.24],
            hspace=0.10,
        )
        ax_cost = fig.add_subplot(gs[0, 0])
        ax_stop = fig.add_subplot(gs[1, 0])
        ax_graph = fig.add_subplot(gs[2, 0])
        ax_cbar = fig.add_subplot(gs[3, 0])
    else:
        gs = gridspec.GridSpec(
            3, 1, figure=fig,
            height_ratios=[cost_row_ratio, stop_row_ratio, graph_ratio],
            hspace=0.10,
        )
        ax_cost = fig.add_subplot(gs[0, 0])
        ax_stop = fig.add_subplot(gs[1, 0])
        ax_graph = fig.add_subplot(gs[2, 0])
        ax_cbar = None

    base_fs = int(args.base_fontsize)

    ax_cost.bar(np.arange(t), step_costs_long, color="#b54848", alpha=0.88, width=0.62)
    ax_cost.set_ylabel("Avg\nLong\nCost", fontsize=base_fs + 2, fontweight="bold")
    ax_cost.set_xlim(-0.5, t - 0.5)
    ymax = float(np.max(step_costs_long)) if float(np.max(step_costs_long)) > 0 else 1.0
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
    # Fit stop-prob labels to per-cell space to avoid overlap when T is large.
    stop_pos = ax_stop.get_position()
    cell_w_pts = max(8.0, float(fig.get_figwidth()) * 72.0 * float(stop_pos.width) / max(1, t))
    cell_h_pts = max(8.0, float(fig.get_figheight()) * 72.0 * float(stop_pos.height))
    stop_fs = int(max(9.0, min(float(base_fs + 2), 0.62 * cell_w_pts, 0.55 * cell_h_pts)))
    for tt in range(t):
        if int(term_pct_labels[tt]) > 0:
            ax_stop.text(
                tt, 0, _format_stop_pct_label(term_pct_labels[tt]),
                ha="center", va="center",
                color="black", fontsize=stop_fs, fontweight="bold",
            )

    norm = Normalize(vmin=0.0, vmax=1.0)
    node_cmap = _nonwhite_cmap(args.node_cmap, low_clip=0.18)
    for y in range(f):
        for tt in range(t):
            freq = float(node_freq[tt, y])
            if freq > 0.001:
                ax_graph.scatter(
                    tt, y,
                    s=max(27.0, freq * float(args.edge_node_size_scale)),
                    color=node_cmap(norm(freq)),
                    edgecolors="none",
                    zorder=8,
                )

    if ax_cbar is not None:
        sm = ScalarMappable(norm=norm, cmap=node_cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=ax_cbar, orientation="horizontal")
        cbar.set_label("Acquisition rate", fontsize=int(args.base_fontsize) + 2, fontweight="bold")
        cbar.ax.xaxis.set_label_position("bottom")
        cbar.ax.xaxis.labelpad = 1.0
        cbar.ax.tick_params(labelsize=int(args.base_fontsize), width=1.0)
        cbar.set_ticks([0.0, 0.5, 1.0])
        for lbl in cbar.ax.get_xticklabels():
            lbl.set_fontweight("bold")

    # Adaptive edge styling:
    # - Dense edge sets (common in WOMAC/KLG) get lighter/thinner edges to reduce clutter.
    # - Sparse edge sets keep stronger visibility.
    n_selected = int(len(selected))
    max_selected_prob = max((float(p) for (_, _, p) in selected), default=1.0)
    if max_selected_prob <= 0.0:
        max_selected_prob = 1.0
    density = min(1.0, max(0.0, (float(n_selected) - 40.0) / 260.0))
    alpha_scale = 1.0 - 0.55 * density
    width_scale = 1.0 - 0.35 * density
    edge_color = "#4a4a4a" if density >= 0.55 else "#2f2f2f"

    for (s, e, prob) in selected:
        ts, gs_idx = int(s[0]), int(s[1])
        td, gd_idx = int(e[0]), int(e[1])
        dist = max(1, td - ts)
        prob_norm = float(prob) / max_selected_prob
        alpha_base = 0.20 + 0.48 * (prob_norm ** 0.75)
        alpha = min(0.72, max(0.12, alpha_base * alpha_scale))
        linewidth = max(0.75, max(0.85, float(prob) * float(args.edge_width_scale)) * width_scale)
        arrow = FancyArrowPatch(
            (ts, gs_idx),
            (td, gd_idx),
            connectionstyle=f"arc3,rad=-{0.1 + 0.01 * dist}",
            color=edge_color,
            alpha=alpha,
            linewidth=linewidth,
            arrowstyle="->",
            mutation_scale=5.5,
            zorder=5,
            clip_on=False,
        )
        ax_graph.add_patch(arrow)

    # Keep label sizes consistent across datasets.
    font_size = max(14, base_fs + 1)
    ax_graph.set_yticks(np.arange(f))
    if int(args.label_stride) > 1:
        yt = np.arange(f)[:: int(args.label_stride)]
        ax_graph.set_yticks(yt)
        ax_graph.set_yticklabels(
            [ap.short_text(names[i], 28) for i in yt],
            fontsize=font_size, fontweight="bold",
        )
    else:
        ax_graph.set_yticklabels(
            [ap.short_text(x, 28) for x in names],
            fontsize=font_size, fontweight="bold",
        )
    ax_graph.set_xticks(np.arange(t))
    ax_graph.set_xticklabels([str(i + 1) for i in range(t)], fontsize=font_size, fontweight="bold")
    ax_graph.set_ylim(-0.6, f - 0.4)
    ax_graph.set_xlim(-0.5, t - 0.5)
    ax_graph.grid(linestyle=":", alpha=0.75, color="#3f3f3f", linewidth=0.7)
    ax_graph.spines["top"].set_visible(False)
    ax_graph.spines["right"].set_visible(False)

    for lbl in ax_graph.get_xticklabels() + ax_graph.get_yticklabels():
        lbl.set_fontweight("bold")

    left_margin = _resolve_left_margin(
        max_name_len=max((len(str(x)) for x in names), default=10),
        font_size=font_size,
    )
    right_margin = 0.985
    plt.subplots_adjust(left=left_margin, right=right_margin, top=0.92, bottom=0.095)
    # Center dataset title over plot area only (excluding feature-label margin).
    plot_center_x = left_margin + 0.5 * (right_margin - left_margin)
    fig.text(
        plot_center_x, 0.985, _dataset_title_text(dataset),
        ha="center", va="top",
        fontsize=int(args.title_fontsize), fontweight="bold",
    )
    if ax_cbar is not None:
        _shrink_reposition_per_plot_cbar(ax_cbar=ax_cbar, ax_graph=ax_graph)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=int(args.dpi))
    plt.close(fig)
    return max_freq


def _render_separate_cbar(cbar_path: str, cmap_name: str, vmax: float, args):
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 0.95), dpi=int(args.dpi))
    # Fixed node-frequency legend range requested: [0, 1].
    norm = Normalize(vmin=0.0, vmax=1.0)
    cmap = _nonwhite_cmap(cmap_name, low_clip=0.18)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax, orientation="horizontal")
    cbar.set_label("Acquisition rate", fontsize=int(args.base_fontsize) + 2, fontweight="bold")
    cbar.ax.xaxis.set_label_position("bottom")
    cbar.ax.xaxis.labelpad = 1.0
    cbar.ax.tick_params(labelsize=int(args.base_fontsize), width=1.0)
    cbar.set_ticks([0.0, 0.5, 1.0])
    for lbl in cbar.ax.get_xticklabels():
        lbl.set_fontweight("bold")
    os.makedirs(os.path.dirname(cbar_path), exist_ok=True)
    fig.savefig(cbar_path, dpi=int(args.dpi), bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def _render_selected_context(
    out_path: str,
    dataset: str,
    aux_gate_binary: np.ndarray,
    args,
):
    aux_binary = np.asarray(aux_gate_binary, dtype=np.float32).reshape(-1)
    if aux_binary.size == 0:
        return None

    num_aux = int(aux_binary.size)
    aux_names, _ = apm._resolve_aux_names_for_dataset(dataset=dataset, num_aux=num_aux)
    # Fixed context-plot typography so figures remain comparable across datasets.
    tick_fs = 16
    cbar_fs = 17

    # Keep cell sizes consistent across datasets; figure width adapts from aux count.
    cell_in = 0.34
    left_margin = 0.08
    right_margin = 0.86
    bottom_margin = 0.44
    top_margin = 0.90
    fig_w = max(10.5, (cell_in * float(num_aux)) / max(0.12, right_margin - left_margin))
    fig_h = 6.8

    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=int(args.dpi))
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
    ax.set_xticklabels(aux_names, rotation=58, ha="right", fontsize=tick_fs, fontweight="bold")
    ax.set_xticks(np.arange(-0.5, num_aux, 1.0), minor=True)
    ax.set_yticks(np.array([-0.5, 0.5]), minor=True)
    ax.grid(which="minor", color="#1a1a1a", linestyle="-", linewidth=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)
    plt.subplots_adjust(left=left_margin, right=right_margin, bottom=bottom_margin, top=top_margin)

    plot_w = right_margin - left_margin
    cbar_w = min(0.32, 0.42 * plot_w)
    cbar_h = 0.035
    cbar_x = left_margin + 0.5 * (plot_w - cbar_w)
    cbar_y = 0.08
    cax = fig.add_axes([cbar_x, cbar_y, cbar_w, cbar_h])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("Acquisition rate", fontsize=cbar_fs, fontweight="bold")
    cbar.ax.tick_params(labelsize=max(20, cbar_fs - 3), width=1.0)
    cbar.set_ticks([0.0, 0.5, 1.0])
    for lbl in cbar.ax.get_xticklabels():
        lbl.set_fontweight("bold")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=int(args.dpi))
    plt.close(fig)
    return out_path


def _find_group_dir(plots_root: str, dataset: str, variant: str, mode: str, group_idx: int) -> str:
    base = os.path.join(plots_root, _dataset_dir_name(dataset), variant, mode)
    gdir = os.path.join(base, f"group_{int(group_idx):03d}")
    if os.path.isdir(gdir):
        return gdir
    raise FileNotFoundError(f"Missing group directory: {gdir}")


def _output_base_name(dataset: str, total_cost: float, long_cost: float, baseline: str, group_idx: int) -> str:
    # Keep requested core pattern, add baseline/group suffix for uniqueness.
    return (
        f"{dataset.lower()}_total_{_fmt_cost(total_cost)}_long_{_fmt_cost(long_cost)}"
        f"_{baseline}_group{int(group_idx):03d}"
    )


def _parse_baselines_csv(s: str) -> List[str]:
    raw = [x.strip().lower() for x in str(s).split(",")]
    keep = []
    for b in raw:
        if b in BASELINE_ORDER and b not in keep:
            keep.append(b)
    if len(keep) == 0:
        raise ValueError("No valid baselines requested. Use any of: none,learned,all")
    return keep


def main():
    args = parse_args()
    selected_baselines = _parse_baselines_csv(args.baselines)
    group_dir = _find_group_dir(
        plots_root=args.plots_root,
        dataset=args.dataset,
        variant=args.variant,
        mode=args.mode,
        group_idx=args.group,
    )
    meta = _load_json(os.path.join(group_dir, "group_meta.json"))
    costs = _read_costs_quick(os.path.join(group_dir, "group_costs_quick.tsv"))

    num_time = int(meta.get("num_time"))
    num_groups = int(meta.get("num_groups"))
    order_all = _shared_order_all(meta=meta, num_groups=num_groups)

    # Flatten output: write everything directly under output_root.
    out_dir = args.output_root
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("POSTHOC TRAJECTORY PLOTTER")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Group: {args.group}")
    print(f"Mode: {args.mode}")
    print(f"Variant: {args.variant}")
    print(f"Output dir: {out_dir}")
    print(f"Feature filter mode: {args.feature_filter_mode}")
    print(f"Baselines: {','.join(selected_baselines)}")
    print(f"Node cmap: {args.node_cmap} | Stop cmap: {args.stop_cmap}")

    for baseline in selected_baselines:
        ck = meta["checkpoints"][baseline]
        rollout = _load_rollout(ck["cache_dir"])
        rec = costs.get(baseline, {})
        avg_long = float(rec.get("avg_long_cost", float("nan")))
        avg_total = float(rec.get("avg_cost", float("nan")))
        base = _output_base_name(
            dataset=args.dataset,
            total_cost=avg_total,
            long_cost=avg_long,
            baseline=baseline,
            group_idx=args.group,
        )

        main_path = os.path.join(out_dir, f"{base}.png")
        cbar_path = os.path.join(out_dir, f"{base}_node_frequency_bar.png")

        max_freq = _render_main_trajectory(
            out_path=main_path,
            dataset=args.dataset,
            baseline=baseline,
            group_idx=args.group,
            masks_flat=rollout["masks"],
            num_time=rollout["num_time"],
            num_groups=rollout["num_groups"],
            order_all=order_all,
            group_names=rollout["group_names"],
            group_costs=rollout["group_costs"],
            avg_long_cost=avg_long,
            avg_total_cost=avg_total,
            args=args,
        )
        print(f"[OK] {main_path}")
        if baseline == "learned":
            ctx_path = os.path.join(out_dir, f"{base}_selected_context.png")
            rendered_ctx = _render_selected_context(
                out_path=ctx_path,
                dataset=args.dataset,
                aux_gate_binary=rollout.get("aux_gate_binary", np.zeros((0,), dtype=np.float32)),
                args=args,
            )
            if rendered_ctx is not None:
                print(f"[OK] {rendered_ctx}")

    if str(args.node_bar_mode) == "shared":
        shared_path = str(args.shared_node_bar_path).strip()
        if shared_path == "":
            shared_path = os.path.join(out_dir, "shared_node_frequency_bar.png")
        _render_separate_cbar(
            cbar_path=shared_path,
            cmap_name=args.node_cmap,
            vmax=float(args.shared_node_bar_vmax),
            args=args,
        )
        print(f"[OK] {shared_path}")


if __name__ == "__main__":
    main()
