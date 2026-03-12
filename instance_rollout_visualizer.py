#!/usr/bin/env python3
"""
Render top per-instance rollout trajectory plots from matched-cost artifacts.

Compared with group trajectory plots:
1) No edges/arrows.
2) Node size is constant for selected feature-timestep pairs.
3) Two top rows are text-only labels: prediction and ground truth.
4) Select top-K per dataset with emphasis on correct class transitions.
"""

import argparse
import csv
import glob
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import numpy as np

import analysis_plots as ap


def parse_args():
    p = argparse.ArgumentParser(description="Render top instance rollout trajectory plots.")
    p.add_argument(
        "--plots_root",
        default="/playpen-nvme/scribble/ddinh/ACTOR_LAFAv1/plots",
    )
    p.add_argument(
        "--output_root",
        default="/playpen-nvme/scribble/ddinh/ACTOR_LAFAv1/instance_rollouts",
        help="New root output directory; one subfolder per dataset is created.",
    )
    p.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated datasets, or 'all'. Example: adni,womac",
    )
    p.add_argument(
        "--groups",
        default="all",
        help="Comma-separated group names to keep, or 'all'. Example: group_002,group_005",
    )
    p.add_argument("--variant", default="matched_cost_groups")
    p.add_argument("--mode", default="longitudinal", choices=["longitudinal", "total"])
    p.add_argument(
        "--selection_scope",
        type=str,
        default="dataset",
        choices=["dataset", "group"],
        help="Select top-K across the whole dataset or independently within each group.",
    )
    p.add_argument(
        "--baseline",
        default="learned",
        choices=["learned"],
        help="Only learned context is supported for instance visualization.",
    )
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--max_groups", type=int, default=0, help="<=0 means all groups.")
    p.add_argument(
        "--feature_filter_mode",
        type=str,
        default="used",
        choices=["used", "all"],
        help="Feature set for trajectory axis: used (filtered) or all (unfiltered).",
    )
    p.add_argument("--dpi", type=int, default=240)
    p.add_argument(
        "--fig_height_scale",
        type=float,
        default=1.0,
        help="Multiply the resolved figure height by this factor.",
    )
    p.add_argument("--row_gap", type=float, default=0.03)
    p.add_argument("--label_stride", type=int, default=1, help="Show every Nth feature label.")
    p.add_argument("--node_size", type=float, default=470.0)
    p.add_argument("--node_color", type=str, default="#2b6cb0")
    p.add_argument(
        "--feature_panel_style",
        type=str,
        default="scatter",
        choices=["scatter", "heatmap"],
        help="Feature-time panel style: scatter nodes or compact binary heatmap tiles.",
    )
    p.add_argument(
        "--heatmap_on_color",
        type=str,
        default="#1f5fae",
        help="Tile color for acquired cells in heatmap mode.",
    )
    p.add_argument(
        "--heatmap_off_color",
        type=str,
        default="#edf2f7",
        help="Tile color for non-acquired cells in heatmap mode.",
    )
    p.add_argument("--min_unique_classes", type=int, default=2)
    p.add_argument("--min_transitions", type=int, default=1)
    p.add_argument(
        "--require_perfect_prediction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep only instances with 100% prediction accuracy on valid (non-missing) labels.",
    )
    p.add_argument(
        "--enforce_unique_feature_subset",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep at most one candidate per unique selected-feature subset within each dataset.",
    )
    p.add_argument(
        "--require_transition_exact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prioritize exact GT transition matching.",
    )
    p.add_argument(
        "--allow_non_exact_fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If exact-transition candidates are < top_k, backfill with best non-exact examples.",
    )
    p.add_argument(
        "--missing_label_mode",
        type=str,
        default="any",
        choices=["any", "complete", "has_missing"],
        help=(
            "Filter candidate instances by ground-truth label completeness: "
            "'any' keeps all rows, 'complete' keeps only rows with no missing labels, "
            "and 'has_missing' keeps only rows with at least one missing label."
        ),
    )
    return p.parse_args()


def _dataset_dir_name(dataset: str) -> str:
    return dataset.strip().lower()


def _discover_datasets(plots_root: str, variant: str, mode: str) -> List[str]:
    out = []
    for name in sorted(os.listdir(plots_root)):
        if name.startswith(".") or name == "_logs":
            continue
        d = os.path.join(plots_root, name, variant, mode)
        if os.path.isdir(d):
            out.append(name.lower())
    return out


def _parse_datasets_arg(datasets_arg: str, plots_root: str, variant: str, mode: str) -> List[str]:
    raw = [x.strip().lower() for x in str(datasets_arg).split(",") if x.strip()]
    if len(raw) == 0 or (len(raw) == 1 and raw[0] == "all"):
        return _discover_datasets(plots_root=plots_root, variant=variant, mode=mode)
    keep = []
    for x in raw:
        if x not in keep:
            keep.append(x)
    return keep


def _parse_groups_arg(groups_arg: str) -> Optional[set]:
    raw = [x.strip() for x in str(groups_arg).split(",") if x.strip()]
    if len(raw) == 0 or (len(raw) == 1 and raw[0].lower() == "all"):
        return None
    return set(raw)


def _group_dirs(mode_root: str, max_groups: int, groups_arg: str = "all") -> List[str]:
    gs = sorted(glob.glob(os.path.join(mode_root, "group_*")))
    gs = [g for g in gs if os.path.isdir(g)]
    keep_names = _parse_groups_arg(groups_arg)
    if keep_names is not None:
        gs = [g for g in gs if os.path.basename(g) in keep_names]
    if int(max_groups) > 0:
        return gs[: int(max_groups)]
    return gs


def _load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def _as_probabilities(pred: np.ndarray) -> np.ndarray:
    """
    Ensure predictions are normalized probabilities.
    Some caches store logits, some store probabilities.
    """
    x = np.asarray(pred, dtype=np.float32)
    if x.ndim < 2:
        return x
    c = int(x.shape[-1])
    if c <= 1:
        return x

    flat = x.reshape(-1, c)
    if flat.size == 0:
        return x
    finite_rows = flat[np.all(np.isfinite(flat), axis=1)]
    if finite_rows.size == 0:
        return x

    min_v = float(np.min(finite_rows))
    max_v = float(np.max(finite_rows))
    row_err = float(np.mean(np.abs(np.sum(finite_rows, axis=1) - 1.0)))
    looks_like_prob = (min_v >= -1e-4) and (max_v <= 1.0001) and (row_err <= 1e-2)
    if looks_like_prob:
        return x

    z = x - np.max(x, axis=-1, keepdims=True)
    ez = np.exp(z)
    denom = np.sum(ez, axis=-1, keepdims=True)
    denom = np.where(denom <= 0.0, 1.0, denom)
    return ez / denom


def _load_rollout(cache_dir: str):
    path = os.path.join(cache_dir, "analysis_rollout.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing rollout NPZ: {path}")
    z = np.load(path, allow_pickle=True)
    need = ("masks", "predictions", "labels", "num_time", "num_groups", "group_names")
    for k in need:
        if k not in z.files:
            raise KeyError(f"Missing key '{k}' in {path}. Keys={z.files}")
    return {
        "masks": z["masks"].astype(np.float32),  # (N, T*G)
        "predictions": _as_probabilities(z["predictions"].astype(np.float32)),  # (N, T, C)
        "labels": z["labels"].astype(np.int64),  # (N, T)
        "num_time": int(z["num_time"]),
        "num_groups": int(z["num_groups"]),
        "group_names": [str(x) for x in np.asarray(z["group_names"], dtype=object).reshape(-1).tolist()],
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


def _extract_transitions(valid_times: np.ndarray, seq: np.ndarray) -> List[Tuple[int, int, int]]:
    out: List[Tuple[int, int, int]] = []
    if int(seq.size) < 2:
        return out
    prev = int(seq[0])
    for i in range(1, int(seq.size)):
        cur = int(seq[i])
        if cur != prev:
            out.append((int(valid_times[i]), prev, cur))
        prev = cur
    return out


def _transition_jaccard(
    gt: Sequence[Tuple[int, int, int]],
    pred: Sequence[Tuple[int, int, int]],
) -> float:
    a = set(tuple(x) for x in gt)
    b = set(tuple(x) for x in pred)
    if len(a) == 0 and len(b) == 0:
        return 1.0
    u = len(a.union(b))
    if u == 0:
        return 0.0
    return float(len(a.intersection(b))) / float(u)


def _fmt_seq(seq: np.ndarray) -> str:
    return "|".join(str(int(x)) for x in seq.tolist())


def _fmt_transitions(trans: Sequence[Tuple[int, int, int]]) -> str:
    if len(trans) == 0:
        return ""
    return "|".join(f"t{int(t) + 1}:{int(a)}->{int(b)}" for (t, a, b) in trans)


def _resolve_figsize(num_time: int, num_features: int) -> Tuple[float, float]:
    t = max(1, int(num_time))
    f = max(1, int(num_features))
    # Extra-compact canvas to reduce whitespace.
    w = min(10.8, max(6.0, 4.6 + 0.050 * t + 0.017 * min(f, 55)))
    h = min(8.5, max(3.6, 1.9 + 0.10 * min(f, 55)))
    return float(w), float(h)


def _resolve_graph_ratio(num_features: int) -> float:
    f = max(1, int(num_features))
    return float(min(2.1, max(0.75, 0.35 + 0.052 * f)))


def _resolve_left_margin(max_name_len: int) -> float:
    base = 0.11 + 0.0025 * max(0, int(max_name_len) - 12)
    return float(min(0.28, max(0.09, base)))


def _prepare_feature_view(
    masks_flat: np.ndarray,
    num_time: int,
    num_groups: int,
    group_names: List[str],
    feature_filter_mode: str,
) -> Tuple[np.ndarray, List[str]]:
    n = int(masks_flat.shape[0])
    # Keep canonical feature-index order for consistency across visualizations.
    x3_all = masks_flat.reshape(n, num_time, num_groups)
    if str(feature_filter_mode) == "used":
        usage_all = x3_all.mean(axis=(0, 1))
        keep = np.where(usage_all > 0.0)[0].astype(np.int32)
        if keep.size == 0:
            keep = np.arange(x3_all.shape[2], dtype=np.int32)
    else:
        keep = np.arange(x3_all.shape[2], dtype=np.int32)

    x3 = x3_all[:, :, keep]
    names_ord = [group_names[i] for i in keep.tolist()]
    return x3, names_ord


def _align_mask_row_to_feature_names(
    mask_row: np.ndarray,
    source_feature_names: Sequence[str],
    target_feature_names: Sequence[str],
) -> np.ndarray:
    src_names = [str(x) for x in source_feature_names]
    tgt_names = [str(x) for x in target_feature_names]
    if src_names == tgt_names:
        return np.asarray(mask_row, dtype=np.float32)

    src_lookup = {name: idx for idx, name in enumerate(src_names)}
    aligned = np.zeros((int(mask_row.shape[0]), len(tgt_names)), dtype=np.float32)
    for tgt_idx, tgt_name in enumerate(tgt_names):
        src_idx = src_lookup.get(tgt_name)
        if src_idx is not None:
            aligned[:, tgt_idx] = np.asarray(mask_row[:, src_idx], dtype=np.float32)
    return aligned


def _shared_feature_axis_for_selected(
    selected: Sequence[dict],
    group_cache: Dict[str, dict],
    feature_filter_mode: str,
) -> List[str]:
    if len(group_cache) == 0:
        return []

    if len(selected) > 0:
        ref_group_dir = str(selected[0]["group_dir"])
    else:
        ref_group_dir = sorted(group_cache.keys())[0]
    ref_names = [str(x) for x in group_cache[ref_group_dir]["group_names_all"]]

    if str(feature_filter_mode) != "used":
        return ref_names

    keep_mask = np.zeros(len(ref_names), dtype=bool)
    rows = list(selected) if len(selected) > 0 else [{"group_dir": gdir, "instance_index": None} for gdir in sorted(group_cache.keys())]
    for row in rows:
        cache = group_cache[str(row["group_dir"])]
        x3_all = np.asarray(cache["x3_all"], dtype=np.float32)
        src_names = [str(x) for x in cache["group_names_all"]]

        if row.get("instance_index", None) is None:
            used_src = np.any(x3_all > 0.5, axis=(0, 1))
        else:
            inst_idx = int(row["instance_index"])
            used_src = np.any(x3_all[inst_idx] > 0.5, axis=0)

        if src_names == ref_names:
            keep_mask |= used_src
            continue

        src_lookup = {name: idx for idx, name in enumerate(src_names)}
        for ref_idx, ref_name in enumerate(ref_names):
            src_idx = src_lookup.get(ref_name)
            if src_idx is not None and bool(used_src[src_idx]):
                keep_mask[ref_idx] = True

    keep_idx = np.where(keep_mask)[0].astype(np.int32)
    if keep_idx.size == 0:
        return ref_names
    return [ref_names[i] for i in keep_idx.tolist()]


def _class_color_table(max_class: int) -> Dict[int, str]:
    n = max(3, int(max_class) + 1)
    # Keep class text colors distinct from node color (default node is blue).
    palette = [
        "#b23a48",
        "#8c5e34",
        "#4f772d",
        "#8d5a97",
        "#d17b0f",
        "#6b3a5b",
        "#4c6a2a",
        "#a14322",
        "#7a4eab",
        "#9c2f2f",
    ]
    out = {}
    for i in range(n):
        out[int(i)] = palette[i % len(palette)]
    return out


def _draw_label_row(
    ax,
    row_values: np.ndarray,
    row_name: str,
    class_colors: Dict[int, str],
):
    t = int(row_values.size)
    ax.set_xlim(-0.5, t - 0.5)
    ax.set_ylim(-0.6, 0.6)
    ax.add_patch(
        Rectangle(
            (-0.5, -0.6),
            float(t),
            1.2,
            facecolor="white",
            edgecolor="none",
            zorder=0,
        )
    )
    ax.grid(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False, length=0)
    ax.text(
        -0.012,
        0.5,
        row_name,
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=18,
        fontweight="bold",
    )

    for tt in range(t):
        v = int(row_values[tt])
        if v < 0:
            text = "-"
            color = "#9aa0a6"
        else:
            text = str(v)
            color = class_colors.get(v, "#111111")
        ax.text(
            tt,
            0.0,
            text,
            ha="center",
            va="center",
            fontsize=22,
            color=color,
            fontweight="bold",
            zorder=3,
        )

    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)


def _render_instance_plot(
    out_path: str,
    dataset: str,
    mode: str,
    baseline: str,
    group_name: str,
    instance_index: int,
    labels_row: np.ndarray,
    preds_row: np.ndarray,
    mask_row: np.ndarray,
    feature_names: List[str],
    seq_acc: float,
    num_gt_transitions: int,
    unique_gt_classes: int,
    args,
):
    t = int(mask_row.shape[0])
    f = int(mask_row.shape[1])
    max_class = int(max(np.max(preds_row), np.max(labels_row[labels_row >= 0]) if np.any(labels_row >= 0) else 0))
    class_colors = _class_color_table(max_class=max_class)

    fig_w, fig_h = _resolve_figsize(num_time=t, num_features=f)
    fig_h = float(fig_h) * max(0.25, float(getattr(args, "fig_height_scale", 1.0)))
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=int(args.dpi))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(
        3,
        1,
        figure=fig,
        height_ratios=[0.065, 0.065, _resolve_graph_ratio(f)],
        hspace=float(args.row_gap),
    )
    ax_pred = fig.add_subplot(gs[0, 0])
    ax_gt = fig.add_subplot(gs[1, 0])
    ax_graph = fig.add_subplot(gs[2, 0])
    ax_graph.set_zorder(1)
    ax_pred.set_zorder(3)
    ax_gt.set_zorder(3)
    ax_pred.patch.set_facecolor("white")
    ax_gt.patch.set_facecolor("white")
    ax_pred.patch.set_alpha(1.0)
    ax_gt.patch.set_alpha(1.0)

    pred_row_plot = preds_row.astype(np.int32)
    gt_row_plot = labels_row.astype(np.int32)
    _draw_label_row(ax_pred, pred_row_plot, "Prediction:", class_colors=class_colors)
    _draw_label_row(ax_gt, gt_row_plot, "Ground Truth:", class_colors=class_colors)

    style = str(getattr(args, "feature_panel_style", "scatter")).strip().lower()
    if style == "heatmap":
        tile = (mask_row > 0.5).astype(np.int32).T  # (F, T)
        cmap = ListedColormap([str(args.heatmap_off_color), str(args.heatmap_on_color)])
        ax_graph.imshow(
            tile,
            cmap=cmap,
            interpolation="nearest",
            aspect="auto",
            origin="lower",
            vmin=0.0,
            vmax=1.0,
            extent=(-0.5, t - 0.5, -0.5, f - 0.5),
            zorder=1,
        )
        # Explicit tile grid in heatmap mode.
        ax_graph.vlines(
            np.arange(-0.5, t, 1.0),
            -0.5,
            f - 0.5,
            colors="#c7d0dc",
            linewidth=0.65,
            alpha=0.95,
            zorder=2,
            clip_on=True,
        )
        ax_graph.hlines(
            np.arange(-0.5, f, 1.0),
            -0.5,
            t - 0.5,
            colors="#c7d0dc",
            linewidth=0.65,
            alpha=0.95,
            zorder=2,
            clip_on=True,
        )
    else:
        for y in range(f):
            for tt in range(t):
                if float(mask_row[tt, y]) > 0.5:
                    ax_graph.scatter(
                        tt,
                        y,
                        s=float(args.node_size),
                        color=str(args.node_color),
                        edgecolors="none",
                        zorder=8,
                    )

    # Larger fonts for feature-time panel text.
    if f <= 16:
        font_size = 19
    elif f <= 30:
        font_size = 18
    else:
        font_size = 17

    ax_graph.set_yticks(np.arange(f))
    if int(args.label_stride) > 1:
        yt = np.arange(f)[:: int(args.label_stride)]
        ax_graph.set_yticks(yt)
        ax_graph.set_yticklabels([ap.short_text(feature_names[i], 28) for i in yt], fontsize=font_size, fontweight="bold")
    else:
        ax_graph.set_yticklabels([ap.short_text(x, 28) for x in feature_names], fontsize=font_size, fontweight="bold")
    ax_graph.set_xticks(np.arange(t))
    ax_graph.set_xticklabels([str(i + 1) for i in range(t)], fontsize=font_size)
    y_min, y_max = -0.6, f - 0.4
    x_min, x_max = -0.5, t - 0.5
    ax_graph.set_ylim(y_min, y_max)
    ax_graph.set_xlim(x_min, x_max)
    ax_graph.set_xlabel("Time Index", fontsize=font_size + 1, fontweight="bold", labelpad=6)
    # Draw graph-only grid explicitly so it never appears in label rows.
    if style == "heatmap":
        ax_graph.hlines(
            np.arange(f),
            x_min,
            x_max,
            colors="#8a8f98",
            linestyles=":",
            linewidth=0.55,
            alpha=0.45,
            zorder=0,
            clip_on=True,
        )
    else:
        ax_graph.vlines(
            np.arange(t),
            y_min,
            y_max,
            colors="#3f3f3f",
            linestyles=":",
            linewidth=0.65,
            alpha=0.42,
            zorder=0,
            clip_on=True,
        )
        ax_graph.hlines(
            np.arange(f),
            x_min,
            x_max,
            colors="#3f3f3f",
            linestyles=":",
            linewidth=0.65,
            alpha=0.65,
            zorder=0,
            clip_on=True,
        )
    ax_graph.spines["top"].set_visible(False)
    ax_graph.spines["right"].set_visible(False)

    max_name_len = max((len(str(x)) for x in feature_names), default=10)
    left_margin = _resolve_left_margin(max_name_len=max_name_len)
    plt.subplots_adjust(left=float(left_margin), right=0.998, top=0.998, bottom=0.06)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=int(args.dpi), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _candidate_sort_key(c: dict):
    def _safe_float(x: float) -> float:
        v = float(x)
        return v if np.isfinite(v) else float("-inf")

    return (
        int(c["transition_exact"]),
        _safe_float(c["seq_acc"]),
        _safe_float(c["transition_jaccard"]),
        int(c["num_gt_transitions"]),
        int(c["unique_gt_classes"]),
        _safe_float(c["avg_true_prob"]),
        _safe_float(c["avg_pred_prob"]),
    )


def _write_csv(path: str, rows: List[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if len(rows) == 0:
        with open(path, "w", newline="") as f:
            f.write("")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _safe_name(s: str) -> str:
    keep = []
    for ch in str(s):
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


def _subset_key_from_indices(idxs: Sequence[int]) -> str:
    if len(idxs) == 0:
        return ""
    return "|".join(str(int(x)) for x in idxs)


def _choose_candidate_for_same_idx(items: List[dict]) -> dict:
    """
    Dataset-specific tie-break for duplicate instance indices:
      - adni / iliadd: choose highest number of selected features
      - womac: choose medium (closest to median) number of selected features
      - others: choose lowest number of selected features
    For ties, choose best rollout-quality ranking.
    """
    if len(items) == 1:
        return items[0]

    ds = str(items[0].get("dataset", "")).strip().lower()
    feats = np.asarray([int(c.get("instance_feature_count", 10**9)) for c in items], dtype=np.float32)

    if ds in {"adni", "iliadd"}:
        target = float(np.max(feats))
        pool = [c for c in items if int(c.get("instance_feature_count", 10**9)) == int(target)]
        return max(pool, key=_candidate_sort_key)

    if ds == "womac":
        med = float(np.median(feats))
        dists = [abs(float(int(c.get("instance_feature_count", 10**9))) - med) for c in items]
        min_dist = min(dists)
        pool = [c for c, d in zip(items, dists) if abs(d - min_dist) <= 1e-8]
        return max(pool, key=_candidate_sort_key)

    target = float(np.min(feats))
    pool = [c for c in items if int(c.get("instance_feature_count", 10**9)) == int(target)]
    return max(pool, key=_candidate_sort_key)


def _dedupe_by_instance_index(candidates: List[dict]) -> List[dict]:
    grouped: Dict[int, List[dict]] = {}
    for c in candidates:
        idx = int(c["instance_index"])
        grouped.setdefault(idx, []).append(c)
    out = [_choose_candidate_for_same_idx(v) for v in grouped.values()]
    out.sort(key=_candidate_sort_key, reverse=True)
    return out


def _dedupe_by_feature_subset(candidates: List[dict]) -> List[dict]:
    grouped: Dict[str, List[dict]] = {}
    for c in candidates:
        key = str(c.get("feature_subset_key", ""))
        grouped.setdefault(key, []).append(c)
    out = [max(v, key=_candidate_sort_key) for v in grouped.values()]
    out.sort(key=_candidate_sort_key, reverse=True)
    return out


def _select_candidates(candidates: List[dict], args) -> Tuple[List[dict], List[dict], List[dict]]:
    candidates_sorted = sorted(candidates, key=_candidate_sort_key, reverse=True)
    candidates_unique_idx = _dedupe_by_instance_index(candidates_sorted)
    if bool(args.enforce_unique_feature_subset):
        candidates_unique_idx = _dedupe_by_feature_subset(candidates_unique_idx)
    exact = [c for c in candidates_unique_idx if int(c["transition_exact"]) == 1]

    selected: List[dict] = []
    if bool(args.require_transition_exact):
        selected.extend(exact[: int(args.top_k)])
        if len(selected) < int(args.top_k) and bool(args.allow_non_exact_fallback):
            used_idx = set((str(x["group_dir"]), int(x["instance_index"])) for x in selected)
            for c in candidates_unique_idx:
                key = (str(c["group_dir"]), int(c["instance_index"]))
                if key in used_idx:
                    continue
                selected.append(c)
                used_idx.add(key)
                if len(selected) >= int(args.top_k):
                    break
    else:
        selected = candidates_unique_idx[: int(args.top_k)]
    return candidates_sorted, exact, selected


def _write_selection_outputs(
    dataset: str,
    group_cache: Dict[str, dict],
    candidates: List[dict],
    selected: List[dict],
    out_dir: str,
    args,
):
    shared_feature_names = _shared_feature_axis_for_selected(
        selected=selected,
        group_cache=group_cache,
        feature_filter_mode=str(args.feature_filter_mode),
    )

    os.makedirs(out_dir, exist_ok=True)

    summary_rows: List[dict] = []
    manifest = []

    for rank, c in enumerate(selected, start=1):
        gdir = c["group_dir"]
        cache = group_cache[gdir]
        ro = cache["rollout"]
        x3_all = cache["x3_all"]  # (N, T, G)
        idx = int(c["instance_index"])

        labels_row = ro["labels"][idx].astype(np.int64)
        preds_row = np.argmax(ro["predictions"][idx], axis=-1).astype(np.int64)
        mask_row = _align_mask_row_to_feature_names(
            mask_row=x3_all[idx],
            source_feature_names=cache["group_names_all"],
            target_feature_names=shared_feature_names,
        )

        fname = (
            f"{dataset.lower()}_rank_{rank:02d}_{_safe_name(c['group_name'])}_"
            f"idx_{int(idx):05d}_acc_{float(c['seq_acc']):.3f}_"
            f"trans_{int(c['num_gt_transitions'])}.png"
        )
        out_png = os.path.join(out_dir, fname)

        _render_instance_plot(
            out_path=out_png,
            dataset=dataset,
            mode=args.mode,
            baseline=args.baseline,
            group_name=str(c["group_name"]),
            instance_index=idx,
            labels_row=labels_row,
            preds_row=preds_row,
            mask_row=mask_row,
            feature_names=shared_feature_names,
            seq_acc=float(c["seq_acc"]),
            num_gt_transitions=int(c["num_gt_transitions"]),
            unique_gt_classes=int(c["unique_gt_classes"]),
            args=args,
        )

        row = dict(c)
        row["rank"] = int(rank)
        row["output_png"] = out_png
        summary_rows.append(row)
        manifest.append(
            {
                "rank": int(rank),
                "group_name": str(c["group_name"]),
                "instance_index": int(idx),
                "seq_acc": float(c["seq_acc"]),
                "transition_exact": int(c["transition_exact"]),
                "num_gt_transitions": int(c["num_gt_transitions"]),
                "unique_gt_classes": int(c["unique_gt_classes"]),
                "output_png": out_png,
                "gt_seq_valid": str(c["gt_seq_valid"]),
                "pred_seq_valid": str(c["pred_seq_valid"]),
                "gt_transitions": str(c["gt_transitions"]),
                "pred_transitions": str(c["pred_transitions"]),
            }
        )
        print(f"[OK] {out_png}")

    pool_csv = os.path.join(out_dir, "candidate_pool.csv")
    top_csv = os.path.join(out_dir, "top_selected.csv")
    manifest_json = os.path.join(out_dir, "top_selected_manifest.json")
    _write_csv(pool_csv, candidates)
    _write_csv(top_csv, summary_rows)
    with open(manifest_json, "w") as f:
        json.dump(manifest, f, indent=2)


def _process_dataset(dataset: str, args) -> Dict[str, int]:
    dataset = _dataset_dir_name(dataset)
    mode_root = os.path.join(args.plots_root, dataset, args.variant, args.mode)
    if not os.path.isdir(mode_root):
        print(f"[SKIP] Missing mode directory: {mode_root}")
        return {"dataset": dataset, "groups": 0, "candidates": 0, "selected": 0}

    group_dirs = _group_dirs(mode_root=mode_root, max_groups=int(args.max_groups), groups_arg=str(args.groups))
    if len(group_dirs) == 0:
        print(f"[SKIP] No groups found for {dataset} at {mode_root}")
        return {"dataset": dataset, "groups": 0, "candidates": 0, "selected": 0}

    group_cache: Dict[str, dict] = {}
    candidates: List[dict] = []

    for gdir in group_dirs:
        gname = os.path.basename(gdir)
        meta_path = os.path.join(gdir, "group_meta.json")
        if not os.path.isfile(meta_path):
            print(f"[WARN] Missing group_meta.json: {gdir}")
            continue

        try:
            meta = _load_json(meta_path)
            if "checkpoints" not in meta or args.baseline not in meta["checkpoints"]:
                print(f"[WARN] Missing baseline '{args.baseline}' in {meta_path}")
                continue
            cache_dir = str(meta["checkpoints"][args.baseline]["cache_dir"])
            ro = _load_rollout(cache_dir=cache_dir)
        except Exception as e:
            print(f"[WARN] Failed to load group {gname}: {e}")
            continue

        num_time = int(ro["num_time"])
        num_groups = int(ro["num_groups"])
        x3_all = ro["masks"].reshape(-1, num_time, num_groups).astype(np.float32)
        group_cache[gdir] = {
            "meta": meta,
            "rollout": ro,
            "x3_all": x3_all,
            "group_names_all": [str(x) for x in ro["group_names"]],
        }

        preds_prob = ro["predictions"]  # (N, T, C)
        labels = ro["labels"]  # (N, T)
        preds_cls = np.argmax(preds_prob, axis=-1).astype(np.int64)  # (N, T)
        n = int(labels.shape[0])
        t = int(labels.shape[1])
        if preds_cls.shape[0] != n or preds_cls.shape[1] != t:
            print(f"[WARN] Shape mismatch in {gname}: preds={preds_cls.shape} labels={labels.shape}")
            continue
        mask3d_full = ro["masks"].reshape(n, num_time, num_groups)

        for i in range(n):
            y = labels[i]
            p = preds_cls[i]
            valid = np.where(y >= 0)[0].astype(np.int32)
            missing_steps = int(np.sum(y < 0))
            has_missing_labels = int(missing_steps > 0)
            if str(args.missing_label_mode) == "complete" and has_missing_labels:
                continue
            if str(args.missing_label_mode) == "has_missing" and (not has_missing_labels):
                continue
            if valid.size < 1:
                continue

            y_valid = y[valid].astype(np.int64)
            p_valid = p[valid].astype(np.int64)
            unique_gt = int(np.unique(y_valid).size)
            if unique_gt < int(args.min_unique_classes):
                continue

            gt_trans = _extract_transitions(valid_times=valid, seq=y_valid)
            if len(gt_trans) < int(args.min_transitions):
                continue
            pred_trans = _extract_transitions(valid_times=valid, seq=p_valid)

            transition_exact = int(gt_trans == pred_trans)
            seq_acc = float(np.mean(p_valid == y_valid))
            if bool(args.require_perfect_prediction) and (seq_acc < 1.0 - 1e-12):
                continue
            trans_j = float(_transition_jaccard(gt=gt_trans, pred=pred_trans))

            probs_i = preds_prob[i]
            try:
                true_prob = probs_i[valid, y_valid]
                pred_prob = probs_i[valid, p_valid]
                avg_true_prob = float(np.mean(true_prob))
                avg_pred_prob = float(np.mean(pred_prob))
            except Exception:
                avg_true_prob = float("nan")
                avg_pred_prob = float("nan")
            instance_feature_count = int(np.sum(np.any(mask3d_full[i] > 0.5, axis=0)))
            feature_subset_idx = np.where(np.any(mask3d_full[i] > 0.5, axis=0))[0].astype(np.int32).tolist()
            feature_subset_key = _subset_key_from_indices(feature_subset_idx)

            candidates.append(
                {
                    "dataset": dataset,
                    "group_name": gname,
                    "group_dir": gdir,
                    "group_index": int(meta.get("group_index", -1)),
                    "instance_index": int(i),
                    "total_steps": int(t),
                    "valid_steps": int(valid.size),
                    "missing_steps": int(missing_steps),
                    "has_missing_labels": int(has_missing_labels),
                    "unique_gt_classes": int(unique_gt),
                    "num_gt_transitions": int(len(gt_trans)),
                    "num_pred_transitions": int(len(pred_trans)),
                    "transition_exact": int(transition_exact),
                    "transition_jaccard": float(trans_j),
                    "seq_acc": float(seq_acc),
                    "avg_true_prob": float(avg_true_prob),
                    "avg_pred_prob": float(avg_pred_prob),
                    "instance_feature_count": int(instance_feature_count),
                    "feature_subset_key": str(feature_subset_key),
                    "gt_seq_valid": _fmt_seq(y_valid),
                    "pred_seq_valid": _fmt_seq(p_valid),
                    "gt_transitions": _fmt_transitions(gt_trans),
                    "pred_transitions": _fmt_transitions(pred_trans),
                }
            )

    if str(args.selection_scope) == "group":
        dataset_out_dir = os.path.join(args.output_root, dataset)
        os.makedirs(dataset_out_dir, exist_ok=True)

        group_summary_rows: List[dict] = []
        total_selected = 0
        for gdir in sorted(group_cache.keys()):
            gname = os.path.basename(gdir)
            group_candidates = [c for c in candidates if str(c["group_dir"]) == str(gdir)]
            group_candidates_sorted, exact, selected = _select_candidates(group_candidates, args=args)
            group_out_dir = os.path.join(dataset_out_dir, gname)
            _write_selection_outputs(
                dataset=dataset,
                group_cache={gdir: group_cache[gdir]},
                candidates=group_candidates_sorted,
                selected=selected,
                out_dir=group_out_dir,
                args=args,
            )
            total_selected += len(selected)
            group_summary_rows.append(
                {
                    "dataset": dataset,
                    "group_name": gname,
                    "group_index": int(group_cache[gdir]["meta"].get("group_index", -1)),
                    "candidates": int(len(group_candidates_sorted)),
                    "exact": int(len(exact)),
                    "selected": int(len(selected)),
                    "output_dir": group_out_dir,
                }
            )
            print(
                f"[DONE] dataset={dataset} group={gname} "
                f"candidates={len(group_candidates_sorted)} exact={len(exact)} "
                f"selected={len(selected)} dir={group_out_dir}"
            )

        _write_csv(os.path.join(dataset_out_dir, "group_summary.csv"), group_summary_rows)
        return {
            "dataset": dataset,
            "groups": len(group_cache),
            "candidates": len(candidates),
            "selected": total_selected,
        }

    candidates_sorted, exact, selected = _select_candidates(candidates, args=args)
    out_dir = os.path.join(args.output_root, dataset)
    _write_selection_outputs(
        dataset=dataset,
        group_cache=group_cache,
        candidates=candidates_sorted,
        selected=selected,
        out_dir=out_dir,
        args=args,
    )

    print(
        f"[DONE] dataset={dataset} groups={len(group_cache)} "
        f"candidates={len(candidates_sorted)} exact={len(exact)} selected={len(selected)} "
        f"dir={out_dir}"
    )
    return {
        "dataset": dataset,
        "groups": len(group_cache),
        "candidates": len(candidates_sorted),
        "selected": len(selected),
    }


def main():
    args = parse_args()
    datasets = _parse_datasets_arg(
        datasets_arg=args.datasets,
        plots_root=args.plots_root,
        variant=args.variant,
        mode=args.mode,
    )
    if len(datasets) == 0:
        raise FileNotFoundError("No datasets found to process.")

    print("=" * 90)
    print("INSTANCE ROLLOUT VISUALIZER")
    print("=" * 90)
    print(f"Datasets: {','.join(datasets)}")
    print(f"Variant: {args.variant}")
    print(f"Mode: {args.mode}")
    print(f"Baseline: {args.baseline}")
    print(f"Selection scope: {args.selection_scope}")
    print(f"Top-K per {args.selection_scope}: {int(args.top_k)}")
    print(f"Groups: {args.groups}")
    print(f"Feature filter mode: {args.feature_filter_mode}")
    print(f"Feature panel style: {args.feature_panel_style}")
    print(f"Figure height scale: {args.fig_height_scale}")
    print(f"Missing label mode: {args.missing_label_mode}")
    print(f"Require perfect prediction: {bool(args.require_perfect_prediction)}")
    print(f"Enforce unique feature subset: {bool(args.enforce_unique_feature_subset)}")
    print(f"Output root: {args.output_root}")

    stats = []
    for ds in datasets:
        stats.append(_process_dataset(dataset=ds, args=args))

    print("-" * 90)
    print("SUMMARY")
    for s in stats:
        print(
            f"  {s['dataset']}: groups={s['groups']} "
            f"candidates={s['candidates']} selected={s['selected']}"
        )


if __name__ == "__main__":
    main()
