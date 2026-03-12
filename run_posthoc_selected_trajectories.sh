#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/playpen-nvme/scribble/ddinh/ACTOR_LAFAv1"
PYTHON_BIN="/playpen-nvme/scribble/ddinh/miniconda3/envs/nocta/bin/python"
SCRIPT="${ROOT_DIR}/posthoc_trajectory_plotter.py"

# Fixed posthoc selection requested:
#   ILIADD group 2
#   cheears group 4
#
# Output root:
#   /playpen-nvme/scribble/ddinh/ACTOR_LAFAv1/posthoc
#
# Optional env overrides:
#   MODE=longitudinal
#   VARIANT=matched_cost_groups
#   FEATURE_FILTER_MODE=used
#   NODE_CMAP=BuPu
#   STOP_CMAP=YlOrBr
#   DPI=280
#   BASE_FONTSIZE=18
#   TITLE_FONTSIZE=32
#   EDGE_MIN_FREQ=0.01
#   EDGE_MAX_EDGES=0
#   EDGE_NODE_SIZE_SCALE=540
#   EDGE_WIDTH_SCALE=3.6
#   EDGE_TRANSITION_MODE=next_observed
#   LABEL_STRIDE=1
#   BASELINES_CSV=learned
#   FIG_W=0     # <=0 => auto per dataset/group
#   FIG_H=0     # <=0 => auto per dataset/group
#   NODE_BAR_MODE=per_plot
#   SHARED_NODE_BAR_PATH=/playpen-nvme/scribble/ddinh/ACTOR_LAFAv1/posthoc/shared_node_frequency_bar.png
#   SHARED_NODE_BAR_VMAX=1.0
#   OMP_THREADS=1
#   GPU_ID=0
#   OUTPUT_ROOT=/playpen-nvme/scribble/ddinh/ACTOR_LAFAv1/posthoc
#   LIMIT_ITEMS=0   # 0 = run all, >0 = run first N selections (debug)

MODE="${MODE:-longitudinal}"
VARIANT="${VARIANT:-matched_cost_groups}"
FEATURE_FILTER_MODE="${FEATURE_FILTER_MODE:-used}"
NODE_CMAP="${NODE_CMAP:-BuPu}"
STOP_CMAP="${STOP_CMAP:-YlOrBr}"
DPI="${DPI:-280}"
BASE_FONTSIZE="${BASE_FONTSIZE:-18}"
TITLE_FONTSIZE="${TITLE_FONTSIZE:-32}"
EDGE_MIN_FREQ="${EDGE_MIN_FREQ:-0.01}"
EDGE_MAX_EDGES="${EDGE_MAX_EDGES:-0}"
EDGE_NODE_SIZE_SCALE="${EDGE_NODE_SIZE_SCALE:-540.0}"
EDGE_WIDTH_SCALE="${EDGE_WIDTH_SCALE:-3.6}"
EDGE_TRANSITION_MODE="${EDGE_TRANSITION_MODE:-next_observed}"
LABEL_STRIDE="${LABEL_STRIDE:-1}"
BASELINES_CSV="${BASELINES_CSV:-learned}"
FIG_W="${FIG_W:-0.0}"
FIG_H="${FIG_H:-0.0}"
NODE_BAR_MODE="${NODE_BAR_MODE:-per_plot}"
SHARED_NODE_BAR_PATH="${SHARED_NODE_BAR_PATH:-/playpen-nvme/scribble/ddinh/ACTOR_LAFAv1/posthoc/shared_node_frequency_bar.png}"
SHARED_NODE_BAR_VMAX="${SHARED_NODE_BAR_VMAX:-1.0}"
OMP_THREADS="${OMP_THREADS:-1}"
GPU_ID="${GPU_ID:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/playpen-nvme/scribble/ddinh/ACTOR_LAFAv1/posthoc}"
LIMIT_ITEMS="${LIMIT_ITEMS:-0}"

DATASETS=(ILIADD cheears)
TARGET_GROUPS=(2 4)

echo "=============================================================="
echo "Posthoc Selected Trajectory Rendering"
echo "Mode=${MODE} Variant=${VARIANT}"
echo "Filter=${FEATURE_FILTER_MODE} Node cmap=${NODE_CMAP} Stop cmap=${STOP_CMAP}"
echo "Baselines=${BASELINES_CSV}"
echo "Figure size=${FIG_W}x${FIG_H}"
echo "Node bar mode=${NODE_BAR_MODE} Shared node bar=${SHARED_NODE_BAR_PATH}"
echo "Output root=${OUTPUT_ROOT}"
echo "=============================================================="

count=0
for i in "${!DATASETS[@]}"; do
  if [[ "${LIMIT_ITEMS}" != "0" && "${count}" -ge "${LIMIT_ITEMS}" ]]; then
    break
  fi
  DS="${DATASETS[$i]}"
  GP="${TARGET_GROUPS[$i]}"
  echo
  echo "[run] dataset=${DS} group=${GP}"
  (
    cd "${ROOT_DIR}"
    env \
      KMP_USE_SHM=0 \
      OMP_NUM_THREADS="${OMP_THREADS}" \
      MKL_NUM_THREADS="${OMP_THREADS}" \
      ACTOR_DATASET="${DS}" \
      CUDA_VISIBLE_DEVICES="${GPU_ID}" \
      "${PYTHON_BIN}" "${SCRIPT}" \
        --dataset "${DS}" \
        --group "${GP}" \
        --mode "${MODE}" \
        --variant "${VARIANT}" \
        --output_root "${OUTPUT_ROOT}" \
        --feature_filter_mode "${FEATURE_FILTER_MODE}" \
        --node_cmap "${NODE_CMAP}" \
        --stop_cmap "${STOP_CMAP}" \
        --dpi "${DPI}" \
        --base_fontsize "${BASE_FONTSIZE}" \
        --title_fontsize "${TITLE_FONTSIZE}" \
        --edge_min_freq "${EDGE_MIN_FREQ}" \
        --edge_max_edges "${EDGE_MAX_EDGES}" \
        --edge_node_size_scale "${EDGE_NODE_SIZE_SCALE}" \
        --edge_width_scale "${EDGE_WIDTH_SCALE}" \
        --edge_transition_mode "${EDGE_TRANSITION_MODE}" \
        --label_stride "${LABEL_STRIDE}" \
        --baselines "${BASELINES_CSV}" \
        --fig_w "${FIG_W}" \
        --fig_h "${FIG_H}" \
        --node_bar_mode "${NODE_BAR_MODE}" \
        --shared_node_bar_path "${SHARED_NODE_BAR_PATH}" \
        --shared_node_bar_vmax "${SHARED_NODE_BAR_VMAX}"
  )
  count=$((count + 1))
done

echo
echo "[done] posthoc trajectories generated at ${OUTPUT_ROOT}"
