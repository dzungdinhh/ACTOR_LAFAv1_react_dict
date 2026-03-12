#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/playpen-nvme/scribble/ddinh/ACTOR_LAFAv1"
PYTHON_BIN="/playpen-nvme/scribble/ddinh/miniconda3/envs/nocta/bin/python"
PLOT_SCRIPT="${ROOT_DIR}/merge_group_trajectory_plots.py"

# Usage:
#   ./run_trajectory_only_longitudinal.sh                  # run all datasets
#   ./run_trajectory_only_longitudinal.sh womac            # single dataset
# Optional env:
#   DATASETS_CSV="cheears,womac,klg,adni,ILIADD"
#   GROUP=1
#   VARIANTS_CSV="matched_cost_groups,matched_cost_groups_k1"
#   FEATURE_FILTER_MODES_CSV="used,all"
#   MERGED_ROOT="/playpen-nvme/scribble/ddinh/ACTOR_LAFAv1/plots_merged"
#   NODE_CMAP="Blues"
#   STOP_CMAP="YlOrBr"
#   DPI=240
#   EDGE_MIN_FREQ=0.01
#   EDGE_MAX_EDGES=0
#   EDGE_NODE_SIZE_SCALE=120
#   EDGE_WIDTH_SCALE=2.2
#   EDGE_TRANSITION_MODE=next_observed
#   LABEL_STRIDE=1
#   OMP_THREADS=1
#   GPU_ID=0

DATASETS=(cheears womac klg adni ILIADD)
if [[ -n "${DATASETS_CSV:-}" ]]; then
  DATASETS=()
  IFS=',' read -r -a _RAW_DATASETS <<< "${DATASETS_CSV}"
  for _ds in "${_RAW_DATASETS[@]}"; do
    _ds="${_ds//[[:space:]]/}"
    [[ -n "${_ds}" ]] && DATASETS+=("${_ds}")
  done
fi
if [[ $# -gt 0 && -n "${1:-}" ]]; then
  DATASETS=("$1")
fi

MODE="longitudinal"
GROUP="${GROUP:-}"
VARIANTS_CSV="${VARIANTS_CSV:-matched_cost_groups,matched_cost_groups_k1}"
FEATURE_FILTER_MODES_CSV="${FEATURE_FILTER_MODES_CSV:-used,all}"
MERGED_ROOT="${MERGED_ROOT:-/playpen-nvme/scribble/ddinh/ACTOR_LAFAv1/plots_merged}"
NODE_CMAP="${NODE_CMAP:-Blues}"
STOP_CMAP="${STOP_CMAP:-YlOrBr}"
DPI="${DPI:-240}"
EDGE_MIN_FREQ="${EDGE_MIN_FREQ:-0.01}"
EDGE_MAX_EDGES="${EDGE_MAX_EDGES:-0}"
EDGE_NODE_SIZE_SCALE="${EDGE_NODE_SIZE_SCALE:-120.0}"
EDGE_WIDTH_SCALE="${EDGE_WIDTH_SCALE:-2.2}"
EDGE_TRANSITION_MODE="${EDGE_TRANSITION_MODE:-next_observed}"
LABEL_STRIDE="${LABEL_STRIDE:-1}"
OMP_THREADS="${OMP_THREADS:-1}"
GPU_ID="${GPU_ID:-0}"

IFS=',' read -r -a VARIANTS <<< "${VARIANTS_CSV}"
IFS=',' read -r -a FEATURE_FILTER_MODES <<< "${FEATURE_FILTER_MODES_CSV}"

echo "=============================================================="
echo "Trajectory-only plotting (longitudinal mode)"
echo "Datasets: ${DATASETS[*]}"
echo "Mode: ${MODE}"
echo "Variants: ${VARIANTS_CSV}"
echo "Feature filter modes: ${FEATURE_FILTER_MODES_CSV}"
echo "Merged root: ${MERGED_ROOT}"
echo "Node cmap: ${NODE_CMAP} | Stop cmap: ${STOP_CMAP}"
echo "Node size scale (frequency): ${EDGE_NODE_SIZE_SCALE} | Edge width scale: ${EDGE_WIDTH_SCALE}"
if [[ -n "${GROUP}" ]]; then
  echo "Group: ${GROUP}"
else
  echo "Group: all"
fi
echo "=============================================================="

for DATASET in "${DATASETS[@]}"; do
  echo
  echo "-------------------- dataset=${DATASET} --------------------"
  for VARIANT in "${VARIANTS[@]}"; do
    VARIANT="$(echo "${VARIANT}" | xargs)"
    [[ -z "${VARIANT}" ]] && continue

    MODE_DIR="${ROOT_DIR}/plots/${DATASET,,}/${VARIANT}/${MODE}"
    if [[ ! -d "${MODE_DIR}" ]]; then
      echo "[skip] missing mode directory: ${MODE_DIR}"
      continue
    fi

    for FEATURE_FILTER_MODE in "${FEATURE_FILTER_MODES[@]}"; do
      FEATURE_FILTER_MODE="$(echo "${FEATURE_FILTER_MODE}" | xargs)"
      [[ -z "${FEATURE_FILTER_MODE}" ]] && continue

      CMD=(
        "${PYTHON_BIN}" "${PLOT_SCRIPT}"
        --dataset "${DATASET}"
        --mode "${MODE}"
        --variant "${VARIANT}"
        --merged_root "${MERGED_ROOT}"
        --feature_filter_mode "${FEATURE_FILTER_MODE}"
        --node_cmap "${NODE_CMAP}"
        --stop_cmap "${STOP_CMAP}"
        --dpi "${DPI}"
        --edge_min_freq "${EDGE_MIN_FREQ}"
        --edge_max_edges "${EDGE_MAX_EDGES}"
        --edge_node_size_scale "${EDGE_NODE_SIZE_SCALE}"
        --edge_width_scale "${EDGE_WIDTH_SCALE}"
        --edge_transition_mode "${EDGE_TRANSITION_MODE}"
        --label_stride "${LABEL_STRIDE}"
      )
      if [[ -n "${GROUP}" ]]; then
        CMD+=(--group "${GROUP}")
      fi

      echo "[run] dataset=${DATASET} variant=${VARIANT} feature_filter_mode=${FEATURE_FILTER_MODE}"
      (
        cd "${ROOT_DIR}"
        env \
          KMP_USE_SHM=0 \
          OMP_NUM_THREADS="${OMP_THREADS}" \
          MKL_NUM_THREADS="${OMP_THREADS}" \
          ACTOR_DATASET="${DATASET}" \
          CUDA_VISIBLE_DEVICES="${GPU_ID}" \
          "${CMD[@]}"
      )
    done
  done
done

echo "[done] trajectory-only longitudinal plotting finished."
