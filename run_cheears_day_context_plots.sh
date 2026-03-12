#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/playpen-nvme/scribble/ddinh/ACTOR_LAFAv1"
PYTHON_BIN="/playpen-nvme/scribble/ddinh/miniconda3/envs/nocta/bin/python"
DATASET="cheears_day_context"
TEST_DATA_PATH="/playpen-nvme/scribble/ddinh/aaco/cheears_indiv/cheears_day_context_ver3/test_data.npz"

OUTPUT_BUNDLE_ROOT="${OUTPUT_BUNDLE_ROOT:-${ROOT_DIR}/cheears_day_context_plots}"
PLOTS_ROOT="${PLOTS_ROOT:-${OUTPUT_BUNDLE_ROOT}/plots}"
MERGED_ROOT="${MERGED_ROOT:-${OUTPUT_BUNDLE_ROOT}/plots_merged}"
INSTANCE_SCATTER_ROOT="${INSTANCE_SCATTER_ROOT:-${OUTPUT_BUNDLE_ROOT}/instance_rollouts_scatter}"
INSTANCE_HEATMAP_ROOT="${INSTANCE_HEATMAP_ROOT:-${OUTPUT_BUNDLE_ROOT}/instance_rollouts_heatmap}"
POSTHOC_ROOT="${POSTHOC_ROOT:-${OUTPUT_BUNDLE_ROOT}/posthoc}"

MODE="${MODE:-both}"
TOL_PRIMARY="${TOL_PRIMARY:-0.05}"
TOL_SECONDARY="${TOL_SECONDARY:-0.10}"
MAX_GROUPS="${MAX_GROUPS:-5}"
BATCH_SIZE="${BATCH_SIZE:-512}"
CLUSTER_K="${CLUSTER_K:-0}"
SEED="${SEED:-42}"
ADD_CLUSTER1="${ADD_CLUSTER1:-1}"
INCLUDE_WARMUP="${INCLUDE_WARMUP:-0}"

FEATURE_FILTER_MODES_CSV="${FEATURE_FILTER_MODES_CSV:-used,all}"
NODE_CMAP="${NODE_CMAP:-BuPu}"
STOP_CMAP="${STOP_CMAP:-YlOrBr}"
DPI="${DPI:-240}"
EDGE_MIN_FREQ="${EDGE_MIN_FREQ:-0.01}"
EDGE_MAX_EDGES="${EDGE_MAX_EDGES:-0}"
EDGE_NODE_SIZE_SCALE="${EDGE_NODE_SIZE_SCALE:-120.0}"
EDGE_WIDTH_SCALE="${EDGE_WIDTH_SCALE:-2.2}"
EDGE_TRANSITION_MODE="${EDGE_TRANSITION_MODE:-next_observed}"
LABEL_STRIDE="${LABEL_STRIDE:-1}"

INSTANCE_VARIANT="${INSTANCE_VARIANT:-matched_cost_groups}"
INSTANCE_MODE="${INSTANCE_MODE:-longitudinal}"
INSTANCE_TOP_K="${INSTANCE_TOP_K:-10}"
INSTANCE_MAX_GROUPS="${INSTANCE_MAX_GROUPS:-0}"
INSTANCE_FEATURE_FILTER_MODE="${INSTANCE_FEATURE_FILTER_MODE:-used}"

POSTHOC_VARIANT="${POSTHOC_VARIANT:-matched_cost_groups}"
POSTHOC_MODE="${POSTHOC_MODE:-longitudinal}"
POSTHOC_FEATURE_FILTER_MODE="${POSTHOC_FEATURE_FILTER_MODE:-used}"
POSTHOC_BASELINES="${POSTHOC_BASELINES:-learned}"
POSTHOC_BASE_FONTSIZE="${POSTHOC_BASE_FONTSIZE:-18}"
POSTHOC_TITLE_FONTSIZE="${POSTHOC_TITLE_FONTSIZE:-32}"
POSTHOC_NODE_BAR_MODE="${POSTHOC_NODE_BAR_MODE:-per_plot}"
POSTHOC_SHARED_NODE_BAR_PATH="${POSTHOC_SHARED_NODE_BAR_PATH:-${POSTHOC_ROOT}/shared_node_frequency_bar.png}"
POSTHOC_SHARED_NODE_BAR_VMAX="${POSTHOC_SHARED_NODE_BAR_VMAX:-1.0}"

RUN_MATCHED="${RUN_MATCHED:-1}"
RUN_MERGED="${RUN_MERGED:-1}"
RUN_TRAJECTORY="${RUN_TRAJECTORY:-1}"
RUN_INSTANCE="${RUN_INSTANCE:-1}"
RUN_POSTHOC="${RUN_POSTHOC:-1}"

OMP_THREADS="${OMP_THREADS:-1}"
GPU_ID="${GPU_ID:-0}"

IFS=',' read -r -a FEATURE_FILTER_MODES <<< "${FEATURE_FILTER_MODES_CSV}"

mkdir -p "${PLOTS_ROOT}" "${MERGED_ROOT}" "${INSTANCE_SCATTER_ROOT}" "${INSTANCE_HEATMAP_ROOT}" "${POSTHOC_ROOT}"

run_python() {
  (
    cd "${ROOT_DIR}"
    env \
      KMP_USE_SHM=0 \
      OMP_NUM_THREADS="${OMP_THREADS}" \
      MKL_NUM_THREADS="${OMP_THREADS}" \
      ACTOR_DATASET="${DATASET}" \
      CUDA_VISIBLE_DEVICES="${GPU_ID}" \
      "${PYTHON_BIN}" "$@"
  )
}

MATCHED_ROOT="${PLOTS_ROOT}/${DATASET}/matched_cost_groups"
MATCHED_K1_ROOT="${PLOTS_ROOT}/${DATASET}/matched_cost_groups_k1"

echo "============================================================"
echo "CHEEARS day-context plotting bundle"
echo "dataset=${DATASET}"
echo "test_data_path=${TEST_DATA_PATH}"
echo "output_bundle_root=${OUTPUT_BUNDLE_ROOT}"
echo "plots_root=${PLOTS_ROOT}"
echo "merged_root=${MERGED_ROOT}"
echo "instance_scatter_root=${INSTANCE_SCATTER_ROOT}"
echo "instance_heatmap_root=${INSTANCE_HEATMAP_ROOT}"
echo "posthoc_root=${POSTHOC_ROOT}"
echo "============================================================"

if [[ "${RUN_MATCHED}" == "1" ]]; then
  echo
  echo "[1/5] matched-cost plots"
  CMD=(
    "${ROOT_DIR}/analysis_plots_cheears_matched.py"
    --dataset "${DATASET}"
    --test_data_path "${TEST_DATA_PATH}"
    --output_root "${MATCHED_ROOT}"
    --mode "${MODE}"
    --tol_primary "${TOL_PRIMARY}"
    --tol_secondary "${TOL_SECONDARY}"
    --max_groups "${MAX_GROUPS}"
    --batch_size "${BATCH_SIZE}"
    --cluster_k "${CLUSTER_K}"
    --seed "${SEED}"
  )
  if [[ "${INCLUDE_WARMUP}" == "1" ]]; then
    CMD+=(--include_warmup)
  fi
  run_python "${CMD[@]}"

  if [[ "${ADD_CLUSTER1}" == "1" && "${CLUSTER_K}" != "1" ]]; then
    CMD_K1=(
      "${ROOT_DIR}/analysis_plots_cheears_matched.py"
      --dataset "${DATASET}"
      --test_data_path "${TEST_DATA_PATH}"
      --output_root "${MATCHED_K1_ROOT}"
      --mode "${MODE}"
      --tol_primary "${TOL_PRIMARY}"
      --tol_secondary "${TOL_SECONDARY}"
      --max_groups "${MAX_GROUPS}"
      --batch_size "${BATCH_SIZE}"
      --cluster_k "1"
      --seed "${SEED}"
    )
    if [[ "${INCLUDE_WARMUP}" == "1" ]]; then
      CMD_K1+=(--include_warmup)
    fi
    run_python "${CMD_K1[@]}"
  fi
fi

if [[ "${RUN_MERGED}" == "1" ]]; then
  echo
  echo "[2/5] merged group panels"
  for VARIANT in matched_cost_groups matched_cost_groups_k1; do
    for SUBMODE in longitudinal total; do
      SRC_DIR="${PLOTS_ROOT}/${DATASET}/${VARIANT}/${SUBMODE}"
      if [[ ! -d "${SRC_DIR}" ]]; then
        echo "[skip] missing ${SRC_DIR}"
        continue
      fi
      run_python \
        "${ROOT_DIR}/merge_group_plots.py" \
        --dataset "${DATASET}" \
        --mode "${SUBMODE}" \
        --variant "${VARIANT}" \
        --plots_root "${PLOTS_ROOT}" \
        --merged_root "${MERGED_ROOT}"
    done
  done
fi

if [[ "${RUN_TRAJECTORY}" == "1" ]]; then
  echo
  echo "[3/5] trajectory-only panels"
  for VARIANT in matched_cost_groups matched_cost_groups_k1; do
    MODE_DIR="${PLOTS_ROOT}/${DATASET}/${VARIANT}/longitudinal"
    if [[ ! -d "${MODE_DIR}" ]]; then
      echo "[skip] missing ${MODE_DIR}"
      continue
    fi
    for FEATURE_FILTER_MODE in "${FEATURE_FILTER_MODES[@]}"; do
      FEATURE_FILTER_MODE="${FEATURE_FILTER_MODE//[[:space:]]/}"
      [[ -z "${FEATURE_FILTER_MODE}" ]] && continue
      run_python \
        "${ROOT_DIR}/merge_group_trajectory_plots.py" \
        --dataset "${DATASET}" \
        --mode longitudinal \
        --variant "${VARIANT}" \
        --plots_root "${PLOTS_ROOT}" \
        --merged_root "${MERGED_ROOT}" \
        --feature_filter_mode "${FEATURE_FILTER_MODE}" \
        --node_cmap "${NODE_CMAP}" \
        --stop_cmap "${STOP_CMAP}" \
        --dpi "${DPI}" \
        --edge_min_freq "${EDGE_MIN_FREQ}" \
        --edge_max_edges "${EDGE_MAX_EDGES}" \
        --edge_node_size_scale "${EDGE_NODE_SIZE_SCALE}" \
        --edge_width_scale "${EDGE_WIDTH_SCALE}" \
        --edge_transition_mode "${EDGE_TRANSITION_MODE}" \
        --label_stride "${LABEL_STRIDE}"
    done
  done
fi

if [[ "${RUN_INSTANCE}" == "1" ]]; then
  echo
  echo "[4/5] instance rollouts"
  INSTANCE_MODE_DIR="${PLOTS_ROOT}/${DATASET}/${INSTANCE_VARIANT}/${INSTANCE_MODE}"
  if [[ -d "${INSTANCE_MODE_DIR}" ]]; then
    run_python \
      "${ROOT_DIR}/instance_rollout_visualizer.py" \
      --plots_root "${PLOTS_ROOT}" \
      --output_root "${INSTANCE_SCATTER_ROOT}" \
      --datasets "${DATASET}" \
      --variant "${INSTANCE_VARIANT}" \
      --mode "${INSTANCE_MODE}" \
      --top_k "${INSTANCE_TOP_K}" \
      --max_groups "${INSTANCE_MAX_GROUPS}" \
      --feature_filter_mode "${INSTANCE_FEATURE_FILTER_MODE}" \
      --feature_panel_style scatter \
      --dpi "${DPI}" \
      --label_stride "${LABEL_STRIDE}"

    run_python \
      "${ROOT_DIR}/instance_rollout_visualizer.py" \
      --plots_root "${PLOTS_ROOT}" \
      --output_root "${INSTANCE_HEATMAP_ROOT}" \
      --datasets "${DATASET}" \
      --variant "${INSTANCE_VARIANT}" \
      --mode "${INSTANCE_MODE}" \
      --top_k "${INSTANCE_TOP_K}" \
      --max_groups "${INSTANCE_MAX_GROUPS}" \
      --feature_filter_mode "${INSTANCE_FEATURE_FILTER_MODE}" \
      --feature_panel_style heatmap \
      --dpi "${DPI}" \
      --label_stride "${LABEL_STRIDE}"
  else
    echo "[skip] missing ${INSTANCE_MODE_DIR}"
  fi
fi

if [[ "${RUN_POSTHOC}" == "1" ]]; then
  echo
  echo "[5/5] posthoc trajectories"
  POSTHOC_MODE_DIR="${PLOTS_ROOT}/${DATASET}/${POSTHOC_VARIANT}/${POSTHOC_MODE}"
  if [[ -d "${POSTHOC_MODE_DIR}" ]]; then
    shopt -s nullglob
    GROUP_DIRS=("${POSTHOC_MODE_DIR}"/group_*)
    shopt -u nullglob
    for GROUP_DIR in "${GROUP_DIRS[@]}"; do
      GROUP_BASENAME="$(basename "${GROUP_DIR}")"
      GROUP_ID="${GROUP_BASENAME#group_}"
      GROUP_ID="$((10#${GROUP_ID}))"
      run_python \
        "${ROOT_DIR}/posthoc_trajectory_plotter.py" \
        --dataset "${DATASET}" \
        --group "${GROUP_ID}" \
        --mode "${POSTHOC_MODE}" \
        --variant "${POSTHOC_VARIANT}" \
        --plots_root "${PLOTS_ROOT}" \
        --output_root "${POSTHOC_ROOT}" \
        --feature_filter_mode "${POSTHOC_FEATURE_FILTER_MODE}" \
        --baselines "${POSTHOC_BASELINES}" \
        --node_cmap "${NODE_CMAP}" \
        --stop_cmap "${STOP_CMAP}" \
        --dpi "${DPI}" \
        --base_fontsize "${POSTHOC_BASE_FONTSIZE}" \
        --title_fontsize "${POSTHOC_TITLE_FONTSIZE}" \
        --edge_min_freq "${EDGE_MIN_FREQ}" \
        --edge_max_edges "${EDGE_MAX_EDGES}" \
        --edge_node_size_scale "540.0" \
        --edge_width_scale "3.6" \
        --edge_transition_mode "${EDGE_TRANSITION_MODE}" \
        --label_stride "${LABEL_STRIDE}" \
        --node_bar_mode "${POSTHOC_NODE_BAR_MODE}" \
        --shared_node_bar_path "${POSTHOC_SHARED_NODE_BAR_PATH}" \
        --shared_node_bar_vmax "${POSTHOC_SHARED_NODE_BAR_VMAX}"
    done
  else
    echo "[skip] missing ${POSTHOC_MODE_DIR}"
  fi
fi

echo
echo "[done] outputs written under ${OUTPUT_BUNDLE_ROOT}"
