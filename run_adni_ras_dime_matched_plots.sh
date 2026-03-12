#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/playpen-nvme/scribble/ddinh/ACTOR_LAFAv1"
PYTHON_BIN="/playpen-nvme/scribble/ddinh/miniconda3/envs/nocta/bin/python"
DIME_ROOT_DIR="/playpen-nvme/scribble/ddinh/baseline_DIME"
DIME_ACTION_SCRIPT="${DIME_ROOT_DIR}/save_dime_actions.py"
DIME_PYTHON="${DIME_PYTHON:-/playpen-nvme/scribble/ddinh/miniconda3/envs/tafa/bin/python}"
ACTOR_METRICS_PYTHON="${ACTOR_METRICS_PYTHON:-/playpen-nvme/scribble/ddinh/miniconda3/envs/tafa/bin/python}"

PREPARE_SCRIPT="${ROOT_DIR}/prepare_adni_ras_dime_matched_groups.py"
MERGE_SCRIPT="${ROOT_DIR}/merge_group_plots_ras_dime.py"
TRAJ_SCRIPT="${ROOT_DIR}/merge_group_trajectory_plots_ras_dime.py"

DATASET="adni"
SOURCE_VARIANT="${SOURCE_VARIANT:-matched_cost_groups}"
VARIANT="${VARIANT:-matched_cost_groups_ras_dime_all_features_all_baseline_available}"
VARIANT_K1="${VARIANT_K1:-${VARIANT}_k1}"
PLOTS_ROOT="${PLOTS_ROOT:-${ROOT_DIR}/plots}"
MERGED_ROOT="${MERGED_ROOT:-${ROOT_DIR}/plots_merged}"
GROUP="${GROUP:-}"
FEATURE_FILTER_MODES_CSV="${FEATURE_FILTER_MODES_CSV:-all}"
DIME_ACTION_ROOT="${DIME_ACTION_ROOT:-${DIME_ROOT_DIR}/dime_actions_all_baseline_available}"
DIME_BUDGETS_CSV="${DIME_BUDGETS_CSV:-2,4,6,8,10}"
METRICS_CACHE_ROOT="${METRICS_CACHE_ROOT:-${PLOTS_ROOT}/${DATASET}/${VARIANT}_metrics}"
DIME_METRIC_STYLE="${DIME_METRIC_STYLE:-ras}"
LABEL_STRIDE="${LABEL_STRIDE:-1}"
DPI="${DPI:-240}"
EDGE_MIN_FREQ="${EDGE_MIN_FREQ:-0.01}"
EDGE_MAX_EDGES="${EDGE_MAX_EDGES:-0}"
EDGE_NODE_SIZE_SCALE="${EDGE_NODE_SIZE_SCALE:-120.0}"
EDGE_WIDTH_SCALE="${EDGE_WIDTH_SCALE:-2.2}"
EDGE_TRANSITION_MODE="${EDGE_TRANSITION_MODE:-next_observed}"
NODE_CMAP="${NODE_CMAP:-Blues}"
STOP_CMAP="${STOP_CMAP:-YlOrBr}"
OMP_THREADS="${OMP_THREADS:-1}"
MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-SEQUENTIAL}"
FORCE_REFRESH_RAS="${FORCE_REFRESH_RAS:-0}"
FORCE_REFRESH_DIME="${FORCE_REFRESH_DIME:-0}"
FORCE_REFRESH_METRICS="${FORCE_REFRESH_METRICS:-0}"
RAS_BASELINE_AT_T0_ONLY="${RAS_BASELINE_AT_T0_ONLY:-0}"
DIME_BASELINE_AT_T0_ONLY="${DIME_BASELINE_AT_T0_ONLY:-0}"

SOURCE_ROOT="${PLOTS_ROOT}/${DATASET}/${SOURCE_VARIANT}"
OUTPUT_ROOT="${PLOTS_ROOT}/${DATASET}/${VARIANT}"
OUTPUT_ROOT_K1="${PLOTS_ROOT}/${DATASET}/${VARIANT_K1}"

echo "=============================================================="
echo "ADNI learned/RAS/DIME matched-cost plotting"
echo "Source root: ${SOURCE_ROOT}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Output root k=1: ${OUTPUT_ROOT_K1}"
echo "Merged root: ${MERGED_ROOT}"
echo "DIME action root: ${DIME_ACTION_ROOT}"
echo "Metrics cache root: ${METRICS_CACHE_ROOT}"
echo "DIME metric style: ${DIME_METRIC_STYLE}"
echo "RAS baseline-at-t0-only: ${RAS_BASELINE_AT_T0_ONLY}"
if [[ -n "${GROUP}" ]]; then
  echo "Group: ${GROUP}"
else
  echo "Group: all"
fi
echo "=============================================================="

run_in_env() {
  (
    cd "${ROOT_DIR}"
    env \
      KMP_USE_SHM=0 \
      OMP_NUM_THREADS="${OMP_THREADS}" \
      MKL_NUM_THREADS="${OMP_THREADS}" \
      MKL_THREADING_LAYER="${MKL_THREADING_LAYER}" \
      "$@"
  )
}

run_in_dime_env() {
  (
    cd "${DIME_ROOT_DIR}"
    env \
      KMP_USE_SHM=0 \
      OMP_NUM_THREADS="${OMP_THREADS}" \
      MKL_NUM_THREADS="${OMP_THREADS}" \
      MKL_THREADING_LAYER="${MKL_THREADING_LAYER}" \
      "$@"
  )
}

ensure_dime_actions() {
  local need_refresh="${FORCE_REFRESH_DIME}"
  IFS=',' read -r -a DIME_BUDGETS <<< "${DIME_BUDGETS_CSV}"
  for BUDGET in "${DIME_BUDGETS[@]}"; do
    BUDGET="$(echo "${BUDGET}" | xargs)"
    [[ -z "${BUDGET}" ]] && continue
    if [[ ! -f "${DIME_ACTION_ROOT}/dime_actions_budget_${BUDGET}.npz" ]]; then
      need_refresh="1"
      break
    fi
  done

  if [[ "${need_refresh}" != "1" ]]; then
    echo "[dime] using existing all-baseline-available action caches"
    return
  fi

  CMD=(
    "${DIME_PYTHON}" "${DIME_ACTION_SCRIPT}"
    --output-dir "${DIME_ACTION_ROOT}"
  )
  if [[ "${DIME_BASELINE_AT_T0_ONLY}" == "1" ]]; then
    CMD+=(--baseline-at-t0-only)
  else
    CMD+=(--all-baseline-available)
  fi
  CMD+=(--budgets)
  for BUDGET in "${DIME_BUDGETS[@]}"; do
    BUDGET="$(echo "${BUDGET}" | xargs)"
    [[ -z "${BUDGET}" ]] && continue
    CMD+=("${BUDGET}")
  done

  if [[ "${DIME_BASELINE_AT_T0_ONLY}" == "1" ]]; then
    echo "[dime] generating baseline-at-t0-only action caches"
  else
    echo "[dime] generating all-baseline-available action caches"
  fi
  run_in_dime_env "${CMD[@]}"
}

prepare_variant() {
  local output_root="$1"
  local cluster_override="$2"

  PREPARE_CMD=(
    "${PYTHON_BIN}" "${PREPARE_SCRIPT}"
    --source_root "${SOURCE_ROOT}"
    --output_root "${output_root}"
    --dime_action_root "${DIME_ACTION_ROOT}"
    --metrics_cache_root "${METRICS_CACHE_ROOT}"
    --actor_python "${ACTOR_METRICS_PYTHON}"
    --dime_python "${DIME_PYTHON}"
    --dime_metric_style "${DIME_METRIC_STYLE}"
    --all_features
  )
  if [[ "${FORCE_REFRESH_RAS}" == "1" ]]; then
    PREPARE_CMD+=(--force_refresh_ras)
  fi
  if [[ "${RAS_BASELINE_AT_T0_ONLY}" == "1" ]]; then
    PREPARE_CMD+=(--ras_baseline_at_t0_only)
  fi
  if [[ "${DIME_BASELINE_AT_T0_ONLY}" == "1" ]]; then
    PREPARE_CMD+=(--dime_baseline_at_t0_only)
  fi
  if [[ "${FORCE_REFRESH_METRICS}" == "1" ]]; then
    PREPARE_CMD+=(--force_refresh_metrics)
  fi
  if [[ -n "${cluster_override}" ]]; then
    PREPARE_CMD+=(--cluster_k_override "${cluster_override}")
  fi

  echo "[prepare] output_root=${output_root} cluster_k_override=${cluster_override:-source}"
  run_in_env "${PREPARE_CMD[@]}"
}

render_variant() {
  local variant_name="$1"
  local cluster_arg="$2"

  for MODE in longitudinal total; do
    CMD=(
      "${PYTHON_BIN}" "${MERGE_SCRIPT}"
      --dataset "${DATASET}"
      --mode "${MODE}"
      --variant "${variant_name}"
      --merged_root "${MERGED_ROOT}"
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
    if [[ -n "${cluster_arg}" ]]; then
      CMD+=(--cluster_k "${cluster_arg}")
    fi

    echo "[run] variant=${variant_name} merged mode=${MODE}"
    run_in_env "${CMD[@]}"
  done

  IFS=',' read -r -a FEATURE_FILTER_MODES <<< "${FEATURE_FILTER_MODES_CSV}"
  for FEATURE_FILTER_MODE in "${FEATURE_FILTER_MODES[@]}"; do
    FEATURE_FILTER_MODE="$(echo "${FEATURE_FILTER_MODE}" | xargs)"
    [[ -z "${FEATURE_FILTER_MODE}" ]] && continue

    CMD=(
      "${PYTHON_BIN}" "${TRAJ_SCRIPT}"
      --dataset "${DATASET}"
      --mode longitudinal
      --variant "${variant_name}"
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

    echo "[run] variant=${variant_name} trajectory longitudinal feature_filter_mode=${FEATURE_FILTER_MODE}"
    run_in_env "${CMD[@]}"
  done
}

ensure_dime_actions
prepare_variant "${OUTPUT_ROOT}" ""
prepare_variant "${OUTPUT_ROOT_K1}" "1"
render_variant "${VARIANT}" ""
render_variant "${VARIANT_K1}" "1"

echo "[done] ADNI learned/RAS/DIME matched-cost plotting finished."
