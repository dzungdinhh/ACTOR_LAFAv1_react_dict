#!/usr/bin/env bash
set -u
set -o pipefail

ROOT_DIR="/playpen-nvme/scribble/ddinh/ACTOR_LAFAv1"
PYTHON_BIN="/playpen-nvme/scribble/ddinh/miniconda3/envs/nocta/bin/python"
PLOT_SCRIPT="analysis_plots_cheears_matched.py"
SELF_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

# Defaults (override by env vars).
MAX_GROUPS="${MAX_GROUPS:-5}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TOL_PRIMARY="${TOL_PRIMARY:-0.05}"
TOL_SECONDARY="${TOL_SECONDARY:-0.10}"
SEED="${SEED:-42}"
INCLUDE_WARMUP="${INCLUDE_WARMUP:-0}"
REQUIRE_MIN_GROUPS="${REQUIRE_MIN_GROUPS:-1}"   # minimum rows (excluding header) per summary file
DRY_RUN="${DRY_RUN:-0}"                         # 1 => print commands only
USE_GPU="${USE_GPU:-1}"                         # 1 => set CUDA_VISIBLE_DEVICES
GPU_ID="${GPU_ID:-0}"                           # ignored when USE_GPU=0

# Optional tmux wrapper.
USE_TMUX="${USE_TMUX:-1}"                       # 1 => run this script in detached tmux
TMUX_ATTACH="${TMUX_ATTACH:-0}"                 # 1 => attach right away
TMUX_SESSION_NAME="${TMUX_SESSION_NAME:-matched_cost_missing_$(date +%Y%m%d_%H%M%S)}"

DATASETS=(
  "cheears"
  "womac"
  "klg"
  "adni"
  "ILIADD"
)

declare -A TEST_PATHS
TEST_PATHS["cheears"]="/playpen-nvme/scribble/ddinh/aaco/cheears_indiv/cheears_ver_2/test_data.npz"
TEST_PATHS["womac"]="/playpen-nvme/scribble/ddinh/aaco/input_data/womac/test_data.npz"
TEST_PATHS["klg"]="/playpen-nvme/scribble/ddinh/aaco/input_data/womac/test_data.npz"
TEST_PATHS["adni"]="/playpen-nvme/scribble/ddinh/aaco/input_data/test_data.npz"
TEST_PATHS["ILIADD"]="/playpen-nvme/scribble/ddinh/aaco/cheears_indiv/ILIADD_v3/test_data.npz"

EXTRA_ARGS=("$@")

if [[ "${USE_TMUX}" == "1" && -z "${TMUX:-}" ]]; then
  if command -v tmux >/dev/null 2>&1; then
    if tmux has-session -t "${TMUX_SESSION_NAME}" 2>/dev/null; then
      TMUX_SESSION_NAME="${TMUX_SESSION_NAME}_$RANDOM"
    fi

    LOG_DIR="${ROOT_DIR}/plots/_logs"
    mkdir -p "${LOG_DIR}"
    LOG_PATH="${LOG_DIR}/${TMUX_SESSION_NAME}.log"

    ROOT_Q="$(printf '%q' "${ROOT_DIR}")"
    SELF_Q="$(printf '%q' "${SELF_PATH}")"
    LOG_Q="$(printf '%q' "${LOG_PATH}")"

    TMUX_CMD="cd ${ROOT_Q} && \
MAX_GROUPS=$(printf '%q' "${MAX_GROUPS}") \
BATCH_SIZE=$(printf '%q' "${BATCH_SIZE}") \
TOL_PRIMARY=$(printf '%q' "${TOL_PRIMARY}") \
TOL_SECONDARY=$(printf '%q' "${TOL_SECONDARY}") \
SEED=$(printf '%q' "${SEED}") \
INCLUDE_WARMUP=$(printf '%q' "${INCLUDE_WARMUP}") \
REQUIRE_MIN_GROUPS=$(printf '%q' "${REQUIRE_MIN_GROUPS}") \
DRY_RUN=$(printf '%q' "${DRY_RUN}") \
USE_GPU=$(printf '%q' "${USE_GPU}") \
GPU_ID=$(printf '%q' "${GPU_ID}") \
USE_TMUX=0 \
${SELF_Q}"
    if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
      for arg in "${EXTRA_ARGS[@]}"; do
        TMUX_CMD+=" $(printf '%q' "${arg}")"
      done
    fi
    TMUX_CMD+=" 2>&1 | tee ${LOG_Q}"

    tmux new-session -d -s "${TMUX_SESSION_NAME}" "${TMUX_CMD}"
    echo "Started tmux session: ${TMUX_SESSION_NAME}"
    echo "Log file: ${LOG_PATH}"
    echo "Attach with: tmux attach -t ${TMUX_SESSION_NAME}"
    if [[ "${TMUX_ATTACH}" == "1" ]]; then
      tmux attach -t "${TMUX_SESSION_NAME}"
    fi
    exit 0
  else
    echo "WARNING: tmux not found; running in current shell."
  fi
fi

mode_complete() {
  local root="$1"
  local mode="$2"
  local summary="${root}/summary_groups_${mode}.tsv"
  if [[ ! -f "${summary}" ]]; then
    return 1
  fi

  local lines
  lines=$(wc -l < "${summary}")
  local groups
  groups=$((lines - 1))
  if (( groups < REQUIRE_MIN_GROUPS )); then
    return 1
  fi

  shopt -s nullglob
  local dirs=("${root}/${mode}"/group_*)
  shopt -u nullglob
  if (( ${#dirs[@]} == 0 )); then
    return 1
  fi

  local d
  for d in "${dirs[@]}"; do
    if [[ -f "${d}/group_meta.json" ]]; then
      return 0
    fi
  done
  return 1
}

run_one() {
  local ds="$1"
  local mode="$2"
  local cluster_k="$3"
  local output_root="$4"
  local test_path="$5"

  local cmd=(
    "${PYTHON_BIN}" "${PLOT_SCRIPT}"
    --dataset "${ds}"
    --mode "${mode}"
    --tol_primary "${TOL_PRIMARY}"
    --tol_secondary "${TOL_SECONDARY}"
    --max_groups "${MAX_GROUPS}"
    --batch_size "${BATCH_SIZE}"
    --cluster_k "${cluster_k}"
    --seed "${SEED}"
    --output_root "${output_root}"
  )

  if [[ -n "${test_path}" && -f "${test_path}" ]]; then
    cmd+=(--test_data_path "${test_path}")
  fi
  if [[ "${INCLUDE_WARMUP}" == "1" ]]; then
    cmd+=(--include_warmup)
  fi
  if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
    cmd+=("${EXTRA_ARGS[@]}")
  fi

  echo "[RUN] dataset=${ds} mode=${mode} cluster_k=${cluster_k}"
  echo "      output_root=${output_root}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "      DRY_RUN=1 -> skipped"
    return 0
  fi

  (
    cd "${ROOT_DIR}" || exit 1
    if [[ "${USE_GPU}" == "1" ]]; then
      env KMP_USE_SHM=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ACTOR_DATASET="${ds}" CUDA_VISIBLE_DEVICES="${GPU_ID}" "${cmd[@]}"
    else
      env KMP_USE_SHM=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ACTOR_DATASET="${ds}" "${cmd[@]}"
    fi
  )
}

echo "============================================================"
echo "Check-and-run missing matched-cost plot variants"
echo "Root: ${ROOT_DIR}"
echo "max_groups=${MAX_GROUPS} batch_size=${BATCH_SIZE} seed=${SEED}"
echo "tol_primary=${TOL_PRIMARY} tol_secondary=${TOL_SECONDARY}"
echo "require_min_groups=${REQUIRE_MIN_GROUPS} dry_run=${DRY_RUN}"
echo "use_gpu=${USE_GPU} gpu_id=${GPU_ID}"
echo "Need 4 variants per dataset:"
echo "  1) auto + longitudinal"
echo "  2) auto + total"
echo "  3) k1   + longitudinal"
echo "  4) k1   + total"
echo "============================================================"

SUCCESS=()
FAILED=()
SKIPPED=()

for ds in "${DATASETS[@]}"; do
  echo
  echo "------------------------------------------------------------"
  echo "Dataset: ${ds}"
  echo "------------------------------------------------------------"

  ds_lower="$(echo "${ds}" | tr '[:upper:]' '[:lower:]')"
  root_auto="${ROOT_DIR}/plots/${ds_lower}/matched_cost_groups"
  root_k1="${ROOT_DIR}/plots/${ds_lower}/matched_cost_groups_k1"
  test_path="${TEST_PATHS[$ds]:-}"

  missing_auto=()
  missing_k1=()

  for mode in longitudinal total; do
    if mode_complete "${root_auto}" "${mode}"; then
      echo "[OK] auto mode=${mode} exists"
    else
      echo "[MISS] auto mode=${mode}"
      missing_auto+=("${mode}")
    fi

    if mode_complete "${root_k1}" "${mode}"; then
      echo "[OK] k1   mode=${mode} exists"
    else
      echo "[MISS] k1   mode=${mode}"
      missing_k1+=("${mode}")
    fi
  done

  rc_ds=0

  if (( ${#missing_auto[@]} == 0 )); then
    SKIPPED+=("${ds}:auto")
  elif (( ${#missing_auto[@]} == 2 )); then
    run_one "${ds}" "both" "0" "${root_auto}" "${test_path}" || rc_ds=1
  else
    run_one "${ds}" "${missing_auto[0]}" "0" "${root_auto}" "${test_path}" || rc_ds=1
  fi

  if (( rc_ds == 0 )); then
    if (( ${#missing_k1[@]} == 0 )); then
      SKIPPED+=("${ds}:k1")
    elif (( ${#missing_k1[@]} == 2 )); then
      run_one "${ds}" "both" "1" "${root_k1}" "${test_path}" || rc_ds=1
    else
      run_one "${ds}" "${missing_k1[0]}" "1" "${root_k1}" "${test_path}" || rc_ds=1
    fi
  fi

  if (( rc_ds == 0 )); then
    SUCCESS+=("${ds}")
  else
    FAILED+=("${ds}")
  fi
done

echo
echo "============================================================"
echo "Summary"
echo "============================================================"
echo "Succeeded (${#SUCCESS[@]}): ${SUCCESS[*]:-none}"
echo "Failed    (${#FAILED[@]}): ${FAILED[*]:-none}"
echo "Skipped   (${#SKIPPED[@]}): ${SKIPPED[*]:-none}"

if (( ${#FAILED[@]} > 0 )); then
  exit 1
fi
exit 0
