#!/usr/bin/env bash
set -u
set -o pipefail

ROOT_DIR="/playpen-nvme/scribble/ddinh/ACTOR_LAFAv1"
PYTHON_BIN="/playpen-nvme/scribble/ddinh/miniconda3/envs/nocta/bin/python"
PLOT_SCRIPT="analysis_plots_cheears_matched.py"
SELF_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

# Default knobs (override via env vars).
MODE="${MODE:-both}"                    # longitudinal | total | both
TOL_PRIMARY="${TOL_PRIMARY:-0.05}"
TOL_SECONDARY="${TOL_SECONDARY:-0.10}"
MAX_GROUPS="${MAX_GROUPS:-5}"
BATCH_SIZE="${BATCH_SIZE:-512}"
CLUSTER_K="${CLUSTER_K:-0}"
SEED="${SEED:-42}"
INCLUDE_WARMUP="${INCLUDE_WARMUP:-0}"   # 1 => pass --include_warmup
ADD_CLUSTER1="${ADD_CLUSTER1:-1}"       # 1 => also run a fixed cluster_k=1 pass
USE_GPU="${USE_GPU:-1}"                 # 1 => set CUDA_VISIBLE_DEVICES
GPU_ID="${GPU_ID:-0}"                   # backward-compatible single-GPU selector
USE_TMUX="${USE_TMUX:-1}"               # 1 => auto-run inside tmux
TMUX_ATTACH="${TMUX_ATTACH:-0}"         # 1 => attach to created tmux session
TMUX_PER_DATASET="${TMUX_PER_DATASET:-1}"   # 1 => one tmux session per dataset
TMUX_SERIAL="${TMUX_SERIAL:-1}"             # 1 => run dataset sessions one-by-one (safe for 1 GPU)
TMUX_SESSION_NAME="${TMUX_SESSION_NAME:-matched_cost_plots_$(date +%Y%m%d_%H%M%S)}"

# Datasets to run.
DATASETS=(
  "cheears"
  "womac"
  "klg"
  "adni"
  "ILIADD"
)
if [[ -n "${DATASETS_CSV:-}" ]]; then
  DATASETS=()
  IFS=',' read -r -a _DATASETS_RAW <<< "${DATASETS_CSV}"
  for _ds in "${_DATASETS_RAW[@]}"; do
    _ds="${_ds//[[:space:]]/}"
    if [[ -n "${_ds}" ]]; then
      DATASETS+=("${_ds}")
    fi
  done
  if [[ "${#DATASETS[@]}" -eq 0 ]]; then
    echo "ERROR: DATASETS_CSV was provided but no valid datasets were parsed."
    exit 1
  fi
fi

# Optional explicit test-data overrides.
declare -A TEST_PATHS
TEST_PATHS["cheears"]="/playpen-nvme/scribble/ddinh/aaco/cheears_indiv/cheears_ver_2/test_data.npz"
TEST_PATHS["womac"]="/playpen-nvme/scribble/ddinh/aaco/input_data/womac/test_data.npz"
TEST_PATHS["klg"]="/playpen-nvme/scribble/ddinh/aaco/input_data/womac/test_data.npz"
TEST_PATHS["adni"]="/playpen-nvme/scribble/ddinh/aaco/input_data/test_data.npz"
TEST_PATHS["ILIADD"]="/playpen-nvme/scribble/ddinh/aaco/cheears_indiv/ILIADD_v3/test_data.npz"

# Any extra args passed to this script are forwarded to analysis_plots_cheears_matched.py.
EXTRA_ARGS=("$@")

if [[ "${USE_TMUX}" == "1" && -z "${TMUX:-}" ]]; then
  if command -v tmux >/dev/null 2>&1; then
    LOG_DIR="${ROOT_DIR}/plots/_logs"
    mkdir -p "${LOG_DIR}"
    ROOT_Q="$(printf '%q' "${ROOT_DIR}")"
    SELF_Q="$(printf '%q' "${SELF_PATH}")"
    if [[ "${TMUX_PER_DATASET}" == "1" ]]; then
      echo "Launching one tmux session per dataset (TMUX_SERIAL=${TMUX_SERIAL})"
      for DS in "${DATASETS[@]}"; do
        DS_LOWER="$(echo "${DS}" | tr '[:upper:]' '[:lower:]')"
        SESSION_NAME="${TMUX_SESSION_NAME}_${DS_LOWER}"
        if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
          SESSION_NAME="${SESSION_NAME}_$RANDOM"
        fi
        LOG_PATH="${LOG_DIR}/${SESSION_NAME}.log"
        LOG_Q="$(printf '%q' "${LOG_PATH}")"

        TMUX_CMD="cd ${ROOT_Q} && \
MODE=$(printf '%q' "${MODE}") \
TOL_PRIMARY=$(printf '%q' "${TOL_PRIMARY}") \
TOL_SECONDARY=$(printf '%q' "${TOL_SECONDARY}") \
MAX_GROUPS=$(printf '%q' "${MAX_GROUPS}") \
BATCH_SIZE=$(printf '%q' "${BATCH_SIZE}") \
CLUSTER_K=$(printf '%q' "${CLUSTER_K}") \
SEED=$(printf '%q' "${SEED}") \
INCLUDE_WARMUP=$(printf '%q' "${INCLUDE_WARMUP}") \
ADD_CLUSTER1=$(printf '%q' "${ADD_CLUSTER1}") \
USE_GPU=$(printf '%q' "${USE_GPU}") \
GPU_ID=$(printf '%q' "${GPU_ID}") \
DATASETS_CSV=$(printf '%q' "${DS}") \
USE_TMUX=0 \
${SELF_Q}"

        if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
          for arg in "${EXTRA_ARGS[@]}"; do
            TMUX_CMD+=" $(printf '%q' "${arg}")"
          done
        fi
        TMUX_CMD+=" 2>&1 | tee ${LOG_Q}"

        tmux new-session -d -s "${SESSION_NAME}" "${TMUX_CMD}"
        echo "Started tmux session: ${SESSION_NAME}"
        echo "Log file: ${LOG_PATH}"
        echo "Attach with: tmux attach -t ${SESSION_NAME}"
        if [[ "${TMUX_ATTACH}" == "1" ]]; then
          tmux attach -t "${SESSION_NAME}"
        fi

        if [[ "${TMUX_SERIAL}" == "1" ]]; then
          while tmux has-session -t "${SESSION_NAME}" 2>/dev/null; do
            sleep 2
          done
        fi
      done
      exit 0
    else
      if tmux has-session -t "${TMUX_SESSION_NAME}" 2>/dev/null; then
        TMUX_SESSION_NAME="${TMUX_SESSION_NAME}_$RANDOM"
      fi
      LOG_PATH="${LOG_DIR}/${TMUX_SESSION_NAME}.log"
      LOG_Q="$(printf '%q' "${LOG_PATH}")"

      TMUX_CMD="cd ${ROOT_Q} && \
MODE=$(printf '%q' "${MODE}") \
TOL_PRIMARY=$(printf '%q' "${TOL_PRIMARY}") \
TOL_SECONDARY=$(printf '%q' "${TOL_SECONDARY}") \
MAX_GROUPS=$(printf '%q' "${MAX_GROUPS}") \
BATCH_SIZE=$(printf '%q' "${BATCH_SIZE}") \
CLUSTER_K=$(printf '%q' "${CLUSTER_K}") \
SEED=$(printf '%q' "${SEED}") \
INCLUDE_WARMUP=$(printf '%q' "${INCLUDE_WARMUP}") \
ADD_CLUSTER1=$(printf '%q' "${ADD_CLUSTER1}") \
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
    fi
  else
    echo "WARNING: tmux not found; running in current shell."
  fi
fi

GPU_SLOT="${GPU_ID//[[:space:]]/}"
if [[ -z "${GPU_SLOT}" ]]; then
  GPU_SLOT="0"
fi
if [[ "${USE_GPU}" != "1" ]]; then
  GPU_SLOT="cpu"
fi

STATUS_DIR="${ROOT_DIR}/plots/_logs/.status_${TMUX_SESSION_NAME}_$$"
mkdir -p "${STATUS_DIR}"

run_dataset() {
  local DS="$1"
  local GPU_SLOT="$2"
  local RC=0

  echo
  echo "------------------------------------------------------------"
  echo "Running dataset: ${DS} (gpu=${GPU_SLOT})"
  echo "------------------------------------------------------------"

  local CMD=(
    "${PYTHON_BIN}" "${PLOT_SCRIPT}"
    --dataset "${DS}"
    --mode "${MODE}"
    --tol_primary "${TOL_PRIMARY}"
    --tol_secondary "${TOL_SECONDARY}"
    --max_groups "${MAX_GROUPS}"
    --batch_size "${BATCH_SIZE}"
    --cluster_k "${CLUSTER_K}"
    --seed "${SEED}"
  )

  local TEST_PATH="${TEST_PATHS[$DS]:-}"
  if [[ -n "${TEST_PATH}" && -f "${TEST_PATH}" ]]; then
    CMD+=(--test_data_path "${TEST_PATH}")
    echo "Using explicit test_data_path: ${TEST_PATH}"
  else
    echo "No explicit test_data_path (or missing file), script default resolver will be used."
  fi

  if [[ "${INCLUDE_WARMUP}" == "1" ]]; then
    CMD+=(--include_warmup)
  fi

  if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
  fi

  local RUN_ENV=(KMP_USE_SHM=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ACTOR_DATASET="${DS}")
  if [[ "${USE_GPU}" == "1" ]]; then
    RUN_ENV+=(CUDA_VISIBLE_DEVICES="${GPU_SLOT}")
  fi

  (
    cd "${ROOT_DIR}" || exit 1
    echo "[run:auto] dataset=${DS} cluster_k=${CLUSTER_K} gpu=${GPU_SLOT}"
    env "${RUN_ENV[@]}" "${CMD[@]}"
  ) || RC=$?

  if [[ ${RC} -eq 0 && "${ADD_CLUSTER1}" == "1" && "${CLUSTER_K}" != "1" ]]; then
    local DS_LOWER
    DS_LOWER="$(echo "${DS}" | tr '[:upper:]' '[:lower:]')"
    local OUT_K1="${ROOT_DIR}/plots/${DS_LOWER}/matched_cost_groups_k1"
    local CMD_K1=("${CMD[@]}")

    for i in "${!CMD_K1[@]}"; do
      if [[ "${CMD_K1[$i]}" == "--cluster_k" ]]; then
        CMD_K1[$((i + 1))]="1"
        break
      fi
    done
    CMD_K1+=(--output_root "${OUT_K1}")

    (
      cd "${ROOT_DIR}" || exit 1
      echo "[run:k1]   dataset=${DS} cluster_k=1 output_root=${OUT_K1} gpu=${GPU_SLOT}"
      env "${RUN_ENV[@]}" "${CMD_K1[@]}"
    ) || RC=$?
  fi

  printf '%s\n' "${RC}" > "${STATUS_DIR}/${DS}.rc"
}

echo "============================================================"
echo "Matched-cost plotting for all datasets"
echo "Root: ${ROOT_DIR}"
echo "Python: ${PYTHON_BIN}"
echo "Mode: ${MODE} | tol_primary=${TOL_PRIMARY} | tol_secondary=${TOL_SECONDARY}"
echo "max_groups=${MAX_GROUPS} | batch_size=${BATCH_SIZE} | cluster_k=${CLUSTER_K} | seed=${SEED}"
echo "include_warmup=${INCLUDE_WARMUP}"
echo "add_cluster1=${ADD_CLUSTER1}"
echo "use_gpu=${USE_GPU} | gpu_id=${GPU_SLOT}"
echo "parallel_jobs=1 (single-GPU mode)"
echo "datasets=${DATASETS[*]}"
echo "use_tmux=${USE_TMUX}"
echo "============================================================"

for DS in "${DATASETS[@]}"; do
  echo "[queue] dataset=${DS} gpu=${GPU_SLOT}"
  run_dataset "${DS}" "${GPU_SLOT}"
done

SUCCESS_DATASETS=()
FAILED_DATASETS=()
for DS in "${DATASETS[@]}"; do
  RC_FILE="${STATUS_DIR}/${DS}.rc"
  RC="1"
  if [[ -f "${RC_FILE}" ]]; then
    RC="$(cat "${RC_FILE}")"
  fi

  if [[ "${RC}" == "0" ]]; then
    SUCCESS_DATASETS+=("${DS}")
    echo "[OK] ${DS}"
  else
    FAILED_DATASETS+=("${DS}")
    echo "[FAILED] ${DS} (exit=${RC})"
  fi
done

echo
echo "============================================================"
echo "Run summary"
echo "============================================================"
echo "Succeeded (${#SUCCESS_DATASETS[@]}): ${SUCCESS_DATASETS[*]:-none}"
echo "Failed    (${#FAILED_DATASETS[@]}): ${FAILED_DATASETS[*]:-none}"
echo "status_dir=${STATUS_DIR}"

if [[ "${#FAILED_DATASETS[@]}" -gt 0 ]]; then
  exit 1
fi
exit 0
