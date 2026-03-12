#!/usr/bin/env bash
set -u
set -o pipefail

ROOT_DIR="/playpen-nvme/scribble/ddinh/ACTOR_LAFAv1"
PYTHON_BIN="/playpen-nvme/scribble/ddinh/miniconda3/envs/nocta/bin/python"
SCRIPT_PATH="${ROOT_DIR}/merge_group_plots.py"
PLOTS_ROOT="${ROOT_DIR}/plots"
MERGED_ROOT="${ROOT_DIR}/plots_merged"

DATASETS=(
  "cheears"
  "womac"
  "klg"
  "adni"
  "ILIADD"
)

VARIANTS=(
  "matched_cost_groups"
  "matched_cost_groups_k1"
)

MODES=(
  "longitudinal"
  "total"
)

echo "============================================================"
echo "Regenerate merged group plots"
echo "plots_root:  ${PLOTS_ROOT}"
echo "merged_root: ${MERGED_ROOT}"
echo "============================================================"

SUCCESS=0
SKIP=0
FAIL=0

for ds in "${DATASETS[@]}"; do
  ds_dir="$(echo "${ds}" | tr '[:upper:]' '[:lower:]')"
  for v in "${VARIANTS[@]}"; do
    for m in "${MODES[@]}"; do
      src_dir="${PLOTS_ROOT}/${ds_dir}/${v}/${m}"
      if [[ ! -d "${src_dir}" ]]; then
        echo "[SKIP] missing: ${src_dir}"
        SKIP=$((SKIP + 1))
        continue
      fi

      echo "[RUN] dataset=${ds} variant=${v} mode=${m}"
      (
        cd "${ROOT_DIR}" || exit 1
        KMP_USE_SHM=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        "${PYTHON_BIN}" "${SCRIPT_PATH}" \
          --dataset "${ds}" \
          --mode "${m}" \
          --variant "${v}" \
          --plots_root "${PLOTS_ROOT}" \
          --merged_root "${MERGED_ROOT}"
      )
      rc=$?
      if [[ ${rc} -eq 0 ]]; then
        SUCCESS=$((SUCCESS + 1))
      else
        echo "[FAIL] dataset=${ds} variant=${v} mode=${m} (exit=${rc})"
        FAIL=$((FAIL + 1))
      fi
    done
  done
done

echo
echo "============================================================"
echo "Summary"
echo "============================================================"
echo "Success: ${SUCCESS}"
echo "Skip:    ${SKIP}"
echo "Fail:    ${FAIL}"

if [[ ${FAIL} -gt 0 ]]; then
  exit 1
fi
exit 0

