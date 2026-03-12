#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/playpen-nvme/scribble/ddinh/ACTOR_LAFAv1"
BASE_RUNNER="${ROOT_DIR}/run_trajectory_only_longitudinal.sh"

# BuPu trajectory run:
# - uses same directory structure style as plots_merged, but rooted at /traj
# - runs all datasets by default (or pass one dataset as arg)
#
# Usage:
#   ./run_trajectory_only_longitudinal_bupu.sh
#   ./run_trajectory_only_longitudinal_bupu.sh womac
#
# Optional env (same as base runner):
#   DATASETS_CSV, GROUP, VARIANTS_CSV, FEATURE_FILTER_MODES_CSV, OMP_THREADS, GPU_ID, etc.

if [[ ! -x "${BASE_RUNNER}" ]]; then
  echo "ERROR: base runner not found or not executable: ${BASE_RUNNER}"
  exit 1
fi

export MERGED_ROOT="${MERGED_ROOT:-/playpen-nvme/scribble/ddinh/ACTOR_LAFAv1/traj}"
export NODE_CMAP="${NODE_CMAP:-BuPu}"
export STOP_CMAP="${STOP_CMAP:-YlOrBr}"

exec "${BASE_RUNNER}" "$@"
