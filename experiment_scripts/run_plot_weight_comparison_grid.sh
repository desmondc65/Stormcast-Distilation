#!/bin/bash
# Render the 4-variable x 4-weight-type comparison grid (one figure per date).
#
#   rows    = t2m, u10, v10, qpepre        (physical units)
#   columns = truth | legacy EDM | cleaned EDM | cleaned FlowCast
#
# Column wiring (see experiment_scripts/plot_weight_comparison_grid.py):
#   1. truth         -- cleaned 192x96 / log1p dataset target M_{t+1}
#   2. legacy EDM    -- uncropped 224x128 raw dataset
#                       StormCastUNet.0.7500 + EDMPrecond.0.70000
#   3. cleaned EDM   -- cleaned 192x96 log1p dataset
#                       StormCastUNet.0.8000 + EDMPrecond.0.20000
#   4. cleaned Flow  -- cleaned 192x96 log1p dataset
#                       StormCastUNet.0.8000 + FlowCastPrecond.0.20000
#
# Each panel is a single +1h generative step on the frozen regression mean
# (mu_{t+1} + r_{t+1}), NOT an autoregressive rollout.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"

# Prefer the in-repo venv (no conda on this workstation); fall back to PATH python.
if [ -x "${REPO_ROOT}/stormcast_env/bin/python" ]; then
    PY="${REPO_ROOT}/stormcast_env/bin/python"
else
    PY="$(command -v python)"
fi

# Pin to a free GPU by default (0 is usually busy on this box; 1/2 are free).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

OUT_DIR="${OUT_DIR:-${REPO_ROOT}/experiment_scripts/results/weight_comparison_grid}"

# Default dates: 4 representative initial times spanning the 2022 valid year
# (winter / spring / typhoon season / NE-monsoon). Override with DATES="...".
# Leave DATES empty to let the script auto-pick 4 evenly-spaced dates instead.
DATES="${DATES:-2022-02-14T00 2022-06-04T00 2022-09-12T00 2022-11-20T00}"

echo "[run] python       = ${PY}"
echo "[run] CUDA devices = ${CUDA_VISIBLE_DEVICES}"
echo "[run] out_dir      = ${OUT_DIR}"
echo "[run] dates        = ${DATES:-<auto>}"

mkdir -p "${OUT_DIR}"

ARGS=(--output-dir "${OUT_DIR}")
if [ -n "${DATES}" ]; then
    # shellcheck disable=SC2206
    DATE_ARR=(${DATES})
    ARGS+=(--dates "${DATE_ARR[@]}")
fi

"${PY}" -u "${REPO_ROOT}/experiment_scripts/plot_weight_comparison_grid.py" "${ARGS[@]}" \
    2>&1 | tee "${OUT_DIR}/run.log"

echo "[run] done -- figures under ${OUT_DIR}"
