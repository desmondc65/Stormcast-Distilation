#!/bin/bash
# Run the FlowCast-vs-EDM × log1p-vs-NO_log1p ablation on a zettabyte worker.
#
# This is the stormcast/zettabyte_scripts/ablation.md "headline" comparison:
#   D1 = edm_log1p          on cleaned dataset (qpepre stored as log1p(mm/h))
#   F1 = flowcast_log1p     on cleaned dataset
#   D2 = edm_NO_log1p       on cleaned dataset whose qpepre is in raw mm/h
#   F2 = flowcast_NO_log1p  on cleaned dataset whose qpepre is in raw mm/h
#
# Each row uses the regression checkpoint trained on the *same* qpepre
# representation (R0 -> log1p, R0_raw -> raw mm/h). Mixing log1p data with
# qpepre_log1p=false (or vice versa) is silently miscalibrated and is the
# single biggest footgun this script protects against.
#
# Pull both datasets onto the worker first (azcopy, see ../zettabyte/zettabyte.md):
#   export SAS_URL="https://zbstore2026.blob.core.windows.net/g-019c8ca2-...?se=...sig=..."
#   for SUB in zarr_exp3_L_24_H_24_train_2_5_years_full_cleaned_4_27_2026 \
#              zarr_exp3_L_24_H_24_train_2_5_years_full_cleaned_4_27_2026_raw; do
#     SRC="${SAS_URL%%\?*}/desmond/dataset/${SUB}?${SAS_URL#*\?}"
#     azcopy copy "$SRC" /workspace/downloads --recursive
#   done
#
# Run dirs from training are expected to have already been written under
#   /data/exp_3_train_2_5_yrs_val_1yr_tp1/
# by train_diffusion{,_no_log1p}.sh, train_flowcast{,_no_log1p}.sh, and
# train_regression{,_no_log1p}.sh. If you pulled them down from blob, point
# RUNS_ROOT at wherever azcopy landed them.

set -euo pipefail

source $(conda info --base)/etc/profile.d/conda.sh
conda activate stormcast_env

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# --- Paths (override via env if your layout differs) -------------------------
REPO_ROOT="${REPO_ROOT:-/workspace/Stormcast-Distilation}"
RUNS_ROOT="${RUNS_ROOT:-/data/exp_3_train_2_5_yrs_val_1yr_tp1}"
DATA_LOG1P="${DATA_LOG1P:-/workspace/downloads/zarr_exp3_L_24_H_24_train_2_5_years_full_cleaned_4_27_2026}"
DATA_RAW="${DATA_RAW:-/workspace/downloads/zarr_exp3_L_24_H_24_train_2_5_years_full_cleaned_4_27_2026_raw}"

ABLATION_SCRIPT="${REPO_ROOT}/experiment_scripts/log1p_ablation.py"
OUT_DIR="${OUT_DIR:-${RUNS_ROOT}/ablation_results/log1p_compare_diffusion_vs_flowcast}"

# --- Checkpoint selection ---------------------------------------------------
# Step number of the EDM / FlowCast checkpoints to evaluate. ablation.md
# expects D1/D2 around step 70k (EDM teacher converges fast) and F1/F2 in the
# 100k-400k window (FlowCast plateau). 20000 matches the local pilot in
# experiment_scripts/ablation.md; bump for the final thesis run.
CHECKPOINT_STEP="${CHECKPOINT_STEP:-20000}"

# Regression checkpoints (one per qpepre representation). Defaults match
# train_regression.sh / train_regression_no_log1p.sh.
REG_LOG1P="${REG_LOG1P:-${RUNS_ROOT}/regression_zettabyte_v1_cleaned_4_27_2026/regression_zettabyte_cleaned_4_27_2026/run_0/checkpoints_regression/StormCastUNet.0.8000.mdlus}"
REG_NO_LOG1P="${REG_NO_LOG1P:-${RUNS_ROOT}/regression_zettabyte_v1_cleaned_4_27_2026_NO_log1p/regression_cleaned_NO_log1p/run_0/checkpoints_regression/StormCastUNet.0.10000.mdlus}"

# Run dirs (the script picks
# checkpoints_diffusion/EDMPrecond.0.${CHECKPOINT_STEP}.mdlus and
# ema_state.pt out of each).
DIFF_LOG1P_DIR="${DIFF_LOG1P_DIR:-${RUNS_ROOT}/diffusion_zettabyte_v1_cleaned_4_27_2026/diffusion_zettabyte_cleaned_4_27_2026/run_0}"
DIFF_NO_LOG1P_DIR="${DIFF_NO_LOG1P_DIR:-${RUNS_ROOT}/diffusion_zettabyte_v1_cleaned_4_27_2026_NO_log1p/edm_cleaned_NO_log1p/run_0}"
# train_flowcast.sh writes to .../flowcast_zettabyte_cleaned_4_27_2026/run_0 (with the
# _zettabyte_ infix). zettabyte_scripts/inference_flowcast.sh hard-codes a stale
# .../flowcast_cleaned_4_27_2026/run_0 path -- ignore that, the training script is
# authoritative. Override FLOW_LOG1P_DIR if you re-ran training under a different name.
FLOW_LOG1P_DIR="${FLOW_LOG1P_DIR:-${RUNS_ROOT}/flowcast_zettabyte_v1_cleaned_4_27_2026/flowcast_zettabyte_cleaned_4_27_2026/run_0}"
FLOW_NO_LOG1P_DIR="${FLOW_NO_LOG1P_DIR:-${RUNS_ROOT}/flowcast_zettabyte_v1_cleaned_4_27_2026_NO_log1p/flowcast_cleaned_NO_log1p/run_0}"

# --- Evaluation knobs -------------------------------------------------------
# Pass MAX_SAMPLES=200 for a 5-minute smoke test before committing to the
# full 8,759-hour validation year.
MAX_SAMPLES="${MAX_SAMPLES:-}"
SPECTRUM_STRIDE="${SPECTRUM_STRIDE:-24}"   # PS1D every N samples; 24 = 1/day
ROLLOUT_STEPS="${ROLLOUT_STEPS:-12}"        # autoregressive horizon per case
NO_ROLLOUTS="${NO_ROLLOUTS:-0}"             # set 1 to skip rollouts entirely
INCLUDE_REGRESSION="${INCLUDE_REGRESSION:-0}"  # set 1 to add regression-only

# Case-study initial times (ISO-8601). Each one launches an autoregressive
# rollout for every model. Cover convective, typhoon-season and NE-monsoon
# regimes so the qualitative panels span the 2022 valid year.
CASES=(
    "2022-07-15T00:00:00"
    "2022-09-12T00:00:00"
    "2022-12-08T00:00:00"
)

mkdir -p "${OUT_DIR}"

# --- Pre-flight: bail loudly if any input is missing ------------------------
for path in "${DATA_LOG1P}" "${DATA_RAW}" \
            "${REG_LOG1P}" "${REG_NO_LOG1P}" \
            "${DIFF_LOG1P_DIR}/checkpoints_diffusion/EDMPrecond.0.${CHECKPOINT_STEP}.mdlus" \
            "${DIFF_NO_LOG1P_DIR}/checkpoints_diffusion/EDMPrecond.0.${CHECKPOINT_STEP}.mdlus" \
            "${FLOW_LOG1P_DIR}/ema_state.pt" \
            "${FLOW_NO_LOG1P_DIR}/ema_state.pt"; do
    if [ ! -e "${path}" ]; then
        echo "[run_compare] missing required path: ${path}" >&2
        exit 1
    fi
done

# --- Build the python invocation --------------------------------------------
ARGS=(
    --out-dir "${OUT_DIR}"
    --runs-root "${RUNS_ROOT}"
    --data-log1p "${DATA_LOG1P}"
    --data-raw "${DATA_RAW}"
    --checkpoint-step "${CHECKPOINT_STEP}"
    --reg-log1p "${REG_LOG1P}"
    --reg-no-log1p "${REG_NO_LOG1P}"
    --diff-log1p-dir "${DIFF_LOG1P_DIR}"
    --diff-no-log1p-dir "${DIFF_NO_LOG1P_DIR}"
    --flow-log1p-dir "${FLOW_LOG1P_DIR}"
    --flow-no-log1p-dir "${FLOW_NO_LOG1P_DIR}"
    --rollout-steps "${ROLLOUT_STEPS}"
    --spectrum-stride "${SPECTRUM_STRIDE}"
    --cases "${CASES[@]}"
)

if [ -n "${MAX_SAMPLES}" ]; then
    ARGS+=(--max-samples "${MAX_SAMPLES}")
fi
if [ "${NO_ROLLOUTS}" = "1" ]; then
    ARGS+=(--no-rollouts)
fi
if [ "${INCLUDE_REGRESSION}" = "1" ]; then
    ARGS+=(--include-regression)
fi

echo "[run_compare] launching log1p_ablation.py"
echo "  out_dir         = ${OUT_DIR}"
echo "  runs_root       = ${RUNS_ROOT}"
echo "  data (log1p)    = ${DATA_LOG1P}"
echo "  data (raw)      = ${DATA_RAW}"
echo "  checkpoint_step = ${CHECKPOINT_STEP}"
echo "  cases           = ${CASES[*]}"

python -u "${ABLATION_SCRIPT}" "${ARGS[@]}" 2>&1 | tee "${OUT_DIR}/run.log"

echo
echo "[run_compare] done — outputs at ${OUT_DIR}"
