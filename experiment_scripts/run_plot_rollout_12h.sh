#!/bin/bash
# 12-hour autoregressive rollout: one figure per variable comparing RWRF truth
# against three model rows -- StormCast on the legacy domain, StormCast on the
# cleaned domain, FlowCast (NFE=10). All three checkpoints are pinned to the
# matched ~2 M training-sample budget that drives run_main_experiment.sh and
# run_single_time_exp.sh.
#
# Same plotting style as run_single_time_exp.sh: shared colour scale per
# figure, origin=lower, per-figure colorbar; one PNG per channel.
#
# Models / inputs (same defaults as run_main_experiment.sh / run_single_time_exp.sh):
#   A. Legacy StormCast    -- exp_3 legacy regression + EDM (224x128, raw qpepre)
#   B. Cleaned StormCast   -- runs/diffusion_zettabyte_v1_cleaned_4_27_2026 EDM
#                             (192x96, log1p qpepre)
#   C. FlowCast            -- runs/flowcast_zettabyte_v1_cleaned_4_27_2026 CFM
#                             (192x96, log1p qpepre)
#
# Output (default ${OUT_DIR}=${REPO_ROOT}/experiment_scripts/results/rollout_12h):
#   ${OUT_DIR}/rollout_12h_<channel>.png  -- one per channel in CHANNELS
#   ${OUT_DIR}/run.log                    -- full python stdout+stderr

set -euo pipefail

SCRIPT_T0=$(date +%s)
log()    { printf '[%s] [rollout] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
warn()   { printf '[%s] [rollout][WARN] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2; }
banner() {
    local msg="$1"
    printf '\n================================================================\n'
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$msg"
    printf '================================================================\n'
}
elapsed_s() {
    local t0="$1"; local now
    now=$(date +%s); printf '%ds' $((now - t0))
}

banner "12 h autoregressive rollout (3-model comparison)"

# Optional conda activation (best-effort; matches run_main_experiment.sh).
if [ "${SKIP_CONDA:-0}" != "1" ] && \
   { [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "${CONDA_DEFAULT_ENV:-base}" = "base" ]; }; then
    log "trying conda activate ${STORMCAST_ENV_NAME:-stormcast_env} ..."
    _CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    if [ -n "${_CONDA_BASE}" ] && [ -f "${_CONDA_BASE}/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "${_CONDA_BASE}/etc/profile.d/conda.sh" >/dev/null 2>&1 || true
        conda activate "${STORMCAST_ENV_NAME:-stormcast_env}" >/dev/null 2>&1 || \
            warn "conda activate failed; continuing with current python"
    else
        warn "conda not usable; continuing with current python"
    fi
else
    log "skipping conda activation (already in '${CONDA_DEFAULT_ENV:-?}' / SKIP_CONDA=${SKIP_CONDA:-0})"
fi
log "python : $(command -v python)"
log "version: $(python --version 2>&1)"
log "host   : $(hostname)  user=$(whoami)  pid=$$"
log "cwd    : $(pwd)"
log "cuda   : ${CUDA_VISIBLE_DEVICES:-<unset>}"

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/experiment_scripts/results/rollout_12h}"
mkdir -p "${OUT_DIR}"

# ---- Run knobs -------------------------------------------------------------
T0_IDX="${T0_IDX:-4128}"                  # valid index, default 2022-06-23 00 UTC
N_STEPS="${N_STEPS:-12}"
HOURS_TO_PLOT="${HOURS_TO_PLOT:-1 3 6 12}"
CHANNELS="${CHANNELS:-t2m u10 v10 qpepre}"

DIFFUSION_NFE="${DIFFUSION_NFE:-18}"      # Heun => 36 NFE
FLOWCAST_NFE="${FLOWCAST_NFE:-10}"
SEED="${SEED:-0}"

# ---- Pinned ~2 M-sample checkpoints (overridable) --------------------------
# Defaults follow run_main_experiment.sh / run_single_time_exp.sh.
# Legacy domain (224x128, raw qpepre)
LEGACY_DATA="${LEGACY_DATA:-${REPO_ROOT}/exp_3_train_2_5_yrs_val_1yr_tp1/zarr_exp3_L_24_H_24_train_2_5_years_full}"
LEGACY_REG="${LEGACY_REG:-${REPO_ROOT}/exp_3_train_2_5_yrs_val_1yr_tp1/exp_3_reg_L_24_H_4_train_2_5_years/0/checkpoints_regression/StormCastUNet.0.7500.mdlus}"
LEGACY_EDM="${LEGACY_EDM:-${REPO_ROOT}/exp_3_train_2_5_yrs_val_1yr_tp1/exp_3_dif_L_24_H_4_train_2_5_years/0/checkpoints_diffusion/EDMPrecond.0.70000.mdlus}"

# Cleaned domain (192x96, log1p qpepre) -- both EDM and FlowCast at ~2 M samples
CLEANED_DATA="${CLEANED_DATA:-${REPO_ROOT}/exp_3_train_2_5_yrs_val_1yr_tp1/zarr_exp3_L_24_H_24_train_2_5_years_full_cleaned_4_27_2026}"
CLEANED_REG="${CLEANED_REG:-${REPO_ROOT}/runs/regression_zettabyte_v1_cleaned_4_27_2026/regression_zettabyte_cleaned_4_27_2026/run_0/checkpoints_regression/StormCastUNet.0.8000.mdlus}"
CLEANED_EDM="${CLEANED_EDM:-${REPO_ROOT}/runs/diffusion_zettabyte_v1_cleaned_4_27_2026/diffusion_zettabyte_cleaned_4_27_2026/run_0/checkpoints_diffusion/EDMPrecond.0.31000.mdlus}"
CLEANED_FLOW="${CLEANED_FLOW:-${REPO_ROOT}/runs/flowcast_zettabyte_v1_cleaned_4_27_2026/flowcast_zettabyte_cleaned_4_27_2026/run_0/checkpoints_flowcast/FlowCastPrecond.0.20000.mdlus}"

banner "Configuration"
log "REPO_ROOT     = ${REPO_ROOT}"
log "OUT_DIR       = ${OUT_DIR}"
log "T0_IDX        = ${T0_IDX}    (validation hourly index for rollout start)"
log "N_STEPS       = ${N_STEPS} h (autoregressive horizon)"
log "HOURS_TO_PLOT = ${HOURS_TO_PLOT}  (lead-time columns)"
log "CHANNELS      = ${CHANNELS}      (one PNG per channel)"
log "DIFFUSION_NFE = ${DIFFUSION_NFE}   (Heun steps; NFE = 2 * this)"
log "FLOWCAST_NFE  = ${FLOWCAST_NFE}   (Euler steps)"
log "SEED          = ${SEED}"
log ""
log "Inputs:"
log "  LEGACY_DATA  = ${LEGACY_DATA}"
log "  LEGACY_REG   = ${LEGACY_REG}"
log "  LEGACY_EDM   = ${LEGACY_EDM}"
log "  CLEANED_DATA = ${CLEANED_DATA}"
log "  CLEANED_REG  = ${CLEANED_REG}"
log "  CLEANED_EDM  = ${CLEANED_EDM}"
log "  CLEANED_FLOW = ${CLEANED_FLOW}"

banner "Pre-flight checks"
_PREFLIGHT_FAIL=0
for kv in \
    "LEGACY_DATA=${LEGACY_DATA}" \
    "LEGACY_REG=${LEGACY_REG}" \
    "LEGACY_EDM=${LEGACY_EDM}" \
    "CLEANED_DATA=${CLEANED_DATA}" \
    "CLEANED_REG=${CLEANED_REG}" \
    "CLEANED_EDM=${CLEANED_EDM}" \
    "CLEANED_FLOW=${CLEANED_FLOW}"; do
    name="${kv%%=*}"
    path="${kv#*=}"
    if [ -e "${path}" ]; then
        log "  OK     ${name}=${path}"
    else
        warn "  MISSING ${name}=${path}"
        _PREFLIGHT_FAIL=1
    fi
done
if [ "${_PREFLIGHT_FAIL}" = "1" ]; then
    warn "pre-flight failed -- aborting before any python invocation"
    exit 1
fi
log "pre-flight OK"

banner "Running plot_rollout_12h.py"
T0=$(date +%s)
python -u "${REPO_ROOT}/experiment_scripts/plot_rollout_12h.py" \
    --original-data "${LEGACY_DATA}" \
    --original-regression "${LEGACY_REG}" \
    --original-diffusion "${LEGACY_EDM}" \
    --new-data "${CLEANED_DATA}" \
    --new-regression "${CLEANED_REG}" \
    --new-diffusion "${CLEANED_EDM}" \
    --new-flowcast "${CLEANED_FLOW}" \
    --valid-dates 2022/01/01 2022/12/31 \
    --output-dir "${OUT_DIR}" \
    --t0-idx "${T0_IDX}" \
    --n-steps "${N_STEPS}" \
    --hours-to-plot ${HOURS_TO_PLOT} \
    --channels-to-plot ${CHANNELS} \
    --seed "${SEED}" \
    --diffusion-num-steps "${DIFFUSION_NFE}" \
    --diffusion-solver heun \
    --flowcast-num-steps "${FLOWCAST_NFE}" \
    --flowcast-solver euler \
    2>&1 | tee "${OUT_DIR}/run.log"
RC=${PIPESTATUS[0]}
if [ "${RC}" != "0" ]; then
    warn "python failed with exit code ${RC} after $(elapsed_s ${T0})"
    exit "${RC}"
fi
log "python done in $(elapsed_s ${T0})"

banner "All done — total elapsed $(elapsed_s ${SCRIPT_T0})"
log "outputs:"
log "  PNGs    : ${OUT_DIR}/rollout_12h_*.png"
log "  run log : ${OUT_DIR}/run.log"
NPNG=$(find "${OUT_DIR}" -maxdepth 1 -name 'rollout_12h_*.png' 2>/dev/null | wc -l)
log "rendered ${NPNG} PNG(s)"
