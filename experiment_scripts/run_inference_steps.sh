#!/bin/bash
# Per-step inference visualizations: for N=3 evenly-spaced timestamps from the
# 2022 validation year, run both the cleaned StormCast (EDM, 18 Heun steps)
# and FlowCast (CFM, 10 Euler steps) and save the sample state after EVERY
# ODE step as its own PNG. Same plotting style as run_single_time_exp.sh.
#
# Output layout: each variable plotted separately, no titles / labels / cbars.
# Two views per sampler step:
#   forecast   = M_t + R_t denormalized (the one-hour-ahead prediction)
#   residual   = forecast - regression (the raw residual R_t in physical units)
#
#   ${OUT_DIR}/time_NN_<timestamp>/
#       ├── truth/<channel>.png
#       ├── regression/<channel>.png
#       ├── truth_residual/<channel>.png             # truth - regression
#       ├── stormcast/<channel>/step_01.png .. step_18.png            # forecast
#       ├── stormcast_residual/<channel>/step_01.png .. step_18.png   # residual
#       ├── flowcast/<channel>/step_01.png .. step_10.png             # forecast
#       └── flowcast_residual/<channel>/step_01.png .. step_10.png    # residual
#
# Each PNG is a bare imshow (origin=lower, viridis, no axes / colorbar / frame).
# vmin/vmax is locked globally per channel -- forecast view uses the truth's
# range, residual view uses (truth - regression)'s range. Early diffusion
# steps will saturate (that is the sigma_max noise floor, by design).

set -euo pipefail

SCRIPT_T0=$(date +%s)
log()    { printf '[%s] [infer_steps] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
warn()   { printf '[%s] [infer_steps][WARN] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2; }
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

banner "Per-step inference visualization"

# Optional conda activation (best-effort).
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
log "cuda   : ${CUDA_VISIBLE_DEVICES:-<unset>}"

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/experiment_scripts/results/inference_steps}"
mkdir -p "${OUT_DIR}"

N_TIMES="${N_TIMES:-3}"
SEED="${SEED:-0}"
DIFFUSION_NFE="${DIFFUSION_NFE:-18}"
FLOWCAST_NFE="${FLOWCAST_NFE:-10}"

# Cleaned-grid defaults (so StormCast and FlowCast are apples-to-apples;
# legacy 224x128 StormCast cannot share a checkpoint with FlowCast since
# FlowCast was never trained at that grid / encoding).
DATA="${DATA:-${REPO_ROOT}/exp_3_train_2_5_yrs_val_1yr_tp1/zarr_exp3_L_24_H_24_train_2_5_years_full_cleaned_4_27_2026}"
REG="${REG:-${REPO_ROOT}/runs/regression_zettabyte_v1_cleaned_4_27_2026/regression_zettabyte_cleaned_4_27_2026/run_0/checkpoints_regression/StormCastUNet.0.8000.mdlus}"
EDM="${EDM:-${REPO_ROOT}/runs/diffusion_zettabyte_v1_cleaned_4_27_2026/diffusion_zettabyte_cleaned_4_27_2026/run_0/checkpoints_diffusion/EDMPrecond.0.31000.mdlus}"
FLOW="${FLOW:-${REPO_ROOT}/runs/flowcast_zettabyte_v1_cleaned_4_27_2026/flowcast_zettabyte_cleaned_4_27_2026/run_0/checkpoints_flowcast/FlowCastPrecond.0.20000.mdlus}"

banner "Configuration"
log "REPO_ROOT     = ${REPO_ROOT}"
log "OUT_DIR       = ${OUT_DIR}"
log "DATA          = ${DATA}"
log "REG           = ${REG}"
log "EDM           = ${EDM}"
log "FLOW          = ${FLOW}"
log "N_TIMES       = ${N_TIMES}"
log "DIFFUSION_NFE = ${DIFFUSION_NFE}  (Heun steps)"
log "FLOWCAST_NFE  = ${FLOWCAST_NFE}  (Euler steps)"
log "SEED          = ${SEED}"

banner "Pre-flight checks"
_PREFLIGHT_FAIL=0
for kv in "DATA=${DATA}" "REG=${REG}" "EDM=${EDM}" "FLOW=${FLOW}"; do
    name="${kv%%=*}"; path="${kv#*=}"
    if [ -e "${path}" ]; then
        log "  OK     ${name}=${path}"
    else
        warn "  MISSING ${name}=${path}"
        _PREFLIGHT_FAIL=1
    fi
done
if [ "${_PREFLIGHT_FAIL}" = "1" ]; then
    warn "pre-flight failed -- aborting"
    exit 1
fi

banner "Running inference_steps.py"
T0=$(date +%s)
python -u "${REPO_ROOT}/experiment_scripts/inference_steps.py" \
    --data "${DATA}" \
    --regression "${REG}" \
    --diffusion "${EDM}" \
    --flowcast  "${FLOW}" \
    --valid-dates 2022/01/01 2022/12/31 \
    --output-dir "${OUT_DIR}" \
    --n-times "${N_TIMES}" \
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

banner "All done -- total elapsed $(elapsed_s ${SCRIPT_T0})"
log "outputs:"
log "  per-time dir : ${OUT_DIR}/time_NN_<timestamp>/"
log "  reference    : .../{truth,regression,truth_residual}/<channel>.png"
log "  stormcast    : .../stormcast{,_residual}/<channel>/step_01.png .. step_${DIFFUSION_NFE}.png"
log "  flowcast     : .../flowcast{,_residual}/<channel>/step_01.png .. step_${FLOWCAST_NFE}.png"
log "  metadata     : ${OUT_DIR}/metadata.txt"
log "  run log      : ${OUT_DIR}/run.log"
NPNG=$(find "${OUT_DIR}" -type f -name '*.png' 2>/dev/null | wc -l)
log "rendered ${NPNG} PNG(s)"
