#!/bin/bash
# FlowCast NFE Pareto sweep -- vary the Euler step count S from 1 .. 50 on a
# fixed FlowCast checkpoint and record the resulting quality/latency curve.
#
# This is the §6 "NFE Pareto sweep" of `metrics.md` (Slide 25 of the
# presentation): tests whether the FlowCast paper's "saturates at 3-10 NFE"
# headline holds on the Taiwan RWRF domain.
#
# Logging style and env-var conventions match run_main_experiment.sh.

set -euo pipefail

# --- Logging helpers --------------------------------------------------------
SCRIPT_T0=$(date +%s)
log()    { printf '[%s] [nfe_sweep] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
warn()   { printf '[%s] [nfe_sweep][WARN] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2; }
banner() {
    printf '\n================================================================\n'
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
    printf '================================================================\n'
}
elapsed_s() {
    local t0="$1"; local now; now=$(date +%s); printf '%ds' $((now - t0))
}

banner "FlowCast NFE Pareto sweep"

# --- Optional conda activate -------------------------------------------------
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
    log "skipping conda activation (in '${CONDA_DEFAULT_ENV:-?}' / SKIP_CONDA=${SKIP_CONDA:-0})"
fi
log "python : $(command -v python)"
log "version: $(python --version 2>&1)"
log "host   : $(hostname)  user=$(whoami)  pid=$$"

# --- Config ------------------------------------------------------------------
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"

# Which students to sweep. Default sweeps BOTH FlowCast (Euler ODE) and MeanFlow
# (few-step average velocity) so the two NFE-quality curves can be overlaid --
# MeanFlow's headline claim is that it saturates at 1-2 NFE where FlowCast needs
# ~3-10. Each method writes results/<method>_nfe_sweep/. Set METHODS="flowcast"
# to restore the legacy single-method behaviour.
METHODS="${METHODS:-flowcast meanflow}"

# Sweep set -- FlowCast paper Fig. 5 extended to the diffusion regime.
NFES="${NFES:-1 2 3 4 5 6 8 10 12 16 20 25 32 40 50}"

# Sample budget per NFE point. Defaults are tuned so the full sweep takes
# ~20-40 min on one A6000: S sequences x T steps x K members repeated
# len(NFES) times, with the per-NFE cost roughly proportional to its NFE.
N_SEQUENCES="${N_SEQUENCES:-12}"
N_STEPS="${N_STEPS:-1}"       # single-step skill, matches FlowCast Fig. 5
ENSEMBLE="${ENSEMBLE:-10}"    # 10-member standard (matches run_main_experiment.sh)
SEED="${SEED:-0}"
SOLVER="${SOLVER:-euler}"

# Default = latest cleaned FlowCast (~13 M samples). For the matched-2M
# scoreboard comparison use FlowCastPrecond.0.20000.mdlus instead.
DATA="${DATA:-${REPO_ROOT}/exp_3_train_2_5_yrs_val_1yr_tp1/zarr_exp3_L_24_H_24_train_2_5_years_full_cleaned_4_27_2026}"
REG="${REG:-${REPO_ROOT}/runs/regression_zettabyte_v1_cleaned_4_27_2026/regression_zettabyte_cleaned_4_27_2026/run_0/checkpoints_regression/StormCastUNet.0.8000.mdlus}"
FLOW="${FLOW:-${REPO_ROOT}/runs/flowcast_zettabyte_v1_cleaned_4_27_2026/flowcast_zettabyte_cleaned_4_27_2026/run_0/checkpoints_flowcast/FlowCastPrecond.0.140000.mdlus}"
MEANFLOW="${MEANFLOW:-${REPO_ROOT}/runs/meanflow_zettabyte_v1_cleaned_4_27_2026/meanflow_zettabyte_cleaned_4_27_2026/run_0/checkpoints_meanflow/MeanFlowPrecond.0.20000.mdlus}"

banner "Configuration"
log "REPO_ROOT   = ${REPO_ROOT}"
log "METHODS     = ${METHODS}"
log "NFES        = ${NFES}"
log "N_SEQUENCES = ${N_SEQUENCES}    N_STEPS = ${N_STEPS}    ENSEMBLE = ${ENSEMBLE}"
log "SOLVER      = ${SOLVER}    SEED = ${SEED}"
log "DATA        = ${DATA}"
log "REG         = ${REG}"
log "FLOW        = ${FLOW}"
log "MEANFLOW    = ${MEANFLOW}"

# --- Pre-flight --------------------------------------------------------------
banner "Pre-flight"
FAIL=0
PREFLIGHT=("DATA=${DATA}" "REG=${REG}")
for method in ${METHODS}; do
    case "${method}" in
        flowcast) PREFLIGHT+=("FLOW=${FLOW}") ;;
        meanflow) PREFLIGHT+=("MEANFLOW=${MEANFLOW}") ;;
        *) warn "unknown method '${method}' in METHODS"; FAIL=1 ;;
    esac
done
for kv in "${PREFLIGHT[@]}"; do
    name="${kv%%=*}"; path="${kv#*=}"
    if [ -e "${path}" ]; then
        log "  OK     ${name}=${path}"
    else
        warn "  MISSING ${name}=${path}"; FAIL=1
    fi
done
[ "${FAIL}" = "1" ] && { warn "pre-flight failed"; exit 1; }

# --- Run (one sweep per method) ---------------------------------------------
for method in ${METHODS}; do
    OUT_DIR="${REPO_ROOT}/experiment_scripts/results/${method}_nfe_sweep"
    mkdir -p "${OUT_DIR}"
    if [ "${method}" = "meanflow" ]; then
        CKPT_FLAG=(--meanflow-checkpoint "${MEANFLOW}")
    else
        CKPT_FLAG=(--flowcast-checkpoint "${FLOW}")
    fi

    banner "Running ${method} sweep across NFEs: ${NFES}"
    log "log -> ${OUT_DIR}/run.log"
    LEG_T0=$(date +%s)
    # shellcheck disable=SC2086   # NFES intentionally word-splits
    python -u "${REPO_ROOT}/experiment_scripts/flowcast_nfe_sweep.py" \
        --method "${method}" \
        --output-dir "${OUT_DIR}" \
        --data-location "${DATA}" \
        --valid-dates 2022/01/01 2022/12/31 \
        --hr-size 192 96 \
        --qpepre-log1p \
        --regression-checkpoint "${REG}" \
        "${CKPT_FLAG[@]}" \
        --nfes ${NFES} \
        --solver "${SOLVER}" \
        --n-sequences "${N_SEQUENCES}" \
        --n-steps "${N_STEPS}" \
        --ensemble "${ENSEMBLE}" \
        --seed "${SEED}" \
        2>&1 | tee "${OUT_DIR}/run.log"

    RC=${PIPESTATUS[0]}
    if [ "${RC}" != "0" ]; then
        warn "${method} sweep failed with exit code ${RC} after $(elapsed_s ${LEG_T0})"
        exit "${RC}"
    fi
    log "${method} sweep done in $(elapsed_s ${LEG_T0})"
    if [ -f "${OUT_DIR}/nfe_sweep.md" ]; then
        log "preview of ${method} nfe_sweep.md:"
        sed 's/^/[nfe_sweep]     | /' "${OUT_DIR}/nfe_sweep.md"
    fi
done

# --- Summary ---------------------------------------------------------------
banner "All done -- total elapsed $(elapsed_s ${SCRIPT_T0})"
log "outputs (per method):"
for method in ${METHODS}; do
    OUT_DIR="${REPO_ROOT}/experiment_scripts/results/${method}_nfe_sweep"
    log "  ${method}: ${OUT_DIR}/nfe_sweep.{csv,md} + plot_{quality_vs_nfe,time_vs_nfe,pareto}.png"
done
