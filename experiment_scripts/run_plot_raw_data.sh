#!/bin/bash
# Render raw HighRes + LowRes snapshots at N evenly-spaced timestamps across
# the 2022 validation year, using the same plotting style as
# run_single_time_exp.sh (viridis, origin=lower, per-panel colorbar).
#
# Each variable is rendered as its own single-panel PNG. For each timestamp
# we write 4 HighRes PNGs (t2m, u10, v10, qpepre) and 24 LowRes PNGs, grouped
# by channel so the same variable across time lives in one directory. The
# two invariant fields (land-sea mask, orography) are time-independent and
# are written once:
#
#   ${OUT_DIR}/highres/<channel>/time_NN.png
#   ${OUT_DIR}/lowres/<channel>/time_NN.png
#   ${OUT_DIR}/invariants/<channel>.png
#
# Defaults point at the cleaned dataset (192x96, log1p qpepre); override
# DATA / HR_H / HR_W / QPEPRE_LOG1P for the legacy raw dataset.

set -euo pipefail

SCRIPT_T0=$(date +%s)
log()    { printf '[%s] [plot_raw] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
warn()   { printf '[%s] [plot_raw][WARN] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2; }
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

banner "Raw HighRes + LowRes plotting"

# Optional conda activation (best-effort; matches run_single_time_exp.sh).
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

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/experiment_scripts/results/raw_data_plottings}"
mkdir -p "${OUT_DIR}"

N_TIMES="${N_TIMES:-10}"

# Defaults: cleaned dataset (192x96, log1p qpepre). Override DATA / HR_H /
# HR_W / QPEPRE_LOG1P to point at the legacy raw 224x128 dataset.
DATA="${DATA:-${REPO_ROOT}/exp_3_train_2_5_yrs_val_1yr_tp1/zarr_exp3_L_24_H_24_train_2_5_years_full_cleaned_4_27_2026}"
HR_H="${HR_H:-192}"
HR_W="${HR_W:-96}"
QPEPRE_LOG1P="${QPEPRE_LOG1P:-true}"

banner "Configuration"
log "REPO_ROOT     = ${REPO_ROOT}"
log "OUT_DIR       = ${OUT_DIR}"
log "DATA          = ${DATA}"
log "HR_SIZE       = ${HR_H} x ${HR_W}"
log "QPEPRE_LOG1P  = ${QPEPRE_LOG1P}"
log "N_TIMES       = ${N_TIMES}"

banner "Pre-flight checks"
if [ ! -e "${DATA}" ]; then
    warn "MISSING DATA=${DATA}"
    exit 1
fi
log "  OK     DATA=${DATA}"

banner "Running plot_raw_data.py"
T0=$(date +%s)
python -u "${REPO_ROOT}/experiment_scripts/plot_raw_data.py" \
    --data "${DATA}" \
    --hr-size "${HR_H}" "${HR_W}" \
    --qpepre-log1p "${QPEPRE_LOG1P}" \
    --valid-dates 2022/01/01 2022/12/31 \
    --output-dir "${OUT_DIR}" \
    --n-times "${N_TIMES}" \
    2>&1 | tee "${OUT_DIR}/run.log"
RC=${PIPESTATUS[0]}
if [ "${RC}" != "0" ]; then
    warn "python failed with exit code ${RC} after $(elapsed_s ${T0})"
    exit "${RC}"
fi
log "python done in $(elapsed_s ${T0})"

banner "All done -- total elapsed $(elapsed_s ${SCRIPT_T0})"
log "outputs:"
log "  HighRes PNGs   : ${OUT_DIR}/highres/<channel>/time_*.png"
log "  LowRes  PNGs   : ${OUT_DIR}/lowres/<channel>/time_*.png"
log "  Invariant PNGs : ${OUT_DIR}/invariants/<channel>.png"
log "  metadata       : ${OUT_DIR}/metadata.txt"
log "  run log        : ${OUT_DIR}/run.log"
NPNG=$(find "${OUT_DIR}" -type f -name '*.png' 2>/dev/null | wc -l)
log "rendered ${NPNG} PNG(s)"
