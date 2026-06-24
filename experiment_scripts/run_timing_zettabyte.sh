#!/bin/bash
# Inference-timing benchmark — ZETTABYTE-cluster variant of run_timing.sh.
#
# Identical methodology (stormcast EDM / flowcast @ 10,15,20 / meanflow @ 1,2,
# ENSEMBLE=50, 24 h rollout, one GPU, timing only) but pointed at the zettabyte
# cluster's paths + environment, which differ from the local workstation:
#
#   local workstation           zettabyte cluster
#   ---------------------------------------------------------------------------
#   in-repo stormcast_env venv  conda env  `stormcast_env`
#   <repo>/runs/<run>/...       /data/exp_3_train_2_5_yrs_val_1yr_tp1/<run>/...
#   <repo>/exp_.../zarr_...     /workspace/downloads/zarr_..._cleaned_4_27_2026
#   auto-pick a free A6000      CUDA_VISIBLE_DEVICES=0  (matches the zettabyte
#                               inference/train scripts)
#
# Path nesting is taken from the zettabyte TRAIN scripts (the authoritative
# source for where checkpoints are written), which agree with the downloaded
# local mirror under runs/:
#   ${ZB_DATA_ROOT}/<method>_zettabyte_v1_cleaned_4_27_2026/
#                   <method>_zettabyte_cleaned_4_27_2026/run_0/
#                   checkpoints_<method>/<Prefix>.0.<step>.mdlus
# (The lone `flowcast_cleaned_4_27_2026` ema path in inference_flowcast.sh is a
#  one-off and is NOT the trained-checkpoint layout — ignored here.)
#
# Checkpoints are resolved to the LATEST step present in each dir at run time
# (timing is step-independent — same architecture — so "latest" is always fine);
# a canonical step is used only as a fallback when the glob finds nothing.
#
# Everything is overridable; this is a thin wrapper that sets zettabyte defaults
# and then exec's run_timing.sh (the single source of truth for the actual
# timing loop). All run_timing.sh knobs pass straight through, e.g.:
#   ENSEMBLE=10 N_STEPS=12 GPU=1 ./run_timing_zettabyte.sh
#   PRINT_ONLY=1 ./run_timing_zettabyte.sh        # show resolved paths, don't run
#   SKIP_CONDA=1 ./run_timing_zettabyte.sh        # don't touch conda

set -euo pipefail

log()  { printf '[%s] [run_timing_zb] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
warn() { printf '[%s] [run_timing_zb][WARN] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CORE="${SCRIPT_DIR}/run_timing.sh"
if [ ! -x "${CORE}" ]; then
    warn "core timing script not found/executable: ${CORE}"
    exit 1
fi

# --- Zettabyte roots (override if your cluster layout differs) ---------------
ZB_DATA_ROOT="${ZB_DATA_ROOT:-/data/exp_3_train_2_5_yrs_val_1yr_tp1}"
ZB_ZARR="${ZB_ZARR:-/workspace/downloads/zarr_exp3_L_24_H_24_train_2_5_years_full_cleaned_4_27_2026}"
RUN_ID="${RUN_ID:-0}"
CONDA_ENV="${CONDA_ENV:-stormcast_env}"

# --- Activate the zettabyte conda env (best-effort) --------------------------
# The zettabyte train/inference scripts all do exactly this. Best-effort so a
# missing/broken conda doesn't hard-fail; run_timing.sh falls through to
# whatever python ends up on PATH.
if [ "${SKIP_CONDA:-0}" != "1" ]; then
    log "conda activate ${CONDA_ENV} ..."
    _CB="$(conda info --base 2>/dev/null || true)"
    if [ -n "${_CB}" ] && [ -f "${_CB}/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "${_CB}/etc/profile.d/conda.sh" >/dev/null 2>&1 || true
        conda activate "${CONDA_ENV}" >/dev/null 2>&1 || \
            warn "conda activate ${CONDA_ENV} failed; continuing with current python"
    else
        warn "conda not usable; continuing with current python"
    fi
else
    log "SKIP_CONDA=1 — not touching conda"
fi
# Pin run_timing.sh to the (now conda-activated) python rather than letting it
# look for an in-repo stormcast_env venv that doesn't exist on zettabyte.
export PYTHON="${PYTHON:-$(command -v python)}"

# --- Pin ONE GPU (zettabyte inference convention = GPU 0) --------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU:-0}}"

# --- Resolve latest .mdlus per method (fallback = canonical step) ------------
# Filename layout: <Prefix>.0.<step>.mdlus  -> sort numerically on the 3rd
# dot-field to get the highest step.
resolve_ckpt() {
    # $1 run_name  $2 experiment_name  $3 ckpt_subdir  $4 prefix  $5 fallback_step
    local dir="${ZB_DATA_ROOT}/$1/$2/run_${RUN_ID}/$3"
    local latest
    latest="$(ls -1 "${dir}/$4".0.*.mdlus 2>/dev/null | sort -t. -k3,3n | tail -1 || true)"
    if [ -n "${latest}" ]; then
        printf '%s' "${latest}"
    else
        printf '%s' "${dir}/$4.0.$5.mdlus"
    fi
}

export DATA_LOCATION="${DATA_LOCATION:-${ZB_ZARR}}"
export REGRESSION_CKPT="${REGRESSION_CKPT:-$(resolve_ckpt regression_zettabyte_v1_cleaned_4_27_2026 regression_zettabyte_cleaned_4_27_2026 checkpoints_regression StormCastUNet 16000)}"
export DIFFUSION_CKPT="${DIFFUSION_CKPT:-$(resolve_ckpt diffusion_zettabyte_v1_cleaned_4_27_2026 diffusion_zettabyte_cleaned_4_27_2026 checkpoints_diffusion EDMPrecond 31000)}"
export FLOWCAST_CKPT="${FLOWCAST_CKPT:-$(resolve_ckpt flowcast_zettabyte_v1_cleaned_4_27_2026 flowcast_zettabyte_cleaned_4_27_2026 checkpoints_flowcast FlowCastPrecond 140000)}"
export MEANFLOW_CKPT="${MEANFLOW_CKPT:-$(resolve_ckpt meanflow_zettabyte_v1_cleaned_4_27_2026 meanflow_zettabyte_cleaned_4_27_2026 checkpoints_meanflow MeanFlowPrecond 20000)}"

# Separate output dir so zettabyte timings don't overwrite the local ones.
export OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/results/timing_zettabyte}"

log "ZB_DATA_ROOT = ${ZB_DATA_ROOT}"
log "DATA_LOCATION= ${DATA_LOCATION}"
log "PYTHON       = ${PYTHON}"
log "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
log "OUT_DIR      = ${OUT_DIR}"
log "regression   = ${REGRESSION_CKPT}"
log "diffusion    = ${DIFFUSION_CKPT}"
log "flowcast     = ${FLOWCAST_CKPT}"
log "meanflow     = ${MEANFLOW_CKPT}"

if [ "${PRINT_ONLY:-0}" = "1" ]; then
    log "PRINT_ONLY=1 — resolved config printed above; not running."
    exit 0
fi

log "handing off to ${CORE}"
exec "${CORE}"
