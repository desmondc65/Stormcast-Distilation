#!/bin/bash
# Main experiment: legacy old-stormcast baseline vs. new (cleaned + log1p +
# qpw-weighted) stormcast and FlowCast students at matched 2 M-sample budget.
#
# Produces multiple rows in the final scoreboard (one per I-CFM NFE):
#   A. legacy_edm            -- old-dataset EDM teacher (NVIDIA-style 224x128,
#                               raw qpepre, channel order [t2m, u10, v10, qpepre]).
#                               Regression : exp_3_reg_L_24_H_4_train_2_5_years/0/
#                                            checkpoints_regression/StormCastUNet.0.7500.mdlus
#                               Teacher    : exp_3_dif_L_24_H_4_train_2_5_years/0/
#                                            checkpoints_diffusion/EDMPrecond.0.70000.mdlus
#                               Dataset    : zarr_exp3_L_24_H_24_train_2_5_years_full
#                               (NB: the .mdlus archives carry the same weights as the
#                               .pt training-state dumps at step 7500 / 70000; the .pt
#                               files are for training-resume only and won't load via
#                               Module.from_checkpoint.)
#   B. cleaned_edm           -- new EDM teacher at ~2 M samples (cleaned 192x96, log1p
#                               qpepre, channel order [u10, v10, t2m, qpepre]).
#                               Checkpoint : runs/diffusion_zettabyte_v1_cleaned_4_27_2026/
#                                            .../EDMPrecond.0.31000.mdlus (batch 64 * 31250
#                                            ~= 2 M samples).
#   C. cleaned_flow_nfe{N}   -- I-CFM (FlowCast) student (qpw=2.0, spectral on qpepre,
#                               log1p) at ~2 M samples (batch 96 * 20833), one row per
#                               sampler-NFE in $FLOWCAST_NFES (default "10 15 20").
#                               Checkpoint : runs/flowcast_zettabyte_v1_cleaned_4_27_2026/
#                                            .../FlowCastPrecond.0.20000.mdlus
#
# The legacy leg uses the old 224x128 grid + raw mm/h qpepre, so it cannot share
# the same dataset config as the cleaned legs. We therefore run
# compare_diffusion_vs_flowcast.py TWICE -- once per dataset layout -- and a
# post-process step glues the two scoreboards into a single 3-row table.
#
# All threshold-based qpepre metrics are computed AFTER `denormalize_state` so
# the mm/h numbers ARE comparable across the two grids; per-pixel RMSE is also
# comparable because it averages over the whole field. The remaining apples-to-
# apples concern is the field-of-view difference (28,672 cells vs 18,432 cells)
# -- worth a footnote when reporting, not a deal-breaker.

set -euo pipefail

# --- Logging helpers --------------------------------------------------------
SCRIPT_T0=$(date +%s)
log()  { printf '[%s] [run_main] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
warn() { printf '[%s] [run_main][WARN] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2; }
banner() {
    local msg="$1"
    printf '\n'
    printf '================================================================\n'
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$msg"
    printf '================================================================\n'
}
elapsed_s() {
    local t0="$1"
    local now
    now=$(date +%s)
    printf '%ds' $((now - t0))
}

banner "Main experiment — legacy vs. cleaned-2M (3-way)"

# Conda activation is optional and best-effort. Many machines have a broken
# conda install (e.g. the OpenSSL X509_V_FLAG_NOTIFY_POLICY mismatch on the
# joinet anaconda); the script must fall through to whatever python is on PATH
# in that case. Skip activation entirely if a conda env is already active or
# if SKIP_CONDA=1 is set. All stderr from `conda info` is suppressed so a
# broken conda can't leak its multi-page traceback into the run log.
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
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/experiment_scripts/results/main_experiment}"
mkdir -p "${OUT_DIR}"

N_SEQUENCES="${N_SEQUENCES:-10}"
N_STEPS="${N_STEPS:-12}"                  # autoregressive horizon, hours
ENSEMBLE="${ENSEMBLE:-4}"
DIFFUSION_NFE="${DIFFUSION_NFE:-18}"      # 18 Heun steps = 36 NFE
# Space-separated list — each NFE becomes its own row flowcast_nfe<N>.
FLOWCAST_NFES="${FLOWCAST_NFES:-10 15 20}"
SEED="${SEED:-0}"

# Number of sequences to render as PNG panels per leg (truth + predictions +
# prediction-minus-truth diffs, with units). PANEL_STEPS is space-separated;
# leave empty to default to first + last step. The actual files end up in
#   ${OUT_DIR}/<leg>/panels/<channel>/seq{NN}_step{KK}.png       (combined)
#   ${OUT_DIR}/<leg>/panels/diff/<channel>/seq{NN}_step{KK}.png  (diff only)
N_PANELS_SEQ="${N_PANELS_SEQ:-6}"
PANEL_STEPS="${PANEL_STEPS:-}"

LEGACY_DATA="${LEGACY_DATA:-${REPO_ROOT}/exp_3_train_2_5_yrs_val_1yr_tp1/zarr_exp3_L_24_H_24_train_2_5_years_full}"
# NOTE: Module.from_checkpoint loads the .mdlus archive (model + metadata),
# not the .pt training-state dict (state_dict + optimizer) saved at the same
# step. The .pt files are for training-resume only; use the .mdlus sibling at
# the same step number for inference. Step 7500 / 70000 match the user's
# original request -- only the file extension changes.
LEGACY_REG="${LEGACY_REG:-${REPO_ROOT}/exp_3_train_2_5_yrs_val_1yr_tp1/exp_3_reg_L_24_H_4_train_2_5_years/0/checkpoints_regression/StormCastUNet.0.7500.mdlus}"
LEGACY_EDM="${LEGACY_EDM:-${REPO_ROOT}/exp_3_train_2_5_yrs_val_1yr_tp1/exp_3_dif_L_24_H_4_train_2_5_years/0/checkpoints_diffusion/EDMPrecond.0.70000.mdlus}"

CLEANED_DATA="${CLEANED_DATA:-${REPO_ROOT}/exp_3_train_2_5_yrs_val_1yr_tp1/zarr_exp3_L_24_H_24_train_2_5_years_full_cleaned_4_27_2026}"
CLEANED_REG="${CLEANED_REG:-${REPO_ROOT}/runs/regression_zettabyte_v1_cleaned_4_27_2026/regression_zettabyte_cleaned_4_27_2026/run_0/checkpoints_regression/StormCastUNet.0.8000.mdlus}"
CLEANED_EDM="${CLEANED_EDM:-${REPO_ROOT}/runs/diffusion_zettabyte_v1_cleaned_4_27_2026/diffusion_zettabyte_cleaned_4_27_2026/run_0/checkpoints_diffusion/EDMPrecond.0.31000.mdlus}"
CLEANED_FLOW="${CLEANED_FLOW:-${REPO_ROOT}/runs/flowcast_zettabyte_v1_cleaned_4_27_2026/flowcast_zettabyte_cleaned_4_27_2026/run_0/checkpoints_flowcast/FlowCastPrecond.0.20000.mdlus}"

banner "Configuration"
log "REPO_ROOT     = ${REPO_ROOT}"
log "OUT_DIR       = ${OUT_DIR}"
log "N_SEQUENCES   = ${N_SEQUENCES}    (initial times across 2022)"
log "N_STEPS       = ${N_STEPS}    (autoregressive horizon, hours)"
log "ENSEMBLE      = ${ENSEMBLE}    (members per sequence)"
log "DIFFUSION_NFE = ${DIFFUSION_NFE}   (Heun steps; NFE = 2 * this)"
log "FLOWCAST_NFES = ${FLOWCAST_NFES}   (Euler steps; one row per value)"
log "N_PANELS_SEQ  = ${N_PANELS_SEQ}    PANEL_STEPS = '${PANEL_STEPS}'"
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

# --- Pre-flight: report ALL missing paths before exiting --------------------
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

leg_summary() {
    # $1 = leg label, $2 = leg output dir
    local label="$1"
    local dir="$2"
    if [ -d "${dir}" ]; then
        local nfiles
        nfiles=$(find "${dir}" -type f 2>/dev/null | wc -l)
        local size
        size=$(du -sh "${dir}" 2>/dev/null | awk '{print $1}')
        log "[${label}] outputs: ${nfiles} files (${size}) under ${dir}"
        if [ -f "${dir}/scoreboard.md" ]; then
            log "[${label}] scoreboard.md:"
            sed 's/^/[run_main]     | /' "${dir}/scoreboard.md"
        fi
        local npanels
        npanels=$(find "${dir}/panels" -type f -name '*.png' 2>/dev/null | wc -l)
        log "[${label}] panels rendered: ${npanels} PNGs"
    else
        warn "[${label}] expected output dir ${dir} not found"
    fi
}

# -----------------------------------------------------------------------------
# Leg A -- legacy old-stormcast baseline on the old 224x128 dataset.
# -----------------------------------------------------------------------------
banner "Leg A — legacy old-stormcast on old 224x128 dataset"
log "regression : ${LEGACY_REG}"
log "diffusion  : ${LEGACY_EDM}"
log "dataset    : ${LEGACY_DATA}"
log "log -> ${OUT_DIR}/legacy.log"
LEG_T0=$(date +%s)
python -u "${REPO_ROOT}/experiment_scripts/compare_diffusion_vs_flowcast.py" \
    --output-dir "${OUT_DIR}/legacy" \
    --data-location "${LEGACY_DATA}" \
    --valid-dates 2022/01/01 2022/12/31 \
    --hr-size 224 128 \
    --no-qpepre-log1p \
    --kept-channels t2m u10 v10 qpepre \
    --regression-checkpoint "${LEGACY_REG}" \
    --diffusion-checkpoint "${LEGACY_EDM}" \
    --skip-flowcast \
    --n-sequences "${N_SEQUENCES}" \
    --n-steps "${N_STEPS}" \
    --ensemble "${ENSEMBLE}" \
    --seed "${SEED}" \
    --diffusion-num-steps "${DIFFUSION_NFE}" \
    --diffusion-solver heun \
    --n-panels-seq "${N_PANELS_SEQ}" \
    ${PANEL_STEPS:+--panel-steps ${PANEL_STEPS}} \
    2>&1 | tee "${OUT_DIR}/legacy.log"
LEG_RC=${PIPESTATUS[0]}
if [ "${LEG_RC}" != "0" ]; then
    warn "leg A failed with exit code ${LEG_RC} after $(elapsed_s ${LEG_T0})"
    exit "${LEG_RC}"
fi
log "leg A done in $(elapsed_s ${LEG_T0})"
leg_summary "legacy" "${OUT_DIR}/legacy"

# -----------------------------------------------------------------------------
# Leg B -- new EDM + new FlowCast on the cleaned 192x96 + log1p dataset.
# -----------------------------------------------------------------------------
banner "Leg B — cleaned EDM + FlowCast (both @ ~2M samples)"
log "regression : ${CLEANED_REG}"
log "diffusion  : ${CLEANED_EDM}"
log "flowcast   : ${CLEANED_FLOW}"
log "dataset    : ${CLEANED_DATA}"
log "log -> ${OUT_DIR}/cleaned_2M.log"
LEG_T0=$(date +%s)
python -u "${REPO_ROOT}/experiment_scripts/compare_diffusion_vs_flowcast.py" \
    --output-dir "${OUT_DIR}/cleaned_2M" \
    --data-location "${CLEANED_DATA}" \
    --valid-dates 2022/01/01 2022/12/31 \
    --hr-size 192 96 \
    --qpepre-log1p \
    --kept-channels u10 v10 t2m qpepre \
    --regression-checkpoint "${CLEANED_REG}" \
    --diffusion-checkpoint "${CLEANED_EDM}" \
    --flowcast-checkpoint "${CLEANED_FLOW}" \
    --n-sequences "${N_SEQUENCES}" \
    --n-steps "${N_STEPS}" \
    --ensemble "${ENSEMBLE}" \
    --seed "${SEED}" \
    --diffusion-num-steps "${DIFFUSION_NFE}" \
    --diffusion-solver heun \
    --flowcast-num-steps ${FLOWCAST_NFES} \
    --flowcast-solver euler \
    --n-panels-seq "${N_PANELS_SEQ}" \
    ${PANEL_STEPS:+--panel-steps ${PANEL_STEPS}} \
    2>&1 | tee "${OUT_DIR}/cleaned_2M.log"
LEG_RC=${PIPESTATUS[0]}
if [ "${LEG_RC}" != "0" ]; then
    warn "leg B failed with exit code ${LEG_RC} after $(elapsed_s ${LEG_T0})"
    exit "${LEG_RC}"
fi
log "leg B done in $(elapsed_s ${LEG_T0})"
leg_summary "cleaned_2M" "${OUT_DIR}/cleaned_2M"

# -----------------------------------------------------------------------------
# Stitch the two scoreboards into a single 3-row table.
# -----------------------------------------------------------------------------
banner "Stitching 3-way scoreboard"
STITCH_T0=$(date +%s)
OUT_DIR="${OUT_DIR}" FLOWCAST_NFES_ENV="${FLOWCAST_NFES}" python -u - <<'PY'
import csv
import os
import sys
from pathlib import Path

ROOT = Path(os.environ["OUT_DIR"])
NFES = [int(x) for x in os.environ["FLOWCAST_NFES_ENV"].split()]

LEGS = [
    ("legacy_edm",   ROOT / "legacy"     / "scoreboard.csv", "diffusion"),
    ("cleaned_edm",  ROOT / "cleaned_2M" / "scoreboard.csv", "diffusion"),
]
for nfe in NFES:
    LEGS.append(
        (f"cleaned_flow_nfe{nfe}", ROOT / "cleaned_2M" / "scoreboard.csv", f"flowcast_nfe{nfe}")
    )

print(f"[stitch] root = {ROOT}")
for label, path, key in LEGS:
    print(f"[stitch]   {label:<22} <- {path}::{key}")


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


rows = []
header = None
for label, path, key in LEGS:
    try:
        rs = {r["method"]: r for r in load(path)}
    except FileNotFoundError:
        print(f"[stitch] WARN missing {path} -- skipping {label}")
        continue
    if key not in rs:
        print(f"[stitch] WARN {path} has no '{key}' row -- skipping {label}")
        continue
    r = dict(rs[key])
    r["method"] = label
    rows.append(r)
    header = list(r.keys())
    print(f"[stitch]   collected {label}")
if header is None:
    print("[stitch] FATAL no rows collected", file=sys.stderr)
    sys.exit(1)

csv_path = ROOT / "scoreboard_3way.csv"
with open(csv_path, "w", newline="") as fo:
    w = csv.DictWriter(fo, fieldnames=header)
    w.writeheader()
    w.writerows(rows)
print(f"[stitch] wrote {csv_path}")

md = ROOT / "scoreboard_3way.md"
with open(md, "w") as f:
    f.write("# Main experiment scoreboard\n\n")
    f.write(
        "Multi-method comparison: legacy old-stormcast baseline vs. cleaned EDM "
        "and I-CFM (FlowCast) students at matched ~2 M-sample budget. Each I-CFM "
        f"row reports a different sampler NFE ({', '.join(str(n) for n in NFES)}).\n\n"
    )
    f.write("| " + " | ".join(header) + " |\n")
    f.write("|" + "|".join("---" for _ in header) + "|\n")
    for r in rows:
        f.write("| " + " | ".join(r[h] for h in header) + " |\n")
print(f"[stitch] wrote {md}")
print("[stitch] preview:")
for r in rows:
    print("[stitch]   " + " | ".join(f"{r[h]}" for h in header))


# --- Per-lead-time RMSE: glue legs together, relabelling rows the same way --
def stitch_per_step():
    out = ROOT / "rmse_per_step_3way.csv"
    out_rows = []
    out_header = None
    for label, leg_csv, key in LEGS:
        per_step = leg_csv.parent / "rmse_per_step.csv"
        if not per_step.exists():
            print(f"[stitch] WARN no per-step file {per_step} -- skipping {label}")
            continue
        with open(per_step) as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                if r["method"] != key:
                    continue
                r2 = dict(r)
                r2["method"] = label
                out_rows.append(r2)
                out_header = list(r2.keys())
    if out_header is None:
        print("[stitch] WARN no per-step rows assembled")
        return
    with open(out, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=out_header)
        w.writeheader()
        w.writerows(out_rows)
    print(f"[stitch] wrote {out}  ({len(out_rows)} rows)")


stitch_per_step()
PY
STITCH_RC=$?
if [ "${STITCH_RC}" != "0" ]; then
    warn "stitch step failed with exit code ${STITCH_RC}"
    exit "${STITCH_RC}"
fi
log "stitch done in $(elapsed_s ${STITCH_T0})"

banner "All done — total elapsed $(elapsed_s ${SCRIPT_T0})"
log "outputs:"
log "  scoreboard         : ${OUT_DIR}/scoreboard_3way.{md,csv}"
log "  per-step RMSE      : ${OUT_DIR}/rmse_per_step_3way.csv"
log "  per-leg            : ${OUT_DIR}/{legacy,cleaned_2M}/scoreboard.{md,csv}"
log "  per-leg per-step   : ${OUT_DIR}/{legacy,cleaned_2M}/rmse_per_step.csv"
log "  per-leg PNG panels : ${OUT_DIR}/{legacy,cleaned_2M}/panels/<channel>/seq*_step*.png"
log "  per-leg DIFF panels: ${OUT_DIR}/{legacy,cleaned_2M}/panels/diff/<channel>/seq*_step*.png"
log "  per-leg run logs   : ${OUT_DIR}/{legacy,cleaned_2M}.log"
log "tail the 3-way scoreboard:"
if [ -f "${OUT_DIR}/scoreboard_3way.md" ]; then
    sed 's/^/[run_main]     | /' "${OUT_DIR}/scoreboard_3way.md"
fi
