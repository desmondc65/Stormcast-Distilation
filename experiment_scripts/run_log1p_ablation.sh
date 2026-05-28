#!/bin/bash
# log1p vs NO_log1p qpepre encoding ablation. Mirrors run_main_experiment.sh
# but uses the matched ~2M-sample-budget checkpoints documented in CLAUDE.md
# §6.1 / §6.2:
#
#   Leg L (log1p)    : cleaned 192x96 zarr + qpepre_log1p=True
#       regression : runs/regression_zettabyte_v1_cleaned_4_27_2026/.../StormCastUNet.0.8000.mdlus
#       diffusion  : runs/diffusion_zettabyte_v1_cleaned_4_27_2026/.../EDMPrecond.0.31000.mdlus
#       flowcast   : runs/flowcast_zettabyte_v1_cleaned_4_27_2026/.../FlowCastPrecond.0.20000.mdlus
#   Leg N (NO_log1p) : cleaned 192x96 *_raw zarr + qpepre_log1p=False
#       regression : runs/regression_zettabyte_v1_cleaned_4_27_2026_NO_log1p/.../StormCastUNet.0.10000.mdlus
#       diffusion  : runs/diffusion_zettabyte_v1_cleaned_4_27_2026_NO_log1p/.../EDMPrecond.0.20000.mdlus
#       flowcast   : runs/flowcast_zettabyte_v1_cleaned_4_27_2026_NO_log1p/.../FlowCastPrecond.0.20000.mdlus
#
# qpepre is reported in mm/h for both legs (denormalize_state undoes log1p
# when applicable), so threshold-based metrics ARE directly comparable across
# encodings. Per-pixel RMSE is averaged over the same grid (192x96).
#
# Outputs:
#   ${OUT_DIR}/scoreboard_log1p.{md,csv}     — 4 stitched rows
#   ${OUT_DIR}/rmse_per_step_log1p.csv       — per-(method, +Nh) RMSE for line plots
#   ${OUT_DIR}/{log1p,NO_log1p}/             — full per-leg scoreboards + panels

set -euo pipefail

SCRIPT_T0=$(date +%s)
log()  { printf '[%s] [run_log1p] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
warn() { printf '[%s] [run_log1p][WARN] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2; }
banner() {
    local msg="$1"
    printf '\n================================================================\n'
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$msg"
    printf '================================================================\n'
}
elapsed_s() { local t0="$1"; printf '%ds' $(( $(date +%s) - t0 )); }

banner "log1p vs NO_log1p qpepre encoding ablation"

# Conda activation — same best-effort logic as run_main_experiment.sh.
if [ "${SKIP_CONDA:-0}" != "1" ] && \
   { [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "${CONDA_DEFAULT_ENV:-base}" = "base" ]; }; then
    log "trying conda activate ${STORMCAST_ENV_NAME:-stormcast_env} ..."
    _CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    if [ -n "${_CONDA_BASE}" ] && [ -f "${_CONDA_BASE}/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "${_CONDA_BASE}/etc/profile.d/conda.sh" >/dev/null 2>&1 || true
        conda activate "${STORMCAST_ENV_NAME:-stormcast_env}" >/dev/null 2>&1 || \
            warn "conda activate failed; continuing with current python"
    fi
fi
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"

# Project ships a venv at stormcast_env/ (not a conda env). Activate it if
# present and we're not already inside it.
_VENV="${STORMCAST_VENV:-${REPO_ROOT}/stormcast_env}"
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "${_VENV}/bin/activate" ]; then
    log "activating venv ${_VENV}"
    # shellcheck disable=SC1091
    source "${_VENV}/bin/activate"
fi

log "python : $(command -v python)"
log "host   : $(hostname)  user=$(whoami)  pid=$$"
log "cuda   : ${CUDA_VISIBLE_DEVICES:-<unset>}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/experiment_scripts/results/log1p_ablation}"
mkdir -p "${OUT_DIR}"

# Same defaults as run_main_experiment.sh so headline budgets line up.
N_SEQUENCES="${N_SEQUENCES:-10}"
N_STEPS="${N_STEPS:-12}"
ENSEMBLE="${ENSEMBLE:-4}"
DIFFUSION_NFE="${DIFFUSION_NFE:-18}"
FLOWCAST_NFE="${FLOWCAST_NFE:-10}"
SEED="${SEED:-0}"
N_PANELS_SEQ="${N_PANELS_SEQ:-3}"
PANEL_STEPS="${PANEL_STEPS:-}"
# Set to 1 for an I-CFM-only run (skips both EDM teachers; stitched scoreboard
# will have 2 rows: log1p_flowcast and NO_log1p_flowcast).
SKIP_DIFFUSION="${SKIP_DIFFUSION:-0}"
SKIP_DIFFUSION_FLAG=""
if [ "${SKIP_DIFFUSION}" = "1" ]; then
    SKIP_DIFFUSION_FLAG="--skip-diffusion"
    N_PANELS_SEQ=0      # panels need a diffusion column
fi

# --- Leg L (log1p) paths ---
L_DATA="${L_DATA:-${REPO_ROOT}/exp_3_train_2_5_yrs_val_1yr_tp1/zarr_exp3_L_24_H_24_train_2_5_years_full_cleaned_4_27_2026}"
L_REG="${L_REG:-${REPO_ROOT}/runs/regression_zettabyte_v1_cleaned_4_27_2026/regression_zettabyte_cleaned_4_27_2026/run_0/checkpoints_regression/StormCastUNet.0.8000.mdlus}"
L_EDM="${L_EDM:-${REPO_ROOT}/runs/diffusion_zettabyte_v1_cleaned_4_27_2026/diffusion_zettabyte_cleaned_4_27_2026/run_0/checkpoints_diffusion/EDMPrecond.0.31000.mdlus}"
L_FLOW="${L_FLOW:-${REPO_ROOT}/runs/flowcast_zettabyte_v1_cleaned_4_27_2026/flowcast_zettabyte_cleaned_4_27_2026/run_0/checkpoints_flowcast/FlowCastPrecond.0.20000.mdlus}"

# --- Leg N (NO_log1p) paths ---
N_DATA="${N_DATA:-${REPO_ROOT}/exp_3_train_2_5_yrs_val_1yr_tp1/zarr_exp3_L_24_H_24_train_2_5_years_full_cleaned_4_27_2026_raw}"
N_REG="${N_REG:-${REPO_ROOT}/runs/regression_zettabyte_v1_cleaned_4_27_2026_NO_log1p/regression_cleaned_NO_log1p/run_0/checkpoints_regression/StormCastUNet.0.10000.mdlus}"
N_EDM="${N_EDM:-${REPO_ROOT}/runs/diffusion_zettabyte_v1_cleaned_4_27_2026_NO_log1p/edm_cleaned_NO_log1p/run_0/checkpoints_diffusion/EDMPrecond.0.20000.mdlus}"
N_FLOW="${N_FLOW:-${REPO_ROOT}/runs/flowcast_zettabyte_v1_cleaned_4_27_2026_NO_log1p/flowcast_cleaned_NO_log1p/run_0/checkpoints_flowcast/FlowCastPrecond.0.20000.mdlus}"

banner "Configuration"
log "REPO_ROOT     = ${REPO_ROOT}"
log "OUT_DIR       = ${OUT_DIR}"
log "N_SEQUENCES   = ${N_SEQUENCES}"
log "N_STEPS       = ${N_STEPS}"
log "ENSEMBLE      = ${ENSEMBLE}"
log "DIFFUSION_NFE = ${DIFFUSION_NFE}"
log "FLOWCAST_NFE  = ${FLOWCAST_NFE}"
log ""
log "Inputs:"
log "  L_DATA = ${L_DATA}"
log "  L_REG  = ${L_REG}"
log "  L_EDM  = ${L_EDM}"
log "  L_FLOW = ${L_FLOW}"
log "  N_DATA = ${N_DATA}"
log "  N_REG  = ${N_REG}"
log "  N_EDM  = ${N_EDM}"
log "  N_FLOW = ${N_FLOW}"

banner "Pre-flight checks"
_FAIL=0
for kv in \
    "L_DATA=${L_DATA}" "L_REG=${L_REG}" "L_EDM=${L_EDM}" "L_FLOW=${L_FLOW}" \
    "N_DATA=${N_DATA}" "N_REG=${N_REG}" "N_EDM=${N_EDM}" "N_FLOW=${N_FLOW}"; do
    name="${kv%%=*}"; path="${kv#*=}"
    if [ -e "${path}" ]; then
        log "  OK     ${name}=${path}"
    else
        warn "  MISSING ${name}=${path}"
        _FAIL=1
    fi
done
if [ "${_FAIL}" = "1" ]; then
    warn "pre-flight failed -- aborting before any python invocation"
    exit 1
fi
log "pre-flight OK"

# -----------------------------------------------------------------------------
# Leg L — log1p
# -----------------------------------------------------------------------------
banner "Leg L — log1p (cleaned 192x96, qpepre_log1p=True)"
LEG_T0=$(date +%s)
python -u "${REPO_ROOT}/experiment_scripts/compare_diffusion_vs_flowcast.py" \
    --output-dir "${OUT_DIR}/log1p" \
    --data-location "${L_DATA}" \
    --valid-dates 2022/01/01 2022/12/31 \
    --hr-size 192 96 \
    --qpepre-log1p \
    --kept-channels u10 v10 t2m qpepre \
    --regression-checkpoint "${L_REG}" \
    --diffusion-checkpoint "${L_EDM}" \
    --flowcast-checkpoint "${L_FLOW}" \
    ${SKIP_DIFFUSION_FLAG} \
    --n-sequences "${N_SEQUENCES}" \
    --n-steps "${N_STEPS}" \
    --ensemble "${ENSEMBLE}" \
    --seed "${SEED}" \
    --diffusion-num-steps "${DIFFUSION_NFE}" \
    --diffusion-solver heun \
    --flowcast-num-steps "${FLOWCAST_NFE}" \
    --flowcast-solver euler \
    --n-panels-seq "${N_PANELS_SEQ}" \
    ${PANEL_STEPS:+--panel-steps ${PANEL_STEPS}} \
    2>&1 | tee "${OUT_DIR}/log1p.log"
RC=${PIPESTATUS[0]}
[ "${RC}" = "0" ] || { warn "leg L failed (rc=${RC}) after $(elapsed_s ${LEG_T0})"; exit "${RC}"; }
log "leg L done in $(elapsed_s ${LEG_T0})"

# -----------------------------------------------------------------------------
# Leg N — NO_log1p
# -----------------------------------------------------------------------------
banner "Leg N — NO_log1p (cleaned 192x96 raw, qpepre_log1p=False)"
LEG_T0=$(date +%s)
python -u "${REPO_ROOT}/experiment_scripts/compare_diffusion_vs_flowcast.py" \
    --output-dir "${OUT_DIR}/NO_log1p" \
    --data-location "${N_DATA}" \
    --valid-dates 2022/01/01 2022/12/31 \
    --hr-size 192 96 \
    --no-qpepre-log1p \
    --kept-channels u10 v10 t2m qpepre \
    --regression-checkpoint "${N_REG}" \
    --diffusion-checkpoint "${N_EDM}" \
    --flowcast-checkpoint "${N_FLOW}" \
    ${SKIP_DIFFUSION_FLAG} \
    --n-sequences "${N_SEQUENCES}" \
    --n-steps "${N_STEPS}" \
    --ensemble "${ENSEMBLE}" \
    --seed "${SEED}" \
    --diffusion-num-steps "${DIFFUSION_NFE}" \
    --diffusion-solver heun \
    --flowcast-num-steps "${FLOWCAST_NFE}" \
    --flowcast-solver euler \
    --n-panels-seq "${N_PANELS_SEQ}" \
    ${PANEL_STEPS:+--panel-steps ${PANEL_STEPS}} \
    2>&1 | tee "${OUT_DIR}/NO_log1p.log"
RC=${PIPESTATUS[0]}
[ "${RC}" = "0" ] || { warn "leg N failed (rc=${RC}) after $(elapsed_s ${LEG_T0})"; exit "${RC}"; }
log "leg N done in $(elapsed_s ${LEG_T0})"

# -----------------------------------------------------------------------------
# Stitch the two per-leg scoreboards into a single 4-row table.
# -----------------------------------------------------------------------------
banner "Stitching log1p ablation scoreboard"
OUT_DIR="${OUT_DIR}" python -u - <<'PY'
import csv
import os
import sys
from pathlib import Path

ROOT = Path(os.environ["OUT_DIR"])
LEGS = [
    ("log1p_diffusion",   ROOT / "log1p"    / "scoreboard.csv", "diffusion"),
    ("log1p_flowcast",    ROOT / "log1p"    / "scoreboard.csv", "flowcast"),
    ("NO_log1p_diffusion",ROOT / "NO_log1p" / "scoreboard.csv", "diffusion"),
    ("NO_log1p_flowcast", ROOT / "NO_log1p" / "scoreboard.csv", "flowcast"),
]

def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))

rows, header = [], None
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
    rows.append(r); header = list(r.keys())
    print(f"[stitch]   collected {label}")

if header is None:
    print("[stitch] FATAL no rows", file=sys.stderr); sys.exit(1)

csv_path = ROOT / "scoreboard_log1p.csv"
with open(csv_path, "w", newline="") as fo:
    w = csv.DictWriter(fo, fieldnames=header)
    w.writeheader(); w.writerows(rows)
print(f"[stitch] wrote {csv_path}")

md = ROOT / "scoreboard_log1p.md"
with open(md, "w") as f:
    f.write("# log1p vs NO_log1p qpepre encoding ablation\n\n")
    f.write(
        "Matched ~2M-sample-budget checkpoints. qpepre is reported in mm/h on both "
        "sides (denormalize_state undoes log1p when applicable), so threshold-based "
        "metrics are directly comparable across encodings.\n\n"
    )
    f.write("| " + " | ".join(header) + " |\n")
    f.write("|" + "|".join("---" for _ in header) + "|\n")
    for r in rows:
        f.write("| " + " | ".join(r[h] for h in header) + " |\n")
print(f"[stitch] wrote {md}")
print("[stitch] preview:")
for r in rows:
    print("[stitch]   " + " | ".join(f"{r[h]}" for h in header))


# Per-lead-time RMSE stitched the same way.
def stitch_per_step():
    out = ROOT / "rmse_per_step_log1p.csv"
    out_rows, out_header = [], None
    for label, leg_csv, key in LEGS:
        per_step = leg_csv.parent / "rmse_per_step.csv"
        if not per_step.exists():
            print(f"[stitch] WARN no per-step file {per_step} -- skipping {label}")
            continue
        with open(per_step) as f:
            for r in csv.DictReader(f):
                if r["method"] != key: continue
                r2 = dict(r); r2["method"] = label
                out_rows.append(r2); out_header = list(r2.keys())
    if out_header is None:
        print("[stitch] WARN no per-step rows assembled"); return
    with open(out, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=out_header)
        w.writeheader(); w.writerows(out_rows)
    print(f"[stitch] wrote {out}  ({len(out_rows)} rows)")

stitch_per_step()
PY

banner "All done — total elapsed $(elapsed_s ${SCRIPT_T0})"
log "outputs:"
log "  scoreboard      : ${OUT_DIR}/scoreboard_log1p.{md,csv}"
log "  per-step RMSE   : ${OUT_DIR}/rmse_per_step_log1p.csv"
log "  per-leg         : ${OUT_DIR}/{log1p,NO_log1p}/scoreboard.{md,csv}"
log "  per-leg panels  : ${OUT_DIR}/{log1p,NO_log1p}/panels/<channel>/seq*_step*.png"
log "  per-leg run logs: ${OUT_DIR}/{log1p,NO_log1p}.log"
if [ -f "${OUT_DIR}/scoreboard_log1p.md" ]; then
    sed 's/^/[run_log1p]     | /' "${OUT_DIR}/scoreboard_log1p.md"
fi
