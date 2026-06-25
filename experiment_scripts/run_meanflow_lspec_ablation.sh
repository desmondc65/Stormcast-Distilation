#!/bin/bash
# MeanFlow spectral-loss (log-PSD) ablation: MeanFlow WITH the radial log-PSD
# term vs. MeanFlow WITHOUT it, all else equal, on the cleaned 192x96 + log1p
# 2022 validation year. This isolates the spectral regularizer's contribution
# to the MeanFlow student's skill (RMSE / CRPS / precip metrics / spectra).
#
# Produces one row per sampler-NFE per leg in the stitched scoreboard:
#   meanflow_lspec_nfe{N}     -- MeanFlow trained WITH spectral_weight=0.1 on qpepre
#                                (canonical run; checkpoint step 20000, ~2.24M samples).
#                                runs/meanflow_zettabyte_v1_cleaned_4_27_2026/.../
#                                    checkpoints_meanflow/MeanFlowPrecond.0.20000.mdlus
#   meanflow_no_lspec_nfe{N}  -- MeanFlow trained WITHOUT the spectral term
#                                (train_meanflow_no_lspec.sh; checkpoint step 18000,
#                                ~2.02M samples).
#                                runs/meanflow_no_lspec_zettabyte_v1_cleaned_4_27_2026/.../
#                                    checkpoints_meanflow/MeanFlowPrecond.0.18000.mdlus
#
# The two legs share dataset / regression mean / channel order / sigma_data /
# sampler, so the ONLY difference is the spectral loss term used at training
# time -- a clean A/B on the regularizer. Because the comparison harness scores
# a single MeanFlow checkpoint per invocation, we run
# compare_diffusion_vs_flowcast.py TWICE (diffusion + flowcast legs skipped) and
# stitch the two scoreboards into one ablation table.
#
# NOTE on budget: the no-lspec run's final checkpoint is at step 18000 (batch
# 112 -> ~2.02M samples); the canonical with-lspec checkpoint used here is the
# thesis-headline step 20000 (~2.24M samples, ~11% more budget). The gap
# slightly favors the with-lspec leg -- footnote when reporting, or override
# WITH_LSPEC_CKPT to the budget-matched step-17500 checkpoint (~1.96M) for a
# tighter A/B.

set -euo pipefail

# --- Logging helpers --------------------------------------------------------
SCRIPT_T0=$(date +%s)
log()  { printf '[%s] [lspec_abl] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
warn() { printf '[%s] [lspec_abl][WARN] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2; }
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

banner "MeanFlow spectral-loss ablation — lspec vs no-lspec"

# Conda activation is optional and best-effort (mirrors run_main_experiment.sh):
# skip if already in a non-base env or if SKIP_CONDA=1, and never let a broken
# conda leak a traceback into the run log.
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
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/experiment_scripts/results/meanflow_lspec_ablation}"
mkdir -p "${OUT_DIR}"

# Run on GPU 1 by default (overridable), matching run_main_experiment.sh.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

# Evaluation knobs -- kept identical to run_main_experiment.sh so this ablation
# is consistent with the rest of the thesis eval (5-member, +1..+6h rollout).
N_SEQUENCES="${N_SEQUENCES:-10}"
N_STEPS="${N_STEPS:-6}"
ENSEMBLE="${ENSEMBLE:-5}"
# MeanFlow average-velocity sampler NFEs -- each becomes a row meanflow_*_nfe<N>.
# 1 = headline one-step sampler, 2 = config default.
MEANFLOW_NFES="${MEANFLOW_NFES:-1 2}"
SEED="${SEED:-0}"
SIGMA_DATA="${SIGMA_DATA:-0.5}"
N_PANELS_SEQ="${N_PANELS_SEQ:-3}"
PANEL_STEPS="${PANEL_STEPS:-}"

# Shared inputs (cleaned 192x96 + log1p, channel order [u10, v10, t2m, qpepre]).
CLEANED_DATA="${CLEANED_DATA:-${REPO_ROOT}/exp_3_train_2_5_yrs_val_1yr_tp1/zarr_exp3_L_24_H_24_train_2_5_years_full_cleaned_4_27_2026}"
CLEANED_REG="${CLEANED_REG:-${REPO_ROOT}/runs/regression_zettabyte_v1_cleaned_4_27_2026/regression_zettabyte_cleaned_4_27_2026/run_0/checkpoints_regression/StormCastUNet.0.8000.mdlus}"

# The two MeanFlow students being compared. Override either to score a
# different step (e.g. WITH_LSPEC_CKPT=.../MeanFlowPrecond.0.17500.mdlus for a
# budget-matched A/B).
WITH_LSPEC_CKPT="${WITH_LSPEC_CKPT:-${REPO_ROOT}/runs/meanflow_zettabyte_v1_cleaned_4_27_2026/meanflow_zettabyte_cleaned_4_27_2026/run_0/checkpoints_meanflow/MeanFlowPrecond.0.20000.mdlus}"
NO_LSPEC_CKPT="${NO_LSPEC_CKPT:-${REPO_ROOT}/runs/meanflow_no_lspec_zettabyte_v1_cleaned_4_27_2026/meanflow_no_lspec_zettabyte_cleaned_4_27_2026/run_0/checkpoints_meanflow/MeanFlowPrecond.0.18000.mdlus}"

banner "Configuration"
log "REPO_ROOT       = ${REPO_ROOT}"
log "OUT_DIR         = ${OUT_DIR}"
log "CUDA_VISIBLE    = ${CUDA_VISIBLE_DEVICES}"
log "N_SEQUENCES     = ${N_SEQUENCES}    (initial times across 2022)"
log "N_STEPS         = ${N_STEPS}    (autoregressive horizon, hours)"
log "ENSEMBLE        = ${ENSEMBLE}    (members per sequence)"
log "MEANFLOW_NFES   = ${MEANFLOW_NFES}   (avg-velocity NFE; one row per value)"
log "SEED            = ${SEED}    SIGMA_DATA = ${SIGMA_DATA}"
log "N_PANELS_SEQ    = ${N_PANELS_SEQ}    PANEL_STEPS = '${PANEL_STEPS}'"
log ""
log "Inputs:"
log "  CLEANED_DATA    = ${CLEANED_DATA}"
log "  CLEANED_REG     = ${CLEANED_REG}"
log "  WITH_LSPEC_CKPT = ${WITH_LSPEC_CKPT}"
log "  NO_LSPEC_CKPT   = ${NO_LSPEC_CKPT}"

# --- Pre-flight: report ALL missing paths before exiting --------------------
banner "Pre-flight checks"
_PREFLIGHT_FAIL=0
for kv in \
    "CLEANED_DATA=${CLEANED_DATA}" \
    "CLEANED_REG=${CLEANED_REG}" \
    "WITH_LSPEC_CKPT=${WITH_LSPEC_CKPT}" \
    "NO_LSPEC_CKPT=${NO_LSPEC_CKPT}"; do
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
        local nfiles size npanels
        nfiles=$(find "${dir}" -type f 2>/dev/null | wc -l)
        size=$(du -sh "${dir}" 2>/dev/null | awk '{print $1}')
        log "[${label}] outputs: ${nfiles} files (${size}) under ${dir}"
        if [ -f "${dir}/scoreboard.md" ]; then
            log "[${label}] scoreboard.md:"
            sed 's/^/[lspec_abl]     | /' "${dir}/scoreboard.md"
        fi
        # panels/ only exists when a diffusion leg renders it; this ablation
        # skips diffusion, so guard the find or set -e + pipefail aborts here
        # (find on a missing dir returns 1, which pipefail propagates).
        if [ -d "${dir}/panels" ]; then
            npanels=$(find "${dir}/panels" -type f -name '*.png' 2>/dev/null | wc -l)
        else
            npanels=0
        fi
        log "[${label}] panels rendered: ${npanels} PNGs"
    else
        warn "[${label}] expected output dir ${dir} not found"
    fi
}

# run_meanflow_leg <leg_subdir> <meanflow_checkpoint>
# Scores a single MeanFlow checkpoint on the cleaned dataset, with the diffusion
# and flowcast legs skipped. The MeanFlow leg sets the ground-truth rollout, so
# skipping the other two is fine (see compare_diffusion_vs_flowcast.py:1006).
run_meanflow_leg() {
    local subdir="$1"
    local ckpt="$2"
    local t0
    t0=$(date +%s)
    log "leg '${subdir}' : ${ckpt}"
    log "log -> ${OUT_DIR}/${subdir}.log"
    python -u "${REPO_ROOT}/experiment_scripts/compare_diffusion_vs_flowcast.py" \
        --output-dir "${OUT_DIR}/${subdir}" \
        --data-location "${CLEANED_DATA}" \
        --valid-dates 2022/01/01 2022/12/31 \
        --hr-size 192 96 \
        --qpepre-log1p \
        --kept-channels u10 v10 t2m qpepre \
        --regression-checkpoint "${CLEANED_REG}" \
        --skip-diffusion \
        --skip-flowcast \
        --meanflow-checkpoint "${ckpt}" \
        --meanflow-num-steps ${MEANFLOW_NFES} \
        --sigma-data "${SIGMA_DATA}" \
        --n-sequences "${N_SEQUENCES}" \
        --n-steps "${N_STEPS}" \
        --ensemble "${ENSEMBLE}" \
        --seed "${SEED}" \
        --n-panels-seq "${N_PANELS_SEQ}" \
        ${PANEL_STEPS:+--panel-steps ${PANEL_STEPS}} \
        2>&1 | tee "${OUT_DIR}/${subdir}.log"
    local rc=${PIPESTATUS[0]}
    if [ "${rc}" != "0" ]; then
        warn "leg '${subdir}' failed with exit code ${rc} after $(elapsed_s ${t0})"
        exit "${rc}"
    fi
    log "leg '${subdir}' done in $(elapsed_s ${t0})"
    leg_summary "${subdir}" "${OUT_DIR}/${subdir}"
}

# -----------------------------------------------------------------------------
# Leg 1 -- MeanFlow WITH the spectral (log-PSD) loss term.
# -----------------------------------------------------------------------------
banner "Leg 1 — MeanFlow WITH spectral loss (canonical)"
run_meanflow_leg "with_lspec" "${WITH_LSPEC_CKPT}"

# -----------------------------------------------------------------------------
# Leg 2 -- MeanFlow WITHOUT the spectral loss term.
# -----------------------------------------------------------------------------
banner "Leg 2 — MeanFlow WITHOUT spectral loss"
run_meanflow_leg "no_lspec" "${NO_LSPEC_CKPT}"

# -----------------------------------------------------------------------------
# Stitch the two scoreboards into a single ablation table.
# -----------------------------------------------------------------------------
banner "Stitching spectral-loss ablation scoreboard"
STITCH_T0=$(date +%s)
OUT_DIR="${OUT_DIR}" MEANFLOW_NFES_ENV="${MEANFLOW_NFES}" python -u - <<'PY'
import csv
import os
import sys
from pathlib import Path

ROOT = Path(os.environ["OUT_DIR"])
MF_NFES = [int(x) for x in os.environ["MEANFLOW_NFES_ENV"].split()]

# compare_diffusion_vs_flowcast.py names the MeanFlow row "meanflow" when only
# one NFE is requested, else "meanflow_nfe<N>" -- mirror that to find the right
# source row in each leg's scoreboard.
def src_key(nfe):
    return "meanflow" if len(MF_NFES) == 1 else f"meanflow_nfe{nfe}"

# (output_label, leg_scoreboard_csv, source_row_key)
LEGS = []
for nfe in MF_NFES:
    LEGS.append((f"meanflow_lspec_nfe{nfe}",    ROOT / "with_lspec" / "scoreboard.csv", src_key(nfe)))
for nfe in MF_NFES:
    LEGS.append((f"meanflow_no_lspec_nfe{nfe}", ROOT / "no_lspec"   / "scoreboard.csv", src_key(nfe)))

print(f"[stitch] root = {ROOT}")
for label, path, key in LEGS:
    print(f"[stitch]   {label:<26} <- {path}::{key}")


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

csv_path = ROOT / "scoreboard_lspec_ablation.csv"
with open(csv_path, "w", newline="") as fo:
    w = csv.DictWriter(fo, fieldnames=header)
    w.writeheader()
    w.writerows(rows)
print(f"[stitch] wrote {csv_path}")

md = ROOT / "scoreboard_lspec_ablation.md"
with open(md, "w") as f:
    f.write("# MeanFlow spectral-loss (log-PSD) ablation\n\n")
    f.write(
        "MeanFlow WITH the radial log-PSD term (`meanflow_lspec_*`) vs. WITHOUT "
        "it (`meanflow_no_lspec_*`), cleaned 192x96 + log1p, 2022 validation "
        f"year. One row per sampler NFE ({', '.join(str(n) for n in MF_NFES)}). "
        "Everything except the spectral training term is held equal.\n\n"
    )
    f.write("| " + " | ".join(header) + " |\n")
    f.write("|" + "|".join("---" for _ in header) + "|\n")
    for r in rows:
        f.write("| " + " | ".join(r[h] for h in header) + " |\n")
print(f"[stitch] wrote {md}")
print("[stitch] preview:")
for r in rows:
    print("[stitch]   " + " | ".join(f"{r[h]}" for h in header))


# --- Per-lead-time RMSE: glue the two legs, relabelling rows the same way ----
def stitch_per_step():
    out = ROOT / "rmse_per_step_lspec_ablation.csv"
    out_rows = []
    out_header = None
    for label, leg_csv, key in LEGS:
        per_step = leg_csv.parent / "rmse_per_step.csv"
        if not per_step.exists():
            print(f"[stitch] WARN no per-step file {per_step} -- skipping {label}")
            continue
        with open(per_step) as f:
            for r in csv.DictReader(f):
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
log "  ablation scoreboard : ${OUT_DIR}/scoreboard_lspec_ablation.{md,csv}"
log "  per-step RMSE       : ${OUT_DIR}/rmse_per_step_lspec_ablation.csv"
log "  per-leg             : ${OUT_DIR}/{with_lspec,no_lspec}/scoreboard.{md,csv}"
log "  per-leg PNG panels  : ${OUT_DIR}/{with_lspec,no_lspec}/panels/<channel>/seq*_step*.png"
log "  per-leg run logs    : ${OUT_DIR}/{with_lspec,no_lspec}.log"
log "tail the ablation scoreboard:"
if [ -f "${OUT_DIR}/scoreboard_lspec_ablation.md" ]; then
    sed 's/^/[lspec_abl]     | /' "${OUT_DIR}/scoreboard_lspec_ablation.md"
fi
