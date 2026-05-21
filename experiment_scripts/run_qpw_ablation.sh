#!/bin/bash
# qpepre channel-weight (qpw) ablation. Mirrors run_main_experiment.sh but
# sweeps the FlowCast channel_weight on qpepre while everything else (dataset,
# regression mean, EDM teacher, NFE) stays fixed.
#
# Layout:
#   - One cleaned_edm reference row (matched ~2M-sample-budget, same as main_experiment Leg B)
#   - One I-CFM row per qpw in {1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4}, picking
#     the FlowCastPrecond.0.<step>.mdlus closest to step 20000 in each qpw run dir.
#     qpw=2.0 = canonical FlowCastPrecond.0.20000.mdlus from
#     runs/flowcast_zettabyte_v1_cleaned_4_27_2026/ (no qpw2.0/ subdir under
#     flowcast_qpw_ablation/ because the canonical run IS qpw=2.0).
#
# I-CFM legs use --skip-diffusion (the EDM reference is computed once at the
# top), so total work is ~1 EDM + 8 I-CFM rollouts.
#
# Outputs:
#   ${OUT_DIR}/scoreboard_qpw.{md,csv}        — 9 stitched rows (1 EDM + 8 qpw)
#   ${OUT_DIR}/rmse_per_step_qpw.csv          — per-(method, +Nh) RMSE for line plots
#   ${OUT_DIR}/{baseline,qpw_<value>}/        — full per-leg scoreboards

set -euo pipefail

SCRIPT_T0=$(date +%s)
log()  { printf '[%s] [run_qpw] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
warn() { printf '[%s] [run_qpw][WARN] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2; }
banner() {
    local msg="$1"
    printf '\n================================================================\n'
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$msg"
    printf '================================================================\n'
}
elapsed_s() { local t0="$1"; printf '%ds' $(( $(date +%s) - t0 )); }

banner "qpepre channel-weight (qpw) ablation"

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

_VENV="${STORMCAST_VENV:-${REPO_ROOT}/stormcast_env}"
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "${_VENV}/bin/activate" ]; then
    log "activating venv ${_VENV}"
    # shellcheck disable=SC1091
    source "${_VENV}/bin/activate"
fi

log "python : $(command -v python)"
log "host   : $(hostname)  user=$(whoami)  pid=$$"
log "cuda   : ${CUDA_VISIBLE_DEVICES:-<unset>}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/experiment_scripts/results/qpw_ablation_compare}"
mkdir -p "${OUT_DIR}"

N_SEQUENCES="${N_SEQUENCES:-10}"
N_STEPS="${N_STEPS:-12}"
ENSEMBLE="${ENSEMBLE:-4}"
DIFFUSION_NFE="${DIFFUSION_NFE:-18}"
FLOWCAST_NFE="${FLOWCAST_NFE:-10}"
SEED="${SEED:-0}"
TARGET_STEP="${TARGET_STEP:-20000}"  # picks closest FlowCastPrecond.<step> per qpw

DATA="${DATA:-${REPO_ROOT}/exp_3_train_2_5_yrs_val_1yr_tp1/zarr_exp3_L_24_H_24_train_2_5_years_full_cleaned_4_27_2026}"
REG="${REG:-${REPO_ROOT}/runs/regression_zettabyte_v1_cleaned_4_27_2026/regression_zettabyte_cleaned_4_27_2026/run_0/checkpoints_regression/StormCastUNet.0.8000.mdlus}"
EDM="${EDM:-${REPO_ROOT}/runs/diffusion_zettabyte_v1_cleaned_4_27_2026/diffusion_zettabyte_cleaned_4_27_2026/run_0/checkpoints_diffusion/EDMPrecond.0.31000.mdlus}"

# Canonical qpw=2.0 lives outside flowcast_qpw_ablation/ — point at it explicitly.
QPW20_DIR="${QPW20_DIR:-${REPO_ROOT}/runs/flowcast_zettabyte_v1_cleaned_4_27_2026/flowcast_zettabyte_cleaned_4_27_2026/run_0/checkpoints_flowcast}"
ABL_ROOT="${ABL_ROOT:-${REPO_ROOT}/runs/flowcast_qpw_ablation}"

# Map qpw value -> directory holding FlowCastPrecond.*.mdlus checkpoints.
declare -A QPW_DIRS=(
    ["1.0"]="${ABL_ROOT}/qpw1.0/flowcast_qpw1.0_cleaned_4_27_2026/run_0/checkpoints_flowcast"
    ["1.2"]="${ABL_ROOT}/qpw1.2/flowcast_qpw1.2_cleaned_4_27_2026/run_0/checkpoints_flowcast"
    ["1.4"]="${ABL_ROOT}/qpw1.4/flowcast_qpw1.4_cleaned_4_27_2026/run_0/checkpoints_flowcast"
    ["1.6"]="${ABL_ROOT}/qpw1.6/flowcast_qpw1.6_cleaned_4_27_2026/run_0/checkpoints_flowcast"
    ["1.8"]="${ABL_ROOT}/qpw1.8/flowcast_qpw1.8_cleaned_4_27_2026/run_0/checkpoints_flowcast"
    ["2.0"]="${QPW20_DIR}"
    ["2.2"]="${ABL_ROOT}/qpw2.2/flowcast_qpw2.2_cleaned_4_27_2026/run_0/checkpoints_flowcast"
    ["2.4"]="${ABL_ROOT}/qpw2.4/flowcast_qpw2.4_cleaned_4_27_2026/run_0/checkpoints_flowcast"
)
QPW_VALUES=(1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4)

# Pick the FlowCastPrecond.0.<step>.mdlus whose <step> is closest to TARGET_STEP.
closest_ckpt() {
    local dir="$1"
    local target="${TARGET_STEP}"
    local best="" best_diff=9999999
    while IFS= read -r f; do
        local step
        step=$(basename "$f" | sed -E 's/^FlowCastPrecond\.0\.([0-9]+)\.mdlus$/\1/')
        [ "$step" = "$(basename "$f")" ] && continue   # regex didn't match
        local d=$(( step - target ))
        d=${d#-}
        if (( d < best_diff )); then
            best_diff=$d; best="$f"
        fi
    done < <(find "$dir" -maxdepth 1 -name 'FlowCastPrecond.0.*.mdlus' -type f 2>/dev/null)
    echo "$best"
}

# Resolve checkpoint per qpw and pre-flight.
declare -A QPW_CKPT
banner "Resolving qpw checkpoints (closest to step ${TARGET_STEP})"
for q in "${QPW_VALUES[@]}"; do
    d="${QPW_DIRS[$q]}"
    if [ ! -d "$d" ]; then
        warn "qpw=$q  missing dir  $d"
        QPW_CKPT[$q]=""
        continue
    fi
    c="$(closest_ckpt "$d")"
    if [ -z "$c" ]; then
        warn "qpw=$q  no FlowCastPrecond.*.mdlus under $d"
        QPW_CKPT[$q]=""
    else
        log "  qpw=$q -> $c"
        QPW_CKPT[$q]="$c"
    fi
done

banner "Configuration"
log "REPO_ROOT     = ${REPO_ROOT}"
log "OUT_DIR       = ${OUT_DIR}"
log "N_SEQUENCES   = ${N_SEQUENCES}"
log "N_STEPS       = ${N_STEPS}"
log "ENSEMBLE      = ${ENSEMBLE}"
log "DIFFUSION_NFE = ${DIFFUSION_NFE}"
log "FLOWCAST_NFE  = ${FLOWCAST_NFE}"
log "TARGET_STEP   = ${TARGET_STEP}"
log "DATA          = ${DATA}"
log "REG           = ${REG}"
log "EDM           = ${EDM}"

banner "Pre-flight checks"
_FAIL=0
for kv in "DATA=${DATA}" "REG=${REG}" "EDM=${EDM}"; do
    name="${kv%%=*}"; path="${kv#*=}"
    if [ -e "${path}" ]; then log "  OK     ${name}=${path}"
    else warn "  MISSING ${name}=${path}"; _FAIL=1; fi
done
HAVE_QPW=()
for q in "${QPW_VALUES[@]}"; do
    if [ -n "${QPW_CKPT[$q]}" ]; then HAVE_QPW+=("$q"); else _FAIL=1; fi
done
if [ "${_FAIL}" = "1" ] && [ "${ALLOW_PARTIAL:-0}" != "1" ]; then
    warn "pre-flight failed -- aborting (set ALLOW_PARTIAL=1 to run only the qpw values that resolved)"
    exit 1
fi
log "pre-flight OK; will run qpw values: ${HAVE_QPW[*]}"

# -----------------------------------------------------------------------------
# Baseline EDM reference
# -----------------------------------------------------------------------------
banner "EDM baseline (cleaned_edm @ ~2M samples; --skip-flowcast)"
LEG_T0=$(date +%s)
python -u "${REPO_ROOT}/experiment_scripts/compare_diffusion_vs_flowcast.py" \
    --output-dir "${OUT_DIR}/baseline" \
    --data-location "${DATA}" \
    --valid-dates 2022/01/01 2022/12/31 \
    --hr-size 192 96 \
    --qpepre-log1p \
    --kept-channels u10 v10 t2m qpepre \
    --regression-checkpoint "${REG}" \
    --diffusion-checkpoint "${EDM}" \
    --skip-flowcast \
    --n-sequences "${N_SEQUENCES}" \
    --n-steps "${N_STEPS}" \
    --ensemble "${ENSEMBLE}" \
    --seed "${SEED}" \
    --diffusion-num-steps "${DIFFUSION_NFE}" \
    --diffusion-solver heun \
    --n-panels-seq 0 \
    2>&1 | tee "${OUT_DIR}/baseline.log"
RC=${PIPESTATUS[0]}
[ "${RC}" = "0" ] || { warn "baseline failed (rc=${RC}) after $(elapsed_s ${LEG_T0})"; exit "${RC}"; }
log "baseline done in $(elapsed_s ${LEG_T0})"

# -----------------------------------------------------------------------------
# Per-qpw I-CFM runs (--skip-diffusion)
# -----------------------------------------------------------------------------
for q in "${HAVE_QPW[@]}"; do
    safe_q="${q/./_}"   # qpw_1.0 isn't a valid dirname suffix on some FS; use qpw_1_0
    LEG_DIR="${OUT_DIR}/qpw_${safe_q}"
    banner "I-CFM @ qpw=${q}  (--skip-diffusion)"
    log "ckpt: ${QPW_CKPT[$q]}"
    LEG_T0=$(date +%s)
    python -u "${REPO_ROOT}/experiment_scripts/compare_diffusion_vs_flowcast.py" \
        --output-dir "${LEG_DIR}" \
        --data-location "${DATA}" \
        --valid-dates 2022/01/01 2022/12/31 \
        --hr-size 192 96 \
        --qpepre-log1p \
        --kept-channels u10 v10 t2m qpepre \
        --regression-checkpoint "${REG}" \
        --diffusion-checkpoint "${EDM}" \
        --flowcast-checkpoint "${QPW_CKPT[$q]}" \
        --skip-diffusion \
        --n-sequences "${N_SEQUENCES}" \
        --n-steps "${N_STEPS}" \
        --ensemble "${ENSEMBLE}" \
        --seed "${SEED}" \
        --flowcast-num-steps "${FLOWCAST_NFE}" \
        --flowcast-solver euler \
        --n-panels-seq 0 \
        2>&1 | tee "${OUT_DIR}/qpw_${safe_q}.log"
    RC=${PIPESTATUS[0]}
    [ "${RC}" = "0" ] || { warn "qpw=${q} failed (rc=${RC}) after $(elapsed_s ${LEG_T0})"; exit "${RC}"; }
    log "qpw=${q} done in $(elapsed_s ${LEG_T0})"
done

# -----------------------------------------------------------------------------
# Stitch into one scoreboard.
# -----------------------------------------------------------------------------
banner "Stitching qpw ablation scoreboard"
QPW_LIST="${HAVE_QPW[*]}" \
QPW_CKPT_DUMP="$(for q in "${HAVE_QPW[@]}"; do echo "$q ${QPW_CKPT[$q]}"; done)" \
OUT_DIR="${OUT_DIR}" TARGET_STEP="${TARGET_STEP}" python -u - <<'PY'
import csv
import os
import sys
from pathlib import Path

ROOT = Path(os.environ["OUT_DIR"])
QPW_LIST = os.environ["QPW_LIST"].split()

LEGS = [("baseline_edm", ROOT / "baseline" / "scoreboard.csv", "diffusion")]
for q in QPW_LIST:
    safe = q.replace(".", "_")
    LEGS.append((f"flowcast_qpw{q}", ROOT / f"qpw_{safe}" / "scoreboard.csv", "flowcast"))

def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))

rows, header = [], None
for label, path, key in LEGS:
    try:
        rs = {r["method"]: r for r in load(path)}
    except FileNotFoundError:
        print(f"[stitch] WARN missing {path} -- skipping {label}"); continue
    if key not in rs:
        print(f"[stitch] WARN {path} has no '{key}' row -- skipping {label}"); continue
    r = dict(rs[key]); r["method"] = label
    rows.append(r); header = list(r.keys())
    print(f"[stitch]   collected {label}")

if header is None:
    print("[stitch] FATAL no rows", file=sys.stderr); sys.exit(1)

csv_path = ROOT / "scoreboard_qpw.csv"
with open(csv_path, "w", newline="") as fo:
    w = csv.DictWriter(fo, fieldnames=header)
    w.writeheader(); w.writerows(rows)
print(f"[stitch] wrote {csv_path}")

md = ROOT / "scoreboard_qpw.md"
with open(md, "w") as f:
    f.write("# qpepre channel-weight (qpw) ablation\n\n")
    f.write(
        "FlowCast student trained with `channel_weights=[1, 1, 1, qpw]`. "
        f"Each row uses the FlowCast checkpoint nearest step {os.environ.get('TARGET_STEP', '?')} "
        "for that qpw. `baseline_edm` is the matched-2M cleaned EDM teacher (same "
        "checkpoint as main_experiment leg B) — included as an anchor for the "
        "I-CFM rows.\n\n"
    )
    f.write("Checkpoints used:\n")
    for ln in os.environ.get("QPW_CKPT_DUMP", "").splitlines():
        if ln.strip():
            f.write(f"- qpw={ln}\n")
    f.write("\n")
    f.write("| " + " | ".join(header) + " |\n")
    f.write("|" + "|".join("---" for _ in header) + "|\n")
    for r in rows:
        f.write("| " + " | ".join(r[h] for h in header) + " |\n")
print(f"[stitch] wrote {md}")
print("[stitch] preview:")
for r in rows:
    print("[stitch]   " + " | ".join(f"{r[h]}" for h in header))


def stitch_per_step():
    out = ROOT / "rmse_per_step_qpw.csv"
    out_rows, out_header = [], None
    for label, leg_csv, key in LEGS:
        per_step = leg_csv.parent / "rmse_per_step.csv"
        if not per_step.exists():
            print(f"[stitch] WARN no per-step file {per_step} -- skipping {label}"); continue
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
log "  scoreboard      : ${OUT_DIR}/scoreboard_qpw.{md,csv}"
log "  per-step RMSE   : ${OUT_DIR}/rmse_per_step_qpw.csv"
log "  per-leg         : ${OUT_DIR}/{baseline,qpw_<value>}/scoreboard.{md,csv}"
log "  per-leg run logs: ${OUT_DIR}/{baseline,qpw_<value>}.log"
if [ -f "${OUT_DIR}/scoreboard_qpw.md" ]; then
    sed 's/^/[run_qpw]     | /' "${OUT_DIR}/scoreboard_qpw.md"
fi
