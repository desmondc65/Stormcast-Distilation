#!/bin/bash
# MeanFlow sampler-K (NFE) ablation: ONE MeanFlow student scored at K = 1..10
# average-velocity steps on the cleaned 192x96 + log1p 2022 validation year.
# Answers "how much skill do we buy per extra NFE, and what does it cost?" —
# the quality/latency trade-off curve behind the 1-2 NFE headline claim.
#
# The checkpoint is FIXED across all K; only the sampler step count varies, so
# every row is the same weights sampled more or less finely.
#
# Two legs, because quality and timing want different measurement regimes:
#
#   1. quality/  compare_diffusion_vs_flowcast.py, diffusion + flowcast legs
#                skipped, --meanflow-num-steps ${K_LIST} in a single invocation
#                (model loaded once). Gives per-channel RMSE (physical units:
#                u10/v10 m/s, t2m K, qpepre mm/h), CRPS and neighbourhood
#                precip scores per K. Its own "Time/Seq.(s)" column is NOT used
#                for the speedup numbers: it has no warmup, so the FIRST K in
#                the sweep absorbs cuDNN autotune and looks artificially slow.
#                It is carried into the table as `metric_leg_s_per_seq` purely
#                as a cross-check.
#
#   2. timing/   run_timing.sh with MEANFLOW_NFES="${K_LIST}", which does an
#                untimed warmup rollout per method before the timed block. This
#                is where `time_s_*` and every speedup column come from. It
#                also times the EDM teacher (18 Heun steps = 36 NFE) and
#                FlowCast at ${FLOW_REF_NFE} NFE as speedup references.
#
# Speedup columns (all wall-clock per member rollout, higher = faster):
#   speedup_vs_kmax      t(K=max K_LIST) / t(K)   -- self-referenced, isolates
#                                                    the sampler's own scaling
#   speedup_vs_flowcast  t(flowcast @ FLOW_REF_NFE) / t(K)
#   speedup_vs_edm       t(EDM teacher @ 36 NFE)  / t(K)
#
# Speedups are smaller than the raw NFE ratio because each rollout step also
# pays a K-independent cost (regression-mean forward + dataset I/O). That is
# the honest end-to-end number; use the slope of `time_s_per_forecast_hour`
# against K if you want the sampler-only cost.
#
# Outputs:
#   ${OUT_DIR}/scoreboard_K_ablation.{md,csv}   headline table (one row per K)
#   ${OUT_DIR}/K_ablation.png                   RMSE-vs-K + time/speedup-vs-K
#   ${OUT_DIR}/quality/scoreboard.{md,csv}      raw quality leg
#   ${OUT_DIR}/timing/timing.{md,csv}           raw timing leg (+ references)
#   ${OUT_DIR}/{quality,timing}.log             run logs
#
# Override anything via env, e.g.:
#   K_LIST="1 2 4 8" GPU=0 ./meanflow_K_ablation.sh
#   SKIP_TIMING=1 ./meanflow_K_ablation.sh        # quality only
#   SKIP_QUALITY=1 ./meanflow_K_ablation.sh       # timing only (re-stitches)
#   MEANFLOW_CKPT=/path/to/MeanFlowPrecond.0.20000.mdlus ./meanflow_K_ablation.sh

set -euo pipefail

# --- Logging helpers --------------------------------------------------------
SCRIPT_T0=$(date +%s)
log()  { printf '[%s] [K_abl] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
warn() { printf '[%s] [K_abl][WARN] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2; }
banner() {
    printf '\n================================================================\n'
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
    printf '================================================================\n'
}
elapsed_s() { printf '%ds' $(( $(date +%s) - $1 )); }

banner "MeanFlow sampler-K (NFE) ablation — quality + timing vs K"

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
EXP_DIR="${REPO_ROOT}/experiment_scripts"
OUT_DIR="${OUT_DIR:-${EXP_DIR}/results/meanflow_K_ablation}"
mkdir -p "${OUT_DIR}"

# --- Python: prefer the in-repo venv (this box has no working conda) ---------
if [ -z "${PYTHON:-}" ]; then
    if [ -x "${REPO_ROOT}/stormcast_env/bin/python" ]; then
        PYTHON="${REPO_ROOT}/stormcast_env/bin/python"
    else
        PYTHON="$(command -v python)"
    fi
fi

# --- GPU: respect a pre-set CUDA_VISIBLE_DEVICES, else $GPU, else 1 ---------
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    log "using pre-set CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
else
    export CUDA_VISIBLE_DEVICES="${GPU:-1}"
fi

# --- Knobs ------------------------------------------------------------------
K_LIST="${K_LIST:-1 2 3 4 5 6 7 8 9 10}"   # MeanFlow average-velocity NFEs to sweep

# Quality leg — matched to run_meanflow_lspec_ablation.sh / run_main_experiment.sh
# so these rows are comparable with the rest of the thesis eval.
N_SEQUENCES="${N_SEQUENCES:-10}"           # evenly-spaced init times across 2022
N_STEPS="${N_STEPS:-6}"                    # autoregressive horizon, hours
ENSEMBLE="${ENSEMBLE:-5}"                  # members per sequence (CRPS needs >1)
SEED="${SEED:-0}"
SIGMA_DATA="${SIGMA_DATA:-0.5}"
N_PANELS_SEQ="${N_PANELS_SEQ:-0}"          # panels are redundant across K; off by default

# Timing leg — smaller than run_timing.sh's defaults because we time up to 12
# methods (K_LIST + 2 references) in one go.
TIMING_ENSEMBLE="${TIMING_ENSEMBLE:-10}"
TIMING_N_STEPS="${TIMING_N_STEPS:-6}"
TIMING_N_SEQUENCES="${TIMING_N_SEQUENCES:-1}"
TIMING_WARMUP="${TIMING_WARMUP:-1}"
FLOW_REF_NFE="${FLOW_REF_NFE:-10}"         # FlowCast speedup reference (config default)
DIFFUSION_NFE="${DIFFUSION_NFE:-18}"       # EDM Heun steps -> 36 NFE

SKIP_QUALITY="${SKIP_QUALITY:-0}"
SKIP_TIMING="${SKIP_TIMING:-0}"

# --- Dataset / checkpoints (cleaned 192x96 + log1p, CLAUDE.md §6.1/§6.6) ----
# MeanFlow default = the no-spectral-loss student that owns cell D of the main
# experiment, so this curve is the one behind the headline MeanFlow numbers.
CLEANED_DATA="${CLEANED_DATA:-${REPO_ROOT}/exp_3_train_2_5_yrs_val_1yr_tp1/zarr_exp3_L_24_H_24_train_2_5_years_full_cleaned_4_27_2026}"
CLEANED_REG="${CLEANED_REG:-${REPO_ROOT}/runs/regression_zettabyte_v1_cleaned_4_27_2026/regression_zettabyte_cleaned_4_27_2026/run_0/checkpoints_regression/StormCastUNet.0.8000.mdlus}"
MEANFLOW_CKPT="${MEANFLOW_CKPT:-${REPO_ROOT}/runs/meanflow_no_lspec_zettabyte_v1_cleaned_4_27_2026/meanflow_no_lspec_zettabyte_cleaned_4_27_2026/run_0/checkpoints_meanflow/MeanFlowPrecond.0.18000.mdlus}"
FLOWCAST_CKPT="${FLOWCAST_CKPT:-${REPO_ROOT}/runs/flowcast_zettabyte_v1_cleaned_4_27_2026/flowcast_zettabyte_cleaned_4_27_2026/run_0/checkpoints_flowcast/FlowCastPrecond.0.140000.mdlus}"
DIFFUSION_CKPT="${DIFFUSION_CKPT:-${REPO_ROOT}/runs/diffusion_zettabyte_v1_cleaned_4_27_2026/diffusion_zettabyte_cleaned_4_27_2026/run_0/checkpoints_diffusion/EDMPrecond.0.73000.mdlus}"

banner "Configuration"
log "PYTHON          = ${PYTHON}"
log "CUDA_VISIBLE    = ${CUDA_VISIBLE_DEVICES}"
log "OUT_DIR         = ${OUT_DIR}"
log "K_LIST          = ${K_LIST}"
log "quality leg     : N_SEQUENCES=${N_SEQUENCES}  N_STEPS=${N_STEPS}h  ENSEMBLE=${ENSEMBLE}  SEED=${SEED}"
log "timing  leg     : ENSEMBLE=${TIMING_ENSEMBLE}  N_STEPS=${TIMING_N_STEPS}h  N_SEQUENCES=${TIMING_N_SEQUENCES}  WARMUP=${TIMING_WARMUP}"
log "references      : flowcast@${FLOW_REF_NFE} NFE, EDM@${DIFFUSION_NFE} Heun steps ($((2 * DIFFUSION_NFE)) NFE)"
log "SKIP_QUALITY    = ${SKIP_QUALITY}   SKIP_TIMING = ${SKIP_TIMING}"
log "data            = ${CLEANED_DATA}"
log "regression      = ${CLEANED_REG}"
log "meanflow        = ${MEANFLOW_CKPT}"
log "flowcast (ref)  = ${FLOWCAST_CKPT}"
log "diffusion (ref) = ${DIFFUSION_CKPT}"

# --- Pre-flight: report ALL missing paths before any python -----------------
banner "Pre-flight checks"
_FAIL=0
_REQUIRED=( \
    "CLEANED_DATA=${CLEANED_DATA}" \
    "CLEANED_REG=${CLEANED_REG}" \
    "MEANFLOW_CKPT=${MEANFLOW_CKPT}" )
if [ "${SKIP_TIMING}" != "1" ]; then
    # run_timing.sh always times the EDM teacher, so both references are needed.
    _REQUIRED+=( "FLOWCAST_CKPT=${FLOWCAST_CKPT}" \
                 "DIFFUSION_CKPT=${DIFFUSION_CKPT}" \
                 "RUN_TIMING_SH=${EXP_DIR}/run_timing.sh" )
fi
for kv in "${_REQUIRED[@]}"; do
    name="${kv%%=*}"; path="${kv#*=}"
    if [ -e "${path}" ]; then log "  OK      ${name}"; else warn "  MISSING ${name}=${path}"; _FAIL=1; fi
done
if [ -z "${K_LIST// /}" ]; then warn "K_LIST is empty"; _FAIL=1; fi
if [ "${_FAIL}" = "1" ]; then warn "pre-flight failed — aborting before any python invocation"; exit 1; fi
log "pre-flight OK"

# -----------------------------------------------------------------------------
# Leg 1 — quality: one invocation, all K (checkpoint loaded once).
# -----------------------------------------------------------------------------
if [ "${SKIP_QUALITY}" = "1" ]; then
    banner "Leg 1 — quality SKIPPED (SKIP_QUALITY=1, reusing existing scoreboard)"
else
    banner "Leg 1 — quality sweep over K = ${K_LIST}"
    Q_T0=$(date +%s)
    log "log -> ${OUT_DIR}/quality.log"
    "${PYTHON}" -u "${EXP_DIR}/compare_diffusion_vs_flowcast.py" \
        --output-dir "${OUT_DIR}/quality" \
        --data-location "${CLEANED_DATA}" \
        --valid-dates 2022/01/01 2022/12/31 \
        --hr-size 192 96 \
        --qpepre-log1p \
        --kept-channels u10 v10 t2m qpepre \
        --regression-checkpoint "${CLEANED_REG}" \
        --skip-diffusion \
        --skip-flowcast \
        --meanflow-checkpoint "${MEANFLOW_CKPT}" \
        --meanflow-num-steps ${K_LIST} \
        --sigma-data "${SIGMA_DATA}" \
        --n-sequences "${N_SEQUENCES}" \
        --n-steps "${N_STEPS}" \
        --ensemble "${ENSEMBLE}" \
        --seed "${SEED}" \
        --n-panels-seq "${N_PANELS_SEQ}" \
        2>&1 | tee "${OUT_DIR}/quality.log"
    RC=${PIPESTATUS[0]}
    if [ "${RC}" != "0" ]; then
        warn "quality leg failed with exit code ${RC} after $(elapsed_s ${Q_T0})"
        exit "${RC}"
    fi
    log "quality leg done in $(elapsed_s ${Q_T0})"
fi

# -----------------------------------------------------------------------------
# Leg 2 — timing: warmed-up wall clock per K, plus flowcast / EDM references.
# -----------------------------------------------------------------------------
if [ "${SKIP_TIMING}" = "1" ]; then
    banner "Leg 2 — timing SKIPPED (SKIP_TIMING=1, reusing existing timing.csv)"
else
    banner "Leg 2 — timing sweep over K = ${K_LIST} (+ flowcast/EDM references)"
    T_T0=$(date +%s)
    mkdir -p "${OUT_DIR}/timing"
    log "log -> ${OUT_DIR}/timing.log"
    OUT_DIR="${OUT_DIR}/timing" \
    PYTHON="${PYTHON}" \
    REPO_ROOT="${REPO_ROOT}" \
    ENSEMBLE="${TIMING_ENSEMBLE}" \
    N_STEPS="${TIMING_N_STEPS}" \
    N_SEQUENCES="${TIMING_N_SEQUENCES}" \
    WARMUP="${TIMING_WARMUP}" \
    SEED="${SEED}" \
    SIGMA_DATA="${SIGMA_DATA}" \
    DIFFUSION_NFE="${DIFFUSION_NFE}" \
    FLOWCAST_NFES="${FLOW_REF_NFE}" \
    MEANFLOW_NFES="${K_LIST}" \
    DATA_LOCATION="${CLEANED_DATA}" \
    REGRESSION_CKPT="${CLEANED_REG}" \
    MEANFLOW_CKPT="${MEANFLOW_CKPT}" \
    FLOWCAST_CKPT="${FLOWCAST_CKPT}" \
    DIFFUSION_CKPT="${DIFFUSION_CKPT}" \
        bash "${EXP_DIR}/run_timing.sh" 2>&1 | tee "${OUT_DIR}/timing.log"
    RC=${PIPESTATUS[0]}
    if [ "${RC}" != "0" ]; then
        warn "timing leg failed with exit code ${RC} after $(elapsed_s ${T_T0})"
        exit "${RC}"
    fi
    log "timing leg done in $(elapsed_s ${T_T0})"
fi

# -----------------------------------------------------------------------------
# Stitch quality + timing into one K-indexed table (and a figure).
# -----------------------------------------------------------------------------
banner "Stitching K-ablation scoreboard"
S_T0=$(date +%s)
OUT_DIR="${OUT_DIR}" K_LIST_ENV="${K_LIST}" FLOW_REF_NFE_ENV="${FLOW_REF_NFE}" \
DIFFUSION_NFE_ENV="${DIFFUSION_NFE}" EXP_DIR="${EXP_DIR}" \
"${PYTHON}" -u - <<'PY'
import csv
import os
import sys
from pathlib import Path

ROOT = Path(os.environ["OUT_DIR"])
KS = [int(x) for x in os.environ["K_LIST_ENV"].split()]
FLOW_REF = int(os.environ["FLOW_REF_NFE_ENV"])
EDM_NFE = 2 * int(os.environ["DIFFUSION_NFE_ENV"])
sys.path.insert(0, os.environ["EXP_DIR"])

CHANNELS = ["u10", "v10", "t2m", "qpepre"]


def read_csv(path):
    if not Path(path).exists():
        print(f"[stitch] WARN missing {path}")
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


# --- quality leg -------------------------------------------------------------
# compare_diffusion_vs_flowcast.py names the row "meanflow" for a single NFE and
# "meanflow_nfe<K>" otherwise — mirror that so a one-K sweep still stitches.
qual = {r["method"]: r for r in read_csv(ROOT / "quality" / "scoreboard.csv")}
qual_key = (lambda k: "meanflow") if len(KS) == 1 else (lambda k: f"meanflow_nfe{k}")

# --- timing leg (always names rows meanflow_nfe<K>) --------------------------
timing = {r["method"]: r for r in read_csv(ROOT / "timing" / "timing.csv")}


def t_of(label):
    r = timing.get(label)
    return float(r["sec_per_member_rollout"]) if r else None


t_by_k = {k: t_of(f"meanflow_nfe{k}") for k in KS}
t_kmax = t_by_k.get(max(KS))
t_flow = t_of(f"flowcast_nfe{FLOW_REF}")
t_edm = t_of("stormcast_edm")
print(f"[stitch] timing references: K={max(KS)} -> {t_kmax}  "
      f"flowcast@{FLOW_REF} -> {t_flow}  edm@{EDM_NFE} -> {t_edm}")


def ratio(num, den):
    if num is None or not den:
        return ""
    return f"{num / den:.2f}"


SP_KMAX = f"speedup_vs_K{max(KS)}"
SP_FLOW = f"speedup_vs_flowcast{FLOW_REF}"
SP_EDM = f"speedup_vs_edm{EDM_NFE}"
HEADER = (
    ["K"]
    + [f"RMSE_{ch}" for ch in CHANNELS]
    + ["CRPS", "time_s_per_member_rollout", "time_s_per_forecast_hour",
       SP_KMAX, SP_FLOW, SP_EDM, "metric_leg_s_per_seq"]
)

rows = []
for k in KS:
    q = qual.get(qual_key(k))
    tr = timing.get(f"meanflow_nfe{k}")
    if q is None and tr is None:
        print(f"[stitch] WARN no quality and no timing row for K={k} — skipping")
        continue
    t_k = t_by_k.get(k)
    row = {"K": str(k)}
    for ch in CHANNELS:
        # The scoreboard header carries a trailing arrow glyph (RMSE_u10↓).
        row[f"RMSE_{ch}"] = q.get(f"RMSE_{ch}↓", "") if q else ""
    row["CRPS"] = q.get("CRPS↓", "") if q else ""
    row["time_s_per_member_rollout"] = f"{t_k:.4f}" if t_k is not None else ""
    row["time_s_per_forecast_hour"] = tr["sec_per_forecast_hour"] if tr else ""
    row[SP_KMAX] = ratio(t_kmax, t_k)
    row[SP_FLOW] = ratio(t_flow, t_k)
    row[SP_EDM] = ratio(t_edm, t_k)
    row["metric_leg_s_per_seq"] = q.get("Time/Seq.(s)", "") if q else ""
    rows.append(row)

if not rows:
    print("[stitch] FATAL no rows assembled — did either leg run?", file=sys.stderr)
    sys.exit(1)

csv_path = ROOT / "scoreboard_K_ablation.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=HEADER)
    w.writeheader()
    w.writerows(rows)
print(f"[stitch] wrote {csv_path}")

md_path = ROOT / "scoreboard_K_ablation.md"
with open(md_path, "w") as f:
    f.write("# MeanFlow sampler-K (NFE) ablation\n\n")
    f.write(
        "One MeanFlow checkpoint sampled at K = "
        f"{', '.join(str(k) for k in KS)} average-velocity steps, cleaned "
        "192x96 + log1p, 2022 validation year. Only the sampler step count "
        "varies.\n\n"
        "RMSE is in physical units (u10/v10 m/s, t2m K, qpepre mm/h) against "
        "the RWRF target, scored on the ensemble mean. Times are warmed-up "
        "wall clock for one member rollout on a single GPU (from "
        "`timing/timing.csv`); `metric_leg_s_per_seq` is the quality leg's own "
        "un-warmed timing, kept only as a cross-check — do not quote it.\n\n"
        f"Speedups are ratios of that wall clock against K={max(KS)}, FlowCast "
        f"at {FLOW_REF} NFE, and the EDM teacher at {EDM_NFE} NFE. They are "
        "smaller than the raw NFE ratio because every rollout step also pays "
        "the K-independent regression-mean forward + I/O cost.\n\n"
    )
    f.write("| " + " | ".join(HEADER) + " |\n")
    f.write("|" + "|".join("---" for _ in HEADER) + "|\n")
    for r in rows:
        f.write("| " + " | ".join(r[h] for h in HEADER) + " |\n")
print(f"[stitch] wrote {md_path}")
print("[stitch] preview:")
print("[stitch]   " + " | ".join(HEADER))
for r in rows:
    print("[stitch]   " + " | ".join(r[h] for h in HEADER))


# --- Figure: RMSE vs K (one panel per channel) + cost / speedup vs K --------
def make_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from thesis_style import apply_rc
        apply_rc()
    except Exception:
        pass

    def col(name):
        return [(int(r["K"]), float(r[name])) for r in rows if r.get(name)]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, ch in zip(axes.flat, CHANNELS):
        pts = col(f"RMSE_{ch}")
        if not pts:
            ax.set_axis_off()
            continue
        ks, vs = zip(*pts)
        ax.plot(ks, vs, "o-", color="C0")
        ax.set_xlabel("sampler steps K (NFE)")
        ax.set_ylabel(f"RMSE {ch}")
        ax.set_title(ch)
        ax.grid(alpha=0.3)

    ax = axes.flat[4]
    pts = col("time_s_per_member_rollout")
    if pts:
        ks, vs = zip(*pts)
        ax.plot(ks, vs, "o-", color="C1")
    if t_flow:
        ax.axhline(t_flow, ls="--", c="gray", label=f"flowcast@{FLOW_REF}")
    if t_edm:
        ax.axhline(t_edm, ls=":", c="k", label=f"EDM@{EDM_NFE}")
    if t_flow or t_edm:
        ax.legend(fontsize="small")
    ax.set_xlabel("sampler steps K (NFE)")
    ax.set_ylabel("s / member rollout")
    ax.set_title("cost vs K")
    ax.grid(alpha=0.3)

    ax = axes.flat[5]
    for name, marker, color in ((SP_FLOW, "o-", "C2"), (SP_EDM, "s-", "C3")):
        pts = col(name)
        if pts:
            ks, vs = zip(*pts)
            ax.plot(ks, vs, marker, color=color, label=name.replace("speedup_vs_", "vs "))
    ax.axhline(1.0, ls="--", c="gray")
    ax.set_xlabel("sampler steps K (NFE)")
    ax.set_ylabel("speedup (x)")
    ax.set_title("speedup vs K")
    ax.grid(alpha=0.3)
    ax.legend(fontsize="small")

    fig.suptitle("MeanFlow sampler-K ablation — skill and cost vs NFE")
    fig.tight_layout()
    out = ROOT / "K_ablation.png"
    fig.savefig(out, dpi=150)
    print(f"[stitch] wrote {out}")


try:
    make_figure()
except Exception as e:  # a plotting failure must not lose the table
    print(f"[stitch] WARN figure skipped: {type(e).__name__}: {e}")
PY
RC=$?
if [ "${RC}" != "0" ]; then
    warn "stitch step failed with exit code ${RC}"
    exit "${RC}"
fi
log "stitch done in $(elapsed_s ${S_T0})"

banner "All done — total elapsed $(elapsed_s ${SCRIPT_T0})"
log "outputs:"
log "  K-ablation table : ${OUT_DIR}/scoreboard_K_ablation.{md,csv}"
log "  figure           : ${OUT_DIR}/K_ablation.png"
log "  quality leg      : ${OUT_DIR}/quality/scoreboard.{md,csv}   log: ${OUT_DIR}/quality.log"
log "  timing leg       : ${OUT_DIR}/timing/timing.{md,csv}        log: ${OUT_DIR}/timing.log"
if [ -f "${OUT_DIR}/scoreboard_K_ablation.md" ]; then
    sed 's/^/[K_abl]     | /' "${OUT_DIR}/scoreboard_K_ablation.md"
fi
