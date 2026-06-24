#!/bin/bash
# Inference-timing benchmark — wall-clock cost of one ENSEMBLE-member,
# N_STEPS-hour autoregressive forecast for each generative head, on a SINGLE
# A6000. Timing ONLY: predictions are discarded and NO metrics / panels /
# scoreboard are produced (that is what compare_diffusion_vs_flowcast.py /
# run_main_experiment.sh are for).
#
# Methods timed (matched 2 M-sample cleaned 192x96 + log1p checkpoints, same as
# run_main_experiment.sh Leg B — only the SAMPLER cost differs between them):
#   stormcast (EDM diffusion teacher) : DIFFUSION_NFE Heun steps  (default 18 -> 36 NFE)
#   flowcast  (I-CFM student)         : Euler, one pass per value in FLOWCAST_NFES (10 15 20)
#   meanflow  (avg-velocity student)  : few-step sampler, one pass per value in MEANFLOW_NFES (1 2)
#
# Defaults match the request: ENSEMBLE=50, N_STEPS=24 (24 h rollout), one init
# time (N_SEQUENCES=1) -> one 50-member 24 h ensemble forecast per method.
#
# The model-loading + autoregressive rollout path is imported verbatim from
# experiment_scripts/compare_diffusion_vs_flowcast.py, so the per-step compute
# is identical to the real comparison/inference harness. A WARMUP rollout (not
# timed) absorbs cuDNN autotune / torch.compile so the timed block is steady-state.
#
# Outputs (timing only):
#   ${OUT_DIR}/timing.csv   one row per method/NFE
#   ${OUT_DIR}/timing.md    same, markdown
#   ${OUT_DIR}/timing.log   full run log
#
# Override any default via env, e.g.:
#   ENSEMBLE=10 N_STEPS=12 GPU=1 ./run_timing.sh
#   FLOWCAST_NFES="10 20" MEANFLOW_NFES="" ./run_timing.sh   # skip meanflow

set -euo pipefail

# --- Logging helpers --------------------------------------------------------
SCRIPT_T0=$(date +%s)
log()  { printf '[%s] [run_timing] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
warn() { printf '[%s] [run_timing][WARN] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2; }
banner() {
    printf '\n================================================================\n'
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
    printf '================================================================\n'
}
elapsed_s() { printf '%ds' $(( $(date +%s) - $1 )); }

banner "Inference-timing benchmark — stormcast / flowcast / meanflow on one A6000"

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
EXP_DIR="${REPO_ROOT}/experiment_scripts"
OUT_DIR="${OUT_DIR:-${EXP_DIR}/results/timing}"
mkdir -p "${OUT_DIR}"

# --- Python: prefer the in-repo venv (this box has no working conda) ---------
if [ -z "${PYTHON:-}" ]; then
    if [ -x "${REPO_ROOT}/stormcast_env/bin/python" ]; then
        PYTHON="${REPO_ROOT}/stormcast_env/bin/python"
    else
        PYTHON="$(command -v python)"
    fi
fi

# --- Pin to ONE A6000 -------------------------------------------------------
# Respect a pre-set CUDA_VISIBLE_DEVICES; else honour $GPU; else auto-pick the
# A6000 with the most free memory.
pick_gpu() {
    nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader,nounits 2>/dev/null \
        | awk -F', ' 'tolower($2) ~ /a6000/ {print $3, $1}' \
        | sort -rn | head -1 | awk '{print $2}'
}
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    log "using pre-set CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
elif [ -n "${GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES="${GPU}"
else
    _g="$(pick_gpu)"
    export CUDA_VISIBLE_DEVICES="${_g:-0}"
fi

# --- Knobs ------------------------------------------------------------------
ENSEMBLE="${ENSEMBLE:-50}"                  # members per forecast
N_STEPS="${N_STEPS:-24}"                    # autoregressive horizon, hours
N_SEQUENCES="${N_SEQUENCES:-1}"             # distinct init times (1 = one ensemble forecast)
WARMUP="${WARMUP:-1}"                       # untimed warmup rollouts (absorb autotune)
SEED="${SEED:-0}"

DIFFUSION_NFE="${DIFFUSION_NFE:-18}"        # Heun steps; NFE = 2 * this
DIFFUSION_SOLVER="${DIFFUSION_SOLVER:-heun}"
FLOWCAST_NFES="${FLOWCAST_NFES:-10 15 20}"  # Euler steps; one row each
FLOWCAST_SOLVER="${FLOWCAST_SOLVER:-euler}"
MEANFLOW_NFES="${MEANFLOW_NFES:-1 2}"       # avg-velocity NFE; one row each (empty = skip)
SIGMA_DATA="${SIGMA_DATA:-0.5}"
SIGMA_MIN="${SIGMA_MIN:-0.002}"
SIGMA_MAX="${SIGMA_MAX:-80.0}"
RHO="${RHO:-7.0}"

# --- Dataset / checkpoints (cleaned 192x96 + log1p, canonical CLAUDE.md §6.1) -
DATA_LOCATION="${DATA_LOCATION:-${REPO_ROOT}/exp_3_train_2_5_yrs_val_1yr_tp1/zarr_exp3_L_24_H_24_train_2_5_years_full_cleaned_4_27_2026}"
VALID_DATES="${VALID_DATES:-2022/01/01 2022/12/31}"
HR_SIZE="${HR_SIZE:-192 96}"
KEPT_CHANNELS="${KEPT_CHANNELS:-u10 v10 t2m qpepre}"
QPEPRE_LOG1P="${QPEPRE_LOG1P:-1}"
REGRESSION_CKPT="${REGRESSION_CKPT:-${REPO_ROOT}/runs/regression_zettabyte_v1_cleaned_4_27_2026/regression_zettabyte_cleaned_4_27_2026/run_0/checkpoints_regression/StormCastUNet.0.8000.mdlus}"
DIFFUSION_CKPT="${DIFFUSION_CKPT:-${REPO_ROOT}/runs/diffusion_zettabyte_v1_cleaned_4_27_2026/diffusion_zettabyte_cleaned_4_27_2026/run_0/checkpoints_diffusion/EDMPrecond.0.73000.mdlus}"
FLOWCAST_CKPT="${FLOWCAST_CKPT:-${REPO_ROOT}/runs/flowcast_zettabyte_v1_cleaned_4_27_2026/flowcast_zettabyte_cleaned_4_27_2026/run_0/checkpoints_flowcast/FlowCastPrecond.0.140000.mdlus}"
MEANFLOW_CKPT="${MEANFLOW_CKPT:-${REPO_ROOT}/runs/meanflow_zettabyte_v1_cleaned_4_27_2026/meanflow_zettabyte_cleaned_4_27_2026/run_0/checkpoints_meanflow/MeanFlowPrecond.0.20000.mdlus}"

banner "Configuration"
log "PYTHON        = ${PYTHON}"
log "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}  (one A6000)"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader 2>/dev/null \
    | sed 's/^/[run_timing]   gpu: /' || true
log "OUT_DIR       = ${OUT_DIR}"
log "ENSEMBLE      = ${ENSEMBLE}   N_STEPS = ${N_STEPS}h   N_SEQUENCES = ${N_SEQUENCES}   WARMUP = ${WARMUP}"
log "DIFFUSION_NFE = ${DIFFUSION_NFE} (${DIFFUSION_SOLVER})   FLOWCAST_NFES = ${FLOWCAST_NFES} (${FLOWCAST_SOLVER})   MEANFLOW_NFES = ${MEANFLOW_NFES:-<skip>}"
log "DATA          = ${DATA_LOCATION}"
log "regression    = ${REGRESSION_CKPT}"
log "diffusion     = ${DIFFUSION_CKPT}"
log "flowcast      = ${FLOWCAST_CKPT}"
log "meanflow      = ${MEANFLOW_CKPT}"

# --- Pre-flight: report ALL missing paths before launching python ----------
banner "Pre-flight checks"
_FAIL=0
_REQUIRED=( \
    "DATA_LOCATION=${DATA_LOCATION}" \
    "REGRESSION_CKPT=${REGRESSION_CKPT}" \
    "DIFFUSION_CKPT=${DIFFUSION_CKPT}" \
    "FLOWCAST_CKPT=${FLOWCAST_CKPT}" )
if [ -n "${MEANFLOW_NFES// /}" ]; then
    _REQUIRED+=( "MEANFLOW_CKPT=${MEANFLOW_CKPT}" )
fi
for kv in "${_REQUIRED[@]}"; do
    name="${kv%%=*}"; path="${kv#*=}"
    if [ -e "${path}" ]; then log "  OK     ${name}"; else warn "  MISSING ${name}=${path}"; _FAIL=1; fi
done
if [ "${_FAIL}" = "1" ]; then warn "pre-flight failed — aborting"; exit 1; fi
log "pre-flight OK"

# --- Run --------------------------------------------------------------------
banner "Timing run"
export REPO_ROOT EXP_DIR OUT_DIR ENSEMBLE N_STEPS N_SEQUENCES WARMUP SEED \
       DIFFUSION_NFE DIFFUSION_SOLVER FLOWCAST_NFES FLOWCAST_SOLVER MEANFLOW_NFES \
       SIGMA_DATA SIGMA_MIN SIGMA_MAX RHO \
       DATA_LOCATION VALID_DATES HR_SIZE KEPT_CHANNELS QPEPRE_LOG1P \
       REGRESSION_CKPT DIFFUSION_CKPT FLOWCAST_CKPT MEANFLOW_CKPT

"${PYTHON}" -u - <<'PY' 2>&1 | tee "${OUT_DIR}/timing.log"
import csv
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

EXP_DIR = Path(os.environ["EXP_DIR"])
sys.path.insert(0, str(EXP_DIR))
import compare_diffusion_vs_flowcast as C  # reuse loaders + rollout verbatim


def envlist(name):
    return os.environ.get(name, "").split()


OUT_DIR = Path(os.environ["OUT_DIR"])
ENSEMBLE = int(os.environ["ENSEMBLE"])
N_STEPS = int(os.environ["N_STEPS"])
N_SEQ = int(os.environ["N_SEQUENCES"])
WARMUP = int(os.environ["WARMUP"])
SEED = int(os.environ["SEED"])
DIFF_NFE = int(os.environ["DIFFUSION_NFE"])
DIFF_SOLVER = os.environ["DIFFUSION_SOLVER"]
FLOW_NFES = [int(x) for x in envlist("FLOWCAST_NFES")]
FLOW_SOLVER = os.environ["FLOWCAST_SOLVER"]
MF_NFES = [int(x) for x in envlist("MEANFLOW_NFES")]
SIGMA_DATA = float(os.environ["SIGMA_DATA"])

args = SimpleNamespace(
    data_location=Path(os.environ["DATA_LOCATION"]),
    valid_dates=envlist("VALID_DATES"),
    hr_size=[int(x) for x in envlist("HR_SIZE")],
    qpepre_log1p=(os.environ["QPEPRE_LOG1P"] == "1"),
    kept_channels=envlist("KEPT_CHANNELS"),
)

C.DistributedManager.initialize()
dist = C.DistributedManager()
device = dist.device
if device.type == "cuda":
    torch.cuda.empty_cache()
gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
print(f"[device] {device}  ({gpu_name})", flush=True)

dataset, invariant = C.build_dataset(args, device)
n_pairs = len(dataset)
print(f"[data] usable pairs={n_pairs} channels={list(dataset.state_channels())}", flush=True)
need = N_SEQ * N_STEPS + 1
if n_pairs < need:
    raise SystemExit(f"Not enough validation samples ({n_pairs}) for {N_SEQ}x{N_STEPS} (need {need}).")

# Evenly-spaced init times, same recipe as compare_diffusion_vs_flowcast.py.
t0_indices = np.linspace(0, n_pairs - N_STEPS - 1, N_SEQ).astype(int).tolist()
print(f"[seqs] {len(t0_indices)} init time(s) x {N_STEPS}h  (t0={t0_indices})", flush=True)

print(f"[load] regression {os.environ['REGRESSION_CKPT']}", flush=True)
regression = C.Module.from_checkpoint(os.environ["REGRESSION_CKPT"]).to(device).eval()


def time_method(*, label, ckpt, method, sampler_kwargs, nfe_per_step):
    """Load `ckpt`, warm up, then time ENSEMBLE x N_SEQ rollouts of N_STEPS. Discards output."""
    print(f"\n[load] {label}: {ckpt}", flush=True)
    model = C.Module.from_checkpoint(str(ckpt)).to(device).eval()

    # Warmup (untimed) — absorbs cuDNN autotune / torch.compile on this shape.
    for _ in range(WARMUP):
        C.rollout(model=model, method=method, regression=regression, invariant=invariant,
                  dataset=dataset, t0_idx=t0_indices[0], n_steps=N_STEPS,
                  sampler_kwargs=sampler_kwargs, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()

    n_roll = ENSEMBLE * N_SEQ
    per_roll = []
    print(f"[time] {label}: {ENSEMBLE} members x {N_SEQ} init x {N_STEPS}h "
          f"= {n_roll} rollouts, {nfe_per_step} NFE/step", flush=True)
    t_all = time.perf_counter()
    for k in range(ENSEMBLE):
        torch.manual_seed(SEED + 1000 * k)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(SEED + 1000 * k)
        for t0 in t0_indices:
            t_w = time.perf_counter()
            C.rollout(model=model, method=method, regression=regression, invariant=invariant,
                      dataset=dataset, t0_idx=t0, n_steps=N_STEPS,
                      sampler_kwargs=sampler_kwargs, device=device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            per_roll.append(time.perf_counter() - t_w)
        done = (k + 1) * N_SEQ
        el = time.perf_counter() - t_all
        print(f"[time] {label}  member {k+1}/{ENSEMBLE}  "
              f"elapsed={el:6.1f}s  eta={el/done*(n_roll-done):6.1f}s", flush=True)
    total = time.perf_counter() - t_all

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    per_member = total / n_roll                     # one member, N_STEPS h
    per_hour = per_member / N_STEPS                  # one autoregressive step
    total_nfe = n_roll * N_STEPS * nfe_per_step
    row = {
        "method": label,
        "nfe_per_step": nfe_per_step,
        "ensemble": ENSEMBLE,
        "n_sequences": N_SEQ,
        "n_steps": N_STEPS,
        "n_rollouts": n_roll,
        "total_s": round(total, 3),
        "sec_per_ensemble_forecast": round(total / N_SEQ, 3),
        "sec_per_member_rollout": round(per_member, 4),
        "sec_per_forecast_hour": round(per_hour, 4),
        "sec_per_nfe": round(total / total_nfe, 5),
        "member_rollout_std_s": round(statistics.pstdev(per_roll) if len(per_roll) > 1 else 0.0, 4),
    }
    print(f"[done] {label}: total={total:.1f}s  "
          f"ensemble_forecast={row['sec_per_ensemble_forecast']:.1f}s  "
          f"member={per_member:.3f}s  hour={per_hour:.4f}s", flush=True)
    return row


rows = []

# stormcast = EDM diffusion teacher.
diff_nfe_per_step = (2 * DIFF_NFE) if DIFF_SOLVER == "heun" else DIFF_NFE
rows.append(time_method(
    label="stormcast_edm",
    ckpt=os.environ["DIFFUSION_CKPT"],
    method="diffusion",
    sampler_kwargs=dict(num_steps=DIFF_NFE, sigma_min=float(os.environ["SIGMA_MIN"]),
                        sigma_max=float(os.environ["SIGMA_MAX"]), rho=float(os.environ["RHO"]),
                        solver=DIFF_SOLVER),
    nfe_per_step=diff_nfe_per_step,
))

# flowcast — one row per requested Euler-step count.
for nfe in FLOW_NFES:
    rows.append(time_method(
        label=f"flowcast_nfe{nfe}",
        ckpt=os.environ["FLOWCAST_CKPT"],
        method="flowcast",
        sampler_kwargs=dict(num_steps=nfe, sigma_data=SIGMA_DATA, solver=FLOW_SOLVER),
        nfe_per_step=(nfe if FLOW_SOLVER == "euler" else 2 * nfe),
    ))

# meanflow — one row per requested NFE (opt-in via MEANFLOW_NFES).
for nfe in MF_NFES:
    rows.append(time_method(
        label=f"meanflow_nfe{nfe}",
        ckpt=os.environ["MEANFLOW_CKPT"],
        method="meanflow",
        sampler_kwargs=dict(num_steps=nfe, sigma_data=SIGMA_DATA),
        nfe_per_step=nfe,
    ))

# --- Export timing only -----------------------------------------------------
header = list(rows[0].keys())
csv_path = OUT_DIR / "timing.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=header)
    w.writeheader()
    w.writerows(rows)
print(f"\n[write] {csv_path}", flush=True)

md_path = OUT_DIR / "timing.md"
with open(md_path, "w") as f:
    f.write("# Inference timing — one A6000\n\n")
    f.write(f"GPU `{gpu_name}` | ensemble {ENSEMBLE} | rollout {N_STEPS} h | "
            f"{N_SEQ} init time(s) | warmup {WARMUP}\n\n")
    f.write("`sec_per_ensemble_forecast` = wall-clock for one "
            f"{ENSEMBLE}-member {N_STEPS} h ensemble forecast (the headline cost).\n\n")
    f.write("| " + " | ".join(header) + " |\n")
    f.write("|" + "|".join("---" for _ in header) + "|\n")
    for r in rows:
        f.write("| " + " | ".join(str(r[h]) for h in header) + " |\n")
print(f"[write] {md_path}", flush=True)

print("\n[summary]")
print("  " + " | ".join(header))
for r in rows:
    print("  " + " | ".join(str(r[h]) for h in header))
PY
RC=${PIPESTATUS[0]}
if [ "${RC}" != "0" ]; then
    warn "timing run failed with exit code ${RC} after $(elapsed_s ${SCRIPT_T0})"
    exit "${RC}"
fi

banner "Done — total elapsed $(elapsed_s ${SCRIPT_T0})"
log "timing CSV : ${OUT_DIR}/timing.csv"
log "timing MD  : ${OUT_DIR}/timing.md"
log "run log    : ${OUT_DIR}/timing.log"
if [ -f "${OUT_DIR}/timing.md" ]; then
    sed 's/^/[run_timing]   | /' "${OUT_DIR}/timing.md"
fi
