#!/usr/bin/env bash
# Run every progressive-distillation experiment in sequence and write results
# under ``experiment_scripts/progressive/results/``.
#
# Usage:
#     bash run_all.sh                       # default budgets, full plan
#     bash run_all.sh --skip rollout        # skip individual experiments by name
#     N_SAMPLES=512 N_INITS=48 bash run_all.sh
#
# Environment:
#     CONDA_ENV         conda env to activate (default: stormcast_env)
#     N_SAMPLES         samples for metrics + failure-mode scripts (default: 256)
#     N_INITS           initial conditions for rollout (default: 32)
#     ROLLOUT_STRIDE    hours between consecutive rollout init times (default: 96)
#     VALID_DATES       e.g. "2022/01/01 2022/12/31"
#     PHASES_DIR        alternative progressive-run directory
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

CONDA_ENV="${CONDA_ENV:-stormcast_env}"
N_SAMPLES="${N_SAMPLES:-256}"
N_INITS="${N_INITS:-32}"
ROLLOUT_STRIDE="${ROLLOUT_STRIDE:-96}"
VALID_DATES="${VALID_DATES:-2022/01/01 2022/12/31}"
PHASES_DIR="${PHASES_DIR:-}"

# ---- conda activation ----------------------------------------------------
CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090,SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
else
    echo "WARNING: conda not found on PATH — assuming env is already active" >&2
fi

echo "[env] python: $(command -v python)"
python -c "import torch; print(f'[env] torch={torch.__version__} cuda={torch.cuda.is_available()}')"

# ---- selection flags -----------------------------------------------------
RUN_METRICS=1
RUN_ROLLOUT=1
RUN_QUAL=1
RUN_FAILURE=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip)
            case "$2" in
                metrics)  RUN_METRICS=0 ;;
                rollout)  RUN_ROLLOUT=0 ;;
                qualitative) RUN_QUAL=0 ;;
                failure)  RUN_FAILURE=0 ;;
                *) echo "unknown skip target: $2" >&2; exit 2 ;;
            esac
            shift 2
            ;;
        --only)
            RUN_METRICS=0 RUN_ROLLOUT=0 RUN_QUAL=0 RUN_FAILURE=0
            case "$2" in
                metrics)  RUN_METRICS=1 ;;
                rollout)  RUN_ROLLOUT=1 ;;
                qualitative) RUN_QUAL=1 ;;
                failure)  RUN_FAILURE=1 ;;
                *) echo "unknown only target: $2" >&2; exit 2 ;;
            esac
            shift 2
            ;;
        *)
            echo "usage: $0 [--only <exp>] [--skip <exp>]..." >&2
            exit 2
            ;;
    esac
done

COMMON_ARGS=(--valid-dates ${VALID_DATES})
if [[ -n "${PHASES_DIR}" ]]; then
    COMMON_ARGS+=(--phases-dir "${PHASES_DIR}")
fi

run_step() {
    local name="$1"; shift
    local log="results/${name}.log"
    mkdir -p results
    echo
    echo "=================================================="
    echo "  ${name}"
    echo "=================================================="
    time python -u "$@" "${COMMON_ARGS[@]}" 2>&1 | tee "${log}"
}

if (( RUN_METRICS )); then
    run_step "01_metrics" 01_eval_metrics.py --n-samples "${N_SAMPLES}"
fi

if (( RUN_ROLLOUT )); then
    run_step "02_rollout" 02_eval_rollout.py --n-inits "${N_INITS}" --stride-hours "${ROLLOUT_STRIDE}"
fi

if (( RUN_QUAL )); then
    run_step "03_qualitative" 03_qualitative_cases.py
fi

if (( RUN_FAILURE )); then
    run_step "04_failure_modes" 04_failure_modes.py --n-samples "${N_SAMPLES}"
fi

echo
echo "✓ all progressive-distillation experiments complete"
echo "  outputs → ${SCRIPT_DIR}/results"
