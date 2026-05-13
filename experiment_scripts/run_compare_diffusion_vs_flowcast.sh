#!/bin/bash
# Drive ``compare_diffusion_vs_flowcast.py`` with the same dataset / channel
# layout the zettabyte training scripts use. Override paths via flags.
#
# Defaults follow ``experiment_scripts/analysis_plan.md``:
#   diffusion: runs/diffusion_zettabyte_v1_cleaned_4_27_2026/.../EDMPrecond.0.30000.mdlus
#   flowcast : runs/flowcast_zettabyte_v1_cleaned_4_27_2026/.../FlowCastPrecond.0.25000.mdlus
#   regress. : runs/regression_zettabyte_v1_cleaned_4_27_2026/.../StormCastUNet.0.8000.mdlus
#
# Run with:
#   bash experiment_scripts/run_compare_diffusion_vs_flowcast.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Pick up the stormcast python env if available; fall back to the active python.
if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh" || true
    if conda env list 2>/dev/null | grep -q "stormcast_env"; then
        conda activate stormcast_env
    fi
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python -u "${REPO_ROOT}/experiment_scripts/compare_diffusion_vs_flowcast.py" \
    --n-sequences "${N_SEQUENCES:-24}" \
    --n-steps "${N_STEPS:-6}" \
    --ensemble "${ENSEMBLE:-4}" \
    --diffusion-num-steps "${DIFFUSION_NFE:-18}" \
    --flowcast-num-steps "${FLOWCAST_NFE:-10}" \
    --output-dir "${OUTPUT_DIR:-${REPO_ROOT}/experiment_scripts/results/diffusion_vs_flowcast}" \
    "$@"
