#!/bin/bash
# Run the StormCast MeanFlow ABLATION SUITE on zettabyte cloud, 7x H100, on top
# of the same cleaned dataset + regression checkpoint (step 8000) as the
# production MeanFlow leg (train_meanflow.sh).
#
# This is a thin entry point: the actual per-ablation recipe lives in
#   ../../experiment_scripts/meanflow_ablations/   (_common.sh + 00_baseline.sh ..
#   13_attn.sh + run_all.sh + README.md)
# so there is a single source of truth for the shared config. Each ablation
# changes exactly ONE knob relative to the production recipe and trains to a
# ~2.0M-sample budget (batch_size=112 x total_train_steps=18000), saving a
# SINGLE checkpoint at the end (checkpoint_freq == total_train_steps). Runs go
# one at a time (each uses all 7 GPUs). See that README for the rationale of
# every ablation and what to report.
#
# Pull the dataset onto the worker first (azcopy, see ../../zettabyte/zettabyte.md):
#   export SAS_URL="https://zbstore2026.blob.core.windows.net/g-019c8ca2-605d-7bb5-b98b-1c53fbdf2b7f?se=...sig=..."
#   SRC="${SAS_URL%%\?*}/desmond/dataset/zarr_exp3_L_24_H_24_train_2_5_years_full_cleaned_4_27_2026?${SAS_URL#*\?}"
#   azcopy copy "$SRC" /workspace/downloads --recursive
# The regression mean it conditions on is already on /data from the production
# MeanFlow run; no extra download is needed beyond what train_meanflow.sh used.
#
# Dataset spatial shape: 192 (y) x 96 (x). qpepre is stored as log1p(mm/h); the
# loader's denormalize_state auto-applies expm1 for downstream metrics.
#
# Usage (args are forwarded verbatim to run_all.sh):
#   ./train_meanflow_ablations.sh           # core subset (recommended first pass)
#   ./train_meanflow_ablations.sh all       # every ablation
#   ./train_meanflow_ablations.sh 01 05 13  # only scripts starting 01_/05_/13_

# Mirror all stdout/stderr of the whole suite (conda activation + every per-run
# launch) into a timestamped log next to this script, so worker output survives
# the session. Each individual ablation ALSO mirrors to its own log under
# experiment_scripts/meanflow_ablations/logs/.
log_dir="$(cd "$(dirname "$0")" && pwd)/zettabyte_logs"
mkdir -p "${log_dir}"
log_file="${log_dir}/train_meanflow_ablations_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to ${log_file}"
exec > >(tee -a "${log_file}") 2>&1

source $(conda info --base)/etc/profile.d/conda.sh
conda activate stormcast_env

# Resolve the ablation-suite orchestrator relative to this script so it works
# whether the repo is at /workspace/Stormcast-Distilation (zettabyte) or local.
suite="$(cd "$(dirname "$0")/../../experiment_scripts/meanflow_ablations" && pwd)/run_all.sh"
if [[ ! -f "${suite}" ]]; then
    echo "ERROR: ablation suite not found at ${suite}" >&2
    exit 1
fi

echo "Launching MeanFlow ablation suite: ${suite} $*"
bash "${suite}" "$@"
