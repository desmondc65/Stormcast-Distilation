#!/bin/bash
# =============================================================================
# Shared launcher for the MeanFlow ablation suite (Taiwan RWRF, zettabyte 7-GPU).
#
# Every ablation script in this directory sources this file and then calls
#   run_meanflow_ablation <experiment_name> [extra ++hydra.overrides ...]
# changing exactly ONE knob relative to the production baseline so each run is
# a clean controlled A/B against 00_baseline.
#
# DESIGN DECISIONS (read before launching):
#   * Budget = ~2.0M samples.  batch_size=112 (7 GPUs x 16/GPU, must stay a
#     multiple of world_size=7) x total_train_steps=18000 = 2,016,000 samples.
#     This matches the "~2M-sample budget" of the deployed headline leg
#     (MeanFlowPrecond.0.20000.mdlus) closely enough to be directly comparable,
#     while keeping every ablation cheap (~hours, not a day).
#   * ONE checkpoint per run.  checkpoint_freq == total_train_steps, so the
#     trainer only writes a checkpoint at the final (done) step ~2M samples.
#     Validation still runs every validation_freq steps for the loss/RMSE/PS1D
#     CSVs + heatmaps, but those do NOT write model weights -- only one
#     MeanFlowPrecond.0.18000.mdlus (+ ema_state.pt) lands on disk per ablation.
#     NB: because nothing is saved mid-run, an interrupted run restarts from
#     scratch (no resume point). Lower CHECKPOINT_FREQ below if you need that.
#   * Self-contained cosine LR schedule that fully decays at the 2M budget
#     (warmup -> cosine -> min_lr at step 18000), identical across all runs, so
#     the only difference between an ablation and 00_baseline is the named knob.
#   * EMA weights (ema_state.pt) are the inference weights for MeanFlow, exactly
#     as in the production run.
#
# The numerical defaults below are copied verbatim from the production launcher
# stormcast/zettabyte_scripts/train_meanflow.sh (the run that produced the
# deployed checkpoint) so the baseline reproduces that recipe at the 2M budget.
# =============================================================================

set -euo pipefail

# --- Environment ---------------------------------------------------------------
# Activate the training env. NB: a `bash <script>` child shell does NOT inherit
# conda's shell *function* (only PATH/CONDA_* env vars), so calling `conda`
# directly fails with "command not found" when this file is sourced from
# run_all.sh / the zettabyte wrapper. Skip if the env is already active (those
# launchers activate and export CONDA_DEFAULT_ENV), else locate conda's profile
# script without relying on `conda` being on PATH.
if [[ "${CONDA_DEFAULT_ENV:-}" != "stormcast_env" ]]; then
    __conda_base=""
    if command -v conda >/dev/null 2>&1; then
        __conda_base="$(conda info --base)"
    elif [[ -n "${CONDA_EXE:-}" ]]; then
        __conda_base="$(dirname "$(dirname "${CONDA_EXE}")")"
    else
        for __c in /opt/conda "${HOME}/miniconda3" "${HOME}/anaconda3" /workspace/miniconda3; do
            [[ -f "${__c}/etc/profile.d/conda.sh" ]] && { __conda_base="${__c}"; break; }
        done
    fi
    if [[ -n "${__conda_base}" && -f "${__conda_base}/etc/profile.d/conda.sh" ]]; then
        source "${__conda_base}/etc/profile.d/conda.sh"
        conda activate stormcast_env
    else
        echo "WARN: could not locate conda to activate stormcast_env; relying on inherited PATH" >&2
    fi
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6

# --- Torchrun ------------------------------------------------------------------
NUMBER_OF_NODES=1
GPUS_PER_NODE=7

# --- Code / output roots -------------------------------------------------------
STORMCAST_TRAIN="/workspace/Stormcast-Distilation/stormcast/train_meanflow.py"
CONFIG="--config-name meanflow"
# All ablation runs land side-by-side under this base; each gets its own subdir.
ABLATION_BASE="/data/exp_3_train_2_5_yrs_val_1yr_tp1/meanflow_ablations_cleaned_4_27_2026"
RUN_ID="0"

# --- Budget (~2.0M samples, ONE checkpoint at the end) -------------------------
BATCH_SIZE=112              # 7 GPUs x 16/GPU; MUST be a multiple of world_size=7
TOTAL_TRAIN_STEPS=18000     # 18000 x 112 = 2,016,000 samples ~= 2.0M
CHECKPOINT_FREQ=18000       # == TOTAL_TRAIN_STEPS -> single checkpoint at done
VALIDATION_FREQ=500         # diagnostics only (CSV + plots), no weights written
PRINT_PROGRESS_FREQ=25
NUM_DATA_WORKERS=4

# --- Optimization (matched to FlowCast: AdamW, cosine w/ short warmup) ----------
LR=5E-4
WEIGHT_DECAY=1.0E-4
ADAM_BETAS="[0.9,0.999]"
LR_WARMUP_STEPS=1000        # ~5.5% of the 2M-sample schedule
MIN_LR_RATIO=0.01           # min_lr = lr * min_lr_ratio (cosine floor at 2M)
CLIP_GRAD_NORM=1.0
FP_OPTIMIZATIONS="fp32"
EMA_DECAY=0.999

# --- MeanFlow objective (Geng et al. 2025) -------------------------------------
LOSS="meanflow"
T_EPS=1.0E-5
MF_RATIO=0.25               # fraction of batch on the r<t average-velocity identity
ADAPTIVE_P=1.0              # adaptive weight w = 1/(mse+eps)^p; 0 = plain MSE
ADAPTIVE_EPS=1.0E-3
SIGMA_DATA=0.5             # std used to standardize the residual (match teacher)
TIME_SCALE=1000.0
SPATIAL_POS_EMBED="True"
ATTN_RESOLUTIONS="[]"       # res = 192>>level in {192,96,48,24,12}; [] = no attn

# --- Sampler (validation only) -------------------------------------------------
VALID_NUM_STEPS=2           # average-velocity segments (NFE) used for valid plots

# --- Hybrid loss (per-channel beta + log-PSD on qpepre) ------------------------
# channel_weights order MUST match kept_HighRes_channels below: u10, v10, t2m, qpepre.
CHANNEL_WEIGHTS="[1.0,1.0,1.0,2.0]"
SPECTRAL_CHANNELS="[qpepre]"
SPECTRAL_WEIGHT=0.1

# --- Validation outputs --------------------------------------------------------
VALIDATION_PLOT_VARIABLES="[t2m,u10,v10,qpepre]"
OUTPUT_NC="false"
OUTPUT_NC_FREQ=5

# --- Dataset (cleaned 192x96, log1p qpepre) ------------------------------------
LOCATION="/workspace/downloads/zarr_exp3_L_24_H_24_train_2_5_years_full_cleaned_4_27_2026"
HIGHRES_IMG_SIZE="[192,96]"
EXP_TRAIN_ZARRS="[stormcast_test_train]"
TRAIN_DATES="[2019/08/01,2021/12/31]"
EXP_VALID_ZARRS="[stormcast_test_valid]"
VALID_DATES="[2022/01/01,2022/12/31]"
KEPT_LOWRES_CHANNELS="all"
KEPT_HIGHRES_CHANNELS="[u10, v10, t2m, qpepre]"
QPEPRE_LOG1P="true"

# --- Model: frozen regression mean (conditioning + residual anchor) ------------
REGRESSION_WEIGHTS="/data/exp_3_train_2_5_yrs_val_1yr_tp1/regression_zettabyte_v1_cleaned_4_27_2026/regression_zettabyte_cleaned_4_27_2026/run_0/checkpoints_regression/StormCastUNet.0.8000.mdlus"

# -----------------------------------------------------------------------------
# run_meanflow_ablation <experiment_name> [extra ++overrides ...]
#   Launches one ablation. The leading args are the run name; everything after
#   is appended to the hydra command line, so a trailing "++training.mf_ratio=0.5"
#   wins over the baseline value set here (Hydra: last ++ occurrence wins).
# -----------------------------------------------------------------------------
run_meanflow_ablation() {
    local experiment_name="$1"; shift
    local rundir="${ABLATION_BASE}/${experiment_name}/run_${RUN_ID}"

    # Mirror this run's stdout/stderr to a timestamped log next to the script.
    local log_dir
    log_dir="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)/logs"
    mkdir -p "${log_dir}"
    local log_file="${log_dir}/${experiment_name}_$(date +%Y%m%d_%H%M%S).log"
    echo "=============================================================="
    echo " MeanFlow ablation : ${experiment_name}"
    echo " budget            : ${TOTAL_TRAIN_STEPS} steps x ${BATCH_SIZE} = $((TOTAL_TRAIN_STEPS * BATCH_SIZE)) samples"
    echo " rundir            : ${rundir}"
    echo " extra overrides   : $*"
    echo " logging to        : ${log_file}"
    echo "=============================================================="

    python -m torch.distributed.run --standalone \
        --nnodes="${NUMBER_OF_NODES}" --nproc_per_node="${GPUS_PER_NODE}" \
        "${STORMCAST_TRAIN}" ${CONFIG} \
        "hydra.run.dir=${rundir}/hydra" \
        "++training.experiment_name=${experiment_name}" \
        "++training.run_id=${RUN_ID}" \
        "++training.rundir=${rundir}" \
        "++training.print_progress_freq=${PRINT_PROGRESS_FREQ}" \
        "++training.checkpoint_freq=${CHECKPOINT_FREQ}" \
        "++training.validation_freq=${VALIDATION_FREQ}" \
        "++training.num_data_workers=${NUM_DATA_WORKERS}" \
        "++training.batch_size=${BATCH_SIZE}" \
        "++training.lr=${LR}" \
        "++training.weight_decay=${WEIGHT_DECAY}" \
        "++training.adam_betas=${ADAM_BETAS}" \
        "++training.lr_warmup_steps=${LR_WARMUP_STEPS}" \
        "++training.min_lr_ratio=${MIN_LR_RATIO}" \
        "++training.total_train_steps=${TOTAL_TRAIN_STEPS}" \
        "++training.clip_grad_norm=${CLIP_GRAD_NORM}" \
        "++training.loss=${LOSS}" \
        "++training.fp_optimizations=${FP_OPTIMIZATIONS}" \
        "++training.t_eps=${T_EPS}" \
        "++training.mf_ratio=${MF_RATIO}" \
        "++training.adaptive_p=${ADAPTIVE_P}" \
        "++training.adaptive_eps=${ADAPTIVE_EPS}" \
        "++training.ema_decay=${EMA_DECAY}" \
        "++training.valid_num_steps=${VALID_NUM_STEPS}" \
        "++training.channel_weights=${CHANNEL_WEIGHTS}" \
        "++training.spectral_channels=${SPECTRAL_CHANNELS}" \
        "++training.spectral_weight=${SPECTRAL_WEIGHT}" \
        "++training.output_nc=${OUTPUT_NC}" \
        "++training.output_nc_freq=${OUTPUT_NC_FREQ}" \
        "++training.validation_plot_variables=${VALIDATION_PLOT_VARIABLES}" \
        "++dataset.location=${LOCATION}" \
        "++dataset.HighRes_img_size=${HIGHRES_IMG_SIZE}" \
        "++dataset.exp_train_zarrs=${EXP_TRAIN_ZARRS}" \
        "++dataset.train_dates=${TRAIN_DATES}" \
        "++dataset.exp_valid_zarrs=${EXP_VALID_ZARRS}" \
        "++dataset.valid_dates=${VALID_DATES}" \
        "++dataset.kept_LowRes_channels=${KEPT_LOWRES_CHANNELS}" \
        "++dataset.kept_HighRes_channels=${KEPT_HIGHRES_CHANNELS}" \
        "++dataset.qpepre_log1p=${QPEPRE_LOG1P}" \
        "++model.regression_weights=${REGRESSION_WEIGHTS}" \
        "++model.sigma_data=${SIGMA_DATA}" \
        "++model.time_scale=${TIME_SCALE}" \
        "++model.spatial_pos_embed=${SPATIAL_POS_EMBED}" \
        "++model.attn_resolutions=${ATTN_RESOLUTIONS}" \
        "$@" 2>&1 | tee -a "${log_file}"
}
