#!/bin/bash
# Train the StormCast GS-MeanFlow student (Gated-Spectral MeanFlow) on zettabyte
# cloud, 7x H100, on top of the cleaned dataset and the regression checkpoint
# trained at step 8000.
#
# GS-MeanFlow keeps the MeanFlow average-velocity residual head (1-2 NFE) and
# adds two pillars (see ../gated-spectral_meanflow/README.md):
#   1. group-decoupled adaptive weighting -- the smooth channels (t2m,u10,v10)
#      and the precip channel (qpepre) are re-weighted independently, removing
#      the cross-channel objective conflict the qpw ablation exposes;
#   2. an occurrence (hurdle) gate -- a tiny deterministic head predicts qpepre
#      wet/dry and forces confidently-dry pixels to exact zero at inference.
#
# EVERYTHING ELSE (dataset, regression mean, AdamW/cosine, channel order,
# sigma_data, mf_ratio, spectral term, sample budget) is kept identical to
# train_meanflow.sh, so GS-MeanFlow vs MeanFlow is a clean A/B on the two added
# pillars. The gate is ~460x smaller than the SongUNet flow, so it adds
# negligible compute/memory on top of MeanFlow (the JVP r < t subset is still
# the dominant cost); drop ++training.batch_size_per_gpu if a worker OOMs.
#
# Pull the dataset onto the worker first (azcopy, see ../../zettabyte/zettabyte.md):
#   export SAS_URL="https://zbstore2026.blob.core.windows.net/g-019c8ca2-605d-7bb5-b98b-1c53fbdf2b7f?se=...sig=..."
#   SRC="${SAS_URL%%\?*}/desmond/dataset/zarr_exp3_L_24_H_24_train_2_5_years_full_cleaned_4_27_2026?${SAS_URL#*\?}"
#   azcopy copy "$SRC" /workspace/downloads --recursive
#
# Dataset spatial shape: 192 (y) x 96 (x). qpepre is stored as log1p(mm/h);
# the loader's denormalize_state auto-applies expm1 for downstream metrics.

# Mirror all stdout/stderr of this script (conda activation included) into a
# timestamped log next to the script, so worker output survives the session.
log_dir="$(cd "$(dirname "$0")" && pwd)/zettabyte_logs"
mkdir -p "${log_dir}"
log_file="${log_dir}/train_gsmeanflow_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to ${log_file}"
exec > >(tee -a "${log_file}") 2>&1

source $(conda info --base)/etc/profile.d/conda.sh
conda activate stormcast_env

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6

# --- Torchrun settings ---
number_of_nodes=1
gpus_per_node=7

# --- General training config ---
stormcast_train="/workspace/Stormcast-Distilation/stormcast/gated-spectral_meanflow/train_gsmeanflow.py"
config="--config-name gsmeanflow"
experiment_name="gsmeanflow_zettabyte_cleaned_4_27_2026"
training_output_dir="/data/exp_3_train_2_5_yrs_val_1yr_tp1/gsmeanflow_zettabyte_v1_cleaned_4_27_2026"
run_id="0"

# --- Logging parameters ---
print_progress_freq=25
checkpoint_freq=2500
validation_freq=250
num_data_workers=4

# --- Training parameters (matched to FlowCast/MeanFlow: AdamW, cosine w/ warmup) ---
batch_size=112        # 7 GPUs x 16/GPU (must be a multiple of world_size=7)
lr=5E-4
weight_decay=1.0E-4
adam_betas="[0.9,0.999]"
lr_warmup_steps=4000
min_lr_ratio=0.01          # min_lr = lr * min_lr_ratio
total_train_steps=40000
clip_grad_norm=1.0
loss='gsmeanflow'
fp_optimizations='fp32'

# --- MeanFlow objective parameters (Geng et al. 2025) ---
t_eps=1.0E-5               # clamp for time samples ~ U(eps, 1-eps)
mf_ratio=0.25              # fraction of batch on the r < t identity
adaptive_p=1.0             # adaptive weight w = 1/(mse + eps)^p; 0 = off
adaptive_eps=1.0E-3
ema_decay=0.999            # same convention as the FlowCast student

# --- Occurrence (hurdle) gate parameters (GS-MeanFlow Pillar 2) ---
gate_base_channels=48      # width of the lightweight occurrence-gate U-Net
gate_threshold_mm=0.1      # physical precip threshold defining "wet"
gate_weight=1.0            # weight on the focal-BCE occurrence term
gate_focal_gamma=2.0       # focal focusing exponent (0 => weighted BCE)
gate_focal_alpha=0.75      # positive-class (wet) weighting; >0.5 emphasizes wet
gate_prob_threshold=0.5    # wet-probability boundary for the validation sampler

# --- Sampler (validation only) ---
valid_num_steps=2          # average-velocity segments (NFE); 1 = one-step

# --- Hybrid loss (per-channel β + log-PSD on qpepre), mirrors FlowCast/MeanFlow ---
# Order of channel_weights MUST match kept_HighRes_channels below: u10, v10, t2m, qpepre.
# Pillar 1 (group-decoupled adaptive weighting) splits qpepre from the smooth
# channels automatically using the qpepre channel index.
channel_weights="[1.0,1.0,1.0,2.0]"
spectral_channels="[qpepre]"          # channels to add radial log-PSD term
spectral_weight=0.1                   # α_spec

# --- Validation parameters ---
validation_plot_variables="[t2m,u10,v10,qpepre]"

# --- Optional outputs ---
output_nc="false"
output_nc_freq=5

# --- Dataset parameters ---
location="/workspace/downloads/zarr_exp3_L_24_H_24_train_2_5_years_full_cleaned_4_27_2026"
HighRes_img_size="[192,96]"
exp_train_zarrs="[stormcast_test_train]"
train_dates="[2019/08/01,2021/12/31]"
exp_valid_zarrs="[stormcast_test_valid]"
valid_dates="[2022/01/01,2022/12/31]"
kept_LowRes_channels="all"
# Match the channel order used by regression / diffusion / flowcast / meanflow
# scripts so the resulting GS-MeanFlow student is drop-in compatible.
kept_HighRes_channels="[u10, v10, t2m, qpepre]"
qpepre_log1p="true"

# --- Model parameters ---
# GS-MeanFlow is a one-stage average-velocity net learning the residual
# R_t = X_t - M_t on top of the frozen regression mean M, plus an occurrence
# gate. No diffusion teacher is used.
regression_weights="/data/exp_3_train_2_5_yrs_val_1yr_tp1/regression_zettabyte_v1_cleaned_4_27_2026/regression_zettabyte_cleaned_4_27_2026/run_0/checkpoints_regression/StormCastUNet.0.8000.mdlus"
sigma_data=0.5             # std used to standardize the residual
time_scale=1000.0          # scales r ∈ [0,1] into SongUNet's positional-embed range
spatial_pos_embed="True"

# Execute training with torchrun
python -m torch.distributed.run --standalone --nnodes="${number_of_nodes}" --nproc_per_node="${gpus_per_node}" "${stormcast_train}" ${config} \
    "hydra.run.dir=${training_output_dir}" \
    "++training.experiment_name=${experiment_name}" \
    "++training.run_id=${run_id}" \
    "++training.rundir=${training_output_dir}/${experiment_name}/run_${run_id}" \
    "++training.print_progress_freq=${print_progress_freq}" \
    "++training.checkpoint_freq=${checkpoint_freq}" \
    "++training.validation_freq=${validation_freq}" \
    "++training.num_data_workers=${num_data_workers}" \
    "++training.batch_size=${batch_size}" \
    "++training.lr=${lr}" \
    "++training.weight_decay=${weight_decay}" \
    "++training.adam_betas=${adam_betas}" \
    "++training.lr_warmup_steps=${lr_warmup_steps}" \
    "++training.min_lr_ratio=${min_lr_ratio}" \
    "++training.total_train_steps=${total_train_steps}" \
    "++training.clip_grad_norm=${clip_grad_norm}" \
    "++training.loss=${loss}" \
    "++training.fp_optimizations=${fp_optimizations}" \
    "++training.t_eps=${t_eps}" \
    "++training.mf_ratio=${mf_ratio}" \
    "++training.adaptive_p=${adaptive_p}" \
    "++training.adaptive_eps=${adaptive_eps}" \
    "++training.ema_decay=${ema_decay}" \
    "++training.gate_threshold_mm=${gate_threshold_mm}" \
    "++training.gate_weight=${gate_weight}" \
    "++training.gate_focal_gamma=${gate_focal_gamma}" \
    "++training.gate_focal_alpha=${gate_focal_alpha}" \
    "++training.gate_prob_threshold=${gate_prob_threshold}" \
    "++training.valid_num_steps=${valid_num_steps}" \
    "++training.channel_weights=${channel_weights}" \
    "++training.spectral_channels=${spectral_channels}" \
    "++training.spectral_weight=${spectral_weight}" \
    "++training.output_nc=${output_nc}" \
    "++training.output_nc_freq=${output_nc_freq}" \
    "++training.validation_plot_variables=${validation_plot_variables}" \
    "++dataset.location=${location}" \
    "++dataset.HighRes_img_size=${HighRes_img_size}" \
    "++dataset.exp_train_zarrs=${exp_train_zarrs}" \
    "++dataset.train_dates=${train_dates}" \
    "++dataset.exp_valid_zarrs=${exp_valid_zarrs}" \
    "++dataset.valid_dates=${valid_dates}" \
    "++dataset.kept_LowRes_channels=${kept_LowRes_channels}" \
    "++dataset.kept_HighRes_channels=${kept_HighRes_channels}" \
    "++dataset.qpepre_log1p=${qpepre_log1p}" \
    "++model.regression_weights=${regression_weights}" \
    "++model.sigma_data=${sigma_data}" \
    "++model.time_scale=${time_scale}" \
    "++model.gate_base_channels=${gate_base_channels}" \
    "++model.spatial_pos_embed=${spatial_pos_embed}"
