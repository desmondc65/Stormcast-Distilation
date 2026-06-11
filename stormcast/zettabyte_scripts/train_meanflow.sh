#!/bin/bash
# Train the StormCast MeanFlow student (average-velocity flow matching on the
# regression residual R_t = X_t - M_t) on zettabyte cloud, on top of the
# cleaned dataset and the regression checkpoint trained at step 8000.
#
# Pull the dataset onto the worker first (azcopy, see ../../zettabyte/zettabyte.md):
#   export SAS_URL="https://zbstore2026.blob.core.windows.net/g-019c8ca2-605d-7bb5-b98b-1c53fbdf2b7f?se=...sig=..."
#   SRC="${SAS_URL%%\?*}/desmond/dataset/zarr_exp3_L_24_H_24_train_2_5_years_full_cleaned_4_27_2026?${SAS_URL#*\?}"
#   azcopy copy "$SRC" /workspace/downloads --recursive
#
# Dataset spatial shape: 192 (y) x 96 (x). qpepre is stored as log1p(mm/h);
# the loader's denormalize_state auto-applies expm1 for downstream metrics.
#
# All optimizer / loss-weighting hyperparameters are kept identical to
# train_flowcast.sh so MeanFlow vs FlowCast is an A/B on the objective alone.
# The MeanFlow-specific knobs are mf_ratio / adaptive_p / adaptive_eps, and
# validation samples with 2 NFE instead of 10 Euler steps. Note the r < t
# subset runs an extra JVP forward, so a step costs ~1.5x a FlowCast step in
# compute and the JVP samples roughly double activation memory; drop
# ++training.batch_size_per_gpu if a worker OOMs.

# Mirror all stdout/stderr of this script (conda activation included) into a
# timestamped log next to the script, so worker output survives the session.
log_dir="$(cd "$(dirname "$0")" && pwd)/zettabyte_logs"
mkdir -p "${log_dir}"
log_file="${log_dir}/train_meanflow_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to ${log_file}"
exec > >(tee -a "${log_file}") 2>&1

source $(conda info --base)/etc/profile.d/conda.sh
conda activate stormcast_env

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6

# --- Torchrun settings ---
number_of_nodes=1
gpus_per_node=4

# --- General training config ---
stormcast_train="/workspace/Stormcast-Distilation/stormcast/train_meanflow.py"
config="--config-name meanflow"
experiment_name="meanflow_zettabyte_cleaned_4_27_2026"
training_output_dir="/data/exp_3_train_2_5_yrs_val_1yr_tp1/meanflow_zettabyte_v1_cleaned_4_27_2026"
run_id="0"

# --- Logging parameters ---
print_progress_freq=25
checkpoint_freq=2500
validation_freq=250
num_data_workers=4

# --- Training parameters (matched to FlowCast: AdamW, cosine w/ 1% warmup) ---
batch_size=96
lr=5E-4
weight_decay=1.0E-4
adam_betas="[0.9,0.999]"
lr_warmup_steps=4000       # ~1% of total_train_steps
min_lr_ratio=0.01          # min_lr = lr * min_lr_ratio
total_train_steps=40000
clip_grad_norm=1.0
loss='meanflow'
fp_optimizations='fp32'

# --- MeanFlow objective parameters (Geng et al. 2025) ---
t_eps=1.0E-5               # clamp for time samples ~ U(eps, 1-eps)
mf_ratio=0.25              # fraction of batch on the r < t identity
adaptive_p=1.0             # adaptive weight w = 1/(mse + eps)^p; 0 = off
adaptive_eps=1.0E-3
ema_decay=0.999            # same convention as the FlowCast student

# --- Sampler (validation only) ---
valid_num_steps=2          # average-velocity segments (NFE); 1 = one-step

# --- Hybrid loss (per-channel β + log-PSD on qpepre), mirrors FlowCast ---
# Order of channel_weights MUST match kept_HighRes_channels below: u10, v10, t2m, qpepre.
# qpepre is in log1p space (std 0.37, in the same regime as the wind
# channels), so the 2x boost just nudges the spectral / loss weight without
# the heavy-tail balancing the raw mm/h channel needed.
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
# Match the channel order used by regression / diffusion / flowcast scripts so
# the resulting MeanFlow student is drop-in compatible with the rest of the run.
kept_HighRes_channels="[u10, v10, t2m, qpepre]"
qpepre_log1p="true"

# --- Model parameters ---
# MeanFlow is a one-stage average-velocity net learning the residual
# R_t = X_t - M_t on top of the frozen regression mean M.
# No diffusion teacher is used; the few-step capability comes from the
# JVP-bootstrapped MeanFlow identity, not from distilling EDM.
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
    "++model.spatial_pos_embed=${spatial_pos_embed}"
