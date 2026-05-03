#!/bin/bash
# Train the StormCast BridgeCast student (mean-anchored Schrödinger bridge on
# the regression residual R_t = X_t - M_t) on zettabyte cloud, using the
# cleaned 4/27/2026 dataset and the regression checkpoint trained at step 8000.
# Method: ../../new_method_plan.md
#
# Pull the dataset onto the worker first (azcopy, see ../../zettabyte/zettabyte.md):
#   export SAS_URL="https://zbstore2026.blob.core.windows.net/g-019c8ca2-605d-7bb5-b98b-1c53fbdf2b7f?se=...sig=..."
#   SRC="${SAS_URL%%\?*}/desmond/dataset/zarr_exp3_L_24_H_24_train_2_5_years_full_cleaned_4_27_2026?${SAS_URL#*\?}"
#   azcopy copy "$SRC" /workspace/downloads --recursive
#
# Dataset spatial shape: 192 (y) x 96 (x). qpepre is stored as log1p(mm/h);
# the loader's denormalize_state auto-applies expm1 for downstream metrics.
# Because qpepre is already log1p-compressed (std ≈ 0.37, in the same regime
# as the wind channels), we DISABLE the BridgeCast asinh path — the log1p
# already buys us the heavy-tail compression asinh was meant to provide.

source $(conda info --base)/etc/profile.d/conda.sh
conda activate stormcast_env

export CUDA_VISIBLE_DEVICES=0,1,2,3

# --- Torchrun settings ---
number_of_nodes=1
gpus_per_node=4

# --- General training config ---
stormcast_train="/workspace/Stormcast-Distilation/stormcast/train_bridgecast.py"
config="--config-name bridgecast"
experiment_name="bridgecast_zettabyte_cleaned_4_27_2026"
training_output_dir="/data/exp_3_train_2_5_yrs_val_1yr_tp1/bridgecast_v2_zettabyte_v1_cleaned_4_27_2026"
run_id="0"

# --- Logging ---
print_progress_freq=25
checkpoint_freq=2500
validation_freq=250
num_data_workers=4

# --- Optimization (FlowCast-paper compatible: AdamW, cosine w/ 1% warmup) ---
batch_size=48
lr=5E-4
weight_decay=1.0E-4
adam_betas="[0.9,0.999]"
lr_warmup_steps=4000
min_lr_ratio=0.01
total_train_steps=400000
clip_grad_norm=1.0
loss='bridgecast'
# H100-native bf16 AMP — halves activation memory vs fp32 with no quality
# regression. Required to fit BridgeCast (antithetic + K-sample ES) on
# 4xH100 at batch_size=64; the FlowCast paper used fp32 because it has no
# antithetic / ES branches.
fp_optimizations='fp32'
ema_decay=0.999

# --- Bridge parameters (plan §6) ---
sigma_bridge=0.15           # σ_b — bridge noise (peaks at t=1/2)
sigma_anchor=0.0            # σ_a — anchor jitter (0 ⇒ deterministic anchor)
t_eps=1.0E-3                # endpoint clip on [0,1] for γ'(t) stability

# --- Climatology-correlated noise prior (plan §2.3) ---
# The first launch will fit and cache ${rundir}/clim_psd.npy from the
# training residuals (~10 minutes on a single GPU walking 512 samples).
# Set white_noise_ablation=true for ablation #4 (i.i.d. white prior).
white_noise_ablation=false
psd_num_samples=512

# --- qpepre channel-adaptive path (plan §2.4) ---
# DISABLED on zettabyte: the loader applies log1p(mm/h) before standardisation,
# which already compresses the heavy tail. asinh on top would be near-linear
# and add no benefit. To run ablation #5 (no asinh / no mask), keep these
# defaults. To experiment with double-compression set qpepre_asinh=true.
qpepre_asinh=false
qpepre_kappa=1.0
# rain_threshold is interpreted in the loader's standardised log1p(mm/h)
# space. Standardised value > 0 → above-channel-mean precipitation, which is
# a reasonable wet/dry binary on the cleaned 4/27/2026 dataset.
rain_threshold=0.0
enable_mask_head=true
# Mask gate / non-neg clamp at inference are SKIPPED automatically when the
# loader uses qpepre_log1p=true (the dataset's qpepre is in standardised
# log1p space, where "0" is the channel mean — NOT no-rain — so a min=0
# clamp / x*0 gate would force every dry pixel up to the mean and produce
# the flat-zero-background-with-spikes failure mode visible in early runs).
# The mask head still trains as an auxiliary signal; just don't apply it
# post-hoc at inference time.
apply_mask_gate=false
mask_threshold=0.5
nonneg_qpepre=false
# Trust-region upper bound on the qpepre channel of x_pred (in the loader's
# standardised log1p space). A 200 mm/h typhoon corresponds to standardised
# ~14 with sigma_log1p=0.37; 25 leaves comfortable headroom while killing
# the runaway-pixel spikes (model outputs of magnitude 100-150 visible in
# early validation images) that otherwise dominate RMSE.
qpepre_clip_std=25.0

# --- Energy-Score ensemble objective (plan §2.6) ---
# K=2 is the smallest strictly-proper estimator; K=4 was the plan default
# but each extra K runs a *full* SongUNet forward pass with retained graph
# and pushes 4xH100 OOM at batch_size=64. K=2 still trains calibration
# correctly with ~half the activation memory. Bump to 4 if memory allows.
# Set es_K=0 for ablation #6 (no ES).
es_K=2
es_pool=4

# --- Loss weights (plan §2.8) ---
lambda_v=1.0
lambda_m=0.1
lambda_e=0.5
lambda_d=1.0E-3

# --- Channel weights / spectral regularizer ---
# Order MUST match kept_HighRes_channels below: u10, v10, t2m, qpepre.
# qpepre weight kept at 1.0 on the log1p loader (its std is 0.37, in the
# same regime as the wind channels). The 2.0 boost was a heavy-tail balance
# for the raw mm/h channel and amplifies the gradient on rare extreme rain
# pixels — exactly the failure mode the spike artifacts in early validation
# images came from.
channel_weights="[1.0,1.0,1.0,1.0]"
spectral_channels="[qpepre]"
spectral_weight=0.1

# --- Soft physics priors (plan §2.7) ---
divergence_channels="[u10,v10]"

# --- Validation sampler (plan §3 Algorithm 1) ---
valid_num_steps=2           # S — Heun-midpoint steps
solver='midpoint'           # 'midpoint' or 'euler'

validation_plot_variables="[t2m,u10,v10,qpepre]"
output_nc="false"
output_nc_freq=5

# --- Dataset (cleaned 4/27/2026 zarr, log1p qpepre, 192x96 grid) ---
location="/workspace/downloads/zarr_exp3_L_24_H_24_train_2_5_years_full_cleaned_4_27_2026"
HighRes_img_size="[192,96]"
exp_train_zarrs="[stormcast_test_train]"
train_dates="[2019/08/01,2021/12/31]"
exp_valid_zarrs="[stormcast_test_valid]"
valid_dates="[2022/01/01,2022/12/31]"
kept_LowRes_channels="all"
# Keep the channel order consistent with the FlowCast / regression / CD scripts
# so checkpoints are drop-in interchangeable across methods.
kept_HighRes_channels="[u10, v10, t2m, qpepre]"
qpepre_log1p="true"

# --- Pretrained regression checkpoint (frozen anchor for BridgeCast) ---
regression_weights="/data/exp_3_train_2_5_yrs_val_1yr_tp1/regression_zettabyte_v1_cleaned_4_27_2026/regression_zettabyte_cleaned_4_27_2026/run_0/checkpoints_regression/StormCastUNet.0.8000.mdlus"
sigma_data=0.5              # standardisation std (match teacher / FlowCast)
time_scale=1000.0           # scales t ∈ [0,1] into SongUNet positional embed
spatial_pos_embed="True"

# --- Launch ---
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
    "++training.ema_decay=${ema_decay}" \
    "++training.sigma_bridge=${sigma_bridge}" \
    "++training.sigma_anchor=${sigma_anchor}" \
    "++training.t_eps=${t_eps}" \
    "++training.white_noise_ablation=${white_noise_ablation}" \
    "++training.psd_num_samples=${psd_num_samples}" \
    "++training.qpepre_asinh=${qpepre_asinh}" \
    "++training.qpepre_kappa=${qpepre_kappa}" \
    "++training.rain_threshold=${rain_threshold}" \
    "++training.enable_mask_head=${enable_mask_head}" \
    "++training.apply_mask_gate=${apply_mask_gate}" \
    "++training.mask_threshold=${mask_threshold}" \
    "++training.nonneg_qpepre=${nonneg_qpepre}" \
    "++training.qpepre_clip_std=${qpepre_clip_std}" \
    "++training.es_K=${es_K}" \
    "++training.es_pool=${es_pool}" \
    "++training.lambda_v=${lambda_v}" \
    "++training.lambda_m=${lambda_m}" \
    "++training.lambda_e=${lambda_e}" \
    "++training.lambda_d=${lambda_d}" \
    "++training.channel_weights=${channel_weights}" \
    "++training.spectral_channels=${spectral_channels}" \
    "++training.spectral_weight=${spectral_weight}" \
    "++training.divergence_channels=${divergence_channels}" \
    "++training.valid_num_steps=${valid_num_steps}" \
    "++training.solver=${solver}" \
    "++training.validation_plot_variables=${validation_plot_variables}" \
    "++training.output_nc=${output_nc}" \
    "++training.output_nc_freq=${output_nc_freq}" \
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
