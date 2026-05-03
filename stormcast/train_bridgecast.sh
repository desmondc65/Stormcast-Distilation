#!/bin/bash
# Launch BridgeCast (mean-anchored Schrödinger bridge) on Taiwan RWRF.
# Plan: ../new_method_plan.md ; paths assume the local workstation layout
# documented in CLAUDE.md §1.

export CUDA_VISIBLE_DEVICES=0,1

# --- Torchrun settings ---
number_of_nodes=1
gpus_per_node=2

# --- General training config ---
stormcast_train="$(dirname "$0")/train_bridgecast.py"
config="--config-name bridgecast"
experiment_name="bridgecast_full"
training_output_dir="$(pwd)/StormCast_bridgecast"
run_id="0"

# --- Logging ---
print_progress_freq=25
checkpoint_freq=5000
validation_freq=500

# --- Optimization (FlowCast-paper compatible) ---
batch_size=16
lr=5E-4
weight_decay=1E-4
lr_warmup_steps=4000
min_lr_ratio=0.01
total_train_steps=400000
clip_grad_norm=1.0
ema_decay=0.999
loss='bridgecast'

# --- Bridge parameters (plan §6) ---
sigma_bridge=0.15
sigma_anchor=0.0
t_eps=1.0E-3
sigma_data=0.5        # match the EDM teacher
time_scale=1000.0

# --- Climatology-correlated noise (plan §2.3) ---
white_noise_ablation=false   # set true for ablation #4 (white prior)
psd_num_samples=512

# --- qpepre channel-adaptive path (plan §2.4) ---
qpepre_asinh=true
qpepre_kappa=1.0
rain_threshold=0.1
enable_mask_head=true
apply_mask_gate=true
mask_threshold=0.5
nonneg_qpepre=true

# --- Energy-Score ensemble objective (plan §2.6) ---
es_K=4                 # set 0 for ablation #6 (no ES)
es_pool=4

# --- Loss weights (plan §2.8) ---
lambda_v=1.0
lambda_m=0.1
lambda_e=0.5
lambda_d=1.0E-3

# --- Channel weights / spectral regularizer (parity with FlowCast) ---
channel_weights="[1.0,1.0,1.0,2.0]"
spectral_channels="[qpepre]"
spectral_weight=0.1

# --- Physics correctors ---
divergence_channels="[u10,v10]"   # null disables the term

# --- Validation sampler (plan §3 Algorithm 1) ---
valid_num_steps=2          # S
solver='midpoint'          # 'midpoint' or 'euler'

# --- Validation outputs ---
validation_plot_variables="[t2m,u10,v10,qpepre]"
output_nc="false"
output_nc_freq=5

# --- Dataset (Taiwan RWRF zarr) ---
location="/home/desmond/Documents/master_thesis/Stormcast-Distilation/exp_3_train_2_5_yrs_val_1yr_tp1/zarr_exp3_L_24_H_24_train_2_5_years_full"
HighRes_img_size="[224,128]"
exp_train_zarrs="[stormcast_test_train]"
train_dates="[2019/08/01,2021/12/31]"
exp_valid_zarrs="[stormcast_test_valid]"
valid_dates="[2022/01/01,2022/12/31]"
kept_LowRes_channels="all"
kept_HighRes_channels="all"

# --- Pretrained regression checkpoint ---
regression_weights="/home/desmond/Documents/master_thesis/Stormcast-Distilation/exp_3_train_2_5_yrs_val_1yr_tp1/exp_3_reg_L_24_H_4_train_2_5_years/0/checkpoints_regression/StormCastUNet.0.14000.mdlus"

# --- Launch ---
torchrun --standalone --nnodes=${number_of_nodes} --nproc_per_node=${gpus_per_node} \
    ${stormcast_train} ${config} \
    hydra.run.dir=${training_output_dir} \
    ++training.experiment_name=${experiment_name} \
    ++training.run_id=${run_id} \
    ++training.rundir=${training_output_dir}/${experiment_name}/run_${run_id} \
    ++training.print_progress_freq=${print_progress_freq} \
    ++training.checkpoint_freq=${checkpoint_freq} \
    ++training.validation_freq=${validation_freq} \
    ++training.batch_size=${batch_size} \
    ++training.lr=${lr} \
    ++training.weight_decay=${weight_decay} \
    ++training.lr_warmup_steps=${lr_warmup_steps} \
    ++training.min_lr_ratio=${min_lr_ratio} \
    ++training.total_train_steps=${total_train_steps} \
    ++training.clip_grad_norm=${clip_grad_norm} \
    ++training.ema_decay=${ema_decay} \
    ++training.loss=${loss} \
    ++training.sigma_bridge=${sigma_bridge} \
    ++training.sigma_anchor=${sigma_anchor} \
    ++training.t_eps=${t_eps} \
    ++training.white_noise_ablation=${white_noise_ablation} \
    ++training.psd_num_samples=${psd_num_samples} \
    ++training.qpepre_asinh=${qpepre_asinh} \
    ++training.qpepre_kappa=${qpepre_kappa} \
    ++training.rain_threshold=${rain_threshold} \
    ++training.enable_mask_head=${enable_mask_head} \
    ++training.apply_mask_gate=${apply_mask_gate} \
    ++training.mask_threshold=${mask_threshold} \
    ++training.nonneg_qpepre=${nonneg_qpepre} \
    ++training.es_K=${es_K} \
    ++training.es_pool=${es_pool} \
    ++training.lambda_v=${lambda_v} \
    ++training.lambda_m=${lambda_m} \
    ++training.lambda_e=${lambda_e} \
    ++training.lambda_d=${lambda_d} \
    ++training.channel_weights=${channel_weights} \
    ++training.spectral_channels=${spectral_channels} \
    ++training.spectral_weight=${spectral_weight} \
    ++training.divergence_channels=${divergence_channels} \
    ++training.valid_num_steps=${valid_num_steps} \
    ++training.solver=${solver} \
    ++training.validation_plot_variables=${validation_plot_variables} \
    ++training.output_nc=${output_nc} \
    ++training.output_nc_freq=${output_nc_freq} \
    ++dataset.location=${location} \
    ++dataset.HighRes_img_size=${HighRes_img_size} \
    ++dataset.exp_train_zarrs=${exp_train_zarrs} \
    ++dataset.train_dates=${train_dates} \
    ++dataset.exp_valid_zarrs=${exp_valid_zarrs} \
    ++dataset.valid_dates=${valid_dates} \
    ++dataset.kept_LowRes_channels=${kept_LowRes_channels} \
    ++dataset.kept_HighRes_channels=${kept_HighRes_channels} \
    ++model.regression_weights=${regression_weights} \
    ++model.sigma_data=${sigma_data} \
    ++model.time_scale=${time_scale}
