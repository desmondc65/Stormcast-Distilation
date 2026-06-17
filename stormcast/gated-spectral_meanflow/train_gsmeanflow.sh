#!/bin/bash
# Launch Gated-Spectral MeanFlow (GS-MeanFlow) training on Taiwan RWRF.
# Run from the stormcast/ directory. Paths below assume the local workstation
# layout (see CLAUDE.md S1); override for the cleaned zettabyte dataset.

export CUDA_VISIBLE_DEVICES=0,1

# --- Torchrun settings ---
number_of_nodes=1
gpus_per_node=2

# --- General training config ---
stormcast_train="$(dirname "$0")/train_gsmeanflow.py"
config="--config-name gsmeanflow"
experiment_name="gsmeanflow"
training_output_dir="$(pwd)/StormCast_gsmeanflow"
run_id="0"

# --- Logging parameters ---
print_progress_freq=25
checkpoint_freq=5000
validation_freq=500

# --- Training parameters (kept identical to MeanFlow / FlowCast for a clean A/B) ---
batch_size=16
lr=5E-4
weight_decay=1E-4
lr_warmup_steps=4000
min_lr_ratio=0.01
total_train_steps=400000
clip_grad_norm=1.0
ema_decay=0.999
loss='gsmeanflow'

# --- MeanFlow objective parameters ---
t_eps=1.0E-5
mf_ratio=0.25
adaptive_p=1.0
adaptive_eps=1.0E-3
sigma_data=0.5
time_scale=1000.0

# --- Occurrence (hurdle) gate parameters ---
gate_base_channels=48
gate_threshold_mm=0.1
gate_weight=1.0
gate_focal_gamma=2.0
gate_focal_alpha=0.75
gate_prob_threshold=0.5

# --- Validation sampling ---
valid_num_steps=2

# --- Hybrid loss (per-channel beta + optional log-PSD on qpepre) ---
channel_weights="[1.0,1.0,1.0,2.0]"
spectral_channels="[qpepre]"
spectral_weight=0.1

# --- Validation parameters ---
validation_plot_variables="[t2m,u10,v10,qpepre]"

# --- Optional outputs ---
output_nc="false"
output_nc_freq=5

# --- Dataset parameters (local Taiwan RWRF zarr) ---
location="/home/desmond/Documents/master_thesis/Stormcast-Distilation/exp_3_train_2_5_yrs_val_1yr_tp1/zarr_exp3_L_24_H_24_train_2_5_years_full"
HighRes_img_size="[224,128]"
exp_train_zarrs="[stormcast_test_train]"
train_dates="[2019/08/01,2021/12/31]"
exp_valid_zarrs="[stormcast_test_valid]"
valid_dates="[2022/01/01,2022/12/31]"
kept_LowRes_channels="all"
kept_HighRes_channels="all"

# --- Model parameters ---
regression_weights="/home/desmond/Documents/master_thesis/Stormcast-Distilation/exp_3_train_2_5_yrs_val_1yr_tp1/exp_3_reg_L_24_H_4_train_2_5_years/0/checkpoints_regression/StormCastUNet.0.14000.mdlus"

# Execute training with torchrun
torchrun --standalone --nnodes=${number_of_nodes} --nproc_per_node=${gpus_per_node} ${stormcast_train} ${config} \
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
    ++training.t_eps=${t_eps} \
    ++training.mf_ratio=${mf_ratio} \
    ++training.adaptive_p=${adaptive_p} \
    ++training.adaptive_eps=${adaptive_eps} \
    ++training.valid_num_steps=${valid_num_steps} \
    ++training.channel_weights=${channel_weights} \
    ++training.spectral_channels=${spectral_channels} \
    ++training.spectral_weight=${spectral_weight} \
    ++training.gate_threshold_mm=${gate_threshold_mm} \
    ++training.gate_weight=${gate_weight} \
    ++training.gate_focal_gamma=${gate_focal_gamma} \
    ++training.gate_focal_alpha=${gate_focal_alpha} \
    ++training.gate_prob_threshold=${gate_prob_threshold} \
    ++training.output_nc=${output_nc} \
    ++training.output_nc_freq=${output_nc_freq} \
    ++training.validation_plot_variables=${validation_plot_variables} \
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
    ++model.time_scale=${time_scale} \
    ++model.gate_base_channels=${gate_base_channels}
