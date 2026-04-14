#!/bin/bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate stormcast_env

export CUDA_VISIBLE_DEVICES=0,1,2,3

# --- Torchrun settings ---
number_of_nodes=1
gpus_per_node=4

# --- General training config ---
stormcast_train="/workspace/Stormcast-Distilation/stormcast/train_consistency.py"
config="--config-name consistency"
experiment_name="consistency_ncdr"
training_output_dir="/data/exp_3_train_2_5_yrs_val_1yr_tp1/consistency_zettabyte_v1"
run_id="0"

# --- Logging parameters ---
print_progress_freq=25
checkpoint_freq=2500
validation_freq=50
num_data_workers=4

# --- Training parameters ---
batch_size=64
lr=1E-4
lr_rampup_steps=500
total_train_steps=400000
clip_grad_norm=1.0
loss='consistency'

# --- Consistency Distillation parameters ---
N_0=2            # Initial discretization steps
N_total=150      # Final discretization steps (grows via sqrt schedule over training)
rho=7.0          # Karras schedule exponent
huber_c=0.00054  # Pseudo-Huber loss constant (set to 0.0 to use MSE instead)
ema_decay_init=0.95  # Base EMA decay mu_0; adaptive: mu_k = mu_0^(N_0/N_k)

# --- Hybrid loss (per-channel β + log-PSD on qpepre), see CD spec §2 ---
channel_weights="[1.0,1.0,1.0,2.0]"   # β for (t2m, u10, v10, qpepre)
spectral_channels="[qpepre]"           # channels to add radial log-PSD term
spectral_weight=0.1                    # α_spec

# --- Validation parameters ---
validation_plot_variables="[t2m,u10,v10,qpepre]"
valid_num_steps=1                     # 1=one-shot; try 2 or 4 for multi-step sampling

# --- Optional outputs ---
output_nc="false"
output_nc_freq=5

# --- Dataset parameters ---
location="/workspace/downloads/zarr_exp3_L_24_H_24_train_2_5_years_full"
HighRes_img_size="[224,128]"
exp_train_zarrs="[stormcast_test_train]"
train_dates="[2019/08/01,2021/12/31]"
exp_valid_zarrs="[stormcast_test_valid]"
valid_dates="[2022/01/01,2022/12/31]"
kept_LowRes_channels="all"
kept_HighRes_channels="[u10, v10, t2m, qpepre]"

# --- Model parameters ---
regression_weights="/workspace/downloads/exp_3_reg_L_24_H_4_train_2_5_years/0/checkpoints_regression/StormCastUNet.0.14000.mdlus"
teacher_weights="/workspace/downloads/exp_3_dif_L_24_H_4_train_2_5_years/0/checkpoints_diffusion/EDMPrecond.0.70000.mdlus"

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
    "++training.lr_rampup_steps=${lr_rampup_steps}" \
    "++training.total_train_steps=${total_train_steps}" \
    "++training.clip_grad_norm=${clip_grad_norm}" \
    "++training.loss=${loss}" \
    "++training.N_0=${N_0}" \
    "++training.N_total=${N_total}" \
    "++training.rho=${rho}" \
    "++training.huber_c=${huber_c}" \
    "++training.ema_decay_init=${ema_decay_init}" \
    "++training.channel_weights=${channel_weights}" \
    "++training.spectral_channels=${spectral_channels}" \
    "++training.spectral_weight=${spectral_weight}" \
    "++training.output_nc=${output_nc}" \
    "++training.output_nc_freq=${output_nc_freq}" \
    "++training.validation_plot_variables=${validation_plot_variables}" \
    "++training.valid_num_steps=${valid_num_steps}" \
    "++dataset.location=${location}" \
    "++dataset.HighRes_img_size=${HighRes_img_size}" \
    "++dataset.exp_train_zarrs=${exp_train_zarrs}" \
    "++dataset.train_dates=${train_dates}" \
    "++dataset.exp_valid_zarrs=${exp_valid_zarrs}" \
    "++dataset.valid_dates=${valid_dates}" \
    "++dataset.kept_LowRes_channels=${kept_LowRes_channels}" \
    "++dataset.kept_HighRes_channels=${kept_HighRes_channels}" \
    "++model.regression_weights=${regression_weights}" \
    "++model.teacher_weights=${teacher_weights}"
