#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# --- Torchrun settings ---
number_of_nodes=1
gpus_per_node=1

# --- General training config ---
stormcast_train="/home/desmond/Documents/master_thesis/Stormcast-Distilation/stormcast/train.py"
config="--config-name diffusion.yaml"
experiment_name="diffusion_ncdr"
training_output_dir="/home/desmond/Documents/master_thesis/Stormcast-Distilation/stormcast/data/Stormcast_test/diffusion_test_2"
run_id="0"

# -- logging parameters ---
print_progress_freq=5
checkpoint_freq=1000
validation_freq=10

# --- Training parameters ---
batch_size=4
lr=4E-4
lr_rampup_steps=1000
total_train_steps=800000
clip_grad_norm=-1 # Threshold for gradient clipping, set to -1 to disable
loss='edm' # 'edm' or 'regression'

# --- Validation parameters ---
validation_plot_variables="[t2m,u10,v10,qpepre]"

# --- Optional outputs ---
# When set to true, NetCDF files for validation fields will be written to
# ${training_output_dir}/${experiment_name}/run_${run_id}/netcdf_outputs/<field>.
# Default is false.
output_nc="true"
# Output NetCDF every X validations (e.g., if set to 5, only output when validation_counter % 5 == 0).
# Default is 1 (output every validation).
output_nc_freq=5

# --- Dataset parameters ---
location="/home/desmond/Documents/master_thesis/Stormcast-Distilation/exp_3_train_2_5_yrs_val_1yr_tp1/zarr_exp3_L_24_H_24_train_2_5_years_full"
HighRes_img_size="[224,128]"
exp_train_zarrs="[stormcast_test_train]"
train_dates="[2019/08/01,2021/12/31]"
exp_valid_zarrs="[stormcast_test_valid]"
valid_dates="[2022/01/01,2022/12/31]"
kept_LowRes_channels="all"
kept_HighRes_channels="[u10, v10, t2m, qpepre]"

# --- Model parameters ---
regression_weights="/home/desmond/Documents/master_thesis/Stormcast-Distilation/exp_3_train_2_5_yrs_val_1yr_tp1/exp_3_reg_L_24_H_4_train_2_5_years/0/checkpoints_regression/StormCastUNet.0.7500.mdlus" 
initial_weights="/home/desmond/Documents/master_thesis/Stormcast-Distilation/exp_3_train_2_5_yrs_val_1yr_tp1/exp_3_dif_L_24_H_4_train_2_5_years/0/checkpoints_diffusion/EDMPrecond.0.70000.mdlus" # Path to pretrained diffusion mdlus, used if 'edm' is included in diffusion_conditions   
# Path to pretrained regression mdlus, used if 'regression' is included in diffusion_conditions

# execute training with torchrun 
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
    ++training.initial_weights=${initial_weights} \
    ++training.lr_rampup_steps=${lr_rampup_steps} \
    ++training.total_train_steps=${total_train_steps} \
    ++training.clip_grad_norm=${clip_grad_norm} \
    ++training.loss=${loss} \
    ++training.output_nc=${output_nc} \
    ++training.output_nc_freq=${output_nc_freq} \
    "++validation.plot_variables=${validation_plot_variables}" \
    ++dataset.location=${location} \
    "++dataset.HighRes_img_size=${HighRes_img_size}" \
    "++dataset.exp_train_zarrs=${exp_train_zarrs}" \
    "++dataset.train_dates=${train_dates}" \
    "++dataset.exp_valid_zarrs=${exp_valid_zarrs}" \
    "++dataset.valid_dates=${valid_dates}" \
    ++dataset.kept_LowRes_channels=${kept_LowRes_channels} \
    "++dataset.kept_HighRes_channels=${kept_HighRes_channels}" \
    ++model.regression_weights=${regression_weights}
