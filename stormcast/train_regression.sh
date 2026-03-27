#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

# --- Torchrun settings ---
number_of_nodes=1
gpus_per_node=4

# --- General training config ---
stormcast_train="/home/master/13/dczy/code/stormcast-ncdr/stormcast/train.py"
config="--config-name regression.yaml"
experiment_name="regression_ncdr"
training_output_dir="/project/n/desmond/diffusion_output/regression"
run_id="0"

# -- logging parameters ---
print_progress_freq=25
checkpoint_freq=1000
validation_freq=50

# --- Training parameters ---
batch_size=12
lr=4E-4
lr_rampup_steps=1000
total_train_steps=16000
clip_grad_norm=-1 # Threshold for gradient clipping, set to -1 to disable
loss='regression'
# seed=42 # Set a positive seed to avoid distributed broadcast issues in single-GPU mode

# --- Validation parameters ---
validation_plot_variables="[t2m,u10,v10,qpepre]"

# --- Optional outputs ---
# When set to true, NetCDF files for validation fields will be written to
# ${training_output_dir}/${experiment_name}/run_${run_id}/netcdf_outputs/<field>.
# Default is false.
output_nc="false"
# Output NetCDF every X validations (e.g., if set to 5, only output when validation_counter % 5 == 0).
# Default is 1 (output every validation).
output_nc_freq=5

# --- Dataset parameters ---
location="/project/n/desmond/Stormcast_test/Zarr_test_optimized_skip_invalid"
HighRes_img_size="[224,128]"
exp_train_zarrs="[train]" # Zarr files to use for training
train_dates="[2019/08/01,2019/08/17]"
exp_valid_zarrs="[valid]" # Zarr files to use for validation
valid_dates="[2019/08/18,2019/08/31]"
kept_LowRes_channels="all"
kept_HighRes_channels="all"

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
    ++training.lr_rampup_steps=${lr_rampup_steps} \
    ++training.total_train_steps=${total_train_steps} \
    ++training.clip_grad_norm=${clip_grad_norm} \
    ++training.loss=${loss} \
    ++training.output_nc=${output_nc} \
    ++training.output_nc_freq=${output_nc_freq} \
    ++validation.plot_variables=${validation_plot_variables} \
    ++dataset.location=${location} \
    ++dataset.HighRes_img_size=${HighRes_img_size} \
    ++dataset.exp_train_zarrs=${exp_train_zarrs} \
    ++dataset.train_dates=${train_dates} \
    ++dataset.exp_valid_zarrs=${exp_valid_zarrs} \
    ++dataset.valid_dates=${valid_dates} \
    ++dataset.kept_LowRes_channels=${kept_LowRes_channels} \
    ++dataset.kept_HighRes_channels=${kept_HighRes_channels}
