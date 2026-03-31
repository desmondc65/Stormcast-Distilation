#!/bin/bash
export CUDA_VISIBLE_DEVICES=4,5

# --- Torchrun settings ---
number_of_nodes=1
gpus_per_node=2

# --- General training config ---
stormcast_train="/home/master/13/dczy/code/stormcast-ncdr/stormcast/train_consistency.py"
config="--config-name consistency"
experiment_name="consistency_ncdr"
training_output_dir="/home/master/13/dczy/code/stormcast-ncdr/data/Stormcast_test/consistency"
run_id="0"

# --- Logging parameters ---
print_progress_freq=25
checkpoint_freq=5000
validation_freq=500

# --- Training parameters ---
batch_size=16
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

# --- Validation parameters ---
validation_plot_variables="[t2m,u10,v10,qpepre]"

# --- Optional outputs ---
output_nc="false"
output_nc_freq=5

# --- Dataset parameters ---
location="/project/n/desmond/Stormcast_test/Zarr_test_optimized_skip_invalid"
HighRes_img_size="[224,128]"
exp_train_zarrs="[train]"
train_dates="[2019/08/01,2019/08/17]"
exp_valid_zarrs="[valid]"
valid_dates="[2019/08/18,2019/08/31]"
kept_LowRes_channels="all"
kept_HighRes_channels="all"

# --- Model parameters ---
regression_weights="/home/master/13/dczy/code/stormcast-ncdr/data/Stormcast_test/regression/regression_ncdr/run_0/checkpoints_regression/StormCastUNet.0.1000.mdlus"
teacher_weights="/home/master/13/dczy/code/stormcast-ncdr/data/Stormcast_test/diffusion/diffusion_ncdr/run_0/checkpoints/EDMPrecond.0.1000.mdlus"

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
    ++training.lr_rampup_steps=${lr_rampup_steps} \
    ++training.total_train_steps=${total_train_steps} \
    ++training.clip_grad_norm=${clip_grad_norm} \
    ++training.loss=${loss} \
    ++training.N_0=${N_0} \
    ++training.N_total=${N_total} \
    ++training.rho=${rho} \
    ++training.huber_c=${huber_c} \
    ++training.ema_decay_init=${ema_decay_init} \
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
    ++model.teacher_weights=${teacher_weights}
