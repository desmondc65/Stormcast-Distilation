#!/bin/bash
export CUDA_VISIBLE_DEVICES=4,5

# --- Torchrun settings ---
number_of_nodes=1
gpus_per_node=2

# --- General training config ---
stormcast_train="/home/master/13/dczy/code/stormcast-ncdr/stormcast/train_add.py"
config="--config-name add"
experiment_name="add_ncdr"
training_output_dir="/home/master/13/dczy/code/stormcast-ncdr/data/Stormcast_test/add"
run_id="0"

# --- Logging parameters ---
print_progress_freq=25
checkpoint_freq=5000
validation_freq=500

# --- Training parameters ---
batch_size=16
lr_G=1E-5
lr_D=4E-5
lr_rampup_steps=1000
total_train_steps=400000
clip_grad_norm=1.0
loss='add'

# --- ADD Distillation parameters ---
num_student_steps=1          # Number of student denoising steps (1-4)
lambda_distill=2.5           # Score distillation loss weight
r1_gamma=0.00001             # R1 gradient penalty weight (1e-5, per ADD design)
use_nfsd=false               # Noise-Free Score Distillation
P_std=1.2                    # Std for score distillation sigma sampling
ema_decay=0.999              # Fixed EMA decay

# --- Discriminator parameters ---
disc_base_ch=64              # Base channel width
disc_num_scales=4            # Number of multi-scale heads
condition_discriminator=true # Condition discriminator on input state

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
    ++training.lr_G=${lr_G} \
    ++training.lr_D=${lr_D} \
    ++training.lr_rampup_steps=${lr_rampup_steps} \
    ++training.total_train_steps=${total_train_steps} \
    ++training.clip_grad_norm=${clip_grad_norm} \
    ++training.loss=${loss} \
    ++training.num_student_steps=${num_student_steps} \
    ++training.lambda_distill=${lambda_distill} \
    ++training.r1_gamma=${r1_gamma} \
    ++training.use_nfsd=${use_nfsd} \
    ++training.P_std=${P_std} \
    ++training.ema_decay=${ema_decay} \
    ++training.disc_base_ch=${disc_base_ch} \
    ++training.disc_num_scales=${disc_num_scales} \
    ++training.condition_discriminator=${condition_discriminator} \
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
