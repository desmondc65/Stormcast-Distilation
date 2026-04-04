#!/bin/bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate stormcast_env

export CUDA_VISIBLE_DEVICES=0

# --- Torchrun settings ---
number_of_nodes=1
gpus_per_node=1

# --- General training config ---
stormcast_train="/home/desmond/Documents/master thesis/Stormcast-Distilation/stormcast/train_progressive.py"
config="--config-name progressive"
experiment_name="progressive_ncdr"
training_output_dir="/home/desmond/Documents/master thesis/Stormcast-Distilation/exp_3_train_2_5_yrs_val_1yr_tp1/progressive"
run_id="0"

# --- Logging parameters ---
print_progress_freq=5
checkpoint_freq=5000
validation_freq=10
num_data_workers=4

# --- Training parameters ---
batch_size=4
lr=1E-4
lr_rampup_steps=500
clip_grad_norm=1.0
loss='progressive'

# --- Progressive Distillation parameters ---
initial_num_steps=18    # Teacher's sampling steps (18 → 9 → 4 → 2 → 1)
target_num_steps=1      # Stop when student reaches 1 step
steps_per_phase=50000   # Optimizer steps per phase
rho=7.0                 # Karras schedule exponent
loss_weighting='edm' # 'uniform' or 'edm'; edm normalizes across noise levels

# --- Validation parameters ---
validation_plot_variables="[t2m,u10,v10,qpepre]"

# --- Optional outputs ---
output_nc="false"
output_nc_freq=5

# --- Dataset parameters ---
location="/home/desmond/Documents/master thesis/Stormcast-Distilation/exp_3_train_2_5_yrs_val_1yr_tp1/zarr_exp3_L_24_H_24_train_2_5_years_full"
HighRes_img_size="[224,128]"
exp_train_zarrs="[stormcast_test_train]"
train_dates="[2019/08/01,2021/12/31]"
exp_valid_zarrs="[stormcast_test_valid]"
valid_dates="[2022/01/01,2022/12/31]"
kept_LowRes_channels="all"
kept_HighRes_channels="[u10, v10, t2m, qpepre]"

# --- Model parameters ---
regression_weights="/home/desmond/Documents/master thesis/Stormcast-Distilation/exp_3_train_2_5_yrs_val_1yr_tp1/exp_3_reg_L_24_H_4_train_2_5_years/0/checkpoints_regression/StormCastUNet.0.14000.mdlus"
teacher_weights="/home/desmond/Documents/master thesis/Stormcast-Distilation/exp_3_train_2_5_yrs_val_1yr_tp1/exp_3_dif_L_24_H_4_train_2_5_years/0/checkpoints_diffusion/EDMPrecond.0.70000.mdlus"

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
    "++training.clip_grad_norm=${clip_grad_norm}" \
    "++training.loss=${loss}" \
    "++training.initial_num_steps=${initial_num_steps}" \
    "++training.target_num_steps=${target_num_steps}" \
    "++training.steps_per_phase=${steps_per_phase}" \
    "++training.rho=${rho}" \
    "++training.loss_weighting=${loss_weighting}" \
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
    "++model.regression_weights=${regression_weights}" \
    "++model.teacher_weights=${teacher_weights}"
