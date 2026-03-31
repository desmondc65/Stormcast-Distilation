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
training_output_dir="/home/desmond/Documents/master thesis/Stormcast-Distilation/exp_4_train_2_5_yrs_val_1yr_tp1/progressive"
run_id="0"

# --- Logging parameters ---
print_progress_freq=1
checkpoint_freq=5000
validation_freq=500
num_data_workers=0

# --- Training parameters ---
batch_size=16
lr=1E-4
lr_rampup_steps=500
clip_grad_norm=1.0
loss='progressive'

# --- Progressive Distillation parameters ---
initial_num_steps=128   # Teacher uses 2x this; number of phases = log2(initial/target)
target_num_steps=4      # Stop when student reaches this step count
steps_per_phase=50000   # Optimizer steps per phase
rho=7.0                 # Karras schedule exponent
loss_weighting='uniform' # 'uniform' or 'edm'

# --- Validation parameters ---
validation_plot_variables="[t2m,u10,v10,qpepre]"

# --- Optional outputs ---
output_nc="false"
output_nc_freq=5

# --- Dataset parameters ---
location="/home/desmond/Documents/master thesis/Stormcast-Distilation/exp_4_train_2_5_yrs_val_1yr_tp1/zarr_exp4_L_24_H_74_train_2_5_years_full"
HighRes_img_size="[224,128]"
exp_train_zarrs="[stormcast_test_train]"
train_dates="[2019/08/01,2021/12/31]"
exp_valid_zarrs="[stormcast_test_valid]"
valid_dates="[2022/01/01,2022/12/31]"
kept_LowRes_channels="[u10,v10,t2m,sp,mslp,tcwv,u50,u100,u150,u200,u250,u300,u400,u500,u600,u700,u850,u925,u1000,v50,v100,v150,v200,v250,v300,v400,v500,v600,v700,v850,v925,v1000,z50,z100,z150,z200,z250,z300,z400,z500,z600,z700,z850,z925,z1000,t50,t100,t150,t200,t250,t300,t400,t500,t600,t700,t850,t925,t1000,q50,q100,q150,q200,q250,q300,q400,q500,q600,q700,q850,q925,q1000,qpepre,lsm,orog]"
kept_HighRes_channels="[u10, v10, t2m, qpepre]"

# --- Model parameters ---
regression_weights="/home/desmond/Documents/master thesis/Stormcast-Distilation/exp_4_train_2_5_yrs_val_1yr_tp1/exp_4_reg_L_24_H_74_train_2_5_years/0/checkpoints_regression/StormCastUNet.0.2500.mdlus"
teacher_weights="/home/desmond/Documents/master thesis/Stormcast-Distilation/exp_4_train_2_5_yrs_val_1yr_tp1/exp_4_dif_L_24_H_74_train_2_5_years/0/checkpoints_diffusion/EDMPrecond.0.40000.mdlus"

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
