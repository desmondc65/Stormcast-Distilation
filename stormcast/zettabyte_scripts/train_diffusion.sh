#!/bin/bash
# Train the StormCast EDM diffusion teacher (residual-denoising network)
# on zettabyte cloud, on top of the cleaned dataset and the regression
# checkpoint trained at step 8000.
#
# Pull the dataset onto the worker first (azcopy, see ../../zettabyte/zettabyte.md):
#   export SAS_URL="https://zbstore2026.blob.core.windows.net/g-019c8ca2-605d-7bb5-b98b-1c53fbdf2b7f?se=...sig=..."
#   SRC="${SAS_URL%%\?*}/desmond/dataset/zarr_exp3_L_24_H_24_train_2_5_years_full_cleaned_4_27_2026?${SAS_URL#*\?}"
#   azcopy copy "$SRC" /workspace/downloads --recursive
#
# Dataset spatial shape: 192 (y) x 96 (x). qpepre is stored as log1p(mm/h);
# the loader's denormalize_state auto-applies expm1 so validation plots and
# RMSE/MAE CSVs are reported in mm/h.

source $(conda info --base)/etc/profile.d/conda.sh
conda activate stormcast_env

export CUDA_VISIBLE_DEVICES=0,1,2,3

# --- Torchrun settings ---
number_of_nodes=1
gpus_per_node=4

# --- General training config ---
stormcast_train="/workspace/Stormcast-Distilation/stormcast/train.py"
config="--config-name diffusion"
experiment_name="diffusion_zettabyte_cleaned_4_27_2026"
training_output_dir="/data/exp_3_train_2_5_yrs_val_1yr_tp1/diffusion_zettabyte_v1_cleaned_4_27_2026"
run_id="0"

# --- Logging parameters ---
print_progress_freq=25
checkpoint_freq=1000
validation_freq=100
num_data_workers=4

# --- Training parameters (StormCast paper diffusion defaults) ---
batch_size=64                # global; with 4 GPUs -> 16 per GPU
lr=4E-4
lr_rampup_steps=1000
total_train_steps=700000      # StormCast pretrained EDMPrecond was at step 70k
clip_grad_norm=-1            # -1 = disable
loss='edm'
fp_optimizations='fp32'

# --- EDM hyperparameters (must match the sampler / any downstream distillation) ---
# P_mean / P_std define the log-normal sigma sampling; sigma_data is the std
# of the standardised target. These are set in config/model/diffusion.yaml +
# config/sampler/edm_deterministic.yaml; we re-state them here so any future
# distillation run (consistency / progressive / flowcast) can copy these
# *exact* numbers off the teacher to avoid silent drift.
P_mean=-1.2
sigma_min=0.002
sigma_max=80.0
sigma_data=0.5
rho=7.0

# --- Validation parameters ---
validation_plot_variables="[t2m,u10,v10,qpepre]"

# --- Optional NetCDF outputs ---
output_nc="false"
output_nc_freq=5

# --- Dataset parameters ---
location="/workspace/downloads/zarr_exp3_L_24_H_24_train_2_5_years_full_cleaned_4_27_2026"
HighRes_img_size="[192,96]"
exp_train_zarrs="[stormcast_test_train]"
train_dates="[2019/08/01,2021/12/31]"
exp_valid_zarrs="[stormcast_test_valid]"
valid_dates="[2022/01/01,2022/12/31]"
kept_LowRes_channels="all"
# Match the channel order used by regression / consistency / flowcast scripts so
# the resulting EDM teacher is drop-in compatible with downstream distillation.
kept_HighRes_channels="[u10, v10, t2m, qpepre]"
qpepre_log1p="true"

# --- Model parameters ---
# Frozen regression mean F_xi: produces M_t which the diffusion model then
# learns to correct via residual denoising. Pulled from the in-progress
# regression run (step 8000 checkpoint).
regression_weights="/data/exp_3_train_2_5_yrs_val_1yr_tp1/regression_zettabyte_v1_cleaned_4_27_2026/regression_zettabyte_cleaned_4_27_2026/run_0/checkpoints_regression/StormCastUNet.0.8000.mdlus"
spatial_pos_embed="True"

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
    "++training.fp_optimizations=${fp_optimizations}" \
    "++training.output_nc=${output_nc}" \
    "++training.output_nc_freq=${output_nc_freq}" \
    "++training.validation_plot_variables=${validation_plot_variables}" \
    "++model.P_mean=${P_mean}" \
    "++model.regression_weights=${regression_weights}" \
    "++model.spatial_pos_embed=${spatial_pos_embed}" \
    "++sampler.args.sigma_min=${sigma_min}" \
    "++sampler.args.sigma_max=${sigma_max}" \
    "++sampler.args.rho=${rho}" \
    "++dataset.location=${location}" \
    "++dataset.HighRes_img_size=${HighRes_img_size}" \
    "++dataset.exp_train_zarrs=${exp_train_zarrs}" \
    "++dataset.train_dates=${train_dates}" \
    "++dataset.exp_valid_zarrs=${exp_valid_zarrs}" \
    "++dataset.valid_dates=${valid_dates}" \
    "++dataset.kept_LowRes_channels=${kept_LowRes_channels}" \
    "++dataset.kept_HighRes_channels=${kept_HighRes_channels}" \
    "++dataset.qpepre_log1p=${qpepre_log1p}"
