#!/bin/bash
# FlowCast inference on the cleaned dataset (192x96 spatial, qpepre log1p).
#
# Pull the dataset onto the worker first (azcopy, see ../../zettabyte/zettabyte.md):
#   export SAS_URL="https://zbstore2026.blob.core.windows.net/g-019c8ca2-605d-7bb5-b98b-1c53fbdf2b7f?se=...sig=..."
#   SRC="${SAS_URL%%\?*}/desmond/dataset/zarr_exp3_L_24_H_24_train_2_5_years_full_cleaned_4_27_2026?${SAS_URL#*\?}"
#   azcopy copy "$SRC" /workspace/downloads --recursive
#
# qpepre is automatically returned in mm/h by the data loader's
# denormalize_state (clean_zarr stores log1p(mm/h) on disk; expm1 is applied
# inside denormalize_state when ``dataset.qpepre_log1p=true``, the default).

source $(conda info --base)/etc/profile.d/conda.sh
conda activate stormcast_env

export CUDA_VISIBLE_DEVICES=0

# --- Inference entry ---
inference_script="/workspace/Stormcast-Distilation/stormcast/inference_flowcast.py"
config="--config-name flowcast_inference"

# --- Output ---
training_output_dir="/data/exp_3_train_2_5_yrs_val_1yr_tp1/flowcast_zettabyte_v1_cleaned_4_27_2026"
experiment_name="flowcast_inference_cleaned_4_27_2026"
run_id="0"
rundir="${training_output_dir}/${experiment_name}/run_${run_id}"
mkdir -p "${rundir}"

# --- Models ---
# Frozen regression mean.
regression_checkpoint="/data/exp_3_train_2_5_yrs_val_1yr_tp1/regression_zettabyte_v1_cleaned_4_27_2026/regression_zettabyte_cleaned_4_27_2026/run_0/checkpoints_regression/StormCastUNet.0.16000.mdlus"

# FlowCast EMA shadow (NOT the raw student) -- saved by trainer_flowcast.py
# at <flowcast_train_rundir>/ema_state.pt.
flowcast_ema_path="/data/exp_3_train_2_5_yrs_val_1yr_tp1/flowcast_zettabyte_v1_cleaned_4_27_2026/flowcast_cleaned_4_27_2026/run_0/ema_state.pt"

# --- Inference window ---
initial_time="2022-07-01T00:00:00"   # must lie inside dataset.valid_dates
n_steps=24                            # autoregressive horizon (each step = 1 h)

# --- FlowCast sampler ---
flowcast_num_steps=10                # ODE steps (FlowCast paper default)
flowcast_solver="euler"              # 'euler' or 'midpoint'

# --- Plotting ---
plot_var_state="qpepre"
plot_var_background="t2m"

# --- Dataset (cleaned, 192x96, qpepre as log1p(mm/h)) ---
location="/workspace/downloads/zarr_exp3_L_24_H_24_train_2_5_years_full_cleaned_4_27_2026"
HighRes_img_size="[192,96]"
exp_train_zarrs="[stormcast_test_train]"
train_dates="[2019/08/01,2021/12/31]"
exp_valid_zarrs="[stormcast_test_valid]"
valid_dates="[2022/01/01,2022/12/31]"
kept_LowRes_channels="all"
kept_HighRes_channels="[u10, v10, t2m, qpepre]"
qpepre_log1p="true"

# --- Model parameters (must match the training run) ---
sigma_data=0.5
time_scale=1000.0
spatial_pos_embed="True"

# Single-process inference (no torchrun -- the FlowCast student is small).
python -u "${inference_script}" ${config} \
    "hydra.run.dir=${training_output_dir}" \
    "++inference.experiment_name=${experiment_name}" \
    "++inference.run_id=${run_id}" \
    "++inference.rundir=${rundir}" \
    "++inference.initial_time=${initial_time}" \
    "++inference.n_steps=${n_steps}" \
    "++inference.regression_checkpoint=${regression_checkpoint}" \
    "++inference.flowcast_ema_path=${flowcast_ema_path}" \
    "++inference.flowcast.num_steps=${flowcast_num_steps}" \
    "++inference.flowcast.solver=${flowcast_solver}" \
    "++inference.plot_var_state=${plot_var_state}" \
    "++inference.plot_var_background=${plot_var_background}" \
    "++dataset.location=${location}" \
    "++dataset.HighRes_img_size=${HighRes_img_size}" \
    "++dataset.exp_train_zarrs=${exp_train_zarrs}" \
    "++dataset.train_dates=${train_dates}" \
    "++dataset.exp_valid_zarrs=${exp_valid_zarrs}" \
    "++dataset.valid_dates=${valid_dates}" \
    "++dataset.kept_LowRes_channels=${kept_LowRes_channels}" \
    "++dataset.kept_HighRes_channels=${kept_HighRes_channels}" \
    "++dataset.qpepre_log1p=${qpepre_log1p}" \
    "++model.sigma_data=${sigma_data}" \
    "++model.time_scale=${time_scale}" \
    "++model.spatial_pos_embed=${spatial_pos_embed}"
