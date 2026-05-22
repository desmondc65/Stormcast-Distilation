#!/bin/bash
# Train the anchored Stochastic-Interpolant Bridge student on zettabyte cloud.
# Direction 1 of flowcast_improvement.md. Same dataset / regression
# checkpoint as the FlowCast zettabyte launcher so the two runs are directly
# comparable at fixed sample budget.
#
# Reframes the residual head as a stochastic interpolant from the regression
# mean mu_{t+1} to the truth M_{t+1}: the integrator starts at
# x_0 = mu + sigma_prior * eps (NOT at white noise) and integrates to M_{t+1}
# directly. The regression network is the structural prior endpoint of the
# flow, not just a conditioning channel + target shift.
#
# Pull the dataset onto the worker first (azcopy, see ../../zettabyte/zettabyte.md):
#   export SAS_URL="https://zbstore2026.blob.core.windows.net/...sig=..."
#   SRC="${SAS_URL%%\?*}/desmond/dataset/zarr_exp3_L_24_H_24_train_2_5_years_full_cleaned_4_27_2026?${SAS_URL#*\?}"
#   azcopy copy "$SRC" /workspace/downloads --recursive
#
# Dataset spatial shape: 192 (y) x 96 (x). qpepre is stored as log1p(mm/h);
# the loader's denormalize_state auto-applies expm1 for downstream metrics.

source $(conda info --base)/etc/profile.d/conda.sh
conda activate stormcast_env

export CUDA_VISIBLE_DEVICES=0,1,2,3

# --- Torchrun settings ---
number_of_nodes=1
gpus_per_node=4

# --- General training config ---
stormcast_train="/workspace/Stormcast-Distilation/stormcast/train_bridge.py"
config="--config-name bridge"
experiment_name="bridge_zettabyte_cleaned_4_27_2026"
training_output_dir="/data/exp_3_train_2_5_yrs_val_1yr_tp1/bridge_zettabyte_v1_cleaned_4_27_2026"
run_id="0"

# --- Logging parameters ---
print_progress_freq=25
checkpoint_freq=2500
validation_freq=250
num_data_workers=4

# --- Training parameters (FlowCast-style defaults: AdamW, cosine w/ 1% warmup) ---
batch_size=96
lr=5E-4
weight_decay=1.0E-4
adam_betas="[0.9,0.999]"
lr_warmup_steps=4000       # ~1% of total_train_steps
min_lr_ratio=0.01
total_train_steps=400000
clip_grad_norm=1.0
loss='bridge'
fp_optimizations='fp32'

# --- Bridge objective parameters ---
# sigma_prior  std of the Gaussian perturbation on the prior endpoint
#              (x_0 = mu + sigma_prior * eps). Small keeps the bridge short.
# sigma_max    peak of the bridge interior noise term gamma(t)*z. Set to 0
#              to ablate the stochastic interpolant (collapses to
#              deterministic residual head).
# schedule     'quadratic' (gamma = sigma_max*t*(1-t)) or 'trig' (sin(pi t)).
# coupling     'iid' or 'ot' (minibatch OT-CFM coupling, Direction 3).
sigma_prior=0.05
sigma_max=0.5
schedule='quadratic'
coupling='iid'
t_eps=1.0E-5
ema_decay=0.999

# --- Sampler (validation only) ---
valid_num_steps=10
solver='euler'

# --- Hybrid loss (per-channel β + log-PSD on qpepre) ---
# Channel order MUST match kept_HighRes_channels below: u10, v10, t2m, qpepre.
channel_weights="[1.0,1.0,1.0,2.0]"
spectral_channels="[qpepre]"
spectral_weight=0.1
# Per-channel prior noise std multipliers. Optional (Direction 2 ablation).
# Boosting the qpepre slot pushes the prior toward a heavier-tail
# approximation of the precipitation residual. Set to [1,1,1,1] for parity
# with the FlowCast baseline.
prior_channel_std="[1.0,1.0,1.0,1.0]"

# --- Validation parameters ---
validation_plot_variables="[t2m,u10,v10,qpepre]"

# --- Optional outputs ---
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
kept_HighRes_channels="[u10, v10, t2m, qpepre]"
qpepre_log1p="true"

# --- Model parameters ---
# Bridge backbone is the same SongUNet as FlowCast / EDM. The frozen
# regression net provides mu_{t+1} as BOTH the prior endpoint of the bridge
# AND a conditioning channel -- the bridge needs it twice over.
regression_weights="/data/exp_3_train_2_5_yrs_val_1yr_tp1/regression_zettabyte_v1_cleaned_4_27_2026/regression_zettabyte_cleaned_4_27_2026/run_0/checkpoints_regression/StormCastUNet.0.8000.mdlus"
sigma_data=0.5             # kept for parity; BridgeMatchingLoss itself operates in raw space
time_scale=1000.0
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
    "++training.weight_decay=${weight_decay}" \
    "++training.adam_betas=${adam_betas}" \
    "++training.lr_warmup_steps=${lr_warmup_steps}" \
    "++training.min_lr_ratio=${min_lr_ratio}" \
    "++training.total_train_steps=${total_train_steps}" \
    "++training.clip_grad_norm=${clip_grad_norm}" \
    "++training.loss=${loss}" \
    "++training.fp_optimizations=${fp_optimizations}" \
    "++training.sigma_prior=${sigma_prior}" \
    "++training.sigma_max=${sigma_max}" \
    "++training.schedule=${schedule}" \
    "++training.coupling=${coupling}" \
    "++training.t_eps=${t_eps}" \
    "++training.ema_decay=${ema_decay}" \
    "++training.valid_num_steps=${valid_num_steps}" \
    "++training.solver=${solver}" \
    "++training.channel_weights=${channel_weights}" \
    "++training.spectral_channels=${spectral_channels}" \
    "++training.spectral_weight=${spectral_weight}" \
    "++training.prior_channel_std=${prior_channel_std}" \
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
    "++dataset.qpepre_log1p=${qpepre_log1p}" \
    "++model.regression_weights=${regression_weights}" \
    "++model.sigma_data=${sigma_data}" \
    "++model.time_scale=${time_scale}" \
    "++model.spatial_pos_embed=${spatial_pos_embed}"
