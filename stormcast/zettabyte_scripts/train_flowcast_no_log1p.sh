#!/bin/bash
# F2 in [ablation.md](ablation.md): `flowcast_cleaned_NO_log1p`.
# Same as [train_flowcast.sh](train_flowcast.sh) (the F1 FlowCast student
# baseline) except qpepre is kept in raw mm/h on disk instead of log1p(mm/h).
# Exactly one knob differs from F1 so the F1 vs F2 paired comparison
# isolates the log1p qpepre transform; pair with D2 to also measure the
# log1p × method interaction.
#
# Prerequisite: a sibling cleaned dataset whose qpepre channel is raw mm/h,
# with HighRes/stats/{means,stds}.npy recomputed on the raw values
# (on the 192x96 Taiwan crop the recomputed qpepre std is ~2 mm/h vs ~0.37
# for the log1p variant — see ablation.md §7 sanity check #2). Build it by
# running clean_zarr.py with --no-qpepre-log1p (the clip-min=0 numerical-
# noise fix stays on, only log1p is gated):
#   python data_preprocessing/clean_zarr/clean_zarr.py \
#       --no-qpepre-log1p \
#       --dst .../zarr_exp3_L_24_H_24_train_2_5_years_full_cleaned_4_27_2026_raw
#
# Pull that dataset onto the worker first (azcopy, see ../../zettabyte/zettabyte.md):
#   export SAS_URL="https://zbstore2026.blob.core.windows.net/g-019c8ca2-605d-7bb5-b98b-1c53fbdf2b7f?se=...sig=..."
#   SRC="${SAS_URL%%\?*}/desmond/dataset/zarr_exp3_L_24_H_24_train_2_5_years_full_cleaned_4_27_2026_raw?${SAS_URL#*\?}"
#   azcopy copy "$SRC" /workspace/downloads --recursive
#
# Dataset spatial shape: 192 (y) x 96 (x). qpepre is stored as raw mm/h;
# qpepre_log1p=false makes denormalize_state the identity on the qpepre
# channel, so validation plots and RMSE/MAE CSVs are already in mm/h.
# Note: RMSE in raw mm/h is NOT comparable to RMSE on the log1p-standardised
# F1 run — for F1 vs F2 use CSI / FSS / radial PSD on qpepre instead
# (ablation.md §5).

source $(conda info --base)/etc/profile.d/conda.sh
conda activate stormcast_env

export CUDA_VISIBLE_DEVICES=0,1,2,3

# --- Torchrun settings ---
number_of_nodes=1
gpus_per_node=4

# --- General training config ---
stormcast_train="/workspace/Stormcast-Distilation/stormcast/train_flowcast.py"
config="--config-name flowcast"
experiment_name="flowcast_cleaned_NO_log1p"
training_output_dir="/data/exp_3_train_2_5_yrs_val_1yr_tp1/flowcast_zettabyte_v1_cleaned_4_27_2026_NO_log1p"
run_id="0"

# --- Logging parameters ---
print_progress_freq=25
checkpoint_freq=2500
validation_freq=250
num_data_workers=4

# --- Training parameters (FlowCast paper defaults: AdamW, cosine w/ 1% warmup) ---
# Held identical to F1 so the only diff vs F1 is the log1p knob.
batch_size=96
lr=5E-4
weight_decay=1.0E-4
adam_betas="[0.9,0.999]"
lr_warmup_steps=4000       # ~1% of total_train_steps
min_lr_ratio=0.01          # min_lr = lr * min_lr_ratio
total_train_steps=400000
clip_grad_norm=1.0
loss='flowcast'
fp_optimizations='fp32'

# --- I-CFM objective parameters ---
sigma_path=0.01            # I-CFM probability-path std (FlowCast paper)
t_eps=1.0E-5               # clamp for t ~ U(eps, 1-eps)
ema_decay=0.999            # FlowCast paper default (fixed)

# --- Sampler (validation only) ---
valid_num_steps=10         # Euler steps at validation (paper default)
solver='euler'             # 'euler' or 'midpoint'

# --- Hybrid loss (per-channel β + log-PSD on qpepre), mirrors CD spec ---
# Order of channel_weights MUST match kept_HighRes_channels below: u10, v10, t2m, qpepre.
# IMPORTANT: held identical to F1 on purpose. In the no-log1p run qpepre is
# in standardised raw mm/h (stats recomputed; std comes back to ~1 after
# normalisation), so the 2x boost is now operating on a channel whose
# pre-normalisation dynamic range is ~5x wider than F1. This is part of the
# log1p ablation, not a confound: changing channel_weights here would muddy
# the F1 vs F2 comparison.
channel_weights="[1.0,1.0,1.0,2.0]"
spectral_channels="[qpepre]"          # channels to add radial log-PSD term
spectral_weight=0.1                   # α_spec

# --- Validation parameters ---
validation_plot_variables="[t2m,u10,v10,qpepre]"

# --- Optional outputs ---
output_nc="false"
output_nc_freq=5

# --- Dataset parameters ---
# Sibling cleaned store with qpepre in raw mm/h and stats recomputed on raw.
location="/workspace/downloads/zarr_exp3_L_24_H_24_train_2_5_years_full_cleaned_4_27_2026_raw"
HighRes_img_size="[192,96]"
exp_train_zarrs="[stormcast_test_train]"
train_dates="[2019/08/01,2021/12/31]"
exp_valid_zarrs="[stormcast_test_valid]"
valid_dates="[2022/01/01,2022/12/31]"
kept_LowRes_channels="all"
kept_HighRes_channels="[u10, v10, t2m, qpepre]"
# THE one-knob diff vs train_flowcast.sh: data on disk is raw mm/h, so the
# loader must NOT apply expm1 in denormalize_state. Pairing this flag with
# log1p data on disk (or vice versa) is silently miscalibrated.
qpepre_log1p="false"

# --- Model parameters ---
# Sibling regression trained on the same `_raw` zarr (see
# [train_regression_no_log1p.sh](train_regression_no_log1p.sh)). Using the
# log1p-trained R0 here would feed F_xi inputs from a different qpepre
# distribution than it was trained on AND produce M_t in log1p space while
# X_t lives in raw space — so the residual on qpepre would conflate the
# log1p ablation with a regression I/O mismatch. Use the `_raw` sibling
# instead, paired by step number with the R0_raw checkpoint that D2 uses
# (ablation.md §1: "the same step for every row" within each log1p group).
# Step 10000 matches the choice in train_diffusion_no_log1p.sh.
regression_weights="/data/exp_3_train_2_5_yrs_val_1yr_tp1/regression_zettabyte_v1_cleaned_4_27_2026_NO_log1p/regression_cleaned_NO_log1p/run_0/checkpoints_regression/StormCastUNet.0.10000.mdlus"
sigma_data=0.5             # std used to standardize the residual
time_scale=1000.0          # scales t ∈ [0,1] into SongUNet's positional-embed range
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
    "++training.sigma_path=${sigma_path}" \
    "++training.t_eps=${t_eps}" \
    "++training.ema_decay=${ema_decay}" \
    "++training.valid_num_steps=${valid_num_steps}" \
    "++training.solver=${solver}" \
    "++training.channel_weights=${channel_weights}" \
    "++training.spectral_channels=${spectral_channels}" \
    "++training.spectral_weight=${spectral_weight}" \
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
