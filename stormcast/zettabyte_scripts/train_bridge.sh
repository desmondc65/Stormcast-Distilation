#!/bin/bash
# Train the StormCast anchored Stochastic-Interpolant Bridge student
# (Direction 1 of flowcast_improvement.md) on zettabyte cloud, on top of the
# cleaned dataset and the regression checkpoint trained at step 8000.
#
# Pull the dataset onto the worker first (azcopy, see ../../zettabyte/zettabyte.md):
#   export SAS_URL="https://zbstore2026.blob.core.windows.net/g-019c8ca2-605d-7bb5-b98b-1c53fbdf2b7f?se=...sig=..."
#   SRC="${SAS_URL%%\?*}/desmond/dataset/zarr_exp3_L_24_H_24_train_2_5_years_full_cleaned_4_27_2026?${SAS_URL#*\?}"
#   azcopy copy "$SRC" /workspace/downloads --recursive
#
# Dataset spatial shape: 192 (y) x 96 (x). qpepre is stored as log1p(mm/h);
# the loader's denormalize_state auto-applies expm1 for downstream metrics.
# Note that *during training* qpepre is in standardised log1p space, which is
# exactly what makes the per-channel MSE / spectral terms well-conditioned
# without any extra weighting beyond ``channel_weights``.
#
# Reframes the residual head as a stochastic interpolant from the regression
# mean mu_{t+1} to the truth M_{t+1}: the integrator starts at
# x_0 = mu + sigma_prior * eps (NOT at white noise) and integrates to M_{t+1}
# directly. The regression network is the structural prior endpoint of the
# flow, not just a conditioning channel + target shift. Same dataset and
# regression checkpoint as train_flowcast.sh so the two are directly
# comparable at fixed sample budget.

source $(conda info --base)/etc/profile.d/conda.sh
conda activate stormcast_env

export CUDA_VISIBLE_DEVICES=0,1,2,3

# --- Log capture + auto-commit ---
# Tee training output to a git-tracked log inside the repo, then on ANY exit
# (success, non-zero, Ctrl-C, SIGTERM, torchrun crash) commit & push just the
# log file. Only the log is staged -- other dirty files in the worktree are
# left alone. Every git step is allowed to fail independently so a broken
# remote / hook / auth doesn't lose the local commit.
repo_root="/workspace/Stormcast-Distilation"
log_file="${repo_root}/zettabyte/bridge_train.log"
mkdir -p "$(dirname "${log_file}")"

commit_and_push_log() {
    local status=$?
    # Disarm immediately so a failure inside this handler can't re-enter it.
    trap - EXIT INT TERM
    echo "[train_bridge] training exited with status ${status}; committing log"

    if [[ ! -f "${log_file}" ]]; then
        echo "[train_bridge] log file ${log_file} missing -- nothing to commit"
        exit "${status}"
    fi
    if [[ ! -d "${repo_root}/.git" ]]; then
        echo "[train_bridge] ${repo_root} is not a git repo -- skipping commit"
        exit "${status}"
    fi

    (
        cd "${repo_root}" || exit 0
        rel="${log_file#${repo_root}/}"
        # -f because zettabyte/* and *.log are both gitignored.
        git add -f -- "${rel}" || echo "[train_bridge] git add failed (continuing)"
        if git diff --cached --quiet -- "${rel}"; then
            echo "[train_bridge] no log changes staged -- skipping commit"
        else
            local outcome
            if [[ "${status}" == "0" ]]; then
                outcome="training ok"
            else
                outcome="training exit ${status}"
            fi
            local msg="chore(log): bridge_train.log @ $(date -u '+%Y-%m-%dT%H:%M:%SZ') (${outcome})"
            if git commit -m "${msg}" -- "${rel}"; then
                git push || echo "[train_bridge] git push failed -- log is committed locally, push it manually later"
            else
                echo "[train_bridge] git commit failed -- log is staged, commit it manually later"
            fi
        fi
    )
    exit "${status}"
}
trap commit_and_push_log EXIT INT TERM

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

# --- Training parameters (FlowCast paper defaults: AdamW, cosine w/ 1% warmup) ---
batch_size=96
lr=5E-4
weight_decay=1.0E-4
adam_betas="[0.9,0.999]"
lr_warmup_steps=4000       # ~1% of total_train_steps
min_lr_ratio=0.01          # min_lr = lr * min_lr_ratio
total_train_steps=400000
clip_grad_norm=1.0
loss='bridge'
fp_optimizations='fp32'

# --- Bridge objective parameters ---
# Replaces I-CFM's sigma_path with a stochastic-interpolant family
# (Albergo & Vanden-Eijnden 2023). x_t = alpha(t) x_0 + beta(t) x_1 + gamma(t) z
# with x_0 = mu + sigma_prior * eps and x_1 = M.
sigma_prior=0.05           # std of the additive Gaussian on the prior endpoint
sigma_max=0.5              # peak of the interior noise term gamma(t)*z (0 disables)
schedule='quadratic'       # 'quadratic' (t*(1-t)) or 'trig' (sin(pi t))
coupling='iid'             # 'iid' or 'ot' (minibatch OT-CFM, Direction 3)
t_eps=1.0E-5               # clamp for t ~ U(eps, 1-eps)
ema_decay=0.999            # FlowCast paper default (fixed)

# --- Sampler (validation only) ---
valid_num_steps=10         # Euler steps at validation (paper default)
solver='euler'             # 'euler' or 'midpoint'

# --- Hybrid loss (per-channel β + log-PSD on qpepre), mirrors CD spec ---
# Order of channel_weights MUST match kept_HighRes_channels below: u10, v10, t2m, qpepre.
# qpepre is now in log1p space (std 0.37, in the same regime as the wind
# channels), so the 2x boost just nudges the spectral / loss weight without
# the heavy-tail balancing the raw mm/h channel needed.
channel_weights="[1.0,1.0,1.0,2.0]"
spectral_channels="[qpepre]"          # channels to add radial log-PSD term
spectral_weight=0.1                   # α_spec
# Per-channel prior noise std multipliers (Direction 2 ablation). Set to
# [1,1,1,1] for parity with the FlowCast baseline; bump the qpepre slot for
# the heavier-tail prior experiment.
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
# Match the channel order used by regression / consistency / diffusion / flowcast
# scripts so the resulting Bridge student is drop-in compatible with the rest
# of the run.
kept_HighRes_channels="[u10, v10, t2m, qpepre]"
qpepre_log1p="true"

# --- Model parameters ---
# Bridge is a one-stage flow-matching net learning to integrate from the
# frozen regression mean mu_{t+1} to M_{t+1} directly. The regression net
# is used BOTH as the prior endpoint AND as a conditioning channel.
# No diffusion teacher is used (unlike CD / PD).
regression_weights="/data/exp_3_train_2_5_yrs_val_1yr_tp1/regression_zettabyte_v1_cleaned_4_27_2026/regression_zettabyte_cleaned_4_27_2026/run_0/checkpoints_regression/StormCastUNet.0.8000.mdlus"
sigma_data=0.5             # kept for parity; BridgeMatchingLoss operates in raw space
time_scale=1000.0          # scales t ∈ [0,1] into SongUNet's positional-embed range
spatial_pos_embed="True"

# Execute training with torchrun. pipefail so tee doesn't mask a torchrun
# failure -- the EXIT trap reads $? to record the real status in the commit.
set -o pipefail
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
    "++model.spatial_pos_embed=${spatial_pos_embed}" 2>&1 | tee "${log_file}"
