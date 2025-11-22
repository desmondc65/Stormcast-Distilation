#!/bin/bash
#SBATCH --job-name=test_1_month_reg   # Changed for clarity
#SBATCH --partition=normal
#SBATCH --account=MST111414
#SBATCH --nodes=1                         # Already set to 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1                 # MODIFIED: Changed from 8 to 4
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x/%j.out # Changed for clarity
#SBATCH --error=slurm_logs/%x/%j.err  # Changed for clarity
#SBATCH --export=ALL

# --- Environment setup ---
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=${MASTER_PORT:-$(( 20000 + (RANDOM % 20000) ))}
export MASTER_ADDR MASTER_PORT


# --- General training config ---
stormcast_train="/work/jasjou71/code/stormcast-ncdr/stormcast/train.py"
config="--config-name regression.yaml"
experiment_name="regression_ncdr"
training_output_dir="/work/jasjou71/code/stormcast-ncdr/stormcast/nano5_output/test_1_month_regression"
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
output_nc="true"
# Output NetCDF every X validations (e.g., if set to 5, only output when validation_counter % 5 == 0).
# Default is 1 (output every validation).
output_nc_freq=5

# --- Dataset parameters ---
location="/work/jasjou71/data/test_1_month_data/stormcast_zarr/"
HighRes_img_size="[224,128]"
exp_train_zarrs="[stormcast_test_train]" # Zarr files to use for training
train_dates="[2022/01/01,2022/01/20]"
exp_valid_zarrs="[stormcast_test_valid]" # Zarr files to use for validation
valid_dates="[2022/01/21,2022/01/31]"
kept_LowRes_channels="all"
kept_HighRes_channels="all"

# execute training with torchrun 
srun --mpi=pmix bash -lc "
torchrun --standalone --nnodes=${SLURM_JOB_NUM_NODES} --nproc_per_node=${NPROC} ${stormcast_train} ${config} \
    --node_rank=\${SLURM_PROCID} \
    hydra.run.dir=${training_output_dir} --rdzv_backend=c10d  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
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
"