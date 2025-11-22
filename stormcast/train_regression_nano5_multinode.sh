#!/bin/bash
#SBATCH --job-name=multinode_reg     # Changed for clarity
#SBATCH --partition=normal
#SBATCH --account=MST111414
#SBATCH --nodes=2                         # <<< CRITICAL: Set to 2 or more nodes for testing
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1                 # <<< CRITICAL: Assuming 1 GPU per node
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs_multinode/%x/%j.out
#SBATCH --error=slurm_logs_multinode/%x/%j.err
#SBATCH --export=ALL

# Load your environment (e.g., source activate stormcast_env or module load)
# source activate stormcast_env # UNCOMMENT if needed

# --- Multi-Node Environment Setup ---
# The rank 0 node address is needed for the rendezvous endpoint.
# We find the hostname of the first node in the Slurm allocation list.
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

# A fixed or predictable port is required. If using a large cluster,
# use $SLURM_JOB_ID to make the port more unique, or rely on a fixed port if safe.
# Using a fixed port for simplicity, assuming it's free.
MASTER_PORT=29500 

export MASTER_ADDR
export MASTER_PORT
export NPROC=$SLURM_GPUS_PER_NODE # Use SLURM_GPUS_PER_NODE for consistency

# The node_rank is supplied by Slurm's job index.
NODE_RANK=$SLURM_NODEID 

# --- General training config (Unchanged) ---
stormcast_train="/work/jasjou71/code/stormcast-ncdr/stormcast/train.py"
config="--config-name regression.yaml"
experiment_name="regression_ncdr"
# ... (all other parameters remain the same) ...

# ------------------------------------------------------------------
# execute training with torchrun using Slurm Rendezvous Backend
# ------------------------------------------------------------------
echo "Running distributed training across $SLURM_NNODES nodes, using $NPROC GPUs per node."
echo "Master Node: $MASTER_ADDR:$MASTER_PORT (Node Rank: $NODE_RANK)"

srun --mpi=pmix bash -lc "
torchrun \
    --nnodes ${SLURM_NNODES} \
    --nproc_per_node ${NPROC} \
    --rdzv_id ${SLURM_JOB_ID} \
    --rdzv_backend c10d \
    --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
    --node_rank ${NODE_RANK} \
    ${stormcast_train} ${config} \
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