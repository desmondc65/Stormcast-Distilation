#!/bin/bash
#SBATCH --job-name=cor_reg_mulnode  # slurm 工作名稱
#SBATCH --partition=normal              # 使用的分區
#SBATCH --account=MST111414             # 帳號名稱
#SBATCH --nodes=2                         # 節點數量
#SBATCH --ntasks-per-node=1               # 每個節點的任務數量
#SBATCH --cpus-per-task=12               # 每個任務使用的 CPU 核心數
#SBATCH --gpus-per-node=2                 # gpu數量
#SBATCH --time=48:00:00       # 工作的最大執行時間
#SBATCH --output=slurm_logs/%x/%j.out # 輸出檔案路徑
#SBATCH --error=slurm_logs/%x/%j.err  # 錯誤檔案路徑
#SBATCH --export=ALL

# --- 多節點環境設定 (Multi-Node Environment Setup) ---
# 需要 rank 0 節點的位址作為 rendezvous 端點。
# 我們在 Slurm 分配列表中找出第一個節點的主機名稱。
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

# 需要一個固定或可預測的連接埠。如果是使用大型叢集，
# 使用 $SLURM_JOB_ID 可以讓連接埠更具唯一性，或者在安全的情況下依賴固定連接埠。
# 為求簡單，這裡假設該連接埠空閒並使用固定連接埠。
MASTER_PORT=29500 

export MASTER_ADDR
export MASTER_PORT
export NPROC=$SLURM_GPUS_PER_NODE # 為了保持一致性，使用 SLURM_GPUS_PER_NODE

# node_rank 由 Slurm 的作業索引提供。
NODE_RANK=$SLURM_NODEID 

# 初始化 Conda（Slurm 腳本的標準方法）
source $(conda info --base)/etc/profile.d/conda.sh
conda activate stormcast_env

# --- 一般訓練配置 ---
stormcast_train="/work/jasjou71/code/stormcast-ncdr/stormcast/train.py"
config="--config-name regression.yaml"
experiment_name="regression_ncdr"
training_output_dir="/work/jasjou71/code/stormcast-ncdr/stormcast/nano5_output/test_1_month_regression"
run_id="0"

# -- 記錄（Logging）參數 ---
print_progress_freq=25
checkpoint_freq=1000
validation_freq=50

# --- 訓練參數 ---
batch_size=64
lr=4E-4
lr_rampup_steps=1000
total_train_steps=16000
clip_grad_norm=-1 # 梯度裁剪（Gradient Clipping）的閾值，設為 -1 以停用
loss='regression'
# seed=42 # 設定正整數種子以避免單 GPU 模式下的分佈式廣播問題

# --- 驗證參數 ---
validation_plot_variables="[t2m,u10,v10,qpepre]"

# --- 可選輸出 ---
# 當設為 true 時，驗證欄位的 NetCDF 檔案將被寫入至
# ${training_output_dir}/${experiment_name}/run_${run_id}/netcdf_outputs/<field>。
# 預設為 false。
output_nc="true"
# 每 X 次驗證輸出一次 NetCDF（例如：若設為 5，則僅當 validation_counter % 5 == 0 時輸出）。
# 預設為 1（每次驗證都輸出）。
output_nc_freq=5

# --- 資料集參數 ---
location="/work/jasjou71/data/test_1_month_data/stormcast_zarr/"
HighRes_img_size="[224,128]"
exp_train_zarrs="[train]" # 用於訓練的 Zarr 檔案
train_dates="[2022/01/01,2022/01/20]"
exp_valid_zarrs="[valid]" # 用於驗證的 Zarr 檔案
valid_dates="[2022/01/21,2022/01/31]"
kept_LowRes_channels="all"
kept_HighRes_channels="all"

# ------------------------------------------------------------------
# 使用 Slurm Rendezvous 後端透過 torchrun 執行訓練
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