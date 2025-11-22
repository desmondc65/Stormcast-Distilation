#!/bin/bash
#SBATCH --job-name=corr_dif    # slurm 工作名稱
#SBATCH --partition=normal2    # 使用的分區
#SBATCH --account=MST111414    # 使用的帳號
#SBATCH --nodes=1              # 使用的節點數量
#SBATCH --ntasks-per-node=1    # 每個節點的任務數量
#SBATCH --cpus-per-task=12     # 每個任務使用的 CPU 核心數
#SBATCH --gpus-per-node=1      # 每個節點使用的 GPU 數量
#SBATCH --time=48:00:00        # 工作的最大執行時間 
#SBATCH --output=stormcast_slurm_logs/%x/%j.out # 輸出log檔案位置，%x 代表工作名稱，%j 代表工作 ID
#SBATCH --error=stormcast_slurm_logs/%x/%j.err  # 輸出錯誤log檔案位置，%x 代表工作名稱，%j 代表工作 ID
#SBATCH --export=ALL

# --- 環境設定 ---
export NPROC=$SLURM_GPUS_ON_NODE

# 初始化 Conda (Slurm 腳本的標準方法)
source $(conda info --base)/etc/profile.d/conda.sh
conda activate stormcast_env

# --- 一般訓練設定 ---
stormcast_train="/work/jasjou71/code/stormcast-ncdr/stormcast/train.py"
config="--config-name diffusion.yaml"
experiment_name="diffusion_ncdr"
training_output_dir="/work/jasjou71/code/stormcast-ncdr/stormcast/nano5_output/test_1_month_diffusion"
run_id="0"

# --- 日誌參數 ---
print_progress_freq=25
checkpoint_freq=1000
validation_freq=50

# --- 訓練參數 ---
batch_size=12
lr=4E-4
lr_rampup_steps=1000
total_train_steps=16000
clip_grad_norm=-1 # 梯度裁剪閾值，設為 -1 表示停用
loss='edm'
# seed=42 # 設定正數種子以避免單 GPU 模式下的分散式廣播問題

# --- 驗證參數 ---
validation_plot_variables="[t2m,u10,v10,qpepre]"

# --- 可選輸出 ---
# 當設為 true 時，驗證欄位的 NetCDF 檔案將寫入
# ${training_output_dir}/${experiment_name}/run_${run_id}/netcdf_outputs/<field>。
# 預設為 false。
output_nc="true"
# 每 X 次驗證輸出一次 NetCDF（例如，若設為 5，只在 validation_counter % 5 == 0 時輸出）。
# 預設為 1（每次驗證都輸出）。
output_nc_freq=5

# --- 資料集參數 ---
location="/work/jasjou71/data/test_1_month_data/stormcast_zarr/"
HighRes_img_size="[224,128]"
exp_train_zarrs="[train]" # 訓練用的 Zarr 檔案
train_dates="[2022/01/01,2022/01/20]"
exp_valid_zarrs="[valid]" # 驗證用的 Zarr 檔案
valid_dates="[2022/01/21,2022/01/31]"
kept_LowRes_channels="all"
kept_HighRes_channels="all"

# --- 模型參數 ---
regression_weights="/work/jasjou71/code/stormcast-ncdr/stormcast/nano5_output/test_1_month_regression/regression_ncdr/run_0/checkpoints_regression/StormCastUNet.0.1000.mdlus" 
# 預訓練回歸模型的路徑，若 'regression' 包含在 diffusion_conditions 中則使用

# 使用 torchrun 執行訓練 
srun --mpi=pmix bash -lc "
torchrun --standalone --nnodes=${SLURM_JOB_NUM_NODES} --nproc_per_node=${NPROC} ${stormcast_train} ${config} \
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
    ++dataset.kept_HighRes_channels=${kept_HighRes_channels} \
    ++model.regression_weights=${regression_weights}
"