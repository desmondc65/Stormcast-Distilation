#!/bin/bash
#SBATCH --job-name=inference    # slurm 工作名稱
#SBATCH --partition=normal2     # 使用的分區
#SBATCH --account=MST111414     # 使用的帳號
#SBATCH --nodes=1               # 使用的節點數量
#SBATCH --ntasks-per-node=1     # 每個節點的任務數量
#SBATCH --cpus-per-task=12      # 每個任務使用的 CPU 核心數
#SBATCH --gpus-per-node=1       # 每個節點使用的 GPU 數量
#SBATCH --time=2:00:00          # 工作的最大執行時間
#SBATCH --output=stormcast_slurm_logs/%x/%j.out # 輸出log檔案位置，%x 代表工作名稱，%j 代表工作 ID
#SBATCH --error=stormcast_slurm_logs/%x/%j.err  # 輸出錯誤log檔案位置，%x 代表工作名稱，%j 代表工作 ID
#SBATCH --export=ALL

# =============================================================================
# StormCast NCDR 推論腳本
# 此腳本整合資料前處理與模型推論，一次完成完整的推論流程
# =============================================================================

# --- 環境設定 ---
export NPROC=$SLURM_GPUS_ON_NODE

# --- 環境設定 ---
# 初始化 Conda (Slurm 腳本的標準方法)

ml load miniconda3

source $(conda info --base)/etc/profile.d/conda.sh
conda activate stormcast_env


# =============================================================================
# 第一部分：資料前處理 (to_zarr.py)
# 將 GRIB + RWRF + QPEPRE 檔案轉換為 Zarr 格式
# =============================================================================

echo "=========================================="
echo "開始資料前處理流程"
echo "=========================================="

# --- 前處理腳本路徑 ---
to_zarr_script="/work/jasjou71/code/stormcast-ncdr/data_preprocessing/inference_data_preprocess/to_zarr.py"

# --- 輸入資料路徑 ---
# GRIB 檔案路徑資料夾 (低解析度全球模式資料，例如 EC-Pangu)
grib_folder="/work/jasjou71/data_ncdr/Global/dynamic_global_test"

# RWRF NetCDF 檔案路徑 (高解析度區域模式資料)
rwrf_path="/work/jasjou71/data_ncdr/RWRF/2025120300/wrfout_d02_2025-12-03_00:00:00"

# QPEPRE 降水文字檔路徑
qpepre_path="/work/jasjou71/data_ncdr/Rain/qpepre_202512030000-202512030100_1_h.txt"

# --- 前處理輸出路徑 ---
# Zarr 資料集輸出目錄
preprocessing_output="/work/jasjou71/data_ncdr/output_test"

# --- 時間戳記設定 ---
# 輸入資料的時間戳記 (格式：YYYY-MM-DDTHH:MM:SS)
# 這是 RWRF 和 QPEPRE 資料的實際時間戳記
input_timestamp="2025-12-03T00:00:00"

# GRIB 預報基準時間 (選用，格式：YYYY-MM-DDTHH:MM:SS)
# 當 GRIB 檔案以「基準時間-預報時數」格式命名時使用
# 例如：EC-pangu_2025120300-12.grb 表示基準時間 2025-12-03T00:00:00，預報 12 小時後
# 如果設定此參數，腳本會自動計算每個時間步所需的預報時數
# 留空則從檔名自動推斷完整時間戳記
grib_base_time=""  # 例如："2025-12-03T00:00:00"


# --- 區域範圍設定 ---
# 網格大小 [高度, 寬度] (緯度, 經度方向的格點數)
domain_height=224
domain_width=128

# 經度範圍 [最小, 最大]
lon_min=119.75
lon_max=122.25

# 緯度範圍 [最小, 最大]
lat_min=21.6
lat_max=25.6

# --- 推論參數設定 ---
# 預報步數 (例如：6 表示預報 t+1 到 t+6)
n_steps=15

# 每步時間間隔 (小時)
dt_hours=1

# --- 變數設定 ---
# 低解析度變數 (必須與訓練資料一致)
lowres_vars="mslp,t2m,u10,v10,q1000,q850,q500,q250,t1000,t850,t500,t250,u1000,u850,u500,u250,v1000,v850,v500,v250,z1000,z850,z500,z250"

# 高解析度變數 (必須與訓練資料一致)
highres_vars="t2m,u10,v10,qpepre"

# 不變場變數
invariant_vars="lsm,orog"

# --- 前處理選項 ---
# 重新採樣模式 (interpolate: 雙線性內插)
resample_mode="interpolate"

# 是否覆寫現有檔案
overwrite_preprocessing="true"

# --- 執行資料前處理 ---
echo "執行 to_zarr.py..."
echo "輸入 GRIB 資料夾: ${grib_folder}"
echo "輸入 RWRF: ${rwrf_path}"
echo "輸入 QPEPRE: ${qpepre_path}"
echo "輸出目錄: ${preprocessing_output}"
echo ""

srun --mpi=pmix bash -lc "
torchrun --standalone --nnodes=${SLURM_JOB_NUM_NODES} --nproc_per_node=${NPROC} ${to_zarr_script} \
    --grib-folder "${grib_folder}" \
    --rwrf-path "${rwrf_path}" \
    --qpepre-path "${qpepre_path}" \
    --output "${preprocessing_output}" \
    --timestamp "${input_timestamp}" \
    --grib-base-time "${grib_base_time}" \
    --domain-size ${domain_height} ${domain_width} \
    --lon-bounds ${lon_min} ${lon_max} \
    --lat-bounds ${lat_min} ${lat_max} \
    --lowres-variables ${lowres_vars} \
    --highres-variables ${highres_vars} \
    --invariant-variables ${invariant_vars} \
    --resample-mode ${resample_mode} \
    --n-steps ${n_steps} \
    --dt-hours ${dt_hours} \
    $([ "${overwrite_preprocessing}" = "true" ] && echo "--overwrite")
"

# 檢查前處理是否成功
if [ $? -ne 0 ]; then
    echo "錯誤：資料前處理失敗"
    exit 1
fi

echo ""
echo "資料前處理完成"
echo ""

# =============================================================================
# 第二部分：模型推論 (inference_ncdr.py)
# 使用預訓練模型進行多步驟自回歸預報
# =============================================================================

echo "=========================================="
echo "開始模型推論流程"
echo "=========================================="

# --- 推論腳本路徑 ---
inference_script="/work/jasjou71/code/stormcast-ncdr/stormcast/inference_ncdr.py"


# 輸出資料夾路徑
FINAL_OUTPUT_DIR="/work/jasjou71/data_ncdr/n_16_inference_output"
# 確保該資料夾存在
mkdir -p "${FINAL_OUTPUT_DIR}"

# --- 模型 Checkpoint 路徑 ---
# 迴歸模型權重檔案路徑
regression_checkpoint="/work/jasjou71/data_ncdr/6hr_checkpoints/StormCastUNet.0.3000.mdlus"

# 擴散模型權重檔案路徑
diffusion_checkpoint="/work/jasjou71/data_ncdr/6hr_checkpoints/EDMPrecond.0.15000.mdlus"

# # --- 推論參數設定 ---
# # 預報步數 (例如：6 表示預報 t+1 到 t+6)
# n_steps=9

# # 每步時間間隔 (小時)
# dt_hours=1

# --- 資料集參數 ---
# Zarr 資料集位置 (使用前處理輸出目錄)
dataset_location="${preprocessing_output}"

# 網格大小 (必須與訓練資料一致)
HighRes_img_size="[${domain_height},${domain_width}]"

# 保留的變數通道 (使用 all 表示全部保留，或指定特定通道列表)
kept_LowRes_channels="all"
kept_HighRes_channels="all"

# 不變場變數列表
invariants="[lsm,orog]"

# --- Sampler 參數 ---
# 擴散模型採樣器設定 (EDM deterministic sampler)
sampler_name="edm_deterministic"
num_steps=18  # 擴散步數
sigma_min=0.002
sigma_max=80.0
rho=7.0
S_churn=0.0
S_min=0.0
S_max=.inf
S_noise=1.0

# --- 執行模型推論 ---
echo "執行 inference_ncdr.py..."
echo "資料集位置: ${dataset_location}"
echo "迴歸模型: ${regression_checkpoint}"
echo "擴散模型: ${diffusion_checkpoint}"
echo "預報步數: ${n_steps}"
echo "輸出目錄: ${FINAL_OUTPUT_DIR}"
echo ""

srun --mpi=pmix bash -lc "
torchrun --standalone --nnodes=${SLURM_JOB_NUM_NODES} --nproc_per_node=${NPROC} ${inference_script} \
    ++inference.rundir="${FINAL_OUTPUT_DIR}" \
    ++inference.regression_checkpoint="${regression_checkpoint}" \
    ++inference.diffusion_checkpoint="${diffusion_checkpoint}" \
    ++inference.n_steps=${n_steps} \
    ++inference.dt_hours=${dt_hours} \
    ++dataset.location="${dataset_location}" \
    ++dataset.HighRes_img_size="${HighRes_img_size}" \
    ++dataset.kept_LowRes_channels="${kept_LowRes_channels}" \
    ++dataset.kept_HighRes_channels="${kept_HighRes_channels}" \
    ++dataset.invariants="${invariants}" \
    ++sampler.name="${sampler_name}" \
    ++sampler.num_steps=${num_steps} \
    ++sampler.sigma_min=${sigma_min} \
    ++sampler.sigma_max=${sigma_max} \
    ++sampler.rho=${rho} \
    ++sampler.S_churn=${S_churn} \
    ++sampler.S_min=${S_min} \
    ++sampler.S_max=${S_max} \
    ++sampler.S_noise=${S_noise}
"

# 檢查推論是否成功
if [ $? -ne 0 ]; then
    echo "錯誤：模型推論失敗"
    exit 1
fi

echo ""
echo "模型推論完成"
echo ""

# =============================================================================
# 第三部分：結果視覺化 (選用)
# =============================================================================

# 是否執行視覺化 (設為 true 以啟用)
enable_visualization="true"

if [ "${enable_visualization}" = "true" ]; then
    echo "=========================================="
    echo "開始結果視覺化"
    echo "=========================================="
    
    # 視覺化腳本路徑
    plotting_script="/work/jasjou71/code/stormcast-ncdr/data_preprocessing/inference_data_preprocess/prediction_plotting.py"
    
    # 推論輸出目錄 (包含 NetCDF 檔案)
    inference_results_dir="${FINAL_OUTPUT_DIR}/NetCDF"
    
    # 圖片儲存目錄
    plot_save_dir="${FINAL_OUTPUT_DIR}/plots"
    
    # 是否包含輸入資料
    include_input="--include_input"
    
    echo "執行 prediction_plotting.py..."
    echo "推論結果目錄: ${inference_results_dir}"
    echo "圖片儲存目錄: ${plot_save_dir}"
    echo ""
    
srun --mpi=pmix bash -lc "
torchrun --standalone --nnodes=${SLURM_JOB_NUM_NODES} --nproc_per_node=${NPROC} ${plotting_script} \
        --output_dir "${inference_results_dir}" \
        --save_dir "${plot_save_dir}" \
        ${include_input}
"

    if [ $? -eq 0 ]; then
        echo "結果視覺化完成"
        echo "圖片已儲存至: ${plot_save_dir}"
    else
        echo "警告：結果視覺化失敗，但推論已完成"
    fi
    
    echo ""
fi

# =============================================================================
# 完成
# =============================================================================

echo "=========================================="
echo "推論流程全部完成"
echo "=========================================="
echo "前處理輸出: ${preprocessing_output}"
echo "推論結果輸出: ${FINAL_OUTPUT_DIR}"
echo ""
echo "輸出檔案包含："
echo "  - step_XX_YYYYMMDDHH.zarr  (Zarr 格式)"
echo "  - step_XX_YYYYMMDDHH.nc    (NetCDF 格式)"
echo ""
echo "變數："
echo "  - t2m: 2公尺溫度 (K)"
echo "  - u10: 10公尺U風 (m/s)"
echo "  - v10: 10公尺V風 (m/s)"
echo "  - qpepre: 降水量 (mm/hr)"
echo "=========================================="
