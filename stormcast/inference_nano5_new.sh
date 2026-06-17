#!/bin/bash
#SBATCH --job-name=inference_new    # slurm 工作名稱
#SBATCH --partition=normal2         # 使用的分區
#SBATCH --account=MST111414         # 使用的帳號
#SBATCH --nodes=1                   # 使用的節點數量
#SBATCH --ntasks-per-node=1         # 每個節點的任務數量
#SBATCH --cpus-per-task=12          # 每個任務使用的 CPU 核心數
#SBATCH --gpus-per-node=1           # 每個節點使用的 GPU 數量
#SBATCH --time=3:00:00              # 工作的最大執行時間
#SBATCH --output=stormcast_slurm_logs/%x/%j.out
#SBATCH --error=stormcast_slurm_logs/%x/%j.err
#SBATCH --export=ALL

# =============================================================================
# StormCast / FlowCast NCDR 推論腳本 (新資料集格式)
#
# 使用 nc_to_zarr_pipeline 將 ERA5 NC + RWRF + QPEPRE 轉為訓練格式 Zarr，
# 再分別呼叫 inference_flowcast.py (FlowCast) 和 inference_ncdr.py (StormCast)
# 進行多步驟自回歸預報。
#
# 與舊版 inference_nano5.sh 的差異：
#   - 前處理：to_zarr.py (GRIB) → nc_to_zarr_pipeline.py (ERA5 NC)
#   - 輸出格式：InferenceDataset 單步 zarr → 訓練格式 zarr (valid.zarr)
#   - 新增 FlowCast 推論段落
# =============================================================================

export NPROC=$SLURM_GPUS_ON_NODE

ml load miniconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate stormcast_env

# =============================================================================
# 全域路徑設定
# =============================================================================

CODE_DIR="/work/jasjou71/code/stormcast-ncdr"

# =============================================================================
# 第一部分：資料前處理 (nc_to_zarr_pipeline.py)
# 將 ERA5 NC 小時檔 + RWRF + QPEPRE 轉換為訓練格式 Zarr
# =============================================================================

echo "=========================================="
echo "開始資料前處理流程 (nc_to_zarr_pipeline)"
echo "=========================================="

# --- 輸入資料路徑 ---
# ERA5 NC 小時檔資料夾 (每小時一個 NC 檔，或月份資料夾)
era5_path="/work/jasjou71/data_ncdr/ERA5_nc_hourly"

# RWRF NC 小時檔資料夾 (wrfout_d02_YYYY-MM-DD_HH 格式)
rwrf_path="/work/jasjou71/data_ncdr/RWRF"

# QPEPRE 降水文字檔資料夾 (qpepre_YYYYMMDDHHMM-..._1_h.txt 格式)
qpepre_path="/work/jasjou71/data_ncdr/obs_1hrRain"

# --- 輸出路徑 ---
# Zarr 資料集輸出目錄 (將產生 LowRes/valid.zarr, HighRes/valid.zarr, invariants/, 統計值)
preprocessing_output="/work/jasjou71/data_ncdr/inference_zarr_output"

# --- 推論時間範圍 ---
# 設定推論所需的起訖日期 (格式: YYYY/MM/DD)
# 只產生 valid 資料集；不需要訓練集
valid_start="2025/12/03"
valid_end="2025/12/04"

# --- 網域設定 (必須與訓練資料一致) ---
domain_height=224
domain_width=128
lon_min=119.75
lon_max=122.25
lat_min=21.6
lat_max=25.6

# --- 最大 CPU 並行數 ---
max_workers=20

echo "執行 nc_to_zarr_pipeline..."
echo "ERA5 路徑:   ${era5_path}"
echo "RWRF 路徑:   ${rwrf_path}"
echo "QPEPRE 路徑: ${qpepre_path}"
echo "輸出目錄:    ${preprocessing_output}"
echo "時間範圍:    ${valid_start} → ${valid_end}"
echo ""

python "${CODE_DIR}/data_preprocessing/nc_to_zarr/nc_to_zarr.py" \
    --era5-path    "${era5_path}" \
    --rwrf-path    "${rwrf_path}" \
    --qpepre-path  "${qpepre_path}" \
    --output-path  "${preprocessing_output}" \
    --valid-ranges "${valid_start},${valid_end}" \
    --split        valid \
    --domain-size  "${domain_height},${domain_width}" \
    --lon-bounds   "${lon_min},${lon_max}" \
    --lat-bounds   "${lat_min},${lat_max}" \
    --compute-stats \
    --overwrite \
    --max-workers  ${max_workers}  2>/dev/null || \
python "${CODE_DIR}/data_preprocessing/nc_to_zarr/nc_to_zarr.py" \
    --era5-path    "${era5_path}" \
    --rwrf-path    "${rwrf_path}" \
    --qpepre-path  "${qpepre_path}" \
    --output-path  "${preprocessing_output}" \
    --valid-ranges "${valid_start},${valid_end}" \
    --split        valid \
    --domain-size  "${domain_height},${domain_width}" \
    --lon-bounds   "${lon_min},${lon_max}" \
    --lat-bounds   "${lat_min},${lat_max}" \
    --compute-stats \
    --overwrite

if [ $? -ne 0 ]; then
    echo "錯誤：資料前處理失敗"
    exit 1
fi

echo ""
echo "資料前處理完成 → ${preprocessing_output}"
echo ""

# =============================================================================
# 共用推論參數
# =============================================================================

# 自回歸起始時間點 (必須在 valid_start ~ valid_end 之間，且存在於 zarr 中)
initial_time="2025-12-03T00:00:00"

# 預報步數與時間間隔
n_steps=15
dt_hours=1

# 資料集位置 (使用前處理輸出)
dataset_location="${preprocessing_output}"

# 新格式 zarr 名稱 (nc_to_zarr_pipeline 以 valid-ranges 建立的 zarr 叫 "valid")
valid_zarr_name="valid"

# valid_dates 格式需與訓練資料集一致 (YYYY/MM/DD)
valid_dates="[${valid_start},${valid_end}]"

# 網格大小 (必須與訓練資料一致)
HighRes_img_size="[${domain_height},${domain_width}]"

# 是否執行視覺化
enable_visualization="true"

# =============================================================================
# 第二部分：StormCast 推論 (EDM diffusion)
# 使用 inference_ncdr.py 搭配 data_loader_rwrf_era5_stable_inference.Dataset
#
# 注意：inference_ncdr.py 預設使用 data_loader_inference.InferenceDataset。
#       若要使用新格式，inference_ncdr.py 的 dataset 呼叫需支援
#       data_loader_rwrf_era5_stable_inference.Dataset 介面。
# =============================================================================

enable_stormcast="true"  # 設為 "false" 以跳過

if [ "${enable_stormcast}" = "true" ]; then
    echo "=========================================="
    echo "開始 StormCast (EDM) 推論"
    echo "=========================================="

    stormcast_output="/work/jasjou71/data_ncdr/inference_output_stormcast"
    mkdir -p "${stormcast_output}"

    # StormCast 模型 checkpoint
    regression_checkpoint="/work/jasjou71/data_ncdr/checkpoints/StormCastUNet.0.3000.mdlus"
    diffusion_checkpoint="/work/jasjou71/data_ncdr/checkpoints/EDMPrecond.0.15000.mdlus"

    # EDM deterministic sampler 設定 (必須與訓練一致)
    sampler_name="edm_deterministic"
    num_sampler_steps=18
    sigma_min=0.002
    sigma_max=80.0
    rho=7.0
    S_churn=0.0
    S_min=0.0
    S_max=.inf
    S_noise=1.0

    echo "迴歸模型:    ${regression_checkpoint}"
    echo "擴散模型:    ${diffusion_checkpoint}"
    echo "輸出目錄:    ${stormcast_output}"
    echo ""

    srun --mpi=pmix bash -lc "
torchrun --standalone --nnodes=${SLURM_JOB_NUM_NODES} --nproc_per_node=${NPROC} \
    ${CODE_DIR}/stormcast/inference_ncdr.py \
    ++dataset.name=data_loader_rwrf_era5_stable_inference.Dataset \
    ++dataset.location='${dataset_location}' \
    ++dataset.HighRes_img_size='${HighRes_img_size}' \
    ++dataset.exp_valid_zarrs='[${valid_zarr_name}]' \
    ++dataset.valid_dates='${valid_dates}' \
    ++dataset.kept_LowRes_channels=all \
    ++dataset.kept_HighRes_channels=all \
    ++dataset.qpepre_log1p=true \
    ++inference.rundir='${stormcast_output}' \
    ++inference.regression_checkpoint='${regression_checkpoint}' \
    ++inference.diffusion_checkpoint='${diffusion_checkpoint}' \
    ++inference.initial_time='${initial_time}' \
    ++inference.n_steps=${n_steps} \
    ++inference.dt_hours=${dt_hours} \
    ++sampler.name='${sampler_name}' \
    ++sampler.num_steps=${num_sampler_steps} \
    ++sampler.sigma_min=${sigma_min} \
    ++sampler.sigma_max=${sigma_max} \
    ++sampler.rho=${rho} \
    ++sampler.S_churn=${S_churn} \
    ++sampler.S_min=${S_min} \
    ++sampler.S_max=${S_max} \
    ++sampler.S_noise=${S_noise}
"

    if [ $? -ne 0 ]; then
        echo "錯誤：StormCast 推論失敗"
        exit 1
    fi

    echo ""
    echo "StormCast 推論完成 → ${stormcast_output}"
    echo ""
fi

# =============================================================================
# 第三部分：FlowCast 推論 (Conditional Flow Matching)
# 使用 inference_flowcast.py 搭配 data_loader_rwrf_era5_stable.Dataset
# (inference_flowcast.py 原生支援訓練格式 zarr)
# =============================================================================

enable_flowcast="true"  # 設為 "false" 以跳過

if [ "${enable_flowcast}" = "true" ]; then
    echo "=========================================="
    echo "開始 FlowCast 推論"
    echo "=========================================="

    flowcast_output="/work/jasjou71/data_ncdr/inference_output_flowcast"
    mkdir -p "${flowcast_output}"

    # FlowCast 模型 checkpoint
    # regression_checkpoint: 凍結的迴歸模型 (StormCastUNet)
    flowcast_regression="/work/jasjou71/data_ncdr/checkpoints/StormCastUNet.0.3000.mdlus"
    # flowcast_ema_path: FlowCast 訓練輸出的 EMA 權重 (ema_state.pt)
    flowcast_ema="/work/jasjou71/data_ncdr/checkpoints/flowcast/ema_state.pt"

    # FlowCast ODE sampler 設定
    flowcast_num_steps=10   # Euler 步數 (FlowCast 論文預設)
    flowcast_solver="euler" # "euler" 或 "midpoint"

    echo "迴歸模型:       ${flowcast_regression}"
    echo "FlowCast EMA:   ${flowcast_ema}"
    echo "輸出目錄:       ${flowcast_output}"
    echo ""

    srun --mpi=pmix bash -lc "
torchrun --standalone --nnodes=${SLURM_JOB_NUM_NODES} --nproc_per_node=${NPROC} \
    ${CODE_DIR}/stormcast/inference_flowcast.py \
    ++dataset.location='${dataset_location}' \
    ++dataset.HighRes_img_size='${HighRes_img_size}' \
    ++dataset.exp_valid_zarrs='[${valid_zarr_name}]' \
    ++dataset.valid_dates='${valid_dates}' \
    ++dataset.kept_LowRes_channels=all \
    ++dataset.kept_HighRes_channels=all \
    ++dataset.qpepre_log1p=true \
    ++inference.rundir='${flowcast_output}' \
    ++inference.regression_checkpoint='${flowcast_regression}' \
    ++inference.flowcast_ema_path='${flowcast_ema}' \
    ++inference.initial_time='${initial_time}' \
    ++inference.n_steps=${n_steps} \
    ++inference.flowcast.num_steps=${flowcast_num_steps} \
    ++inference.flowcast.solver='${flowcast_solver}'
"

    if [ $? -ne 0 ]; then
        echo "錯誤：FlowCast 推論失敗"
        exit 1
    fi

    echo ""
    echo "FlowCast 推論完成 → ${flowcast_output}"
    echo ""
fi

# =============================================================================
# 第四部分：結果視覺化 (選用)
# =============================================================================

if [ "${enable_visualization}" = "true" ]; then
    echo "=========================================="
    echo "開始結果視覺化"
    echo "=========================================="

    plotting_script="${CODE_DIR}/data_preprocessing/inference_data_preprocess/prediction_plotting.py"

    if [ "${enable_stormcast}" = "true" ]; then
        echo "視覺化 StormCast 輸出..."
        python "${plotting_script}" \
            --output_dir "${stormcast_output}/NetCDF" \
            --save_dir   "${stormcast_output}/plots" \
            --include_input || echo "警告：StormCast 視覺化失敗"
    fi

    if [ "${enable_flowcast}" = "true" ]; then
        echo "視覺化 FlowCast 輸出..."
        python "${plotting_script}" \
            --output_dir "${flowcast_output}/NetCDF" \
            --save_dir   "${flowcast_output}/plots" \
            --include_input || echo "警告：FlowCast 視覺化失敗"
    fi
fi

# =============================================================================
# 完成
# =============================================================================

echo "=========================================="
echo "推論流程全部完成"
echo "=========================================="
echo "前處理輸出:       ${preprocessing_output}"
[ "${enable_stormcast}" = "true" ] && echo "StormCast 輸出:   ${stormcast_output}"
[ "${enable_flowcast}"  = "true" ] && echo "FlowCast 輸出:    ${flowcast_output}"
echo ""
echo "輸出變數："
echo "  t2m:    2公尺溫度 (K)"
echo "  u10:    10公尺U風 (m/s)"
echo "  v10:    10公尺V風 (m/s)"
echo "  qpepre: 降水量 (mm/hr)"
echo "=========================================="
