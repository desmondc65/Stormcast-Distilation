這是一份完整的 StormCast NCDR 推論流程操作指南。本指南整合了**環境建置**、**資料準備規範**以及**腳本參數設定**三個關鍵部分。

請務必確認資料檔案名稱符合規範，否則程式可能無法正確讀取時間戳記。

-----

# StormCast NCDR 推論流程完整操作指南

## 第一部分：環境建置與套件安裝

在首次執行推論前，請依序執行以下指令建立運算環境。

**1. 建立並啟動 Conda 虛擬環境**
建立 Python 3.10 環境並啟動：

```bash
conda create -n stormcast_env python=3.10 -y
conda activate stormcast_env
```

**2. 確認 CUDA 版本並安裝 PyTorch**
請先確認您的 CUDA 版本，再安裝對應的 PyTorch。以下範例適用於 **CUDA 11.8**：

```bash
# 檢查 CUDA 版本
nvidia-smi

# 下載並安裝 PyTorch (範例版本: 2.7.1 / CUDA 11.8)
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
```

**3. 安裝專案相依套件**
請切換至專案根目錄 (`stormcast-ncdr`) 並以編輯模式安裝：

```bash
cd stormcast-ncdr
pip install -e .
```

-----

## 第二部分：資料準備與命名規範 (重要)

前處理腳本 (`to_zarr.py`) 會透過正規表示式 (Regex) 自動從檔名解析時間。請確保您的輸入資料符合以下命名規則，以避免讀取錯誤。

### 1\. 低解析度 GRIB 資料 (Low-Resolution GRIB)

  * **用途**：提供全球模式背景場 (如 EC-Pangu)。
  * **命名規則**：檔名須包含 `YYYYMMDDHH` (年4位+月2位+日2位+時2位)，後接分隔符號 (`-`, `.`, `_`, `/`)。
  * **正確範例**：
      * `EC-pangu_2025120300.grb` (推薦)
      * `data_2025120300.grib`

### 2\. 高解析度 RWRF 資料 (High-Resolution RWRF)

  * **用途**：提供初始時刻的高解析度氣象場。
  * **命名規則**：需完全符合 WRF 預設輸出格式 `YYYY-MM-DD_HH:MM:SS`。
  * **正確範例**：
      * `wrfout_d02_2025-12-03_00:00:00`
  * **內部變數要求**：NetCDF 內部變數需包含標準 WRF 變數名 (如 `U10`, `V10`, `T2`, `PSFC`, `PH`, `PHB`)。

### 3\. 雷達降雨資料 (QPEPRE)

  * **用途**：提供降水觀測資料。
  * **命名規則**：檔名**必須**以 `qpepre_` 開頭，後接 `YYYYMMDDHHMM` (至分鐘)。
  * **正確範例**：
      * `qpepre_202512030000-202512030100_1_h.txt`
      * `qpepre_202512030000.txt`

-----

## 第三部分：推論腳本參數設定

請開啟您的 SLURM 腳本 (如 `run_inference.sh`)，並根據當次任務修改以下變數。

### 1\. 程式碼路徑設定

請填入各 Python 腳本的**絕對路徑**：

  * **`to_zarr_script`**: 指向 `data_preprocessing/inference_data_preprocess/to_zarr.py`
  * **`inference_script`**: 指向 `stormcast/inference_ncdr.py`
  * **`plotting_script`**: 指向 `data_preprocessing/inference_data_preprocess/prediction_plotting.py`

### 2\. 資料來源與輸出路徑

請填入實際檔案存放位置：

  * **`grib_folder`**: GRIB 檔案所在的**資料夾路徑**。
  * **`rwrf_path`**: RWRF (NetCDF) 的**完整檔案路徑**。
  * **`qpepre_path`**: QPEPRE (txt) 的**完整檔案路徑**。
  * **`preprocessing_output`**: 前處理產生的 Zarr 中介檔案輸出目錄。
  * **`FINAL_OUTPUT_DIR`**: 最終推論結果 (NetCDF) 與視覺化圖檔的輸出目錄。

### 3\. 時間與空間範圍

  * **`input_timestamp`**: 預報起始時間 (格式：`YYYY-MM-DDTHH:MM:SS`，例如 `2025-12-03T00:00:00`)。
  * **`domain_height` / `domain_width`**: 網格點數 (例如 `224` / `128`)。
  * **`lon_min` / `lon_max`**: 經度範圍 (例如 `119.75` / `122.25`)。
  * **`lat_min` / `lat_max`**: 緯度範圍 (例如 `21.6` / `25.6`)。

### 4\. 模型與推論參數

  * **`n_steps`**: 預報總步數 (例如 `15` 代表預測未來 15 小時)。
  * **`regression_checkpoint`**: 迴歸模型權重檔 (`.mdlus`) 路徑。
  * **`diffusion_checkpoint`**: 擴散模型權重檔 (`.mdlus`) 路徑。
  * **`lowres_vars` / `highres_vars`**: 變數列表 (務必與模型訓練時的設定一致)。

-----

## 第四部分：執行作業

完成上述設定後，請使用 SLURM 指令提交作業：

```bash
sbatch 您的腳本名稱.sh
```

**檢查重點：**

1.  輸出的 Log 檔 (`.out`) 中是否正確顯示 "開始資料前處理流程"。
2.  確認 Log 中是否印出正確的變數數量 (例如 "Extracted 24 variables from GRIB")，若數量過少通常代表檔名解析時間錯誤或路徑錯誤。
3.  最終結果將包含 `.nc` (NetCDF) 檔案與 `.png` (視覺化圖檔)。