# Stormcast-NCDR (客製化訓練資料)

## 虛擬環境設置
1. 使用conda 建立虛擬環境，並安裝所需套件：
```bash
conda create -n stormcast_env python=3.10 -y
# 啟動虛擬環境
conda activate stormcast_env
```

2. 查詢cuda版本，並安裝對應的pytorch版本與相關套件：
```bash
# 查詢cuda版本
nvidia-smi
```

3. 請到 [pytorch官網](https://pytorch.org/get-started/previous-versions/) 查詢適合的pytorch版本
```bash
# 以cuda 11.8為例，安裝指令如下
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
```

4. 安裝其他所需套件：
```bash
# 確保在專案根目錄
cd stormcast-ncdr
pip install -e .
```
---

## 資料預處理
### 資料放置
Stormcast 分成高解析和低解析的影像資料，請依照以下步驟進行資料預處理：
1. **高解析資料小時nc檔 (RWRF)**
    - 請把檔案都放在**同一個主資料夾裡**
    - 檔名**必須**為此格式：wrfout_d01_YYYY-MM-DD_HH_interp , YYYY為年份，MM為月份，DD為日期，HH為小時
2. **高解析資料小時txt檔（QPEPRE）**
    - 請把檔案都放在**同一個主資料夾裡**
    - 檔名**必須**為此格式：qpepre_YYYYMMDDHHMM-YYYYMMDDHHMM_1_h.txt
3. **低解析資料小時nc檔 (ERA5)**
    - 請把檔案都放在**同一個資料夾裡**
    - 檔名**必須**為此格式：var_YYYYMMDDTHH.nc， var為變數名稱，YYYY為年份，MM為月份，DD為日期，HH為小時，T為時間標記，例子：t2m_20190830T11.nc

---
#### 如果ERA5為月檔，因月檔案太龐大直接轉換成zarr過程會不穩定
#### 請先使用 `data_preprocessing/nc_month_to_day/split_era5_monthly_to_daily.py` 轉成小時檔
  - 檔名 **必須** 為此格式：var_YYYYMM.nc， var為變數名稱，YYYY為年份，MM為月份，例子：t2m_201908.nc
  ```yaml
  # 請到data_preprocessing/nc_month_to_day/config.yaml更改
root_dir: # 【需更動】ERA5 nc 月檔資料夾路徑
output_dir: # 【需更動】轉換後  ERA nc 小時檔放置資料夾路徑
time_variable: "valid_time" # ERA5 nc 檔裡時間變數的名稱
workers: 20 # 平行處理的核心數，可依據CPU核心數調整
dt_hours: 1 # 資料時間間隔，單位為小時
  ```
  - 執行轉換腳本
  ```bash
  # 執行前請先修改 config.yaml 裡的路徑參數
  python3 split_era5_monthly_to_daily.py --config config.yaml
  ```
---

### NC轉Zarr格式 : `data_preprocessing/nc_to_zarr/`

1. 請到 `data_preprocessing/nc_to_zarr/config.yaml` 修改以下參數：
```yaml
# Data paths
era5-path: # ERA5 NC 小時檔資料夾路徑
rwrf-path: # RWRF NC 小時檔資料夾路徑
qpepre-path: # QPEPRE txt 小時檔資料路徑
output-path: # zarr檔資料夾放置路徑

train-ranges: # 訓練期間
  - ["2019/08/01", "2019/08/17"]

valid-ranges: # 驗證期間
  - ["2019/08/18", "2019/08/31"]
hours: ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23"]

# Domain configuration
domain-size: [224, 128]  # [緯度(lat), 經度(long)]
lon-bounds: [119.75, 122.25]  # 經度的 [min, max]
lat-bounds: [21.6, 25.6]    # 緯度的 [min, max]

# Processing options
max-workers: 20 # 平行處理的核心數，可依據CPU核心數調整
```
2. 執行轉換腳本
```bash
# 執行前請先修改 config.yaml 裡的路徑參數
python3 nc_to_zarr.py --config config.yaml
```
---

## Zarr檔案檢視
### 使用 `data_preprocessing/print_zarr_info/`
1. 執行檢視腳本
```bash
# 【需更動】執行腳本，請將 <path to zarr> 替換為欲檢視的 zarr 檔案路徑
python3 print_zarr_info.py <path to zarr>
```

2. 使用 `data_preprocessing/plot_zarr/` 進行Zarr資料視覺化

#### 列出所有可用變數
```bash
# 查看 Zarr 檔案中所有可用的變數
python3 plot_zarr.py <zarr檔案路徑> --list-variables
```

#### 列出所有時間步
```bash
# 查看 Zarr 檔案中所有時間步及其有效性
python3 plot_zarr.py <zarr檔案路徑> --list-times
```

#### 繪製特定變數
```bash
# 繪製 ERA5 500hPa 風場 u 分量，時間索引為 0
python3 plot_zarr.py <zarr檔案路徑> --source LowRes --variable "u500" --time 0

# 繪製 StormCast 高解析度 10米風場 u 分量，時間索引為 12
python3 plot_zarr.py <zarr檔案路徑> --source HighRes --variable "u10" --time 12
```

---

## Stormcast模型訓練

### 1. 訓練回歸模型 (Regression Model)
回歸模型用於直接預測氣象變數，是擴散模型的基礎。

#### 修改訓練參數
請到 `stormcast/train_regression.sh` 修改以下參數：

```bash
# --- Torchrun 設定 ---
number_of_nodes=1          # 節點數量
gpus_per_node=2            # 每個節點的 GPU 數量

# --- 一般訓練設定 ---
experiment_name="regression_ncdr"                                              # 實驗名稱
training_output_dir="/project/n/desmond/diffusion_output/regression"         # 訓練輸出目錄
run_id="0"                                                                    # 執行編號

# --- 日誌參數 ---
print_progress_freq=25     # 每 N 步驟印出訓練進度
checkpoint_freq=1000       # 每 N 步驟儲存檢查點
validation_freq=50         # 每 N 步驟進行驗證

# --- 訓練參數 ---
batch_size=12              # 批次大小
lr=4E-4                    # 學習率
lr_rampup_steps=1000       # 學習率暖身步數
total_train_steps=16000    # 總訓練步數
clip_grad_norm=-1          # 梯度裁剪閾值，設為 -1 表示停用
loss='regression'          # 損失函數類型

# --- 驗證參數 ---
validation_plot_variables="[t2m,u10,v10,qpepre]"  # 要繪製的驗證變數

# --- 可選輸出 ---
output_nc="true"           # 是否輸出 NetCDF 檔案
output_nc_freq=5           # 每 N 次驗證輸出一次 NetCDF（5 表示每 5 次驗證輸出一次）

# --- 資料集參數 ---
location="/project/n/desmond/Stormcast_test/Zarr_test_optimized_skip_invalid"  # Zarr 資料位置
HighRes_img_size="[224,128]"                                                    # 高解析度影像大小
exp_train_zarrs="[stormcast_test_train]"                                       # 訓練用 Zarr 檔案
train_dates="[2019/08/01,2019/08/17]"                                          # 訓練日期範圍
exp_valid_zarrs="[stormcast_test_valid]"                                       # 驗證用 Zarr 檔案
valid_dates="[2019/08/18,2019/08/31]"                                          # 驗證日期範圍
kept_LowRes_channels="all"                                                      # 保留的低解析度通道
kept_HighRes_channels="all"                                                     # 保留的高解析度通道
```

#### 執行訓練
```bash
# 確保在 stormcast 目錄下
cd stormcast

# 執行訓練腳本
./train_regression.sh
```

#### 訓練輸出
訓練完成後，會在 `training_output_dir/experiment_name/run_id/` 產生以下檔案：
- `checkpoints_regression/`: 模型檢查點檔案
- `loss_curves.png`: 訓練與驗證損失曲線圖
- `train_loss.csv`: 訓練損失記錄
- `valid_loss.csv`: 驗證損失記錄
- `rmse_<變數名>.csv`: 各變數的 RMSE 記錄
- `mae_<變數名>.csv`: 各變數的 MAE 記錄
- `ps1d_<變數名>.csv`: 各變數的功率譜記錄
- `images/`: 驗證圖片資料夾
- `netcdf_outputs/`: NetCDF 輸出檔案（若 output_nc=true）

---

### 2. 訓練擴散模型 (Diffusion Model)
擴散模型用於生成高品質的氣象預測，需要先訓練好回歸模型。

#### 修改訓練參數
請到 `stormcast/train_diffusion.sh` 修改以下參數：

```bash
# --- Torchrun 設定 ---
number_of_nodes=1          # 節點數量
gpus_per_node=2            # 每個節點的 GPU 數量

# --- 一般訓練設定 ---
experiment_name="diffusion_ncdr"                                                      # 實驗名稱
training_output_dir="/home/master/13/dczy/code/stormcast-ncdr/data/Stormcast_test/diffusion"  # 訓練輸出目錄
run_id="0"                                                                            # 執行編號

# --- 日誌參數 ---
print_progress_freq=25     # 每 N 步驟印出訓練進度
checkpoint_freq=1000       # 每 N 步驟儲存檢查點
validation_freq=50         # 每 N 步驟進行驗證

# --- 訓練參數 ---
batch_size=16              # 批次大小
lr=4E-4                    # 學習率
lr_rampup_steps=1000       # 學習率暖身步數
total_train_steps=800000   # 總訓練步數
clip_grad_norm=-1          # 梯度裁剪閾值，設為 -1 表示停用
loss='edm'                 # 損失函數類型（擴散模型使用 'edm'）

# --- 驗證參數 ---
validation_plot_variables="[t2m,u10,v10,qpepre]"  # 要繪製的驗證變數

# --- 可選輸出 ---
output_nc="true"           # 是否輸出 NetCDF 檔案
output_nc_freq=5           # 每 N 次驗證輸出一次 NetCDF（5 表示每 5 次驗證輸出一次）

# --- 資料集參數 ---
location="/project/n/desmond/Stormcast_test/Zarr_test_optimized_skip_invalid"  # Zarr 資料位置
HighRes_img_size="[224,128]"                                                    # 高解析度影像大小
exp_train_zarrs="[stormcast_test_train]"                                       # 訓練用 Zarr 檔案
train_dates="[2019/08/01,2019/08/17]"                                          # 訓練日期範圍
exp_valid_zarrs="[stormcast_test_valid]"                                       # 驗證用 Zarr 檔案
valid_dates="[2019/08/18,2019/08/31]"                                          # 驗證日期範圍
kept_LowRes_channels="all"                                                      # 保留的低解析度通道
kept_HighRes_channels="all"                                                     # 保留的高解析度通道

# --- 模型參數 ---
# 【重要】預訓練回歸模型的路徑，用於擴散模型的條件輸入
regression_weights="/home/master/13/dczy/code/stormcast-ncdr/data/Stormcast_test/regression/regression_ncdr/run_0/checkpoints_regression/StormCastUNet.0.1000.mdlus"
```

#### 執行訓練
```bash
# 確保在 stormcast 目錄下
cd stormcast

# 執行訓練腳本
./train_diffusion.sh
```

#### 訓練輸出
訓練完成後，會在 `training_output_dir/experiment_name/run_id/` 產生以下檔案：
- `checkpoints_diffusion/`: 模型檢查點檔案
- `loss_curves.png`: 訓練與驗證損失曲線圖
- `train_loss.csv`: 訓練損失記錄
- `valid_loss.csv`: 驗證損失記錄
- `rmse_<變數名>.csv`: 各變數的 RMSE 記錄
- `mae_<變數名>.csv`: 各變數的 MAE 記錄
- `ps1d_<變數名>.csv`: 各變數的功率譜記錄
- `images/`: 驗證圖片資料夾
- `netcdf_outputs/`: NetCDF 輸出檔案（若 output_nc=true）

---

### 訓練注意事項

#### GPU 記憶體設定
- 若遇到 GPU 記憶體不足問題，可調整以下參數：
  - 減少 `batch_size`
  - 減少 `gpus_per_node`
  - 設定環境變數：`export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

#### 訓練監控
- 訓練過程中會自動產生 `loss_curves.png`，可即時查看訓練進度
- CSV 檔案記錄詳細的訓練指標，可用於後續分析

#### 檢查點恢復
- 訓練會自動從最新的檢查點恢復
- 若要從特定檢查點開始，請修改配置檔中的 `resume_checkpoint` 參數

#### NetCDF 輸出
- `output_nc=true` 時會輸出驗證結果的 NetCDF 檔案
- `output_nc_freq` 控制輸出頻率，避免產生過多檔案
- NetCDF 檔案可用 `plot_zarr.py` 或其他 NetCDF 工具視覺化

---