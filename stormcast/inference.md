# StormCast NCDR 推論指南

本指南說明如何使用 StormCast NCDR 進行氣象預報推論，從資料前處理到模型推論的完整流程。

---

## 目錄

1. [系統需求](#系統需求)
2. [資料準備](#資料準備)
3. [資料前處理 (to_zarr.py)](#資料前處理)
4. [推論設定檔](#推論設定檔)
5. [執行推論](#執行推論)
6. [輸出結果視覺化](#輸出結果視覺化)
7. [常見問題](#常見問題)

---

## 系統需求

- Python 3.10+
- CUDA GPU (建議至少 16GB VRAM)
- 必要套件：PyTorch, xarray, zarr, hydra-core, cfgrib

---

## 資料準備

推論需要三種輸入資料：

### 1. 低解析度資料 (LowRes) - GRIB 檔案
- 來源：全球模式預報資料 (如 EC-Pangu)
- 格式：GRIB2
- 變數 (24 個 channels)：
  - 地面變數：`mslp` (海平面氣壓), `t2m` (2公尺溫度), `u10` (10公尺U風), `v10` (10公尺V風)
  - 各氣壓層變數 (1000/850/500/250 hPa)：
    - `q` (比濕)
    - `t` (溫度)
    - `u` (U風)
    - `v` (V風)
    - `z` (位勢高度)

### 2. 高解析度資料 (HighRes) - RWRF NetCDF 檔案
- 來源：區域 WRF 模式輸出
- 格式：NetCDF (wrfout_d02_*)
- 變數 (4 個 channels)：
  - `t2m` (2公尺溫度，WRF 變數名：T2)
  - `u10` (10公尺U風，WRF 變數名：U10)
  - `v10` (10公尺V風，WRF 變數名：V10)
  - `qpepre` (降水量，來自 QPEPRE 檔案)

### 3. 降水資料 - QPEPRE 文字檔
- 來源：QPEPRE 系統
- 格式：純文字 (.txt)
- 檔名範例：`qpepre_202512030000-202512030100_1_h.txt`

### 4. 不變場資料 (Invariants)
- `lsm` (陸海遮罩，WRF 變數名：LANDMASK)
- `orog` (地形高度，WRF 變數名：HGT)

---

## 資料前處理

### 設定檔位置
```
data_preprocessing/inference_data_preprocess/inference_data_preprocess_config.yaml
```

### 設定檔說明

```yaml
# ---------------------------------------------------------------------------
# 輸入路徑設定
# ---------------------------------------------------------------------------

# GRIB 檔案路徑 (低解析度全球模式資料)
grib-path: /path/to/EC-pangu_2025120300-0.grb

# RWRF NetCDF 檔案路徑 (高解析度區域模式資料)
rwrf-path: /path/to/wrfout_d02_2025-12-03_00:00:00

# QPEPRE 降水文字檔路徑
qpepre-path: /path/to/qpepre_202512030000-202512030100_1_h.txt

# ---------------------------------------------------------------------------
# 輸出設定
# ---------------------------------------------------------------------------

# Zarr 輸出目錄
output-path: /path/to/output

# 是否覆蓋現有檔案
overwrite: true

# ---------------------------------------------------------------------------
# 區域範圍設定 (台灣區域)
# ---------------------------------------------------------------------------

# 網格大小 [高度, 寬度] (緯度, 經度方向的格點數)
domain-size:
  - 224  # 緯度方向格點數
  - 128  # 經度方向格點數

# 經度範圍 [最小, 最大]
lon-bounds:
  - 119.75
  - 122.25

# 緯度範圍 [最小, 最大]
lat-bounds:
  - 21.6
  - 25.6
```

### 執行資料前處理

```bash
cd /home/master/13/dczy/code/stormcast-ncdr/data_preprocessing/inference_data_preprocess

# 使用設定檔執行
python to_zarr.py --config inference_data_preprocess_config.yaml

# 或直接指定參數
python to_zarr.py \
    --grib-path /path/to/file.grb \
    --rwrf-path /path/to/wrfout \
    --qpepre-path /path/to/qpepre.txt \
    --output /path/to/output
```

### 輸出結構

前處理完成後，輸出目錄會包含：
```
output/
├── LowRes/
│   └── inference.zarr/     # 低解析度資料 (24 channels)
├── HighRes/
│   └── inference.zarr/     # 高解析度資料 (4 channels)
└── invariants/
    └── invariants.zarr/    # 不變場資料 (2 channels)
```

---

## 推論設定檔

### 主要設定檔位置
```
stormcast/config/inference_ncdr.yaml      # 主設定檔
stormcast/config/inference/stormcast.yaml # 推論參數設定
stormcast/config/dataset/inference.yaml   # 資料集設定
```

### 推論參數設定 (`inference/stormcast.yaml`)

```yaml
# ---------------------------------------------------------------------------
# 輸出路徑設定
# ---------------------------------------------------------------------------
outdir: '/path/to/inference_output'        # 輸出根目錄
experiment_name: 'stormcast-ncdr-inference' # 實驗名稱
run_id: 0                                   # 執行編號
rundir: ${inference.outdir}/${inference.experiment_name}/${inference.run_id}

# ---------------------------------------------------------------------------
# 模型 Checkpoint 路徑
# ---------------------------------------------------------------------------
regression_checkpoint: /path/to/StormCastUNet.0.3000.mdlus   # 迴歸模型
diffusion_checkpoint: /path/to/EDMPrecond.0.15000.mdlus      # 擴散模型

# ---------------------------------------------------------------------------
# 推論設定
# ---------------------------------------------------------------------------
n_steps: 6      # 預報步數 (例如 6 表示預報 t+1 到 t+6)
dt_hours: 1     # 每步時間間隔 (小時)
```

### 資料集設定 (`dataset/inference.yaml`)

```yaml
name: data_loader_inference.InferenceDataset  # 資料載入器類別

# 資料位置 (to_zarr.py 的輸出目錄)
location: '/path/to/output'

# 網格大小
HighRes_img_size: [224, 128]

# 變數設定
invariants: ["lsm", "orog"]
kept_LowRes_channels: all   # 或指定 channel 列表
kept_HighRes_channels: all  # 或指定 channel 列表
```

### 修改設定的方式

#### 方法一：直接修改 YAML 檔案
編輯 `config/inference/stormcast.yaml` 或 `config/dataset/inference.yaml`

#### 方法二：命令列覆蓋
```bash
# 修改預報步數
python inference_ncdr.py inference.n_steps=12

# 修改資料路徑
python inference_ncdr.py dataset.location=/new/path/to/data

# 同時修改多個參數
python inference_ncdr.py \
    inference.n_steps=6 \
    inference.dt_hours=1 \
    inference.rundir=/custom/output/path
```

---

## 執行推論

### 基本執行

```bash
cd /home/master/13/dczy/code/stormcast-ncdr/stormcast

# 使用預設設定
python inference_ncdr.py

# 指定預報步數
python inference_ncdr.py inference.n_steps=6

# 指定輸出目錄
python inference_ncdr.py inference.rundir=/path/to/output
```

### 完整執行範例

```bash
# 1. 先執行資料前處理
cd /home/master/13/dczy/code/stormcast-ncdr/data_preprocessing/inference_data_preprocess
python to_zarr.py --config inference_data_preprocess_config.yaml

# 2. 執行推論
cd /home/master/13/dczy/code/stormcast-ncdr/stormcast
python inference_ncdr.py \
    dataset.location=/home/master/13/dczy/code/stormcast-ncdr/data/stormcast_nano5_inference/data_ncdr/output \
    inference.n_steps=6 \
    inference.dt_hours=1
```

### 推論輸出

推論完成後，輸出目錄會包含：
```
inference_output/stormcast-ncdr-inference/0/
├── step_01_2025120301.zarr/    # t+1 預報 (zarr 格式)
├── step_01_2025120301.nc       # t+1 預報 (NetCDF 格式)
├── step_02_2025120302.zarr/    # t+2 預報
├── step_02_2025120302.nc
├── step_03_2025120303.zarr/    # t+3 預報
├── step_03_2025120303.nc
├── step_04_2025120304.zarr/    # t+4 預報
├── step_04_2025120304.nc
├── step_05_2025120305.zarr/    # t+5 預報
├── step_05_2025120305.nc
├── step_06_2025120306.zarr/    # t+6 預報
├── step_06_2025120306.nc
├── step_00_input_2025120300.zarr/  # 輸入參考資料
└── step_00_input_2025120300.nc
```

每個檔案包含 4 個變數：
- `t2m` - 2公尺溫度 (K)
- `u10` - 10公尺U風 (m/s)
- `v10` - 10公尺V風 (m/s)
- `qpepre` - 降水量 (mm/hr)

---

## 輸出結果視覺化

### 使用繪圖腳本

```bash
cd /home/master/13/dczy/code/stormcast-ncdr

# 使用預設路徑
python data_preprocessing/inference_data_preprocess/prediction_plotting.py

# 指定輸出目錄
python data_preprocessing/inference_data_preprocess/prediction_plotting.py \
    --output_dir /path/to/inference/output \
    --save_dir /path/to/save/plots

# 包含輸入資料
python data_preprocessing/inference_data_preprocess/prediction_plotting.py --include_input
```

### 輸出圖檔

繪圖腳本會為每個變數產生一張圖，包含所有預報時間步的子圖：
```
plots/
├── t2m_timesteps.png    # 2公尺溫度 6 步預報
├── u10_timesteps.png    # 10公尺U風 6 步預報
├── v10_timesteps.png    # 10公尺V風 6 步預報
└── qpepre_timesteps.png # 降水量 6 步預報
```

### 使用 Python 讀取結果

```python
import xarray as xr

# 讀取 NetCDF 檔案
ds = xr.open_dataset('/path/to/step_01_2025120301.nc')

# 查看變數
print(ds.data_vars)  # ['t2m', 'u10', 'v10', 'qpepre']

# 取得特定變數
t2m = ds['t2m'].values[0]  # shape: (224, 128)

# 取得座標
lat = ds['latitude'].values  # shape: (224, 128)
lon = ds['longitude'].values  # shape: (224, 128)
time = ds['time'].values[0]
```

---

## 常見問題

### Q1: CUDA out of memory 錯誤
**解決方案：**
- 減少 batch size
- 使用較少的 diffusion steps (修改 `config/sampler/edm_deterministic.yaml`)
- 確保 GPU 有足夠的 VRAM (建議 16GB+)

### Q2: 找不到 checkpoint 檔案
**解決方案：**
檢查 `config/inference/stormcast.yaml` 中的路徑：
```yaml
regression_checkpoint: /正確/路徑/StormCastUNet.0.3000.mdlus
diffusion_checkpoint: /正確/路徑/EDMPrecond.0.15000.mdlus
```

### Q3: 資料維度不符
**解決方案：**
確保前處理設定與訓練資料一致：
- `domain-size: [224, 128]`
- 變數數量與順序需與訓練時相同

### Q4: Hydra 設定錯誤
**解決方案：**
檢查 YAML 格式是否正確，特別是縮排和引號。

### Q5: 如何修改預報區域？
**解決方案：**
修改 `inference_data_preprocess_config.yaml`：
```yaml
domain-size:
  - 224  # 新的高度
  - 128  # 新的寬度

lon-bounds:
  - 新的西邊界
  - 新的東邊界

lat-bounds:
  - 新的南邊界
  - 新的北邊界
```

---

## 流程總覽

```
┌─────────────────────────────────────────────────────────────┐
│                     資料準備階段                              │
├─────────────────────────────────────────────────────────────┤
│  GRIB 檔案 ──┐                                              │
│              │                                               │
│  RWRF 檔案 ──┼──→ to_zarr.py ──→ Zarr 資料集                │
│              │                                               │
│  QPEPRE 檔案 ┘                                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     模型推論階段                              │
├─────────────────────────────────────────────────────────────┤
│  Zarr 資料集 ──→ inference_ncdr.py ──→ 預報結果              │
│                       │                    │                 │
│                       │                    ├─→ zarr 檔案     │
│                       │                    └─→ NetCDF 檔案   │
│                       │                                      │
│  迴歸模型 ────────────┤                                      │
│  擴散模型 ────────────┘                                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     結果視覺化階段                            │
├─────────────────────────────────────────────────────────────┤
│  預報結果 ──→ prediction_plotting.py ──→ PNG 圖檔           │
└─────────────────────────────────────────────────────────────┘
```

---

## 聯絡資訊

如有問題，請聯繫開發團隊。
