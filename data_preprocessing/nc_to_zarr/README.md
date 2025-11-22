# NetCDF 轉 Zarr 資料預處理

## 設定指南

### 1. 資料路徑
在設定檔中配置輸入和輸出路徑：

```yaml
era5-path: "/path/to/ERA5/nc/hourly/data"     # ERA5 NetCDF 小時檔資料夾
rwrf-path: "/path/to/RWRF/nc/hourly/data"     # RWRF NetCDF 小時檔資料夾
qpepre-path: "/path/to/obs_1hrRain/txt/data"  # QPEPRE txt 小時檔資料夾
output-path: "/path/to/output/zarr"           # 輸出 Zarr 資料夾
```

### 2. 時間範圍
定義訓練和驗證期間：

```yaml
train-ranges:  # 訓練期間
  - ["2022/01/01", "2022/01/20"] # 2022/01/01 到 2022/01/20

valid-ranges:  # 驗證期間
  - ["2022/01/21", "2022/01/31"] # 2022/01/21 到 2022/01/31
```

### 3. 空間範圍
配置地理區域：

```yaml
domain-size: [224, 128]          # [緯度點數, 經度點數]
lon-bounds: [119.75, 122.25]     # [最小經度, 最大經度] 單位：度
lat-bounds: [21.6, 25.6]         # [最小緯度, 最大緯度] 單位：度
```

### 4. 處理選項
調整平行處理設定：

```yaml
max-workers: 20  # 平行處理的 CPU 核心數
                 # 可依據可用的 CPU 核心數調整
```

### 5. 變數設定
配置要處理的變數：

**覆寫特定變數列表：**
```yaml
era5-vars: "mslp,sp,t2m,u10,v10"      # 逗號分隔的 ERA5 變數
rwrf-vars: "u10,v10,t2m,sp,msl"       # 逗號分隔的 RWRF 變數
invariant-vars: "lsm,orog"            # 逗號分隔的不變變數
```

**或使用 null 來載入完整定義：**
```yaml
era5-vars: null      # 使用下方的 era5-variables 列表
rwrf-vars: null      # 使用下方的 rwrf-variables 列表
invariant-vars: null # 使用下方的 invariant-variables 列表

era5-variables:
  - "mslp"
  - "sp"
  # ... 新增更多變數

rwrf-variables:
  - "u10"
  - "v10"
  # ... 新增更多變數

invariant-variables:
  - "lsm"
  - "orog"
```

### 6. 日誌設定
配置日誌行為：

```yaml
log-level: "DEBUG"           # 選項：DEBUG, INFO, WARNING, ERROR
log-file: "nc_to_zarr.log"   # 日誌檔案名稱
```

## 使用方式

1. 進入 `nc_to_zarr` 目錄：

    ```bash
    cd data_preprocessing/nc_to_zarr
    ```

2. 轉換 NetCDF 檔案為 Zarr 格式： 

    a. 一般環境：
    ```bash
    python nc_to_zarr.py -c config.yaml
    ```
    
    b. 在nano5上，使用SLURM提交工作：
    ```bash
    srun --partition=dev --account=MST111414 \
    --gpus-per-node=1 --cpus-per-task=12 --ntasks=1 \
    python3 nc_to_zarr.py -c config_nano5.yaml
    ```