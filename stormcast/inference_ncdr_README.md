以下更新後的完整操作指南，新增了環境建置步驟，並整合原有的腳本設定說明。

-----

# StormCast NCDR 推論流程操作指南

本指南包含兩個部分：**一、環境建置與套件安裝** (僅需執行一次)，以及 **二、推論腳本參數設定** (每次執行需確認)。

### 一、 環境建置與套件安裝

在首次執行推論前，請依序執行以下指令以建立運算環境。

**1. 建立並啟動 Conda 虛擬環境**
建立 Python 3.10 環境並啟動：

```bash
conda create -n stormcast_env python=3.10 -y
conda activate stormcast_env
```

**2. 安裝 PyTorch 與相關運算套件**
請依照您的 CUDA 版本安裝對應的 PyTorch (以下範例為 CUDA 11.8 版本)：

```bash
# 若需確認 CUDA 版本可使用: nvidia-smi
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
```

**3. 安裝專案相依套件**
請切換至專案根目錄 (`stormcast-ncdr`) 並以編輯模式安裝：

```bash
cd stormcast-ncdr
pip install -e .
```

-----

### 二、 推論腳本參數設定

環境建置完成後，請修改 SLURM 腳本 (`.sh`) 中的以下變數以符合當次推論需求。

#### 1\. 程式碼與腳本路徑

請確認以下 Python 執行檔之絕對路徑：

  * **to\_zarr\_script**：資料前處理程式 (`to_zarr.py`)。
  * **inference\_script**：模型推論主程式 (`inference_ncdr.py`)。
  * **plotting\_script**：結果視覺化程式 (`prediction_plotting.py`)。

#### 2\. 資料來源與輸出路徑

請依據資料存放位置修改：

  * **grib\_folder**：低解析度全球模式資料 (GRIB) 資料夾。
  * **rwrf\_path**：高解析度區域模式資料 (RWRF/NetCDF) 檔案路徑。
  * **qpepre\_path**：降水觀測資料 (QPEPRE) 文字檔路徑。
  * **preprocessing\_output**：中介 Zarr 檔案輸出目錄。
  * **FINAL\_OUTPUT\_DIR**：最終結果 (NetCDF) 與圖檔輸出目錄。

#### 3\. 時空範圍與網格設定

  * **input\_timestamp**：預報起始時間 (格式：`YYYY-MM-DDTHH:MM:SS`)。
  * **domain\_height** / **domain\_width**：網格高與寬。
  * **lon\_min** / **lon\_max**：經度範圍。
  * **lat\_min** / **lat\_max**：緯度範圍。

#### 4\. 推論模型與變數

  * **n\_steps**：預報總步數。
  * **lowres\_vars** / **highres\_vars**：輸入變數列表 (須與訓練設定一致)。
  * **regression\_checkpoint**：迴歸模型權重檔 (`.mdlus`)。
  * **diffusion\_checkpoint**：擴散模型權重檔 (`.mdlus`)。

### 三、 執行作業

完成設定後，請使用 SLURM 提交作業：

```bash
sbatch 您的腳本名稱.sh
```