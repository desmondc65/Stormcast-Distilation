# StormCast to CorrDiff Zarr Converter

This folder contains a converter that rewrites the existing StormCast split Zarr stores into CorrDiff-compatible Zarr stores.

## Source and target

Source root expected:

- LowRes/stormcast_test_train.zarr
- LowRes/stormcast_test_valid.zarr
- HighRes/stormcast_test_train.zarr
- HighRes/stormcast_test_valid.zarr
- LowRes/stats/means.npy and stds.npy
- HighRes/stats/means.npy and stds.npy

Target output (per split):

- corrdiff_train.zarr
- corrdiff_valid.zarr
- corrdiff_combined.zarr

Each output store includes CorrDiff keys:

- era5, cwb
- era5_valid, cwb_valid
- era5_center, era5_scale, cwb_center, cwb_scale
- era5_variable, cwb_variable
- era5_pressure, cwb_pressure
- XLAT, XLON, XLONG
- time with units hours since 2015-01-01 00:00:00

## Input compatibility check result

The existing StormCast source at:

- exp_3_train_2_5_yrs_val_1yr_tp1/zarr_exp3_L_24_H_24_train_2_5_years_full

already has all core data CorrDiff needs:

- LowRes and HighRes tensors with aligned time and grid
- channel names
- latitude and longitude grids
- per-channel mean and std stats

The converter adds the CorrDiff-specific combined layout and metadata:

- era5/cwb tensor names
- era5_valid and cwb_valid masks
- XLAT/XLON fields
- pressure metadata arrays
- time-unit conversion to CorrDiff epoch

## Usage

Activate environment first:

```bash
conda activate stormcast_env
```

Run source check only:

```bash
python data_preprocessing/zarr_stormcast_to_zarr_corrdiff/convert_stormcast_to_corrdiff.py \
  --source-root exp_3_train_2_5_yrs_val_1yr_tp1/zarr_exp3_L_24_H_24_train_2_5_years_full \
  --output-root data_preprocessing/zarr_stormcast_to_zarr_corrdiff/output \
  --check-only
```

Run full conversion as one combined split (default behavior):

```bash
python data_preprocessing/zarr_stormcast_to_zarr_corrdiff/convert_stormcast_to_corrdiff.py \
  --source-root exp_3_train_2_5_yrs_val_1yr_tp1/zarr_exp3_L_24_H_24_train_2_5_years_full \
  --output-root data_preprocessing/zarr_stormcast_to_zarr_corrdiff/output \
  --splits combined \
  --workers 8 \
  --memory-budget-mb 4096 \
  --prefetch-batches 1 \
  --overwrite
```

Run full conversion as separate train and valid outputs:

```bash
python data_preprocessing/zarr_stormcast_to_zarr_corrdiff/convert_stormcast_to_corrdiff.py \
  --source-root exp_3_train_2_5_yrs_val_1yr_tp1/zarr_exp3_L_24_H_24_train_2_5_years_full \
  --output-root data_preprocessing/zarr_stormcast_to_zarr_corrdiff/output \
  --splits train,valid \
  --workers 8 \
  --memory-budget-mb 4096 \
  --prefetch-batches 1 \
  --overwrite
```

## Memory-aware controls

- workers: parallel read workers (thread pool)
- memory-budget-mb: converter caps read batch size from this budget
- read-batch-size: optional manual override for timesteps per read batch
- prefetch-batches: extra in-flight batches

If memory is tight, reduce workers and memory-budget-mb, for example:

```bash
python data_preprocessing/zarr_stormcast_to_zarr_corrdiff/convert_stormcast_to_corrdiff.py \
  --source-root exp_3_train_2_5_yrs_val_1yr_tp1/zarr_exp3_L_24_H_24_train_2_5_years_full \
  --output-root data_preprocessing/zarr_stormcast_to_zarr_corrdiff/output \
  --splits train,valid \
  --workers 2 \
  --memory-budget-mb 1024 \
  --overwrite
```

## Notes

- Valid masks are computed as finite checks over spatial dimensions per channel.
- Pressure levels are inferred from channel names like q1000, t850, u500; others are NaN.
- Output metadata is consolidated at the end of each split conversion.
