# clean_zarr

Pre-processing step that turns the raw exp-3 zarr dataset into a training-ready
"cleaned" zarr.

| | path |
|---|---|
| **input**  | [`exp_3_train_2_5_yrs_val_1yr_tp1/zarr_exp3_L_24_H_24_train_2_5_years_full/`](../../exp_3_train_2_5_yrs_val_1yr_tp1) |
| **output** | `exp_3_train_2_5_yrs_val_1yr_tp1/zarr_exp3_L_24_H_24_train_2_5_years_full_cleaned_4_27_2026/` |

The cleaned dataset is consumed via the existing
[`data_loader_rwrf_era5_stable.Dataset`](../../stormcast/datasets/data_loader_rwrf_era5_stable.py)
and only requires the user to override `dataset.location` to the new path.
Zarr filenames (`stormcast_test_train`, `stormcast_test_valid`), variable names,
and channel order are preserved.

---

## What the cleaning script does

[clean_zarr.py](clean_zarr.py) performs four operations in one pass over the dataset.

### 1. Mark RWRF-invalid timestamps (1,803 of 21,216)

The list of timestamps where the RWRF target field is missing or corrupt lives in
[invalid_rwrf.txt](invalid_rwrf.txt) (one `YYYY-MM-DD_HH` line per hour). The
cleaning script writes a 1-D boolean variable `valid` of shape `(time,)` into
both the `LowRes` and `HighRes` train zarrs. `valid[i] = False` for every
listed timestamp.

This matches what the data loader already does at startup
([data_loader_rwrf_era5_stable.py:238-271](../../stormcast/datasets/data_loader_rwrf_era5_stable.py#L238-L271)):

* `_load_valid_mask` reads the `valid` variable from each store
  (or assumes all-True if the variable is missing).
* `_collect_overlapping_valid_datetimes` keeps only timestamps where BOTH
  LowRes and HighRes report valid.
* `compute_total_samples` keeps only `(ts_inp, ts_inp + dt)` pairs where both
  endpoints are jointly valid.

So a timestamp marked invalid here is rejected as either the input *or* the
target of any training pair. The valid-only zarr (2022) is left untouched -- all
1,803 invalid hours fall in 2019/2020/2021. Result, with `dt = 1 h`:

| split | timestamps | jointly valid | usable `(t, t+1h)` pairs |
|---|---:|---:|---:|
| train | 21,216 | 19,413 | **18,765** |
| valid | 8,760 | 8,760 | **8,759** |

The 648-pair drop in train (`19,413 − 18,765`) corresponds to lone-survivor
timestamps whose `+1 h` neighbour is invalid.

### 2. Crop the spatial domain to the qpepre coverage

The original grid extends past Taiwan in every direction; the qpepre product
covers a tighter Taiwan-centred bounding box. Resolution is preserved, so this
is a pure `isel` along `y` and `x`.

| | original | cropped |
|---|---|---|
| latitude  | 21.6000 -> 25.6000  (224 cells) | **21.887 -> 25.313 (192 cells)** |
| longitude | 119.7500 -> 122.2500 (128 cells) | **120.006 -> 121.876 (96 cells)** |
| dlat | 4 / 223 ≈ 0.01794° | unchanged |
| dlon | 2.5 / 127 ≈ 0.01968° | unchanged |

The qpepre native bbox is lon `[120.0000, 122.0125]` (102 source cells), but
102 isn't a clean multiple of 32 -- the StormCastUNet 5-level downsampling
chain (`/2` five times) can't survive that, the encoder/decoder skip
connections diverge by 1 cell at level 2. We therefore clip the eastern edge
to `121.88` (96 cells = 32·3), trimming ~0.13° / ~6 cells of mostly-ocean
pixels off the right. Both 192 and 96 are now clean multiples of 32, so the
U-Net runs without any padding/wrapping in the data loader.

### 3. Re-chunk for hourly random-access training

The source store was sliced into 8-way splits along every dim
(`time=1326, channel=3, y=28, x=16`) -- the worst possible layout for random
sample access, where pulling one timestep touches ~512 chunks.

The cleaned store uses one chunk per day, full channels and full spatial:

| variable | shape | on-disk chunks |
|---|---|---|
| `LowRes`  (train) | (21216, 24, 192, 96) | (24, 24, 192, 96) |
| `LowRes`  (valid) | (8760, 24, 192, 96)  | (24, 24, 192, 96) |
| `HighRes` (train) | (21216, 4, 192, 96)  | (24, 4, 192, 96) |
| `HighRes` (valid) | (8760, 4, 192, 96)   | (24, 4, 192, 96) |
| `valid`           | (T,) bool             | (T,) -- one chunk |

With `dt = 1 h`, both `ts_inp` and `ts_inp + 1 h` land in the same time-chunk
in 23 / 24 cases, so a training pair triggers a single chunk decompression
(per LowRes / HighRes). Chunk count drops from 4,096 (per train zarr) to 884.

Compression: `Blosc(zstd-3, bitshuffle)`. The source was uncompressed
(`compressor: None`), so the cleaned store is much smaller despite preserving
all timestamps -- see "Disk-size reduction" below.

### 4. Pre-process the qpepre channel (clip negatives + log1p)

Hourly precipitation is sparse and heavy-tailed: most pixels are exactly 0,
the wet pixels span 5+ orders of magnitude (a few mm/h up to >300 mm/h in
extreme events), and the source RWRF data also contains tiny negative
numerical artefacts that have no physical meaning. Training MSE/spectral
losses directly on this distribution is unstable -- the model collapses to
"predict zero everywhere" because the loss landscape is dominated by the
abundant zeros and a handful of outliers.

The cleaning pipeline therefore applies, **only on the HighRes `qpepre`
channel**:

1. `max(0, x)`  -- clip negative numerical artefacts to 0 (precipitation is
   physically non-negative).
2. `log1p(x) = log(1 + x)` -- compress the heavy tail. Mapping mm/h:
    `0 -> 0`, `1 -> 0.69`, `10 -> 2.40`, `100 -> 4.62`, `300 -> 5.71`. The
    transformed channel is roughly an order of magnitude tighter than the
    raw mm/h, so MSE / spectral / Huber losses become well-behaved.

This is the standard precipitation pre-processing in atmospheric ML
(CorrDiff, NowcastNet, DGMR all use log1p or asinh on hourly precip). The
transform is exactly invertible: `mm/h = expm1(stored_value)`. Apply it at
inference time when reporting precipitation in physical units, and to the
target inside any precipitation-specific metric (CSI, FSS, exceedance bias).

A side-by-side plot of the qpepre distribution before vs after the transform
is at [plot/qpepre_distribution.png](plot/qpepre_distribution.png), generated
by [plot_qpepre_distribution.py](plot_qpepre_distribution.py).

### 5. Recompute per-channel statistics

`HighRes/stats/{means,stds}.npy` and `LowRes/stats/{means,stds}.npy` are
recomputed on the cropped, validity-masked train data (no invalid timestamps,
no out-of-domain pixels). The data loader uses these for standard-score
normalisation. **For HighRes the qpepre stats are computed on the
log1p-transformed channel**, not the raw mm/h.

| channel | mean   | std   |
|---|---:|---:|
| HighRes `t2m`    | 295.40 | 6.07 |
| HighRes `u10`    |  -1.06 | 3.58 |
| HighRes `v10`    |  -1.87 | 4.86 |
| HighRes `qpepre` (**after log1p**) | 0.103 | 0.366 |

For comparison, the raw mm/h qpepre had `mean=0.265, std=8.850` -- the
log1p transform brings the channel into the same dynamic range as the wind
components, so a single un-weighted MSE on all four channels treats them
on equal footing.

(LowRes has 24 channels, see [clean_zarr.log](clean_zarr.log) for the full vector.)

---

## Disk-size reduction (90 GB -> 36 GB)

| factor | ratio |
|---|---:|
| crop only (192·96) / (224·128) | 0.643 |
| compression only                | ~0.58 |
| combined (observed)             | **~0.38** |

The biggest win is that the source was uncompressed float32, while the cleaned
store uses zstd-3 with bitshuffle -- typical for slowly-varying atmospheric
fields, which compress about 2x after bitshuffle re-groups the most-significant
bits of adjacent floats together.

---

## Files in this directory

| file | purpose |
|---|---|
| [clean_zarr.py](clean_zarr.py)         | the cleaning pipeline |
| [plot_sample.py](plot_sample.py)       | render every channel of one timestamp into [`plot/`](plot/) |
| [invalid_rwrf.txt](invalid_rwrf.txt)   | RWRF-invalid timestamp list (1,803 hours) |
| [qpepre.txt](qpepre.txt)               | source qpepre coverage dump (used to derive the crop bbox) |
| [clean_zarr.log](clean_zarr.log)       | output of the most recent run |
| [plot/](plot/)                         | per-channel renderings of one sample timestamp |

---

## How to run

```bash
# default paths: src and dst as listed above; recompute stats; refuse to overwrite an existing dst
python clean_zarr.py

# custom paths
python clean_zarr.py --src /path/to/src.zarr_root --dst /path/to/dst.zarr_root

# skip stat re-computation (the slowest step at ~5 min)
python clean_zarr.py --skip-stats

# overwrite individual stores in an existing dst (per-store rmtree then re-write)
python clean_zarr.py --allow-existing-dst
```

A full run on this dataset takes ~35 min on the NAS-mounted home (NFS):

```
LowRes  train (21216 hrs)  18 min
LowRes  valid (8760 hrs)    8 min
HighRes train (21216 hrs)   2 min
HighRes valid (8760 hrs)    1 min
invariants                  <2 s
LowRes  stats               5 min
HighRes stats               1 min
```

Required Python packages: `xarray >= 2024`, `zarr >= 2.16`, `numcodecs`,
`dask`, `numpy`. No other framework dependencies.

---

## Plotting one timestamp

[plot_sample.py](plot_sample.py) picks a timestamp and renders every channel.
By default it picks the timestamp with the largest `qpepre` 99th percentile
(i.e. the rainiest hour) so the plots actually show interesting structure;
override with `--ts 2021-07-25T00` or `--first-valid`.

```bash
python plot_sample.py             # rainiest hour (default)
python plot_sample.py --first-valid
python plot_sample.py --ts 2021-07-25T00
```

The default run picks **2020-03-28T15:00** (q99 qpepre ≈ 345 mm/h, a heavy-rain
event over northern Taiwan) and writes:

| file | content |
|---|---|
| [`plot/lowres_2020-03-28T150000.png`](plot/lowres_2020-03-28T150000.png) | 6 × 4 grid of all 24 ERA5 channels |
| [`plot/highres_2020-03-28T150000.png`](plot/highres_2020-03-28T150000.png) | 1 × 4 grid of HighRes t2m, u10, v10, qpepre |
| [`plot/invariants.png`](plot/invariants.png) | land-sea mask + orography (km) |

Colour-scale conventions used by the script:

* wind components (`u*`, `v*`) -> diverging `RdBu_r`, symmetric around 0
  using the 99th-percentile magnitude (so a single hot pixel doesn't crush
  the scale).
* `qpepre` -> sequential `Blues`, clipped at the 99th percentile of *non-zero*
  pixels (the field is sparse and heavy-tailed).
* temperature, pressure, geopotential, specific humidity -> sequential
  (`inferno`, `viridis`, `plasma`, `BuGn`) with a 1-99 percentile range.
