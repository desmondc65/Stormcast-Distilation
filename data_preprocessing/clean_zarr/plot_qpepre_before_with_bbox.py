"""Render the raw qpepre 'before cleaning' snapshot with the native qpepre
coverage drawn as a blue bounding box."""

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parents[2]
SRC_ZARR = REPO / "exp_3_train_2_5_yrs_val_1yr_tp1/zarr_exp3_L_24_H_24_train_2_5_years_full/HighRes/stormcast_test_train.zarr"
OUT_PNG = REPO / "exp_3_train_2_5_yrs_val_1yr_tp1/qpepre_before_after_cleaning.png"
TS = np.datetime64("2020-03-28T15:00:00")

# Native qpepre coverage derived from data_preprocessing/clean_zarr/qpepre.txt
QPEPRE_LON_MIN, QPEPRE_LON_MAX = 120.0000, 122.0125
QPEPRE_LAT_MIN, QPEPRE_LAT_MAX = 21.8875, 25.3125

# Cleaned crop bbox (192x96, U-Net-friendly trim of the qpepre bbox)
CROP_LON_MIN, CROP_LON_MAX = 120.0059, 121.8760
CROP_LAT_MIN, CROP_LAT_MAX = 21.8870, 25.3130

ds = xr.open_zarr(SRC_ZARR)
arr = ds.HighRes.sel(time=TS, channel="qpepre").values
lat2d = ds.latitude.values
lon2d = ds.longitude.values
lat1d = lat2d[:, 0]
lon1d = lon2d[0, :]
n_neg = int((arr < 0).sum())
qmin, qmax, qmean = float(arr.min()), float(arr.max()), float(arr.mean())

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.pcolormesh(lon1d, lat1d, arr, cmap="Blues", vmin=0.0, shading="auto")
ax.set_xlabel("longitude (°E)")
ax.set_ylabel("latitude (°N)")
ax.set_aspect("equal")
ax.set_title(
    f"BEFORE cleaning  (raw)\n"
    f"shape = {arr.shape[0]} × {arr.shape[1]}     "
    f"lat ∈ [{lat1d.min():.4f}, {lat1d.max():.4f}]     "
    f"lon ∈ [{lon1d.min():.4f}, {lon1d.max():.4f}]",
    fontsize=10,
)

rect_qpepre = patches.Rectangle(
    (QPEPRE_LON_MIN, QPEPRE_LAT_MIN),
    QPEPRE_LON_MAX - QPEPRE_LON_MIN,
    QPEPRE_LAT_MAX - QPEPRE_LAT_MIN,
    linewidth=2.2,
    edgecolor="tab:blue",
    facecolor="none",
    linestyle="--",
    label="qpepre native bbox",
)
ax.add_patch(rect_qpepre)

rect_crop = patches.Rectangle(
    (CROP_LON_MIN, CROP_LAT_MIN),
    CROP_LON_MAX - CROP_LON_MIN,
    CROP_LAT_MAX - CROP_LAT_MIN,
    linewidth=2.2,
    edgecolor="red",
    facecolor="none",
    linestyle="--",
    label="cleaned crop (192×96)",
)
ax.add_patch(rect_crop)
ax.legend(loc="lower left", framealpha=0.9)

fig.suptitle(
    f"qpepre @ {str(TS)}    |    BEFORE: {arr.shape[0]}×{arr.shape[1]} raw mm/h "
    f"({n_neg} negative cells in this snapshot)",
    fontsize=11,
    y=0.995,
)
cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.09, fraction=0.05)
cbar.set_label("qpepre (mm/h)")
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT_PNG, dpi=130)
print(f"wrote {OUT_PNG}")
