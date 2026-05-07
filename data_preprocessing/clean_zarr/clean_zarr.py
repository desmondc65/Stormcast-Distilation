#!/usr/bin/env python3
"""Clean and re-shape the StormCast zarr dataset.

Source:
    exp_3_train_2_5_yrs_val_1yr_tp1/zarr_exp3_L_24_H_24_train_2_5_years_full

Destination:
    exp_3_train_2_5_yrs_val_1yr_tp1/zarr_exp3_L_24_H_24_train_2_5_years_full_cleaned_4_27_2026

Tasks performed:
    1. Add a per-timestep ``valid`` boolean variable to the LowRes/HighRes
       train zarrs. Timestamps listed in ``invalid_rwrf.txt`` are marked
       False. The data loader (data_loader_rwrf_era5_stable.Dataset) drops a
       (ts_inp, ts_tar) training pair whenever EITHER endpoint is invalid
       in EITHER LowRes or HighRes -- see ``_load_valid_mask`` and
       ``_collect_overlapping_valid_datetimes``. Marking the timestamp
       invalid in just one store is therefore sufficient, but for
       robustness we mark it in both.
    2. Crop the spatial domain to the original qpepre coverage:
           longitude in [120.0000, 122.0125]
           latitude  in [21.8875,  25.3125]
       Resolution is preserved -- this is a pure ``isel`` along y/x using
       the existing 2D ``latitude`` / ``longitude`` coordinates.
    3. Re-chunk for hourly random-access training (dt = 1 h):
           time    = 24  (one day per chunk)
           channel = full
           y       = full
           x       = full
       With dt = 1 h the input/target timestamps almost always sit in the
       same time-chunk, so a single chunk read serves a whole training
       pair. Every chunk holds all channels and the full spatial extent
       so a ``.sel(time=ts, channel=[...])`` call decompresses exactly
       one chunk per timestep.
    4. Recompute per-channel mean / std on the cropped, validity-masked
       train data and save them to ``LowRes/stats`` and ``HighRes/stats``.

Run::

    python clean_zarr.py                       # use defaults
    python clean_zarr.py --skip-stats          # skip stat re-computation
    python clean_zarr.py --src ... --dst ...   # custom paths
    # Build the `_raw` sibling for the log1p ablation (D2/F2/R0_raw):
    python clean_zarr.py \\
        --no-qpepre-log1p \\
        --dst .../zarr_exp3_L_24_H_24_train_2_5_years_full_cleaned_4_27_2026_raw
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr
from numcodecs import Blosc

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_SRC = (
    PROJECT_ROOT
    / "exp_3_train_2_5_yrs_val_1yr_tp1"
    / "zarr_exp3_L_24_H_24_train_2_5_years_full"
)
DEFAULT_DST = (
    PROJECT_ROOT
    / "exp_3_train_2_5_yrs_val_1yr_tp1"
    / "zarr_exp3_L_24_H_24_train_2_5_years_full_cleaned_4_27_2026"
)
DEFAULT_INVALID_FILE = SCRIPT_DIR / "invalid_rwrf.txt"

# qpepre bounding box (inclusive of both endpoints).
# qpepre target on its native grid is lat [21.8875, 25.3125], lon [120.0000, 122.0125].
# That maps to 192 cells in y but only 102 cells in x on the source ERA5/RWRF grid,
# and 102 is not divisible by the StormCastUNet downsampling factor (32) -- the
# encoder/decoder skip connections fail. We therefore clip the eastern edge so x
# lands on 96 cells (32*3), trimming ~0.13 deg of mostly-ocean cells off the right.
LAT_MIN, LAT_MAX = 21.8875, 25.3125
LON_MIN, LON_MAX = 120.0000, 121.8800
COORD_TOL = 1e-3  # tolerance when matching grid cells against the bounds

TIME_CHUNK = 24  # one day per chunk -- aligns with dt = 1h training pairs

COMPRESSOR = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)

# qpepre is in mm/h; very sparse (most cells = 0) and heavy-tailed
# (extreme events ~300 mm/h). Two preprocessing fixes applied to the HighRes
# qpepre channel only, both reversible at inference (apply ``np.expm1``):
#   1. ``max(0, x)`` -- precipitation is physically non-negative, but the
#      source RWRF data contains tiny negative artifacts (numerical noise).
#   2. ``log1p(x) = log(1 + x)`` -- compresses the heavy tail so the channel
#      becomes near-Gaussian, which MSE / spectral losses handle far better.
#      log1p is the standard choice for hourly precip (CorrDiff, NowcastNet,
#      DGMR). After log1p the per-channel std drops from ~9 to ~0.6, matching
#      the dynamic range of the other HighRes channels.
QPEPRE_TRANSFORM_CHANNEL = "qpepre"

# Each entry: (relative path under src/dst, name of the data variable, apply invalid mask?)
STORES = [
    ("LowRes/stormcast_test_train.zarr", "LowRes", True),
    ("LowRes/stormcast_test_valid.zarr", "LowRes", False),
    ("HighRes/stormcast_test_train.zarr", "HighRes", True),
    ("HighRes/stormcast_test_valid.zarr", "HighRes", False),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_invalid_timestamps(path: Path) -> "frozenset[np.datetime64]":
    """Parse ``invalid_rwrf.txt`` (lines like ``2019-08-05_00``)."""
    out: set = set()
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.lower() == "timestamp":
                continue
            try:
                date_part, hour_part = line.split("_")
            except ValueError:
                print(f"  [warn] skipping malformed line: {line!r}")
                continue
            ts = np.datetime64(
                f"{date_part}T{hour_part.zfill(2)}:00:00", "ns"
            )
            out.add(ts)
    return frozenset(out)


def find_crop_indices(reference_zarr: Path) -> "tuple[slice, slice]":
    """Locate the y/x slice that covers the qpepre bounds.

    ``latitude`` and ``longitude`` are stored as 2D arrays of shape (y, x).
    We assume a regular grid (lat varies along y, lon along x) and pull
    the 1D profile from column 0 / row 0 to look up indices.
    """
    with xr.open_zarr(reference_zarr, consolidated=True) as ds:
        lat2 = np.asarray(ds.latitude.values)
        lon2 = np.asarray(ds.longitude.values)

    lats = lat2[:, 0]
    lons = lon2[0, :]

    # Sanity check: confirm the grid actually is separable.
    if not np.allclose(lat2 - lats[:, None], 0.0, atol=1e-3):
        raise RuntimeError("latitude varies along x; grid is not separable.")
    if not np.allclose(lon2 - lons[None, :], 0.0, atol=1e-3):
        raise RuntimeError("longitude varies along y; grid is not separable.")

    y_keep = np.where((lats >= LAT_MIN - COORD_TOL) & (lats <= LAT_MAX + COORD_TOL))[0]
    x_keep = np.where((lons >= LON_MIN - COORD_TOL) & (lons <= LON_MAX + COORD_TOL))[0]
    if len(y_keep) == 0 or len(x_keep) == 0:
        raise RuntimeError(
            "Crop region is empty -- bounds do not match the source grid"
        )

    y_slice = slice(int(y_keep[0]), int(y_keep[-1]) + 1)
    x_slice = slice(int(x_keep[0]), int(x_keep[-1]) + 1)
    return y_slice, x_slice


def build_valid_array(
    time_values: np.ndarray, invalid_set: "frozenset[np.datetime64]"
) -> np.ndarray:
    if not invalid_set:
        return np.ones(len(time_values), dtype=bool)
    times = np.asarray(time_values, dtype="datetime64[ns]")
    return np.array([t not in invalid_set for t in times], dtype=bool)


# ---------------------------------------------------------------------------
# Main per-store processing
# ---------------------------------------------------------------------------
def write_store(
    src_path: Path,
    dst_path: Path,
    var_name: str,
    invalid_set: "frozenset[np.datetime64]",
    y_slice: slice,
    x_slice: slice,
    apply_qpepre_log1p: bool = True,
) -> None:
    print(f"\n[store] {src_path.relative_to(src_path.parents[2])}")
    t0 = time.time()
    # ``chunks={}`` preserves the on-disk chunking as the dask graph chunking,
    # we re-chunk explicitly below.
    ds = xr.open_zarr(src_path, consolidated=True, chunks={}, mask_and_scale=False)
    try:
        ds_c = ds.isel(y=y_slice, x=x_slice)
        T = ds_c.sizes["time"]
        C = ds_c.sizes["channel"]
        Y = ds_c.sizes["y"]
        X = ds_c.sizes["x"]
        print(f"        cropped dims: time={T}, channel={C}, y={Y}, x={X}")
        print(
            f"        lat in [{float(ds_c.latitude.min()):.4f},"
            f" {float(ds_c.latitude.max()):.4f}],"
            f" lon in [{float(ds_c.longitude.min()):.4f},"
            f" {float(ds_c.longitude.max()):.4f}]"
        )

        # HighRes-only: apply max(0, .) [+ log1p] to the qpepre channel.
        # The clip is a numerical-noise fix (RWRF qpepre has tiny negatives)
        # and is always on. log1p is the heavy-tail compression and is gated
        # by --no-qpepre-log1p so an ablation can build a `_raw` sibling that
        # differs from the standard cleaned zarr in *only* the log1p axis.
        if var_name == "HighRes" and QPEPRE_TRANSFORM_CHANNEL in [
            str(c) for c in ds_c.channel.values
        ]:
            transform_desc = "clip(min=0) + log1p" if apply_qpepre_log1p else "clip(min=0) only (NO log1p)"
            print(f"        transforming '{QPEPRE_TRANSFORM_CHANNEL}': {transform_desc}")
            data = ds_c[var_name]
            original_dims = data.dims  # ("time", "channel", "y", "x")
            is_qpepre = (ds_c.channel == QPEPRE_TRANSFORM_CHANNEL)
            # Clip the whole array to >=0 first (winds/temps unaffected because
            # we only WRITE the result back to the qpepre channel via where()).
            data_clipped = data.where(data >= 0, 0.0)
            qpepre_new = np.log1p(data_clipped) if apply_qpepre_log1p else data_clipped
            transformed = xr.where(is_qpepre, qpepre_new, data)
            # xr.where broadcasts and may reorder dims (channel first); restore
            # the original (time, channel, y, x) layout so chunks align.
            ds_c[var_name] = transformed.transpose(*original_dims)

        valid_arr = build_valid_array(ds_c.time.values, invalid_set)
        n_invalid = int((~valid_arr).sum())
        print(f"        invalid timestamps: {n_invalid} / {T}")

        ds_c = ds_c.assign(valid=("time", valid_arr))

        # Match dask chunks to the on-disk zarr chunks.
        ds_c = ds_c.chunk({"time": TIME_CHUNK, "channel": C, "y": Y, "x": X})
        # ``valid`` is 1D and tiny -- write it as one chunk.
        ds_c["valid"] = ds_c["valid"].chunk({"time": T})

        encoding = {
            var_name: {
                "chunks": (TIME_CHUNK, C, Y, X),
                "compressor": COMPRESSOR,
            },
            "valid": {
                "chunks": (T,),
                "compressor": COMPRESSOR,
            },
        }

        if dst_path.exists():
            shutil.rmtree(dst_path)
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        ds_c.to_zarr(
            dst_path,
            mode="w",
            encoding=encoding,
            consolidated=True,
            zarr_format=2,
        )
    finally:
        ds.close()
    print(f"        wrote -> {dst_path}  ({time.time() - t0:.1f}s)")


def write_invariants(src_path: Path, dst_path: Path, y_slice: slice, x_slice: slice) -> None:
    print(f"\n[invariants] {src_path.name}")
    t0 = time.time()
    with xr.open_zarr(src_path, consolidated=True) as ds:
        ds_c = ds.isel(y=y_slice, x=x_slice)
        if dst_path.exists():
            shutil.rmtree(dst_path)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        ds_c.to_zarr(dst_path, mode="w", consolidated=True, zarr_format=2)
    print(f"            wrote -> {dst_path}  ({time.time() - t0:.1f}s)")


def recompute_stats(zarr_path: Path, var_name: str, stats_dir: Path) -> None:
    """Compute per-channel mean/std on valid timesteps only and dump npy files."""
    print(f"\n[stats] {zarr_path.relative_to(zarr_path.parents[2])}")
    t0 = time.time()
    with xr.open_zarr(zarr_path, consolidated=True) as ds:
        valid = np.asarray(ds["valid"].values, dtype=bool)
        arr = ds[var_name]
        if not valid.all():
            arr = arr.isel(time=np.where(valid)[0])
        mean = arr.mean(dim=("time", "y", "x")).compute().values.astype(np.float32)
        std = arr.std(dim=("time", "y", "x")).compute().values.astype(np.float32)

    stats_dir.mkdir(parents=True, exist_ok=True)
    np.save(stats_dir / "means.npy", mean)
    np.save(stats_dir / "stds.npy", std)
    np.set_printoptions(precision=4, suppress=True)
    print(f"        means = {mean}")
    print(f"        stds  = {std}")
    print(f"        wrote -> {stats_dir}  ({time.time() - t0:.1f}s)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST)
    parser.add_argument("--invalid-file", type=Path, default=DEFAULT_INVALID_FILE)
    parser.add_argument("--skip-stats", action="store_true",
                        help="Do not recompute per-channel mean/std.")
    parser.add_argument("--allow-existing-dst", action="store_true",
                        help="Do not abort if --dst already exists.")
    parser.add_argument("--skip-lowres", action="store_true",
                        help="Skip the LowRes stores (and their stats).")
    parser.add_argument("--skip-highres", action="store_true",
                        help="Skip the HighRes stores (and their stats).")
    parser.add_argument("--skip-invariants", action="store_true",
                        help="Skip the invariants store.")
    parser.add_argument("--no-qpepre-log1p", action="store_true",
                        help="Do not apply log1p to the HighRes qpepre channel "
                             "(clip(min=0) is still applied). Use for the `_raw` "
                             "sibling dataset that the log1p ablation (D2/F2/R0_raw) "
                             "consumes. The launcher must then set "
                             "dataset.qpepre_log1p=false to keep loader semantics in sync.")
    args = parser.parse_args()

    src: Path = args.src
    dst: Path = args.dst

    if not src.exists():
        print(f"ERROR: source {src} does not exist", file=sys.stderr)
        return 1
    if dst.exists() and not args.allow_existing_dst:
        print(
            f"ERROR: destination {dst} already exists. "
            f"Pass --allow-existing-dst to overwrite individual stores in place.",
            file=sys.stderr,
        )
        return 1

    invalid_set = parse_invalid_timestamps(args.invalid_file)
    print(f"loaded {len(invalid_set)} invalid timestamps from {args.invalid_file}")

    reference = src / "LowRes" / "stormcast_test_train.zarr"
    y_slice, x_slice = find_crop_indices(reference)
    print(f"crop: y={y_slice}  x={x_slice}")

    for rel, var_name, apply_invalid in STORES:
        if args.skip_lowres and var_name == "LowRes":
            print(f"\n[skip] {rel}  (--skip-lowres)")
            continue
        if args.skip_highres and var_name == "HighRes":
            print(f"\n[skip] {rel}  (--skip-highres)")
            continue
        write_store(
            src / rel,
            dst / rel,
            var_name,
            invalid_set if apply_invalid else frozenset(),
            y_slice,
            x_slice,
            apply_qpepre_log1p=not args.no_qpepre_log1p,
        )

    if not args.skip_invariants:
        write_invariants(
            src / "invariants" / "invariants.zarr",
            dst / "invariants" / "invariants.zarr",
            y_slice,
            x_slice,
        )

    if not args.skip_stats:
        if not args.skip_lowres:
            recompute_stats(
                dst / "LowRes" / "stormcast_test_train.zarr",
                "LowRes",
                dst / "LowRes" / "stats",
            )
        if not args.skip_highres:
            recompute_stats(
                dst / "HighRes" / "stormcast_test_train.zarr",
                "HighRes",
                dst / "HighRes" / "stats",
            )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
