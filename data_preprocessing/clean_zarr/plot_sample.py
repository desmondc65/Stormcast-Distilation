#!/usr/bin/env python3
"""Plot all channels of a single timestamp from the cleaned zarr dataset.

Usage:
    python plot_sample.py [--ts YYYY-MM-DDTHH] [--rainy] [--src DIR] [--out DIR]

Behaviour:
    --ts        explicit timestamp to plot (must exist in the train zarr)
    --rainy     pick the timestamp with the highest qpepre 99th percentile (default)
    no flag     pick the first valid timestamp

Generates:
    plot/lowres_<ts>.png       4 x 6 grid of all 24 ERA5 channels
    plot/highres_<ts>.png      1 x 4 grid of HighRes (t2m, u10, v10, qpepre)
    plot/invariants.png        1 x 2 grid of (lsm, orog)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

DEFAULT_SRC = Path(
    "/home3/davidlcs/Econ-Rag/Local_LLM/test_meeting/Stormcast-Distilation/"
    "exp_3_train_2_5_yrs_val_1yr_tp1/"
    "zarr_exp3_L_24_H_24_train_2_5_years_full_cleaned_4_27_2026"
)
DEFAULT_OUT = Path(__file__).resolve().parent / "plot"

# Per-channel display configuration.
# Use a diverging cmap (with 0 at the centre) ONLY for fields where 0 is the
# natural midpoint -- wind components. Temperatures, pressure, humidity, and
# geopotential are absolute scalar fields, so we use a sequential cmap with a
# percentile range to actually show variation across the domain.
LOWRES_CMAPS = {
    "mslp": "viridis",
    "t2m": "inferno", "t1000": "inferno", "t850": "inferno",
    "t500": "inferno", "t250": "inferno",
    "u10": "RdBu_r", "u1000": "RdBu_r", "u850": "RdBu_r",
    "u500": "RdBu_r", "u250": "RdBu_r",
    "v10": "RdBu_r", "v1000": "RdBu_r", "v850": "RdBu_r",
    "v500": "RdBu_r", "v250": "RdBu_r",
    "q1000": "BuGn", "q850": "BuGn", "q500": "BuGn", "q250": "BuGn",
    "z1000": "plasma", "z850": "plasma", "z500": "plasma", "z250": "plasma",
}
HIGHRES_CMAPS = {"t2m": "inferno", "u10": "RdBu_r", "v10": "RdBu_r", "qpepre": "Blues"}


def pick_timestamp(ds_high: xr.Dataset, mode: str, explicit: str | None) -> np.datetime64:
    """Return the np.datetime64 of the timestamp to plot."""
    valid = np.asarray(ds_high["valid"].values, dtype=bool)
    times = np.asarray(ds_high.time.values, dtype="datetime64[ns]")
    if explicit is not None:
        ts = np.datetime64(explicit, "ns")
        if ts not in times:
            raise SystemExit(f"timestamp {explicit} not in zarr time axis")
        idx = int(np.where(times == ts)[0][0])
        if not valid[idx]:
            print(f"[warn] {explicit} is marked INVALID -- plotting anyway")
        return ts
    if mode == "rainy":
        # find the timestamp with the largest qpepre p99
        qpepre = ds_high["HighRes"].sel(channel="qpepre")
        # sample ~600 candidates for speed (every 30h ~= 700 picks across 2.5yr)
        stride = max(1, len(times) // 600)
        cand_idx = np.arange(0, len(times), stride)
        cand_idx = cand_idx[valid[cand_idx]]
        scores = qpepre.isel(time=cand_idx).quantile(0.99, dim=("y", "x")).values
        best = cand_idx[int(np.argmax(scores))]
        print(f"[rainy] picked time index {best} ts={times[best]}  q99(qpepre)={float(scores.max()):.2f} mm/h")
        return times[best]
    # default: first valid
    first = int(np.where(valid)[0][0])
    print(f"[first-valid] picked time index {first} ts={times[first]}")
    return times[first]


def plot_grid(
    arr: np.ndarray,           # (C, Y, X)
    channel_names: list[str],
    cmaps: dict[str, str],
    extent: tuple[float, float, float, float],
    title: str,
    out_path: Path,
    n_cols: int,
) -> None:
    n_ch = arr.shape[0]
    n_rows = int(np.ceil(n_ch / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.6 * n_cols, 3.0 * n_rows),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    for i in range(n_rows * n_cols):
        ax = axes.flat[i]
        if i >= n_ch:
            ax.axis("off")
            continue
        name = channel_names[i]
        data = arr[i]
        cmap = cmaps.get(name, "viridis")
        # Symmetric range for diverging maps (winds: 0 is the natural midpoint).
        if cmap == "RdBu_r":
            mag = float(np.nanpercentile(np.abs(data), 99))
            vmin, vmax = -mag, mag
        elif name == "qpepre":
            # Sparse, heavy-tailed -- clip to a sensible upper end so the
            # colour scale isn't dominated by a single hot pixel.
            nonzero = data[data > 0]
            top = float(np.nanpercentile(nonzero, 99)) if nonzero.size else 1.0
            vmin, vmax = 0.0, max(top, 1.0)
        else:
            vmin = float(np.nanpercentile(data, 1))
            vmax = float(np.nanpercentile(data, 99))
        im = ax.imshow(
            data,
            origin="lower",
            extent=extent,
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_title(f"{name}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(title, fontsize=12)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--ts", type=str, default=None,
                        help="ISO timestamp like 2021-07-25T00")
    parser.add_argument("--first-valid", action="store_true",
                        help="pick the first valid train timestamp instead of "
                             "the rainiest one")
    args = parser.parse_args()

    src = args.src
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    print(f"opening {src}")
    low = xr.open_zarr(src / "LowRes" / "stormcast_test_train.zarr", consolidated=True)
    high = xr.open_zarr(src / "HighRes" / "stormcast_test_train.zarr", consolidated=True)
    inv = xr.open_zarr(src / "invariants" / "invariants.zarr", consolidated=True)

    mode = "first" if args.first_valid else "rainy"
    ts = pick_timestamp(high, mode, args.ts)
    ts_str = str(ts).replace(":", "").split(".")[0]  # "2021-07-25T0000"

    # Geographic extent for imshow (match cropped lat/lon ranges)
    lat = low.latitude.values  # (y, x)
    lon = low.longitude.values
    extent = (float(lon.min()), float(lon.max()),
              float(lat.min()), float(lat.max()))
    print(f"extent (lon_min, lon_max, lat_min, lat_max) = {extent}")

    # LowRes
    print("loading LowRes timestep...")
    low_arr = low["LowRes"].sel(time=ts).values  # (24, Y, X)
    low_channels = [str(c) for c in low.channel.values]
    plot_grid(
        low_arr, low_channels, LOWRES_CMAPS, extent,
        title=f"ERA5 LowRes ({len(low_channels)} channels) -- {ts}",
        out_path=out / f"lowres_{ts_str}.png",
        n_cols=6,
    )

    # HighRes
    print("loading HighRes timestep...")
    high_arr = high["HighRes"].sel(time=ts).values  # (4, Y, X)
    high_channels = [str(c) for c in high.channel.values]
    # qpepre is stored as log1p(mm/h); invert for display so the colourbar reads mm/h.
    if "qpepre" in high_channels:
        qi = high_channels.index("qpepre")
        high_arr[qi] = np.expm1(high_arr[qi])
    plot_grid(
        high_arr, high_channels, HIGHRES_CMAPS, extent,
        title=f"RWRF HighRes ({len(high_channels)} channels) -- {ts}",
        out_path=out / f"highres_{ts_str}.png",
        n_cols=4,
    )

    # Invariants
    print("plotting invariants...")
    inv_arr = inv["HighRes_invariants"].values  # (2, Y, X)
    inv_channels = [str(c) for c in inv.channel.values]
    plot_grid(
        inv_arr, inv_channels,
        cmaps={"lsm": "Greens", "orog": "terrain"},
        extent=extent,
        title="Invariants (land-sea mask, orography)",
        out_path=out / "invariants.png",
        n_cols=2,
    )

    low.close(); high.close(); inv.close()
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
