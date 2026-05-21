#!/usr/bin/env python3
"""Raw data plotting: HighRes (target), LowRes (conditioning), and invariant
fields.

For N evenly-spaced timestamps across the validation year, renders ONE PNG
per (channel, timestamp) pair for the time-varying fields, plus one PNG per
invariant channel (time-independent, written once):

    <output-dir>/highres/<channel>/time_NN.png      # 4 channels
    <output-dir>/lowres/<channel>/time_NN.png       # 24 channels
    <output-dir>/invariants/<channel>.png           # 2 channels, no time

Each PNG is a single-panel figure: ``imshow`` with ``origin='lower'``,
default ``viridis`` colormap, no quantile clipping (vmin/vmax = true field
min/max), ticks stripped, ``aspect='auto'``, and a vertical colorbar on the
right with the channel's physical-units label. Title carries the channel
name and timestamp (or just channel name for invariants).

Time-varying fields are denormalized to physical units via the dataset's
``denormalize_state`` / ``denormalize_background`` helpers (which also
invert the ``log1p`` on qpepre when the cleaned dataset is selected).
Invariants come from ``get_invariants`` and are already in physical units.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

# --- repo path bootstrap -----------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
STORMCAST_ROOT = REPO_ROOT / "stormcast"
sys.path.insert(0, str(STORMCAST_ROOT))

from physicsnemo.distributed import DistributedManager  # noqa: E402

from datasets import dataset_classes  # noqa: E402


# HighRes channel render order (rows top -> bottom in the 2x2 grid traversed
# in row-major: [[t2m, u10], [v10, qpepre]]).
HIGHRES_CHANNELS = ["t2m", "u10", "v10", "qpepre"]

HIGHRES_UNITS = {"t2m": "K", "u10": "m/s", "v10": "m/s", "qpepre": "mm/h"}

# LowRes layout: 6 rows x 4 cols. The surface row mixes units, the five
# pressure-level rows each fix one physical variable across the four standard
# levels.
LOWRES_SURFACE = ["mslp", "t2m", "u10", "v10"]
LOWRES_PRESSURE_VARS = ["q", "t", "u", "v", "z"]
LOWRES_LEVELS = [1000, 850, 500, 250]

# Physical units per LowRes channel for colorbar labels.
LOWRES_UNITS = {
    "mslp": "Pa", "t2m": "K", "u10": "m/s", "v10": "m/s",
    "q": "kg/kg", "t": "K", "u": "m/s", "v": "m/s", "z": "m^2/s^2",
}

# Static invariant channels (always rendered).
INVARIANT_CHANNELS = ["lsm", "orog"]
INVARIANT_UNITS = {"lsm": "0-1 (land=1)", "orog": "m"}


def make_dataset_cfg(data_loc, valid_dates, hr_size, qpepre_log1p, kept_HR):
    return OmegaConf.create(
        {
            "location": str(data_loc),
            "HighRes_img_size": list(hr_size),
            "dt": 1,
            "exp_train_zarrs": ["stormcast_test_train"],
            "train_dates": ["2019/08/01", "2021/12/31"],
            "exp_valid_zarrs": ["stormcast_test_valid"],
            "valid_dates": list(valid_dates),
            "invariants": ["lsm", "orog"],
            "input_channels": "all",
            "diffusion_channels": list(kept_HR),
            "kept_LowRes_channels": "all",
            "kept_HighRes_channels": list(kept_HR),
            "qpepre_log1p": qpepre_log1p,
        }
    )


def build_dataset(data_loc, valid_dates, hr_size, qpepre_log1p, kept_HR):
    cfg = make_dataset_cfg(data_loc, valid_dates, hr_size, qpepre_log1p, kept_HR)
    dataset_cls = dataset_classes["data_loader_rwrf_era5_stable.Dataset"]
    return dataset_cls(cfg, train=False)


def lowres_unit(channel):
    """Look up the physical-units string for a LowRes channel name."""
    if channel in LOWRES_UNITS:
        return LOWRES_UNITS[channel]
    # Pressure-level channels are e.g. ``q1000`` / ``t850`` / ``u500`` / etc.
    # The variable letter is the leading non-digit run.
    var = "".join(ch for ch in channel if not ch.isdigit())
    return LOWRES_UNITS.get(var, "")


def plot_single(out_path, field, title, cbar_label):
    """Render one (H, W) field as a single-panel viridis imshow with a
    vertical colorbar on the right. Matches the trainer's validation_plot
    style: ``origin='lower'``, default viridis, true-min/max color limits,
    no ticks, ``aspect='auto'``.
    """
    vmin, vmax = float(np.min(field)), float(np.max(field))
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    im = ax.imshow(
        field, origin="lower", vmin=vmin, vmax=vmax, aspect="auto",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=13)
    cbar = fig.colorbar(
        im, ax=ax, location="right",
        fraction=0.046, pad=0.02, shrink=0.95,
        label=cbar_label,
    )
    cbar.ax.tick_params(labelsize=10)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--data", type=Path, required=True,
                    help="Dataset root (zarr_exp3_... directory).")
    ap.add_argument("--hr-size", type=int, nargs=2, default=[192, 96],
                    help="HighRes image size (H W). 192 96 for cleaned, "
                         "224 128 for legacy.")
    ap.add_argument("--qpepre-log1p", type=lambda s: s.lower() == "true",
                    default=True,
                    help="Set to false for legacy raw-mm/h datasets.")
    ap.add_argument("--valid-dates", nargs=2, default=["2022/01/01", "2022/12/31"])
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--n-times", type=int, default=10,
                    help="Number of evenly-spaced timestamps to render.")

    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    DistributedManager.initialize()

    print(f"[data] building dataset @ {args.data}")
    dataset = build_dataset(
        args.data, tuple(args.valid_dates), tuple(args.hr_size),
        qpepre_log1p=args.qpepre_log1p, kept_HR=HIGHRES_CHANNELS,
    )
    hr_channels = list(dataset.state_channels())
    lr_channels = list(dataset.background_channels())
    print(f"[data] pairs={len(dataset)}  HR={hr_channels}  LR_n={len(lr_channels)}")
    print(f"[data] LR channels: {lr_channels}")

    n = len(dataset)
    if n < args.n_times:
        raise SystemExit(f"not enough samples ({n}) for {args.n_times} times")
    t_indices = np.linspace(0, n - 1, args.n_times).astype(int).tolist()
    print(f"[seqs] n_times={len(t_indices)}  first={t_indices[0]}  last={t_indices[-1]}")

    hr_idx_to_plot = [hr_channels.index(c) for c in HIGHRES_CHANNELS]

    # Pre-make one subdirectory per channel so the same variable across
    # timestamps lives side by side on disk.
    hr_root = args.output_dir / "highres"
    lr_root = args.output_dir / "lowres"
    inv_root = args.output_dir / "invariants"
    for ch in HIGHRES_CHANNELS:
        (hr_root / ch).mkdir(parents=True, exist_ok=True)
    for ch in lr_channels:
        (lr_root / ch).mkdir(parents=True, exist_ok=True)
    inv_root.mkdir(parents=True, exist_ok=True)

    # Invariants are time-independent -- render once before the per-timestamp
    # loop. ``get_invariants`` returns physical units already, indexed in the
    # order of ``dataset.params.invariants``.
    inv_array = dataset.get_invariants()
    if inv_array is not None:
        inv_channels = list(dataset.params.invariants)
        for j, ch in enumerate(inv_channels):
            unit = INVARIANT_UNITS.get(ch, "")
            plot_single(
                inv_root / f"{ch}.png",
                inv_array[j],
                title=f"invariant {ch} [{unit}]" if unit else f"invariant {ch}",
                cbar_label=unit,
            )
        print(f"[inv]  wrote {len(inv_channels)} invariant PNG(s): {inv_channels}")
    else:
        inv_channels = []
        print("[inv]  no invariants returned by dataset")

    t_start = time.perf_counter()
    for i, ti in enumerate(t_indices):
        t_s = time.perf_counter()
        data = dataset[ti]
        # background: (24, H, W) normalized; state: ((4,H,W), (4,H,W))
        bg = data["background"].cpu().numpy().copy()
        hr = data["state"][0].cpu().numpy().copy()

        bg_phys = dataset.denormalize_background(bg)
        hr_phys = dataset.denormalize_state(hr)[hr_idx_to_plot]

        try:
            ts = dataset.valid_samples[ti]
            time_label = ts.strftime("%Y-%m-%d %H:00 UTC")
        except (AttributeError, IndexError):
            time_label = f"idx={ti}"

        stamp = f"time_{i + 1:02d}"

        # HighRes: one PNG per channel.
        for j, ch in enumerate(HIGHRES_CHANNELS):
            unit = HIGHRES_UNITS[ch]
            out = hr_root / ch / f"{stamp}.png"
            plot_single(
                out, hr_phys[j],
                title=f"HighRes {ch} [{unit}]  |  {time_label}",
                cbar_label=unit,
            )

        # LowRes: one PNG per channel.
        for j, ch in enumerate(lr_channels):
            unit = lowres_unit(ch)
            out = lr_root / ch / f"{stamp}.png"
            plot_single(
                out, bg_phys[j],
                title=f"LowRes {ch} [{unit}]  |  {time_label}",
                cbar_label=unit,
            )

        dt = time.perf_counter() - t_s
        total = time.perf_counter() - t_start
        eta = (total / (i + 1)) * (len(t_indices) - (i + 1))
        n_per_t = len(HIGHRES_CHANNELS) + len(lr_channels)
        print(
            f"[plot] {i + 1}/{len(t_indices)} idx={ti:<5}  {time_label}  "
            f"wrote={n_per_t}  dt={dt:4.1f}s  total={total / 60:5.1f}m  "
            f"eta={eta / 60:5.1f}m"
        )

    meta = args.output_dir / "metadata.txt"
    with open(meta, "w") as f:
        f.write(f"data: {args.data}\n")
        f.write(f"hr_size: {tuple(args.hr_size)}\n")
        f.write(f"qpepre_log1p: {args.qpepre_log1p}\n")
        f.write(f"valid_dates: {args.valid_dates}\n")
        f.write(f"n_times: {args.n_times}\n")
        f.write(f"t_indices: {t_indices}\n")
        f.write(f"hr_channels: {hr_channels}\n")
        f.write(f"lr_channels: {lr_channels}\n")
        f.write(f"invariant_channels: {inv_channels}\n")
    print(f"[done] outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
