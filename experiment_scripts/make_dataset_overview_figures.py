#!/usr/bin/env python3
"""Composite dataset-overview figures for the thesis.

Renders two single-PNG grid figures for one validation timestamp:

    <output-dir>/era5_lowres.png   # 6x4 grid of the 24 ERA5 LowRes channels
    <output-dir>/highres.png       # 1x4 row of the 4 HighRes target channels

Both are denormalized to physical units via the dataset's
``denormalize_background`` / ``denormalize_state`` helpers (the latter also
inverts the ``log1p`` on qpepre for the cleaned store). Colour follows
``thesis_style`` (viridis, ``origin='lower'``), matching every other field
figure in the thesis.

By default it renders the HighRes *target* ``M_{t+1}`` (state[1]) at validation
index 0 so the scene lines up with ``qual_qpepre_seq00`` from the method
comparison panel (compare_diffusion_vs_flowcast.py selects sequences via
``np.linspace`` with start=0).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

# --- repo path bootstrap -----------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
STORMCAST_ROOT = REPO_ROOT / "stormcast"
sys.path.insert(0, str(STORMCAST_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiment_scripts"))

import thesis_style  # noqa: E402  (applies the shared rcParams + viridis on import)

from physicsnemo.distributed import DistributedManager  # noqa: E402

from datasets import dataset_classes  # noqa: E402


# HighRes render order (canonical thesis order, independent of on-disk order).
HIGHRES_CHANNELS = ["t2m", "u10", "v10", "qpepre"]
HIGHRES_UNITS = {"t2m": "K", "u10": "m/s", "v10": "m/s", "qpepre": "mm/h"}

# LowRes 6x4 layout: surface row, then one row per upper-air variable across the
# four standard pressure levels.
LOWRES_SURFACE = ["mslp", "t2m", "u10", "v10"]
LOWRES_PRESSURE_VARS = ["q", "t", "u", "v", "z"]
LOWRES_LEVELS = [1000, 850, 500, 250]

LOWRES_UNITS = {
    "mslp": "Pa", "t2m": "K", "u10": "m/s", "v10": "m/s",
    "q": "kg/kg", "t": "K", "u": "m/s", "v": "m/s", "z": "m$^2$/s$^2$",
}

# Pretty per-variable display names for titles.
LOWRES_PRETTY = {
    "mslp": "mslp", "t2m": "t2m", "u10": "u10", "v10": "v10",
    "q": "q", "t": "T", "u": "u", "v": "v", "z": "z",
}


def make_dataset_cfg(data_loc, valid_dates, hr_size, qpepre_log1p, kept_HR):
    return OmegaConf.create({
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
    })


def build_dataset(data_loc, valid_dates, hr_size, qpepre_log1p, kept_HR):
    cfg = make_dataset_cfg(data_loc, valid_dates, hr_size, qpepre_log1p, kept_HR)
    dataset_cls = dataset_classes["data_loader_rwrf_era5_stable.Dataset"]
    return dataset_cls(cfg, train=False)


def lowres_unit(var):
    return LOWRES_UNITS.get(var, "")


def _panel(ax, field, title, cbar_label):
    """One viridis imshow panel with its own colorbar (per-panel clim because
    the 24 LowRes channels span wildly different physical ranges)."""
    vmin, vmax = float(np.min(field)), float(np.max(field))
    im = ax.imshow(
        field, origin="lower", cmap=thesis_style.FIELD_CMAP,
        vmin=vmin, vmax=vmax, aspect="equal",  # square pixels -> true field ratio
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(cbar_label, fontsize=9)


def plot_lowres_grid(out_path, bg_phys, lr_channels, time_label):
    """Presentation layout: columns = variable groups, rows = pressure levels.

    The fields are portrait (192x96, true ratio kept via aspect='equal'), so a
    6-row x 4-col layout renders as a very tall strip. Transposing to
    4 rows (levels) x 6 cols (Surface, q, T, u, v, z) gives a near-square,
    slide-friendly grid while keeping the full physical structure.
    """
    name_to_idx = {str(c): i for i, c in enumerate(lr_channels)}

    # column groups: first the four single-level surface fields, then one column
    # per upper-air variable holding its four pressure levels (top->bottom:
    # 1000, 850, 500, 250 hPa).
    groups = [("Surface", LOWRES_SURFACE, None)]
    for var in LOWRES_PRESSURE_VARS:
        groups.append((LOWRES_PRETTY[var],
                       [f"{var}{lev}" for lev in LOWRES_LEVELS], var))

    nrows, ncols = len(LOWRES_LEVELS), len(groups)   # 4 x 6
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.7 * ncols, 3.4 * nrows),
                             constrained_layout=True)
    for c, (label, chans, var) in enumerate(groups):
        for r in range(nrows):
            ax = axes[r, c]
            ch = chans[r]
            if ch not in name_to_idx:
                ax.axis("off")
                continue
            field = bg_phys[name_to_idx[ch]]
            if var is None:                       # surface column
                title = LOWRES_PRETTY.get(ch, ch)
                unit = lowres_unit(ch)
            else:                                 # var @ level
                title = f"{LOWRES_PRETTY[var]} @ {LOWRES_LEVELS[r]} hPa"
                unit = lowres_unit(var)
            _panel(ax, field, title, unit)
    fig.suptitle(
        f"ERA5 LowRes conditioning $S_t$ (24 channels)  |  {time_label}",
        fontsize=17,
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[lowres] wrote {out_path}")


def plot_highres_grid(out_path, hr_phys_by_name, time_label):
    """1x4 row of the 4 HighRes target channels (portrait, true ratio)."""
    ncols = len(HIGHRES_CHANNELS)
    fig, axes = plt.subplots(1, ncols, figsize=(3.0 * ncols, 6.2),
                             constrained_layout=True)
    for c, ch in enumerate(HIGHRES_CHANNELS):
        unit = HIGHRES_UNITS[ch]
        _panel(axes[c], hr_phys_by_name[ch], f"{ch}  [{unit}]", unit)
    fig.suptitle(
        f"HighRes target $X_{{t+1}}$ (4 channels)  |  {time_label}",
        fontsize=15,
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[highres] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--hr-size", type=int, nargs=2, default=[192, 96])
    ap.add_argument("--qpepre-log1p", type=lambda s: s.lower() == "true",
                    default=True)
    ap.add_argument("--valid-dates", nargs=2,
                    default=["2022/01/01", "2022/12/31"])
    ap.add_argument("--index", type=int, default=0,
                    help="Validation sample index to render (default 0 = "
                         "scene of qual_qpepre_seq00).")
    ap.add_argument("--use-target", action="store_true", default=True,
                    help="Plot the +1h HighRes target (matches comparison "
                         "'truth' panel). Default on.")
    ap.add_argument("--use-input", dest="use_target", action="store_false",
                    help="Plot the input-time HighRes instead of the target.")
    ap.add_argument("--output-dir", type=Path, required=True)
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

    if args.index >= len(dataset):
        raise SystemExit(f"index {args.index} >= dataset size {len(dataset)}")

    data = dataset[args.index]
    bg = data["background"].cpu().numpy().copy()           # (24, H, W) normalized
    state_sel = 1 if args.use_target else 0
    hr = data["state"][state_sel].cpu().numpy().copy()     # (4, H, W) normalized

    bg_phys = dataset.denormalize_background(bg)
    hr_phys = dataset.denormalize_state(hr)
    hr_phys_by_name = {ch: hr_phys[hr_channels.index(ch)] for ch in HIGHRES_CHANNELS}

    # Time label: target time if plotting the target, else input time.
    base_ts = dataset.valid_samples[args.index]
    from datetime import timedelta
    ts = base_ts + timedelta(hours=dataset.dt) if args.use_target else base_ts
    time_label = ts.strftime("%Y-%m-%d %H:00 UTC")
    print(f"[scene] index={args.index}  target={args.use_target}  ts={time_label}")

    plot_lowres_grid(args.output_dir / "era5_lowres.png",
                     bg_phys, lr_channels, time_label)
    plot_highres_grid(args.output_dir / "highres.png",
                      hr_phys_by_name, time_label)
    print(f"[done] outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
