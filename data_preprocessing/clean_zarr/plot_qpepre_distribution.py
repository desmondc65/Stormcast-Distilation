#!/usr/bin/env python3
"""Compare the qpepre distribution before vs after the clean_zarr transform.

The cleaning pipeline applies ``max(0, x)`` then ``log1p(x)`` to qpepre
(mm/h), which is the standard precipitation pre-processing used by
CorrDiff / NowcastNet / DGMR. This script samples both the source zarr (raw
qpepre, mm/h) and the cleaned zarr (log1p(qpepre)) and writes a side-by-side
plot to ``plot/qpepre_distribution.png`` showing:

  * raw histogram (mm/h, log-y axis) -- heavy-tailed, mostly 0
  * raw histogram of NEGATIVE values  -- numerical noise we clip away
  * post-transform histogram          -- compact, near-Gaussian wet portion
  * survival curve of wet pixels in raw mm/h

A small text panel reports the per-channel mean / std before and after.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "exp_3_train_2_5_yrs_val_1yr_tp1/zarr_exp3_L_24_H_24_train_2_5_years_full"
DST = ROOT / "exp_3_train_2_5_yrs_val_1yr_tp1/zarr_exp3_L_24_H_24_train_2_5_years_full_cleaned_4_27_2026"
OUT = Path(__file__).resolve().parent / "plot" / "qpepre_distribution.png"

SAMPLE_HOURS = 2000  # randomly pick this many timesteps to sample (full = 21,216)


def sample_qpepre(zarr_path: Path, n_hours: int = SAMPLE_HOURS) -> np.ndarray:
    print(f"sampling qpepre from {zarr_path}")
    ds = xr.open_zarr(zarr_path / "stormcast_test_train.zarr", consolidated=True)
    arr = ds["HighRes"].sel(channel="qpepre")
    n_total = arr.sizes["time"]
    rng = np.random.default_rng(42)
    idx = np.sort(rng.choice(n_total, size=min(n_hours, n_total), replace=False))
    sampled = arr.isel(time=idx).values  # (n, y, x)
    ds.close()
    return sampled.ravel()


def main() -> int:
    raw = sample_qpepre(SRC / "HighRes")             # mm/h, may be negative
    new = sample_qpepre(DST / "HighRes")             # log1p(max(0, mm/h))

    # Stats
    raw_neg = raw[raw < 0]
    raw_wet = raw[raw > 0]
    print(f"raw  : n={raw.size:,}  mean={raw.mean():.4f}  std={raw.std():.4f}  "
          f"min={raw.min():.4f}  max={raw.max():.4f}  "
          f"negatives={raw_neg.size}  wet (>0)={raw_wet.size}  "
          f"wet fraction={raw_wet.size/raw.size:.4%}")
    print(f"log1p: n={new.size:,}  mean={new.mean():.4f}  std={new.std():.4f}  "
          f"min={new.min():.4f}  max={new.max():.4f}")

    # Cross-check: the inverse transform should recover the clipped raw values.
    inverted_max = float(np.expm1(new.max()))
    print(f"expm1(log1p_max) = {inverted_max:.2f} mm/h (raw max after clip = "
          f"{float(raw[raw >= 0].max()):.2f})")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    # (0,0) raw histogram, log-y, mm/h
    ax = axes[0, 0]
    bins = np.linspace(0, max(raw_wet.max(), 1.0), 80)
    ax.hist(raw_wet, bins=bins, color="#1f6fb4", edgecolor="white")
    ax.set_yscale("log")
    ax.set_xlabel("qpepre (mm/h)")
    ax.set_ylabel("# pixels (log)")
    ax.set_title(f"BEFORE: raw wet pixels  (n={raw_wet.size:,}, "
                 f"{raw_wet.size/raw.size:.2%} of all)")

    # (0,1) negatives we clip
    ax = axes[0, 1]
    if raw_neg.size:
        ax.hist(raw_neg, bins=80, color="#c0392b", edgecolor="white")
        ax.set_yscale("log")
        ax.set_xlabel("qpepre (mm/h, negative only)")
        ax.set_ylabel("# pixels (log)")
        ax.set_title(f"BEFORE: negative artifacts to clip  (n={raw_neg.size:,}, "
                     f"min={raw_neg.min():.4f})")
    else:
        ax.text(0.5, 0.5, "no negative values found", ha="center", va="center")
        ax.set_axis_off()

    # (1,0) post-transform histogram
    ax = axes[1, 0]
    new_wet = new[new > 0]
    bins = np.linspace(0, new.max(), 80)
    ax.hist(new_wet, bins=bins, color="#27ae60", edgecolor="white")
    ax.set_yscale("log")
    ax.set_xlabel("log1p(qpepre)")
    ax.set_ylabel("# pixels (log)")
    ax.set_title(f"AFTER: log1p(max(0, qpepre))  "
                 f"(mean={new.mean():.4f}, std={new.std():.4f})")
    # Annotate landmarks: 1 mm/h, 10, 100
    for v in (1.0, 10.0, 100.0):
        ax.axvline(np.log1p(v), color="black", lw=0.6, linestyle="--", alpha=0.5)
        ax.text(np.log1p(v), 0.9 * ax.get_ylim()[1], f"{v:g} mm/h",
                rotation=90, va="top", ha="right", fontsize=8)

    # (1,1) survival / quantiles
    ax = axes[1, 1]
    qs = np.linspace(0.5, 0.99999, 200)
    pcts_raw = np.quantile(raw_wet, qs)
    pcts_new = np.quantile(new_wet, qs)
    ax2 = ax.twinx()
    ax.plot(1.0 - qs, pcts_raw, color="#1f6fb4", label="raw mm/h (left)")
    ax2.plot(1.0 - qs, pcts_new, color="#27ae60", label="log1p (right)")
    ax.set_xscale("log")
    ax.set_xlabel("Exceedance probability over wet pixels")
    ax.set_ylabel("qpepre (mm/h)")
    ax2.set_ylabel("log1p(qpepre)")
    ax.set_title("Tail behaviour of the wet portion")
    ax.invert_xaxis()
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left")

    fig.suptitle("qpepre pre-processing -- raw mm/h vs log1p(max(0, x))", fontsize=14)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120)
    plt.close(fig)
    print(f"wrote {OUT}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
