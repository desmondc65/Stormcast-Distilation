#!/usr/bin/env python3
"""Compose the per-step PNGs in a directory into a single overview figure.

Reads ``step_NN.png`` files from ``--step-dir`` (numerically ordered), lays
them out in ``--rows`` rows (default 2), and draws a small ``->`` arrow
between consecutive frames in each row. Rows are independent (no arrow
between row-end and the start of the next row -- read top row first, then
the bottom).

Designed for the per-(timestamp, channel) directories produced by
``inference_steps.py``, e.g.

    results/inference_steps/time_01_2022-01-01_01/stormcast/t2m/

Usage:
    python plot_step_sequence.py \\
        --step-dir <path with step_*.png> \\
        [--output <path/to/sequence.png>] \\
        [--rows 2]
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def find_step_pngs(step_dir):
    files = list(step_dir.glob("step_*.png"))
    if not files:
        raise SystemExit(f"no step_*.png files found in {step_dir}")
    pat = re.compile(r"step_(\d+)")
    def keynum(p):
        m = pat.search(p.name)
        return int(m.group(1)) if m else -1
    return sorted(files, key=keynum)


def plot_sequence(step_dir, output, rows=2):
    step_dir = Path(step_dir)
    files = find_step_pngs(step_dir)
    n = len(files)
    ncols = math.ceil(n / rows)

    # Read first PNG to learn the image aspect (so cells aren't stretched).
    first = plt.imread(files[0])
    H, W = first.shape[:2]
    aspect = W / H            # width per unit height

    # Cell widths: image cells + arrow cells alternating in each row.
    img_w, arr_w = 4.0, 0.6   # in inches
    cell_h = img_w / aspect   # keep image cells the right aspect

    width_ratios = []
    for i in range(ncols):
        width_ratios.append(img_w)
        if i < ncols - 1:
            width_ratios.append(arr_w)
    total_cols = 2 * ncols - 1

    fig_w = sum(width_ratios) + 0.4
    fig_h = rows * cell_h + 0.6 * rows + 0.4

    fig, axes = plt.subplots(
        rows, total_cols,
        figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": width_ratios, "wspace": 0.05, "hspace": 0.25},
        squeeze=False,
    )

    # Hide every cell up front -- we'll re-enable only what we use.
    for ax in axes.flat:
        ax.set_axis_off()

    for idx, fp in enumerate(files):
        r = idx // ncols
        cir = idx % ncols     # column index within the row (image-only)
        c = cir * 2           # axes-grid column
        ax = axes[r, c]
        img = plt.imread(fp)
        ax.imshow(img)
        ax.set_title(f"step {idx + 1}", fontsize=22, pad=6)

        # Right-arrow in the next (arrow) cell, only if there's a same-row
        # neighbour after this one.
        if cir < ncols - 1 and idx + 1 < n and (idx + 1) // ncols == r:
            a = axes[r, c + 1]
            a.annotate(
                "", xy=(0.95, 0.5), xytext=(0.05, 0.5),
                xycoords="axes fraction",
                arrowprops=dict(
                    arrowstyle="->", lw=1.4, color="black",
                    mutation_scale=14,
                ),
            )

    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {output}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--step-dir", type=Path, required=True,
                    help="Directory containing step_NN.png files.")
    ap.add_argument("--output", type=Path, default=None,
                    help="Output PNG (default: <step-dir>/sequence.png).")
    ap.add_argument("--rows", type=int, default=2,
                    help="Number of rows to lay the steps out in (default 2).")
    args = ap.parse_args()

    output = args.output or (args.step_dir / "sequence.png")
    plot_sequence(args.step_dir, output, rows=args.rows)


if __name__ == "__main__":
    main()
