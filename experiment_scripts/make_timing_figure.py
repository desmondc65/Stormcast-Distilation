#!/usr/bin/env python
"""Timing figure for thesis Chapter 4: per-method inference wall-clock with the
speedup over the EDM baseline (``X vs baseline``) annotated on every bar.

Reads ``results/timing/a6000.csv`` (falls back to ``timing.csv``) and, when it
exists, ``results/timing/h100.csv``. H100-ready: a second grouped-bar series is
added automatically the moment an ``h100.csv`` is dropped in, so re-running this
script after the cluster timing job completes fills the H100 bars with no edits.

    python make_timing_figure.py            # -> des_master_thesis/figures/results/timing_speedup.png
    python make_timing_figure.py --out /some/where.png
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

try:  # match the rest of the thesis figures (uniform typography) if available
    import thesis_style as ts
    ts.apply_rc()
except Exception:  # pragma: no cover - style is cosmetic only
    pass

REPO = Path(__file__).resolve().parents[1]
TIMING = REPO / "experiment_scripts" / "results" / "timing"

ORDER = ["stormcast_edm", "flowcast_nfe10", "flowcast_nfe15",
         "flowcast_nfe20", "meanflow_nfe1", "meanflow_nfe2"]
LABELS = {
    "stormcast_edm": "EDM\n(35 NFE)",
    "flowcast_nfe10": "FlowCast\nK=10", "flowcast_nfe15": "FlowCast\nK=15",
    "flowcast_nfe20": "FlowCast\nK=20",
    "meanflow_nfe1": "MeanFlow\nK=1", "meanflow_nfe2": "MeanFlow\nK=2",
}
# Cool end = baselines, warm end = the average-velocity student (viridis ramp).
POS = {"stormcast_edm": 0.05, "flowcast_nfe10": 0.32, "flowcast_nfe15": 0.42,
       "flowcast_nfe20": 0.52, "meanflow_nfe1": 0.80, "meanflow_nfe2": 0.92}
VIRIDIS = cm.get_cmap("viridis")


def _load(path):
    with path.open() as f:
        return {r["method"]: r for r in csv.DictReader(f)}


def _devices():
    devs = []
    a = TIMING / "a6000.csv"
    a = a if a.exists() else TIMING / "timing.csv"
    if a.exists():
        devs.append(("A6000", _load(a)))
    h = TIMING / "h100.csv"
    if h.exists():
        devs.append(("H100", _load(h)))
    return devs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=REPO / "des_master_thesis" / "figures" / "results" / "timing_speedup.png")
    args = ap.parse_args()

    devs = _devices()
    if not devs:
        raise SystemExit(f"no timing CSVs found under {TIMING}")
    methods = [m for m in ORDER if any(m in rows for _, rows in devs)]

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    nd = len(devs)
    width = 0.8 / nd
    x = np.arange(len(methods))

    for di, (dev, rows) in enumerate(devs):
        base = (float(rows["stormcast_edm"]["sec_per_ensemble_forecast"])
                if "stormcast_edm" in rows else np.nan)
        vals, sps = [], []
        for m in methods:
            if m in rows:
                v = float(rows[m]["sec_per_ensemble_forecast"])
                vals.append(v)
                sps.append(base / v)
            else:
                vals.append(np.nan)
                sps.append(np.nan)
        colors = [VIRIDIS(POS.get(m, 0.5)) for m in methods]
        off = (di - (nd - 1) / 2) * width
        bars = ax.bar(x + off, vals, width=width * 0.92, color=colors,
                      edgecolor="black", linewidth=0.6,
                      hatch=("" if di == 0 else "//"),
                      label=(dev if nd > 1 else None))
        for b, v, sp in zip(bars, vals, sps):
            if np.isnan(v):
                continue
            ax.annotate(f"{sp:.1f}×", (b.get_x() + b.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 3),
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(m, m) for m in methods], fontsize=8)
    ax.set_ylabel("s per 50-member 24 h forecast (log)")
    devnames = " + ".join(d for d, _ in devs)
    ax.set_title(f"Inference cost and speedup vs. EDM baseline ({devnames})")
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.margins(y=0.20)
    if nd > 1:
        ax.legend(title="GPU", fontsize=8)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight", pad_inches=0.06)
    print(f"[make_timing_figure] wrote {args.out}  ({nd} device series: {devnames})")


if __name__ == "__main__":
    main()
