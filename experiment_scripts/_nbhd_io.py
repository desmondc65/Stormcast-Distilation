"""Shared IO + schema helpers for the neighbourhood precip metrics.

``compare_diffusion_vs_flowcast.py`` now scores CSI/POD/FAR/HSS/FSS at two
neighbourhood kernels on the ~2 km RWRF grid:

    10km     = 5x5  px  (half-width w=2)  ~= 10 km x 10 km
    0p25deg  = 13x13 px  (half-width w=6)  ~= 0.25 deg (ERA5 grid box)

This changed two CSV schemas the figure/results scripts consume:

  * ``per_threshold.csv`` gained a ``kernel`` column:
        new: method,kernel,metric,<thresholds...>
        old: method,metric,<thresholds...>            (single-pixel, pre-2026-06-15)
  * the FSS detail moved from ``fss_p16.csv`` (columns = neighbourhood half-widths)
    to ``fss_nbhd.csv`` (columns = thresholds, one row per kernel).

The readers below auto-detect which schema a directory holds so every
downstream script renders BOTH the legacy single-pixel result dirs and the new
kerneled ones without a coordinated re-run. ``HEADLINE_KERNEL`` is the kernel
used wherever a single panel / column is forced (the thesis "as ERA5" framing).
"""
from __future__ import annotations

import csv
from pathlib import Path

# Canonical kernel order + display labels. Mirrors NBHD_KERNELS / NBHD_PRETTY in
# compare_diffusion_vs_flowcast.py.
KERNELS = ["10km", "0p25deg"]
KERNEL_PRETTY = {"10km": "10 km", "0p25deg": "0.25°"}
# Pixel side length per kernel (for axis labels / captions).
KERNEL_PX = {"10km": 5, "0p25deg": 13}
# Where a single value/figure is forced, report the ERA5-scale 0.25 deg kernel.
HEADLINE_KERNEL = "0p25deg"


def kernel_pretty(label: str | None) -> str:
    if label is None:
        return "single-pixel"
    return KERNEL_PRETTY.get(label, label)


def base_metric(name: str) -> str:
    """Strip a ``@<kernel>`` suffix and the up/down sort arrows -> base name.

    ``"CSI-M@0.25°↑"`` -> ``"CSI-M"``; ``"RMSE_qpepre↓"`` -> ``"RMSE_qpepre"``.
    """
    s = name.replace("↑", "").replace("↓", "").strip()
    return s.split("@", 1)[0].strip()


def read_per_threshold(path: str | Path):
    """Read a ``per_threshold.csv`` (either schema).

    Returns ``(thresholds, records)`` where ``thresholds`` is ``list[float]`` and
    ``records`` is a list of dicts ``{method, kernel, metric, values}``.
    ``kernel`` is the kernel label for the new schema or ``None`` for the legacy
    single-pixel schema. ``values`` is the raw list of per-threshold strings.
    """
    path = Path(path)
    with path.open() as f:
        rows = list(csv.reader(f))
    header, data = rows[0], rows[1:]
    has_kernel = len(header) > 1 and header[1].strip().lower() == "kernel"
    if has_kernel:
        thresholds = [float(t) for t in header[3:]]
        records = [
            {"method": r[0], "kernel": r[1], "metric": r[2], "values": r[3:]}
            for r in data
        ]
    else:
        thresholds = [float(t) for t in header[2:]]
        records = [
            {"method": r[0], "kernel": None, "metric": r[1], "values": r[2:]}
            for r in data
        ]
    return thresholds, records


def read_fss(dirpath: str | Path):
    """Read ``fss_nbhd.csv`` (new) or ``fss_p16.csv`` (legacy) from a leg dir.

    Returns ``(mode, xs, records)``:
      * ``mode == "kernel"``: ``xs`` = ``list[float]`` thresholds (mm/h);
        ``records`` = dicts ``{method, kernel, values}``.
      * ``mode == "window"``: ``xs`` = ``list[int]`` neighbourhood half-widths;
        ``records`` = dicts ``{method, metric, values}`` (legacy fss_p16.csv).
      * ``(None, None, None)`` if neither file exists.
    """
    dirpath = Path(dirpath)
    new = dirpath / "fss_nbhd.csv"
    old = dirpath / "fss_p16.csv"
    if new.exists():
        with new.open() as f:
            rows = list(csv.reader(f))
        header, data = rows[0], rows[1:]
        thresholds = [float(t) for t in header[2:]]
        records = [{"method": r[0], "kernel": r[1], "values": r[2:]} for r in data]
        return "kernel", thresholds, records
    if old.exists():
        with old.open() as f:
            rows = list(csv.reader(f))
        header, data = rows[0], rows[1:]
        scales = [int(s) for s in header[2:]]
        records = [{"method": r[0], "metric": r[1], "values": r[2:]} for r in data]
        return "window", scales, records
    return None, None, None


def kernels_present(records) -> list:
    """Ordered, de-duplicated kernel labels appearing in per-threshold/fss records.

    Falls back to ``[None]`` for legacy single-pixel records so callers can loop
    uniformly over ``kernels_present(...)``.
    """
    seen = []
    for r in records:
        k = r.get("kernel")
        if k not in seen:
            seen.append(k)
    if seen == [None]:
        return [None]
    ordered = [k for k in KERNELS if k in seen]
    ordered += [k for k in seen if k is not None and k not in ordered]
    return ordered or [None]
