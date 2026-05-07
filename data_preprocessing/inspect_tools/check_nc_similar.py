#!/usr/bin/env python3
"""Scan a directory of per-hour NetCDF files and report hours whose data
duplicates an adjacent hour.

Files are expected to be named ``<var>_YYYYMMDDTHH.nc`` (e.g. ``t2m_20190801T00.nc``)
where each file holds a single hour of one variable. The script:

  1. Groups files by variable.
  2. Sorts each group by timestamp.
  3. For every adjacent pair (t, t+1) computes the max-abs-diff once.
  4. Per-hour verdict (``previous`` / ``next`` / ``both``) is derived from those pair diffs.

Multiprocessing splits each variable's timeline into contiguous blocks; each
worker reads its block + one trailing file, computes pair diffs in NumPy, and
returns them.

Usage:
    python check_nc_similar.py [--nc-root /path/to/nc] [--workers 8]
                               [--block-size 200] [--all-rows] [--var t2m]
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import pathlib
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import xarray as xr


DEFAULT_NC_ROOT = pathlib.Path(
    "/home3/davidlcs/Econ-Rag/Local_LLM/test_meeting/Stormcast-Distilation/"
    "exp_3_train_2_5_yrs_val_1yr_tp1/nc"
)

# Filename pattern: <var>_YYYYMMDDTHH.nc  (var may contain letters/digits/underscores)
NC_NAME_RE = re.compile(r"^(?P<var>[A-Za-z0-9]+)_(?P<ts>\d{8}T\d{2})\.nc$")


@dataclass
class FileEntry:
    var: str
    timestamp: np.datetime64
    path: pathlib.Path


@dataclass
class BlockJob:
    var: str
    start: int                 # index into the var's sorted file list
    end: int                   # exclusive — last "current" hour is end-1
    n_total: int
    paths: list[str]           # full sorted path list for this variable


def _parse_files(nc_root: pathlib.Path) -> dict[str, list[FileEntry]]:
    groups: dict[str, list[FileEntry]] = defaultdict(list)
    for p in sorted(nc_root.iterdir()):
        if not p.is_file() or p.suffix != ".nc":
            continue
        m = NC_NAME_RE.match(p.name)
        if not m:
            print(f"  WARN: skip unrecognised filename {p.name}", file=sys.stderr)
            continue
        ts_token = m.group("ts")  # YYYYMMDDTHH
        ts = np.datetime64(
            f"{ts_token[0:4]}-{ts_token[4:6]}-{ts_token[6:8]}T{ts_token[9:11]}:00:00",
            "ns",
        )
        groups[m.group("var")].append(FileEntry(m.group("var"), ts, p))
    for var in groups:
        groups[var].sort(key=lambda e: e.timestamp)
    return groups


def _load_one(path: str, var: str) -> np.ndarray:
    """Open a single nc file and return the variable as a 2D/3D array (squeezed)."""
    ds = xr.open_dataset(path)
    arr = ds[var].values
    ds.close()
    return np.squeeze(arr)


def _block_worker(job: BlockJob) -> tuple[BlockJob, np.ndarray]:
    """Read files [start, end+1) and return per-pair max-abs-diff."""
    end_inclusive = min(job.end + 1, job.n_total)
    arrays = [_load_one(job.paths[i], job.var) for i in range(job.start, end_inclusive)]
    if len(arrays) < 2:
        return job, np.empty(0, dtype=np.float64)

    block = np.stack(arrays, axis=0)
    n_pairs = block.shape[0] - 1
    a = block[:-1]
    b = block[1:]
    nan_a = np.isnan(a)
    nan_b = np.isnan(b)
    nan_match = (nan_a == nan_b).reshape(n_pairs, -1).all(axis=1)
    abs_diff = np.where(nan_a | nan_b, 0.0, np.abs(a - b))
    pair_max = abs_diff.reshape(n_pairs, -1).max(axis=1).astype(np.float64)
    return job, np.where(nan_match, pair_max, np.inf)


def _scan_var(
    var: str,
    entries: list[FileEntry],
    block_size: int,
    workers: int,
) -> np.ndarray:
    """Return pair_max_diff[N-1] for the sorted entry list."""
    n = len(entries)
    if n < 2:
        return np.empty(0, dtype=np.float64)

    paths = [str(e.path) for e in entries]
    jobs: list[BlockJob] = []
    for start in range(0, n - 1, block_size):
        end = min(start + block_size, n - 1)
        jobs.append(BlockJob(var, start, end, n, paths))

    pair_diff = np.full(n - 1, np.nan, dtype=np.float64)
    print(f"\n[{var}] {n} files, {len(jobs)} blocks, {workers} workers")
    t0 = time.time()

    if workers <= 1:
        for j, jb in enumerate(jobs, 1):
            _job, diffs = _block_worker(jb)
            pair_diff[_job.start:_job.start + diffs.size] = diffs
            print(
                f"  [{j:>3d}/{len(jobs)}] block {_job.start}-{_job.end}  "
                f"max_pair_diff={diffs.max() if diffs.size else 0:.4g}",
                flush=True,
            )
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            for j, (job, diffs) in enumerate(
                pool.imap_unordered(_block_worker, jobs), 1
            ):
                pair_diff[job.start:job.start + diffs.size] = diffs
                print(
                    f"  [{j:>3d}/{len(jobs)}] block {job.start}-{job.end}  "
                    f"max_pair_diff={diffs.max() if diffs.size else 0:.4g}",
                    flush=True,
                )
    print(f"[{var}] scan done in {time.time() - t0:.1f}s")
    return pair_diff


def _write_csv(
    out_path: pathlib.Path,
    results: list[tuple[str, list[FileEntry], np.ndarray]],
    rtol: float,
    atol: float,
    write_all: bool,
) -> dict[str, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    threshold = atol + rtol

    def _is_close(d):
        if d is None:
            return None
        if not np.isfinite(d):
            return False
        return d <= threshold

    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "timestamp",
            "variable",
            "file",
            "file_index",
            "prev_timestamp",
            "next_timestamp",
            "same_as_prev",
            "same_as_next",
            "max_abs_diff_prev",
            "max_abs_diff_next",
            "verdict",
        ])
        for var, entries, pair_diff in results:
            T = len(entries)
            for t in range(T):
                d_prev = float(pair_diff[t - 1]) if t - 1 >= 0 else None
                d_next = float(pair_diff[t]) if t < T - 1 else None

                same_prev = _is_close(d_prev)
                same_next = _is_close(d_next)

                if same_prev and same_next:
                    v = "both"
                elif same_prev:
                    v = "previous"
                elif same_next:
                    v = "next"
                elif same_prev is None and same_next is None:
                    v = "no_neighbors"
                elif same_prev is None:
                    v = "neither (no_prev)"
                elif same_next is None:
                    v = "neither (no_next)"
                else:
                    v = "neither"

                counts[v] = counts.get(v, 0) + 1
                if not write_all and v not in ("previous", "next", "both"):
                    continue

                e = entries[t]
                ts_str = np.datetime_as_string(e.timestamp, unit="h").replace("T", "_")
                ts_prev = (
                    np.datetime_as_string(entries[t - 1].timestamp, unit="h").replace("T", "_")
                    if t - 1 >= 0 else ""
                )
                ts_next = (
                    np.datetime_as_string(entries[t + 1].timestamp, unit="h").replace("T", "_")
                    if t + 1 < T else ""
                )
                w.writerow([
                    ts_str,
                    var,
                    e.path.name,
                    t,
                    ts_prev,
                    ts_next,
                    "" if same_prev is None else bool(same_prev),
                    "" if same_next is None else bool(same_next),
                    "" if d_prev is None else f"{d_prev:.6g}",
                    "" if d_next is None else f"{d_next:.6g}",
                    v,
                ])

                if v in ("previous", "next", "both"):
                    print(
                        f"  SIMILAR  {ts_str}  var={var}  idx={t:>5d}  "
                        f"Δprev={'-' if d_prev is None else f'{d_prev:.4g}'}  "
                        f"Δnext={'-' if d_next is None else f'{d_next:.4g}'}  "
                        f"-> {v}",
                        flush=True,
                    )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nc-root", type=pathlib.Path, default=DEFAULT_NC_ROOT)
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path(__file__).parent / "nc_similarity.csv",
    )
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2))
    parser.add_argument(
        "--block-size",
        type=int,
        default=200,
        help="files per worker block (memory ~ block * field_size_bytes)",
    )
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help="write every hour, not just those flagged similar to a neighbor",
    )
    parser.add_argument(
        "--var",
        type=str,
        default=None,
        help="restrict to this variable (default: all variables found in --nc-root)",
    )
    args = parser.parse_args()

    if not args.nc_root.is_dir():
        print(f"ERROR: nc-root not a directory: {args.nc_root}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {args.nc_root}")
    groups = _parse_files(args.nc_root)
    if not groups:
        print("ERROR: no .nc files matched the expected naming pattern", file=sys.stderr)
        sys.exit(1)

    print("Variables found:")
    for v, entries in groups.items():
        print(f"  {v}: {len(entries)} files  "
              f"[{entries[0].timestamp} ... {entries[-1].timestamp}]")

    targets = [args.var] if args.var else list(groups.keys())
    for v in targets:
        if v not in groups:
            print(f"ERROR: variable {v!r} not found", file=sys.stderr)
            sys.exit(1)

    results = []
    for v in targets:
        pair_diff = _scan_var(
            var=v,
            entries=groups[v],
            block_size=args.block_size,
            workers=args.workers,
        )
        results.append((v, groups[v], pair_diff))

    print(f"\nWriting CSV: {args.out}")
    counts = _write_csv(args.out, results, args.rtol, args.atol, args.all_rows)

    print("\nSummary (rtol={}, atol={}):".format(args.rtol, args.atol))
    for k in sorted(counts):
        print(f"  {k:>22s}: {counts[k]}")
    print(f"  {'total':>22s}: {sum(counts.values())}")
    print(f"\nReport written to: {args.out}")


if __name__ == "__main__":
    main()
