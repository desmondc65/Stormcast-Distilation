#!/usr/bin/env python3
"""Scan the LowRes (ERA5) zarr stores for hours whose data duplicates a neighbor.

For every hour t we check whether ``LowRes[t]`` matches ``LowRes[t-1]`` and/or
``LowRes[t+1]`` element-wise. The pair-wise diff is computed once per adjacent
pair, then unrolled into per-hour verdicts (``previous`` / ``next`` / ``both``).
The scan is parallelised with ``multiprocessing``: the time axis is split into
contiguous blocks and each worker reads its block + 1 trailing hour, computes
pair max-abs differences in memory, and returns them.

Usage:
    python check_era5_similar.py [--workers 8] [--block-size 400] [--all-rows]
                                 [--rtol 1e-5] [--atol 1e-6]

By default only hours that have a similar neighbor are written to the CSV.
Pass ``--all-rows`` to dump every hour.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import pathlib
import sys
import time
from dataclasses import dataclass

import numpy as np
import xarray as xr
import zarr


DEFAULT_ROOT = pathlib.Path(
    "/home3/davidlcs/Econ-Rag/Local_LLM/test_meeting/Stormcast-Distilation/"
    "exp_3_train_2_5_yrs_val_1yr_tp1/zarr_exp3_L_24_H_24_train_2_5_years_full"
)


@dataclass
class BlockJob:
    store_name: str
    zarr_path: str
    start: int          # inclusive index of first "current" hour
    end: int            # exclusive index — last current hour is end-1
    n_total: int        # total time length of the store
    channel_idx: int    # integer index along the channel axis
    raw_array_path: str # subpath inside the .zarr to the bare 4D array, "" if xarray-style


def _open_zarr_safe(zarr_path: str) -> xr.Dataset:
    """Try consolidated metadata first; fall back to non-consolidated."""
    try:
        return xr.open_zarr(zarr_path, consolidated=True)
    except (KeyError, ValueError):
        return xr.open_zarr(zarr_path, consolidated=False)


def _block_worker(job: BlockJob) -> tuple[BlockJob, np.ndarray]:
    """Read [start, end+1) for a single channel and return max-abs-diff for each
    adjacent pair (i, i+1) where i in [start, end)."""
    end_inclusive = min(job.end + 1, job.n_total)
    if job.raw_array_path:
        arr = zarr.open(f"{job.zarr_path}/{job.raw_array_path}", mode="r")
        block = arr[job.start:end_inclusive, job.channel_idx]
    else:
        ds = _open_zarr_safe(job.zarr_path)
        block = (
            ds["LowRes"]
            .isel(channel=job.channel_idx, time=slice(job.start, end_inclusive))
            .values
        )
        ds.close()

    n_pairs = block.shape[0] - 1
    if n_pairs <= 0:
        return job, np.empty(0, dtype=np.float64)

    a = block[:-1]
    b = block[1:]
    nan_a = np.isnan(a)
    nan_b = np.isnan(b)
    nan_match = (nan_a == nan_b).reshape(n_pairs, -1).all(axis=1)
    abs_diff = np.where(nan_a | nan_b, 0.0, np.abs(a - b))
    pair_max = abs_diff.reshape(n_pairs, -1).max(axis=1).astype(np.float64)
    diffs = np.where(nan_match, pair_max, np.inf)
    return job, diffs


def _resolve_layout(
    zarr_path: pathlib.Path,
    channel: str | None,
    channel_index: int | None,
    start_date: str,
    data_var: str,
) -> tuple[np.ndarray, int, int, str]:
    """Return (times[T], n_timesteps, channel_idx, raw_array_path).

    Tries xarray (named channels + time coord) first, falls back to a bare
    raw-zarr 4D array under ``zarr_path/<data_var>``.
    """
    # 1) xarray-friendly path
    try:
        ds = _open_zarr_safe(str(zarr_path))
    except Exception:
        ds = None

    if ds is not None and data_var in ds.data_vars:
        try:
            times = ds.time.values.astype("datetime64[ns]")
            n = times.size
            available = [str(c) for c in ds.channel.values]
            if channel_index is not None:
                if not (0 <= channel_index < len(available)):
                    raise ValueError(
                        f"channel-index {channel_index} out of range [0, {len(available)})"
                    )
                ch_idx = channel_index
                ch_name = available[ch_idx]
            else:
                if channel not in available:
                    raise ValueError(f"channel {channel!r} not in {available}")
                ch_idx = available.index(channel)
                ch_name = channel
            print(f"  layout=xarray, channel={ch_name!r} (idx {ch_idx})")
            return times, n, ch_idx, ""
        finally:
            ds.close()

    # 2) raw zarr fallback: <zarr_path>/<data_var> is a bare 4D array
    inner = pathlib.Path(zarr_path) / data_var
    if not inner.exists():
        raise FileNotFoundError(
            f"Could not open {zarr_path} as xarray, and no raw array at {inner}"
        )
    arr = zarr.open(str(inner), mode="r")
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D array at {inner}, got shape {arr.shape}")
    n = arr.shape[0]
    n_channels = arr.shape[1]
    if channel_index is None:
        raise ValueError(
            f"Store at {zarr_path} has no channel-name metadata; pass --channel-index "
            f"(0..{n_channels - 1}) to pick a channel."
        )
    if not (0 <= channel_index < n_channels):
        raise ValueError(
            f"channel-index {channel_index} out of range [0, {n_channels})"
        )
    times = np.arange(n, dtype="timedelta64[h]") + np.datetime64(start_date, "h")
    times = times.astype("datetime64[ns]")
    print(f"  layout=raw-zarr, channel-index={channel_index}, "
          f"synthesized times start={start_date}")
    return times, n, channel_index, data_var


def _scan_store(
    store_name: str,
    zarr_path: pathlib.Path,
    block_size: int,
    workers: int,
    channel: str | None,
    channel_index: int | None,
    start_date: str,
    data_var: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (times[T], pair_max_diff[T-1])."""
    times, n, ch_idx, raw_path = _resolve_layout(
        zarr_path, channel, channel_index, start_date, data_var
    )

    jobs: list[BlockJob] = []
    for start in range(0, n - 1, block_size):
        end = min(start + block_size, n - 1)
        jobs.append(BlockJob(store_name, str(zarr_path), start, end, n, ch_idx, raw_path))

    pair_diff = np.full(n - 1, np.nan, dtype=np.float64)
    print(f"\n[{store_name}] {n} hours, {len(jobs)} blocks, {workers} workers")
    t0 = time.time()

    if workers <= 1:
        for j, jb in enumerate(jobs, 1):
            _job, diffs = _block_worker(jb)
            pair_diff[_job.start:_job.start + diffs.size] = diffs
            print(f"  [{j:>3d}/{len(jobs)}] block {_job.start}-{_job.end}  "
                  f"max_pair_diff={diffs.max() if diffs.size else 0:.4g}",
                  flush=True)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            for j, (job, diffs) in enumerate(
                pool.imap_unordered(_block_worker, jobs), 1
            ):
                pair_diff[job.start:job.start + diffs.size] = diffs
                print(f"  [{j:>3d}/{len(jobs)}] block {job.start}-{job.end}  "
                      f"max_pair_diff={diffs.max() if diffs.size else 0:.4g}",
                      flush=True)
    print(f"[{store_name}] scan done in {time.time() - t0:.1f}s")
    return times, pair_diff


def _write_csv(
    out_path: pathlib.Path,
    results: list[tuple[str, np.ndarray, np.ndarray]],
    rtol: float,
    atol: float,
    write_all: bool,
) -> dict[str, int]:
    """For hour t: same_as_prev = pair_diff[t-1] small; same_as_next = pair_diff[t] small."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    threshold = atol + rtol  # max-abs-diff "is_close" cutoff

    def _is_close(d: float | None) -> bool | None:
        if d is None:
            return None
        if not np.isfinite(d):
            return False
        return d <= threshold

    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "timestamp",
            "store",
            "time_index",
            "prev_timestamp",
            "next_timestamp",
            "same_as_prev",
            "same_as_next",
            "max_abs_diff_prev",
            "max_abs_diff_next",
            "verdict",
        ])

        for store_name, times, pair_diff in results:
            T = times.size
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

                ts = times[t]
                ts_prev = times[t - 1] if t - 1 >= 0 else None
                ts_next = times[t + 1] if t + 1 < T else None
                ts_str = np.datetime_as_string(ts, unit="h").replace("T", "_")
                w.writerow([
                    ts_str,
                    store_name,
                    t,
                    np.datetime_as_string(ts_prev, unit="h").replace("T", "_") if ts_prev is not None else "",
                    np.datetime_as_string(ts_next, unit="h").replace("T", "_") if ts_next is not None else "",
                    "" if same_prev is None else bool(same_prev),
                    "" if same_next is None else bool(same_next),
                    "" if d_prev is None else f"{d_prev:.6g}",
                    "" if d_next is None else f"{d_next:.6g}",
                    v,
                ])

                if v in ("previous", "next", "both"):
                    print(
                        f"  SIMILAR  {ts_str}  store={store_name:5s} idx={t:>6d}  "
                        f"Δprev={'-' if d_prev is None else f'{d_prev:.4g}'}  "
                        f"Δnext={'-' if d_next is None else f'{d_next:.4g}'}  "
                        f"-> {v}",
                        flush=True,
                    )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr-root", type=pathlib.Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path(__file__).parent / "lowres_similarity.csv",
    )
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2))
    parser.add_argument(
        "--block-size",
        type=int,
        default=400,
        help="hours per worker block (memory ~ block * 24 * 224 * 128 * 4 bytes ~ 2.75 MB/hr)",
    )
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help="write every hour, not just those flagged similar to a neighbor",
    )
    parser.add_argument("--store", choices=["train", "valid", "both"], default="both")
    parser.add_argument(
        "--channel",
        type=str,
        default="t2m",
        help="LowRes channel to compare (default: t2m). Used when the store has named channels.",
    )
    parser.add_argument(
        "--channel-index",
        type=int,
        default=None,
        help="Override --channel by integer index along the channel axis. Required for "
             "bare zarr stores that lack channel-name metadata.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2019-08-01T00",
        help="Used to synthesize the time axis for stores without a time coordinate.",
    )
    parser.add_argument(
        "--data-var",
        type=str,
        default="LowRes",
        help="Name of the 4D data variable / inner array path (default: LowRes).",
    )
    args = parser.parse_args()

    paths = {
        "train": args.zarr_root / "LowRes" / "stormcast_test_train.zarr",
        "valid": args.zarr_root / "LowRes" / "stormcast_test_valid.zarr",
    }
    if args.store == "both":
        targets = [t for t in ("train", "valid") if paths[t].exists()]
        if not targets:
            print(f"ERROR: neither train nor valid zarr present under {args.zarr_root}/LowRes",
                  file=sys.stderr)
            sys.exit(1)
        for t in ("train", "valid"):
            if not paths[t].exists():
                print(f"  NOTE: skipping missing store {paths[t]}")
    else:
        targets = [args.store]
        if not paths[args.store].exists():
            print(f"ERROR: missing zarr {paths[args.store]}", file=sys.stderr)
            sys.exit(1)

    if args.channel_index is not None:
        print(f"Comparing channel-index: {args.channel_index}")
    else:
        print(f"Comparing channel: {args.channel!r}")
    results = []
    for t in targets:
        times, pair_diff = _scan_store(
            store_name=t,
            zarr_path=paths[t],
            block_size=args.block_size,
            workers=args.workers,
            channel=args.channel,
            channel_index=args.channel_index,
            start_date=args.start_date,
            data_var=args.data_var,
        )
        results.append((t, times, pair_diff))

    print(f"\nWriting CSV: {args.out}")
    counts = _write_csv(args.out, results, args.rtol, args.atol, args.all_rows)

    print("\nSummary (rtol={}, atol={}):".format(args.rtol, args.atol))
    for k in sorted(counts):
        print(f"  {k:>22s}: {counts[k]}")
    print(f"  {'total':>22s}: {sum(counts.values())}")
    print(f"\nReport written to: {args.out}")


if __name__ == "__main__":
    main()
