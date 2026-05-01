#!/usr/bin/env python3
"""Validate whether HighRes timestamps listed in missing_rwrf.txt are duplicates
of the previous hour, next hour, both, or neither.

For each missing timestamp:
  - Locate it in the appropriate zarr store (train: 2019-2021, valid: 2022).
  - Load its HighRes slice and the slices for t-1h and t+1h (if they exist).
  - Compare element-wise (exact equality, then close-equality with tolerance).
  - Record the verdict in a CSV.

Usage:
    python check_similarity.py \
        --zarr-root /path/to/zarr_exp3_L_24_H_24_train_2_5_years_full \
        --missing  /path/to/missing_rwrf.txt \
        --out      /path/to/similarity_report.csv
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr


DEFAULT_ROOT = pathlib.Path(
    "/home3/davidlcs/Econ-Rag/Local_LLM/test_meeting/Stormcast-Distilation/"
    "exp_3_train_2_5_yrs_val_1yr_tp1/zarr_exp3_L_24_H_24_train_2_5_years_full"
)


@dataclass
class StoreHandle:
    name: str
    ds: xr.Dataset
    times: np.ndarray  # datetime64[ns]
    time_index: dict[np.datetime64, int]


def parse_timestamp(token: str) -> np.datetime64:
    """missing_rwrf.txt uses 'YYYY-MM-DD_HH' (hourly)."""
    date_part, hour_part = token.strip().split("_")
    return np.datetime64(f"{date_part}T{hour_part.zfill(2)}:00:00", "ns")


def load_store(zarr_path: pathlib.Path, name: str) -> StoreHandle:
    ds = xr.open_zarr(zarr_path, consolidated=True)
    times = ds.time.values.astype("datetime64[ns]")
    return StoreHandle(
        name=name,
        ds=ds,
        times=times,
        time_index={t: i for i, t in enumerate(times)},
    )


def pick_store(ts: np.datetime64, train: StoreHandle, valid: StoreHandle) -> StoreHandle | None:
    if ts in train.time_index:
        return train
    if ts in valid.time_index:
        return valid
    return None


def compare_slices(
    a: np.ndarray, b: np.ndarray, rtol: float = 1e-5, atol: float = 1e-6
) -> tuple[bool, bool, float]:
    """Return (exact_equal, close_equal, max_abs_diff). NaN==NaN treated as equal."""
    a_nan = np.isnan(a)
    b_nan = np.isnan(b)
    nan_match = np.array_equal(a_nan, b_nan)
    finite_mask = ~(a_nan | b_nan)
    if not nan_match:
        diffs = np.abs(np.where(finite_mask, a - b, 0.0))
        return False, False, float(diffs.max() if diffs.size else 0.0)
    if finite_mask.any():
        diffs = np.abs(a[finite_mask] - b[finite_mask])
        max_abs = float(diffs.max())
    else:
        max_abs = 0.0
    exact = max_abs == 0.0
    close = bool(np.allclose(a[finite_mask], b[finite_mask], rtol=rtol, atol=atol))
    return exact, close, max_abs


def verdict(prev_close: bool | None, next_close: bool | None) -> str:
    if prev_close and next_close:
        return "both"
    if prev_close:
        return "previous"
    if next_close:
        return "next"
    if prev_close is None and next_close is None:
        return "no_neighbors"
    if prev_close is None:
        return "next" if next_close else "neither (no_prev)"
    if next_close is None:
        return "previous" if prev_close else "neither (no_next)"
    return "neither"


def process(
    missing_path: pathlib.Path,
    train: StoreHandle,
    valid: StoreHandle,
    out_path: pathlib.Path,
    rtol: float,
    atol: float,
) -> None:
    df = pd.read_csv(missing_path)
    if "timestamp" not in df.columns:
        # The file may simply be a list of timestamps; re-read as plain text.
        with open(missing_path) as fh:
            tokens = [ln.strip() for ln in fh if ln.strip() and ln.strip() != "timestamp"]
    else:
        tokens = [str(t) for t in df["timestamp"].tolist()]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "timestamp",
            "store",
            "time_index",
            "prev_timestamp",
            "next_timestamp",
            "same_as_prev_close",
            "same_as_prev_exact",
            "max_abs_diff_prev",
            "same_as_next_close",
            "same_as_next_exact",
            "max_abs_diff_next",
            "verdict",
        ])

        n_total = len(tokens)
        verdict_counts: dict[str, int] = {}
        for i, tok in enumerate(tokens, 1):
            try:
                ts = parse_timestamp(tok)
            except Exception as exc:
                writer.writerow([tok, "PARSE_ERROR", "", "", "", "", "", "", "", "", "", str(exc)])
                continue

            store = pick_store(ts, train, valid)
            if store is None:
                writer.writerow([tok, "NOT_FOUND", "", "", "", "", "", "", "", "", "", "missing_in_zarr"])
                continue

            idx = store.time_index[ts]
            cur = store.ds["HighRes"].isel(time=idx).values

            prev_ts: np.datetime64 | None = None
            next_ts: np.datetime64 | None = None
            prev_exact = next_exact = None
            prev_close = next_close = None
            prev_diff = next_diff = ""

            if idx - 1 >= 0:
                prev_ts = store.times[idx - 1]
                prev_arr = store.ds["HighRes"].isel(time=idx - 1).values
                prev_exact, prev_close, prev_diff_val = compare_slices(cur, prev_arr, rtol, atol)
                prev_diff = f"{prev_diff_val:.6g}"

            if idx + 1 < store.times.size:
                next_ts = store.times[idx + 1]
                next_arr = store.ds["HighRes"].isel(time=idx + 1).values
                next_exact, next_close, next_diff_val = compare_slices(cur, next_arr, rtol, atol)
                next_diff = f"{next_diff_val:.6g}"

            v = verdict(prev_close, next_close)
            verdict_counts[v] = verdict_counts.get(v, 0) + 1

            writer.writerow([
                tok,
                store.name,
                idx,
                np.datetime_as_string(prev_ts, unit="h") if prev_ts is not None else "",
                np.datetime_as_string(next_ts, unit="h") if next_ts is not None else "",
                "" if prev_close is None else bool(prev_close),
                "" if prev_exact is None else bool(prev_exact),
                prev_diff,
                "" if next_close is None else bool(next_close),
                "" if next_exact is None else bool(next_exact),
                next_diff,
                v,
            ])

            prev_diff_repr = prev_diff if prev_diff != "" else "-"
            next_diff_repr = next_diff if next_diff != "" else "-"
            print(
                f"  [{i:>4d}/{n_total}] {tok}  store={store.name:5s} idx={idx:>6d}  "
                f"prev_close={str(prev_close):>5s} (Δ={prev_diff_repr})  "
                f"next_close={str(next_close):>5s} (Δ={next_diff_repr})  "
                f"-> {v}",
                flush=True,
            )

    print("\nSummary (rtol={}, atol={}):".format(rtol, atol))
    for k in sorted(verdict_counts):
        print(f"  {k:>22s}: {verdict_counts[k]}")
    print(f"  {'total':>22s}: {sum(verdict_counts.values())}")
    print(f"\nReport written to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr-root", type=pathlib.Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--missing",
        type=pathlib.Path,
        default=DEFAULT_ROOT / "missing_rwrf.txt",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path(__file__).parent / "missing_rwrf_similarity.csv",
    )
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()

    train_path = args.zarr_root / "HighRes" / "stormcast_test_train.zarr"
    valid_path = args.zarr_root / "HighRes" / "stormcast_test_valid.zarr"
    for p in (train_path, valid_path, args.missing):
        if not p.exists():
            print(f"ERROR: missing path {p}", file=sys.stderr)
            sys.exit(1)

    print(f"Opening train zarr: {train_path}")
    train = load_store(train_path, "train")
    print(f"  -> {train.times.size} steps "
          f"[{train.times[0]} ... {train.times[-1]}]")

    print(f"Opening valid zarr: {valid_path}")
    valid = load_store(valid_path, "valid")
    print(f"  -> {valid.times.size} steps "
          f"[{valid.times[0]} ... {valid.times[-1]}]")

    print(f"Reading missing list: {args.missing}")
    process(args.missing, train, valid, args.out, args.rtol, args.atol)


if __name__ == "__main__":
    main()
