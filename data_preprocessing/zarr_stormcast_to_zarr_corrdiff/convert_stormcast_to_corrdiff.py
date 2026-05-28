#!/usr/bin/env python3
"""Convert StormCast split Zarr stores into CorrDiff-compatible Zarr stores.

This converter reads StormCast datasets from:
- LowRes/stormcast_test_{train,valid}.zarr
- HighRes/stormcast_test_{train,valid}.zarr
- LowRes/stats/{means.npy,stds.npy}
- HighRes/stats/{means.npy,stds.npy}

And writes CorrDiff-style datasets with keys such as:
- era5, cwb
- era5_valid, cwb_valid
- era5_center, era5_scale, cwb_center, cwb_scale
- era5_variable, cwb_variable
- era5_pressure, cwb_pressure
- XLAT, XLON, XLONG
- time (hours since 2015-01-01 00:00:00)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import math
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import zarr


LOGGER = logging.getLogger("stormcast_to_corrdiff")

TARGET_TIME_UNITS = "hours since 2015-01-01 00:00:00"
TARGET_EPOCH = datetime(2015, 1, 1)

_SPLIT_TO_SOURCE = {
    "train": "stormcast_test_train.zarr",
    "valid": "stormcast_test_valid.zarr",
}

_COMBINED_SPLIT_SOURCES = ("train", "valid")

_PRESSURE_LEVELS_HPA = {50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000}
_PRESSURE_PREFIXES = {"q", "t", "u", "v", "z"}


@dataclass(frozen=True)
class SplitPaths:
    split: str
    lowres_store: Path
    highres_store: Path
    output_store: Path


@dataclass(frozen=True)
class CopyPlan:
    workers: int
    batch_size: int
    prefetch: int
    estimated_peak_bytes: int


class ConversionError(RuntimeError):
    """Raised when the source schema is incompatible with conversion."""


class StormcastToCorrdiffConverter:
    """Convert StormCast-format Zarr stores to CorrDiff-format Zarr stores."""

    def __init__(
        self,
        source_root: Path,
        output_root: Path,
        splits: Sequence[str],
        workers: int,
        prefetch_batches: int,
        memory_budget_mb: int,
        read_batch_size: Optional[int],
        output_time_chunk: int,
        overwrite: bool,
        check_only: bool,
        max_timesteps: Optional[int],
    ) -> None:
        self.source_root = source_root
        self.output_root = output_root
        self.splits = [self._normalize_split_name(s) for s in splits]
        self.workers = max(1, workers)
        self.prefetch_batches = max(0, prefetch_batches)
        self.memory_budget_mb = max(256, memory_budget_mb)
        self.read_batch_size = read_batch_size if read_batch_size is None else max(1, read_batch_size)
        self.output_time_chunk = max(1, output_time_chunk)
        self.overwrite = overwrite
        self.check_only = check_only
        self.max_timesteps = max_timesteps if max_timesteps is None else max(1, max_timesteps)

    @staticmethod
    def _normalize_split_name(name: str) -> str:
        split = name.strip().lower()
        allowed = set(_SPLIT_TO_SOURCE) | {"combined"}
        if split not in allowed:
            raise ValueError(f"Unsupported split '{name}'. Use one of: {sorted(allowed)}")
        return split

    @staticmethod
    def _source_splits_for_target_split(split: str) -> Tuple[str, ...]:
        if split == "combined":
            return _COMBINED_SPLIT_SOURCES
        return (split,)

    def run(self) -> None:
        if not self.source_root.exists():
            raise FileNotFoundError(f"Source root does not exist: {self.source_root}")

        self.output_root.mkdir(parents=True, exist_ok=True)

        for split in self.splits:
            source_paths = [
                self._resolve_split_paths(source_split, output_split=split)
                for source_split in self._source_splits_for_target_split(split)
            ]
            LOGGER.info(
                "Checking source split '%s' using %d source store(s)",
                split,
                len(source_paths),
            )
            source_summary = self._check_source(source_paths)

            LOGGER.info(
                "Split %s summary | steps=%d era5_channels=%d cwb_channels=%d grid=%dx%d",
                split,
                source_summary["steps"],
                source_summary["era5_channels"],
                source_summary["cwb_channels"],
                source_summary["y"],
                source_summary["x"],
            )

            if self.check_only:
                continue

            self._convert_split(source_paths)

        if self.check_only:
            LOGGER.info("Check-only mode complete. No output was written.")

    def _resolve_split_paths(self, split: str, output_split: Optional[str] = None) -> SplitPaths:
        lowres_store = self.source_root / "LowRes" / _SPLIT_TO_SOURCE[split]
        highres_store = self.source_root / "HighRes" / _SPLIT_TO_SOURCE[split]
        output_name = output_split or split
        output_store = self.output_root / f"corrdiff_{output_name}.zarr"
        return SplitPaths(
            split=split,
            lowres_store=lowres_store,
            highres_store=highres_store,
            output_store=output_store,
        )

    def _open_read_group(self, store_path: Path) -> zarr.hierarchy.Group:
        if not store_path.exists():
            raise FileNotFoundError(f"Missing source store: {store_path}")

        try:
            return zarr.open_consolidated(str(store_path), mode="r")
        except Exception:
            return zarr.open_group(str(store_path), mode="r")

    def _check_single_source(self, paths: SplitPaths) -> Dict[str, int]:
        lr_group = self._open_read_group(paths.lowres_store)
        hr_group = self._open_read_group(paths.highres_store)

        for key in ("LowRes", "channel", "time", "latitude", "longitude"):
            if key not in lr_group:
                raise ConversionError(f"Missing key '{key}' in {paths.lowres_store}")
        for key in ("HighRes", "channel", "time", "latitude", "longitude"):
            if key not in hr_group:
                raise ConversionError(f"Missing key '{key}' in {paths.highres_store}")

        lowres = lr_group["LowRes"]
        highres = hr_group["HighRes"]

        if lowres.ndim != 4:
            raise ConversionError(f"LowRes must be 4D (time,channel,y,x). Got shape: {lowres.shape}")
        if highres.ndim != 4:
            raise ConversionError(f"HighRes must be 4D (time,channel,y,x). Got shape: {highres.shape}")

        if lowres.shape[0] != highres.shape[0]:
            raise ConversionError(
                f"Time dimension mismatch: LowRes={lowres.shape[0]} HighRes={highres.shape[0]}"
            )
        if lowres.shape[2:] != highres.shape[2:]:
            raise ConversionError(
                f"Spatial shape mismatch: LowRes={lowres.shape[2:]} HighRes={highres.shape[2:]}"
            )

        lr_time = np.asarray(lr_group["time"][:], dtype=np.int64)
        hr_time = np.asarray(hr_group["time"][:], dtype=np.int64)
        if not np.array_equal(lr_time, hr_time):
            raise ConversionError("LowRes/HighRes time vectors are not identical")

        lr_lat = np.asarray(lr_group["latitude"][:], dtype=np.float32)
        hr_lat = np.asarray(hr_group["latitude"][:], dtype=np.float32)
        lr_lon = np.asarray(lr_group["longitude"][:], dtype=np.float32)
        hr_lon = np.asarray(hr_group["longitude"][:], dtype=np.float32)

        if not np.array_equal(lr_lat, hr_lat):
            raise ConversionError("LowRes/HighRes latitude grids are not identical")
        if not np.array_equal(lr_lon, hr_lon):
            raise ConversionError("LowRes/HighRes longitude grids are not identical")

        lowres_means = self.source_root / "LowRes" / "stats" / "means.npy"
        lowres_stds = self.source_root / "LowRes" / "stats" / "stds.npy"
        highres_means = self.source_root / "HighRes" / "stats" / "means.npy"
        highres_stds = self.source_root / "HighRes" / "stats" / "stds.npy"
        for stat_path in (lowres_means, lowres_stds, highres_means, highres_stds):
            if not stat_path.exists():
                raise ConversionError(f"Missing stats file: {stat_path}")

        return {
            "steps": int(lowres.shape[0]),
            "era5_channels": int(lowres.shape[1]),
            "cwb_channels": int(highres.shape[1]),
            "y": int(lowres.shape[2]),
            "x": int(lowres.shape[3]),
        }

    def _check_source(self, paths_list: Sequence[SplitPaths]) -> Dict[str, int]:
        if not paths_list:
            raise ConversionError("No source stores provided for conversion")

        summaries: List[Dict[str, int]] = []
        for paths in paths_list:
            summaries.append(self._check_single_source(paths))

        ref = summaries[0]
        for idx, summary in enumerate(summaries[1:], start=1):
            if summary["era5_channels"] != ref["era5_channels"]:
                raise ConversionError(
                    f"ERA5 channel mismatch between source 0 and source {idx}: "
                    f"{ref['era5_channels']} vs {summary['era5_channels']}"
                )
            if summary["cwb_channels"] != ref["cwb_channels"]:
                raise ConversionError(
                    f"CWB channel mismatch between source 0 and source {idx}: "
                    f"{ref['cwb_channels']} vs {summary['cwb_channels']}"
                )
            if summary["y"] != ref["y"] or summary["x"] != ref["x"]:
                raise ConversionError(
                    f"Grid mismatch between source 0 and source {idx}: "
                    f"({ref['y']},{ref['x']}) vs ({summary['y']},{summary['x']})"
                )

        return {
            "steps": int(sum(s["steps"] for s in summaries)),
            "era5_channels": ref["era5_channels"],
            "cwb_channels": ref["cwb_channels"],
            "y": ref["y"],
            "x": ref["x"],
        }

    def _convert_split(self, paths_list: Sequence[SplitPaths]) -> None:
        if not paths_list:
            raise ConversionError("No source stores provided for conversion")

        paths = paths_list[0]
        target_split = paths.output_store.stem.replace("corrdiff_", "")
        if paths.output_store.exists():
            if not self.overwrite:
                raise FileExistsError(
                    f"Output exists and overwrite is false: {paths.output_store}. "
                    "Use --overwrite to replace it."
                )
            LOGGER.warning("Removing existing output store: %s", paths.output_store)
            shutil.rmtree(paths.output_store)

        source_entries: List[Dict[str, object]] = []
        remaining = self.max_timesteps

        ref_era5_channels: Optional[np.ndarray] = None
        ref_cwb_channels: Optional[np.ndarray] = None
        ref_lat: Optional[np.ndarray] = None
        ref_lon: Optional[np.ndarray] = None

        for source_idx, source_paths in enumerate(paths_list):
            lr_group = self._open_read_group(source_paths.lowres_store)
            hr_group = self._open_read_group(source_paths.highres_store)

            lowres_src = lr_group["LowRes"]
            highres_src = hr_group["HighRes"]

            source_steps = int(lowres_src.shape[0])
            if remaining is not None:
                if remaining <= 0:
                    break
                source_steps = min(source_steps, remaining)
                remaining -= source_steps

            if source_steps <= 0:
                continue

            era5_channels = _to_fixed_unicode(np.asarray(lr_group["channel"][:]))
            cwb_channels = _to_fixed_unicode(np.asarray(hr_group["channel"][:]))
            lat = np.asarray(lr_group["latitude"][:], dtype=np.float32)
            lon = np.asarray(lr_group["longitude"][:], dtype=np.float32)

            if ref_era5_channels is None:
                ref_era5_channels = era5_channels
                ref_cwb_channels = cwb_channels
                ref_lat = lat
                ref_lon = lon
            else:
                if not np.array_equal(ref_era5_channels, era5_channels):
                    raise ConversionError(f"ERA5 channel mismatch across sources at index {source_idx}")
                if not np.array_equal(ref_cwb_channels, cwb_channels):
                    raise ConversionError(f"CWB channel mismatch across sources at index {source_idx}")
                if not np.array_equal(ref_lat, lat):
                    raise ConversionError(f"Latitude grid mismatch across sources at index {source_idx}")
                if not np.array_equal(ref_lon, lon):
                    raise ConversionError(f"Longitude grid mismatch across sources at index {source_idx}")

            source_entries.append(
                {
                    "paths": source_paths,
                    "lr_group": lr_group,
                    "hr_group": hr_group,
                    "lowres_src": lowres_src,
                    "highres_src": highres_src,
                    "steps": source_steps,
                }
            )

        if not source_entries:
            raise ConversionError("No timesteps available to convert after applying max_timesteps")

        assert ref_era5_channels is not None
        assert ref_cwb_channels is not None
        assert ref_lat is not None
        assert ref_lon is not None

        total_steps = int(sum(int(entry["steps"]) for entry in source_entries))

        first_lowres = source_entries[0]["lowres_src"]
        first_highres = source_entries[0]["highres_src"]
        n_era5 = int(first_lowres.shape[1])
        n_cwb = int(first_highres.shape[1])
        y_size = int(first_lowres.shape[2])
        x_size = int(first_lowres.shape[3])

        era5_channels = ref_era5_channels
        cwb_channels = ref_cwb_channels

        if era5_channels.shape[0] != n_era5:
            raise ConversionError("LowRes channel coordinate length does not match data channels")
        if cwb_channels.shape[0] != n_cwb:
            raise ConversionError("HighRes channel coordinate length does not match data channels")

        era5_center = np.load(self.source_root / "LowRes" / "stats" / "means.npy").astype(np.float32)
        era5_scale = np.load(self.source_root / "LowRes" / "stats" / "stds.npy").astype(np.float32)
        cwb_center = np.load(self.source_root / "HighRes" / "stats" / "means.npy").astype(np.float32)
        cwb_scale = np.load(self.source_root / "HighRes" / "stats" / "stds.npy").astype(np.float32)

        if era5_center.shape[0] != n_era5 or era5_scale.shape[0] != n_era5:
            raise ConversionError("LowRes stats length does not match era5 channels")
        if cwb_center.shape[0] != n_cwb or cwb_scale.shape[0] != n_cwb:
            raise ConversionError("HighRes stats length does not match cwb channels")

        source_time_units_list: List[str] = []
        source_time_calendars: List[str] = []
        time_parts: List[np.ndarray] = []
        for entry in source_entries:
            lr_group = entry["lr_group"]
            steps = int(entry["steps"])
            source_time = np.asarray(lr_group["time"][:steps], dtype=np.int64)
            source_time_units = str(lr_group["time"].attrs.get("units", ""))
            source_time_calendar = str(lr_group["time"].attrs.get("calendar", "proleptic_gregorian"))
            source_time_units_list.append(source_time_units)
            source_time_calendars.append(source_time_calendar)
            time_parts.append(_convert_time_hours_to_target_epoch(source_time, source_time_units))

        target_time = np.concatenate(time_parts, axis=0)
        source_time_calendar = source_time_calendars[0]

        lat = ref_lat
        lon = ref_lon

        plan = _build_copy_plan(
            total_steps=total_steps,
            n_era5=n_era5,
            n_cwb=n_cwb,
            y_size=y_size,
            x_size=x_size,
            requested_workers=self.workers,
            requested_prefetch=self.prefetch_batches,
            memory_budget_mb=self.memory_budget_mb,
            requested_batch_size=self.read_batch_size,
        )

        LOGGER.info(
            "Split %s copy plan | workers=%d batch=%d prefetch=%d estimated_peak=%.1f MB",
            target_split,
            plan.workers,
            plan.batch_size,
            plan.prefetch,
            plan.estimated_peak_bytes / (1024.0 * 1024.0),
        )

        group = zarr.open_group(str(paths.output_store), mode="w")

        chunk_t = min(self.output_time_chunk, total_steps)
        era5_dst = group.create_dataset(
            "era5",
            shape=(total_steps, n_era5, y_size, x_size),
            chunks=(chunk_t, n_era5, y_size, x_size),
            dtype=np.float32,
            fill_value=np.nan,
        )
        cwb_dst = group.create_dataset(
            "cwb",
            shape=(total_steps, n_cwb, y_size, x_size),
            chunks=(chunk_t, n_cwb, y_size, x_size),
            dtype=np.float32,
            fill_value=np.nan,
        )
        era5_valid_dst = group.create_dataset(
            "era5_valid",
            shape=(total_steps, n_era5),
            chunks=(chunk_t, n_era5),
            dtype=np.int8,
            fill_value=0,
        )
        cwb_valid_dst = group.create_dataset(
            "cwb_valid",
            shape=(total_steps, n_cwb),
            chunks=(chunk_t, n_cwb),
            dtype=np.int8,
            fill_value=0,
        )

        # Core coordinates and metadata arrays
        group.create_dataset(
            "time",
            data=target_time,
            chunks=(min(total_steps, max(128, chunk_t * 64)),),
            dtype=np.int64,
        )
        group["time"].attrs["units"] = TARGET_TIME_UNITS
        group["time"].attrs["calendar"] = source_time_calendar

        group.create_dataset("era5_channel", data=era5_channels, chunks=(n_era5,))
        group.create_dataset("cwb_channel", data=cwb_channels, chunks=(n_cwb,))

        group.create_dataset("era5_variable", data=era5_channels, chunks=(n_era5,))
        group.create_dataset("cwb_variable", data=cwb_channels, chunks=(n_cwb,))

        group.create_dataset("era5_pressure", data=_infer_pressures(era5_channels), chunks=(n_era5,))
        group.create_dataset("cwb_pressure", data=_infer_pressures(cwb_channels), chunks=(n_cwb,))

        group.create_dataset("era5_center", data=era5_center, chunks=(n_era5,))
        group.create_dataset("era5_scale", data=era5_scale, chunks=(n_era5,))
        group.create_dataset("cwb_center", data=cwb_center, chunks=(n_cwb,))
        group.create_dataset("cwb_scale", data=cwb_scale, chunks=(n_cwb,))

        group.create_dataset("XLAT", data=lat, chunks=(y_size, x_size))
        group.create_dataset("XLON", data=lon, chunks=(y_size, x_size))
        group.create_dataset("XLONG", data=lon, chunks=(y_size, x_size))

        group.create_dataset("south_north", data=np.arange(y_size, dtype=np.int32), chunks=(y_size,))
        group.create_dataset("west_east", data=np.arange(x_size, dtype=np.int32), chunks=(x_size,))

        _attach_array_dimensions(group)

        # Copy tensors in parallel, bounded by memory-aware plan.
        jobs = list(_iter_time_jobs(total_steps, plan.batch_size))
        max_inflight = max(1, plan.workers + plan.prefetch)

        processed_steps = 0
        next_log_at = max(1, total_steps // 20)

        global_offset = 0
        for entry in source_entries:
            source_paths = entry["paths"]
            lowres_src = entry["lowres_src"]
            highres_src = entry["highres_src"]
            source_steps = int(entry["steps"])
            jobs = list(_iter_time_jobs(source_steps, plan.batch_size))

            with concurrent.futures.ThreadPoolExecutor(max_workers=plan.workers) as executor:
                pending: Dict[concurrent.futures.Future, Tuple[int, int]] = {}
                job_iter = iter(jobs)

                for _ in range(min(max_inflight, len(jobs))):
                    start, end = next(job_iter)
                    future = executor.submit(_read_chunk_and_valid, lowres_src, highres_src, start, end)
                    pending[future] = (start, end)

                while pending:
                    done, _ = concurrent.futures.wait(
                        pending,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for future in done:
                        start, end = pending.pop(future)
                        chunk = future.result()

                        dst_start = global_offset + start
                        dst_end = global_offset + end
                        era5_dst[dst_start:dst_end] = chunk[0]
                        cwb_dst[dst_start:dst_end] = chunk[1]
                        era5_valid_dst[dst_start:dst_end] = chunk[2]
                        cwb_valid_dst[dst_start:dst_end] = chunk[3]

                        processed_steps += (end - start)
                        if processed_steps >= next_log_at or processed_steps == total_steps:
                            LOGGER.info(
                                "Split %s progress: %d/%d timesteps",
                                target_split,
                                processed_steps,
                                total_steps,
                            )
                            next_log_at += max(1, total_steps // 20)

                        try:
                            new_start, new_end = next(job_iter)
                        except StopIteration:
                            continue
                        new_future = executor.submit(
                            _read_chunk_and_valid,
                            lowres_src,
                            highres_src,
                            new_start,
                            new_end,
                        )
                        pending[new_future] = (new_start, new_end)

            LOGGER.info(
                "Finished source split %s (%d timesteps) into %s",
                source_paths.split,
                source_steps,
                paths.output_store,
            )
            global_offset += source_steps

        group.attrs["source_lowres"] = [str(entry["paths"].lowres_store) for entry in source_entries]
        group.attrs["source_highres"] = [str(entry["paths"].highres_store) for entry in source_entries]
        group.attrs["source_time_units"] = source_time_units_list
        group.attrs["target_time_units"] = TARGET_TIME_UNITS
        group.attrs["created_utc"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        zarr.consolidate_metadata(str(paths.output_store))

        LOGGER.info("Finished split %s -> %s", target_split, paths.output_store)


def _attach_array_dimensions(group: zarr.hierarchy.Group) -> None:
    """Attach xarray-friendly dimension metadata to key arrays."""
    group["era5"].attrs["_ARRAY_DIMENSIONS"] = ["time", "era5_channel", "south_north", "west_east"]
    group["cwb"].attrs["_ARRAY_DIMENSIONS"] = ["time", "cwb_channel", "south_north", "west_east"]
    group["era5_valid"].attrs["_ARRAY_DIMENSIONS"] = ["time", "era5_channel"]
    group["cwb_valid"].attrs["_ARRAY_DIMENSIONS"] = ["time", "cwb_channel"]
    group["time"].attrs["_ARRAY_DIMENSIONS"] = ["time"]

    group["era5_channel"].attrs["_ARRAY_DIMENSIONS"] = ["era5_channel"]
    group["cwb_channel"].attrs["_ARRAY_DIMENSIONS"] = ["cwb_channel"]
    group["era5_variable"].attrs["_ARRAY_DIMENSIONS"] = ["era5_channel"]
    group["cwb_variable"].attrs["_ARRAY_DIMENSIONS"] = ["cwb_channel"]
    group["era5_pressure"].attrs["_ARRAY_DIMENSIONS"] = ["era5_channel"]
    group["cwb_pressure"].attrs["_ARRAY_DIMENSIONS"] = ["cwb_channel"]
    group["era5_center"].attrs["_ARRAY_DIMENSIONS"] = ["era5_channel"]
    group["era5_scale"].attrs["_ARRAY_DIMENSIONS"] = ["era5_channel"]
    group["cwb_center"].attrs["_ARRAY_DIMENSIONS"] = ["cwb_channel"]
    group["cwb_scale"].attrs["_ARRAY_DIMENSIONS"] = ["cwb_channel"]

    group["XLAT"].attrs["_ARRAY_DIMENSIONS"] = ["south_north", "west_east"]
    group["XLON"].attrs["_ARRAY_DIMENSIONS"] = ["south_north", "west_east"]
    group["XLONG"].attrs["_ARRAY_DIMENSIONS"] = ["south_north", "west_east"]
    group["south_north"].attrs["_ARRAY_DIMENSIONS"] = ["south_north"]
    group["west_east"].attrs["_ARRAY_DIMENSIONS"] = ["west_east"]


def _to_fixed_unicode(arr: np.ndarray) -> np.ndarray:
    values = [str(v) for v in np.asarray(arr).tolist()]
    max_len = max(1, max(len(v) for v in values))
    return np.asarray(values, dtype=f"<U{max_len}")


def _parse_time_units(units: str) -> datetime:
    match = re.match(r"^\s*hours\s+since\s+(.+?)\s*$", units)
    if not match:
        raise ConversionError(f"Unsupported time units: {units!r}")

    date_text = match.group(1).strip().replace("T", " ")

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(date_text, fmt)
        except ValueError:
            continue

    try:
        parsed = datetime.fromisoformat(date_text)
    except ValueError as exc:
        raise ConversionError(f"Could not parse time units base date: {units!r}") from exc

    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(tz=None).replace(tzinfo=None)
    return parsed


def _convert_time_hours_to_target_epoch(hours: np.ndarray, source_units: str) -> np.ndarray:
    source_epoch = _parse_time_units(source_units)
    offset_hours = int((source_epoch - TARGET_EPOCH).total_seconds() // 3600)
    return np.asarray(hours, dtype=np.int64) + offset_hours


def _infer_pressures(channels: np.ndarray) -> np.ndarray:
    pressures = np.full(channels.shape[0], np.nan, dtype=np.float32)
    for idx, raw_name in enumerate(channels.tolist()):
        name = str(raw_name)
        match = re.fullmatch(r"([a-zA-Z]+)(\d+)", name)
        if not match:
            continue

        prefix = match.group(1).lower()
        level = int(match.group(2))
        if prefix in _PRESSURE_PREFIXES and level in _PRESSURE_LEVELS_HPA:
            pressures[idx] = float(level)
    return pressures


def _iter_time_jobs(total_steps: int, batch_size: int) -> Iterator[Tuple[int, int]]:
    start = 0
    while start < total_steps:
        end = min(start + batch_size, total_steps)
        yield start, end
        start = end


def _read_chunk_and_valid(
    lowres_src: zarr.core.Array,
    highres_src: zarr.core.Array,
    start: int,
    end: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    era5 = np.asarray(lowres_src[start:end], dtype=np.float32)
    cwb = np.asarray(highres_src[start:end], dtype=np.float32)

    era5_valid = np.isfinite(era5).all(axis=(2, 3)).astype(np.int8)
    cwb_valid = np.isfinite(cwb).all(axis=(2, 3)).astype(np.int8)

    return era5, cwb, era5_valid, cwb_valid


def _build_copy_plan(
    total_steps: int,
    n_era5: int,
    n_cwb: int,
    y_size: int,
    x_size: int,
    requested_workers: int,
    requested_prefetch: int,
    memory_budget_mb: int,
    requested_batch_size: Optional[int],
) -> CopyPlan:
    workers = max(1, min(requested_workers, os.cpu_count() or requested_workers))
    prefetch = max(0, requested_prefetch)

    bytes_per_timestep = (n_era5 + n_cwb) * y_size * x_size * np.dtype(np.float32).itemsize
    memory_budget_bytes = int(memory_budget_mb * 1024 * 1024)

    # Keep a small safety margin for Python overhead and write buffers.
    usable_budget = int(memory_budget_bytes * 0.70)

    inflight = max(1, workers + prefetch)
    max_batch_by_memory = max(1, usable_budget // max(1, bytes_per_timestep * inflight))

    if requested_batch_size is None:
        batch_size = max_batch_by_memory
    else:
        batch_size = min(requested_batch_size, max_batch_by_memory)

    batch_size = max(1, min(batch_size, total_steps))

    while workers > 1:
        inflight = max(1, workers + prefetch)
        estimated_peak = bytes_per_timestep * batch_size * inflight
        if estimated_peak <= usable_budget:
            break
        workers -= 1

    estimated_peak = bytes_per_timestep * batch_size * max(1, workers + prefetch)
    return CopyPlan(
        workers=workers,
        batch_size=batch_size,
        prefetch=prefetch,
        estimated_peak_bytes=estimated_peak,
    )


def _parse_splits(raw: str) -> List[str]:
    splits = [s.strip().lower() for s in raw.split(",") if s.strip()]
    if not splits:
        raise ValueError("--splits produced an empty list")
    return splits


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert StormCast Zarr stores into CorrDiff-compatible Zarr stores.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="StormCast root containing LowRes/, HighRes/, and stats subfolders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory where corrdiff_train.zarr and corrdiff_valid.zarr are written.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="combined",
        help=(
            "Comma-separated split names from {train,valid,combined}. "
            "'combined' concatenates train then valid into one output store."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) // 2),
        help="Thread count for parallel chunk copy.",
    )
    parser.add_argument(
        "--prefetch-batches",
        type=int,
        default=1,
        help="Extra in-flight read batches beyond worker count.",
    )
    parser.add_argument(
        "--memory-budget-mb",
        type=int,
        default=4096,
        help="Approximate RAM budget used to auto-size read batches.",
    )
    parser.add_argument(
        "--read-batch-size",
        type=int,
        default=None,
        help="Timesteps per read batch. If omitted, chosen automatically from memory budget.",
    )
    parser.add_argument(
        "--output-time-chunk",
        type=int,
        default=1,
        help="Time chunk for era5/cwb arrays in output CorrDiff stores.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output Zarr stores.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate source schema and exit without writing output.",
    )
    parser.add_argument(
        "--max-timesteps",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    converter = StormcastToCorrdiffConverter(
        source_root=args.source_root,
        output_root=args.output_root,
        splits=_parse_splits(args.splits),
        workers=args.workers,
        prefetch_batches=args.prefetch_batches,
        memory_budget_mb=args.memory_budget_mb,
        read_batch_size=args.read_batch_size,
        output_time_chunk=args.output_time_chunk,
        overwrite=args.overwrite,
        check_only=args.check_only,
        max_timesteps=args.max_timesteps,
    )
    converter.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
