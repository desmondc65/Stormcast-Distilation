#!/usr/bin/env python3
"""Direct ERA5/RWRF/QPEPRE NetCDF conversion into StormCast-style Zarr stores."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import logging
import pathlib
import re
import shutil
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from netCDF4 import Dataset, num2date

if __package__ in {None, ""}:
    # Support running as a script (python nc_to_zarr_pipeline.py)
    import pathlib
    import sys

    sys.path.append(str(pathlib.Path(__file__).resolve().parent))
    from common import (
        XR_IMPORT_ERROR,
        ZARR_IMPORT_ERROR,
        ensure_timezone_naive,
        get_fork_context,
        resize_to_domain,
        search_bounds_1d,
        time_function,
        StreamingStats,
        xr,
        zarr,
    )
    from era5 import (
        DEFAULT_ERA5_VARIABLES,
        VARIABLE_CONFIGS,
        ERA5WorkerState,
        _era5_level_selection,
        _era5_worker,
        _era5_worker_init,
        _select_time_index,
    )
    from rwrf import (
        DEFAULT_INVARIANTS,
        DEFAULT_PRES_IDX,
        DEFAULT_RWRF_VARIABLES,
        RWRFWorkerState,
        resolve_variable,
        _extract_qpepre_keys,
        _rwrf_worker,
        _rwrf_worker_init,
    )
else:
    from .common import (
        XR_IMPORT_ERROR,
        ZARR_IMPORT_ERROR,
        ensure_timezone_naive,
        get_fork_context,
        resize_to_domain,
        search_bounds_1d,
        time_function,
        StreamingStats,
        xr,
        zarr,
    )
    from .era5 import (
        DEFAULT_ERA5_VARIABLES,
        VARIABLE_CONFIGS,
        ERA5WorkerState,
        _era5_level_selection,
        _era5_worker,
        _era5_worker_init,
        _select_time_index,
    )
    from .rwrf import (
        DEFAULT_INVARIANTS,
        DEFAULT_PRES_IDX,
        DEFAULT_RWRF_VARIABLES,
        RWRFWorkerState,
        resolve_variable,
        _extract_qpepre_keys,
        _rwrf_worker,
        _rwrf_worker_init,
    )

logger = logging.getLogger("nc_to_zarr")

# Config helpers
# ---------------------------------------------------------------------------


def load_config(path: str) -> Dict:
    import yaml  # lazy import to keep base deps minimal

    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    logger.info("Loaded config %s", path)
    return cfg


def merge_config_with_args(cfg: Dict, args: argparse.Namespace) -> Dict:
    """Merge command-line arguments into config dictionary, overriding config values."""
    merged = dict(cfg)
    for key, value in vars(args).items():
        if value is None:
            continue
        merged[key.replace("_", "-")] = value
    return merged


def parse_date_ranges(spec: Optional[Sequence]) -> Optional[List[Tuple[str, str]]]:
    """Parse date range specifications from config or command-line input."""
    if spec is None:
        return None
    if isinstance(spec, str):
        ranges: List[Tuple[str, str]] = []
        for chunk in spec.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            start, end = chunk.split(",", 1)
            ranges.append((start.strip(), end.strip()))
        return ranges
    parsed: List[Tuple[str, str]] = []
    for item in spec:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            parsed.append((str(item[0]), str(item[1])))
        else:
            raise ValueError(f"Invalid date range entry: {item!r}")
    return parsed


def _split_csv(value) -> Optional[List[str]]:
    """Split comma-separated string into list, or return None."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [v.strip() for v in str(value).split(",") if v.strip()]


# ---------------------------------------------------------------------------
# Pipeline implementation
# ---------------------------------------------------------------------------


class NCToZarrPipeline:
    def __init__(self, config: Dict) -> None:
        # Basic paths and settings
        self.era5_base = pathlib.Path(config.get("era5-path"))
        self.rwrf_base = pathlib.Path(config.get("rwrf-path"))
        self.qpepre_base = pathlib.Path(config.get("qpepre-path"))
        self.output_base = pathlib.Path(config.get("output-path"))

        # Validate paths
        if not self.era5_base.exists():
            raise FileNotFoundError(f"ERA5 path not found: {self.era5_base}")
        if not self.rwrf_base.exists():
            raise FileNotFoundError(f"RWRF path not found: {self.rwrf_base}")
        if not self.qpepre_base.exists():
            logger.warning("QPEPRE path missing: %s", self.qpepre_base)

        self.output_base.mkdir(parents=True, exist_ok=True)
        (self.output_base / "LowRes").mkdir(exist_ok=True)
        (self.output_base / "HighRes").mkdir(exist_ok=True)
        self.invariants_dir = self.output_base / "invariants"
        self.invariants_dir.mkdir(exist_ok=True)

        # Domain size
        domain_size = config.get("domain-size", (256, 256))
        if isinstance(domain_size, str):
            domain_size = tuple(int(x) for x in domain_size.split(","))
        elif isinstance(domain_size, list):
            domain_size = tuple(int(x) for x in domain_size)
        self.domain_size = tuple(int(x) for x in domain_size)

        # Latitude/longitude bounds
        lon_bounds = config.get("lon-bounds", (121.0, 121.75))
        lat_bounds = config.get("lat-bounds", (25.0, 25.75))
        if isinstance(lon_bounds, str):
            lon_bounds = tuple(float(x) for x in lon_bounds.split(","))
        if isinstance(lat_bounds, str):
            lat_bounds = tuple(float(x) for x in lat_bounds.split(","))
        self.lon_min, self.lon_max = lon_bounds
        self.lat_min, self.lat_max = lat_bounds

        # variables of high/low res and invariants
        era5_vars = _split_csv(config.get("era5-vars"))
        rwrf_vars = _split_csv(config.get("rwrf-vars"))
        invariant_vars = _split_csv(config.get("invariant-vars"))
        era5_config_vars = _split_csv(config.get("era5-variables"))
        rwrf_config_vars = _split_csv(config.get("rwrf-variables"))
        invariant_config_vars = _split_csv(config.get("invariant-variables"))
        
        # Build variable lists
        self.era5_variables = era5_vars or era5_config_vars or DEFAULT_ERA5_VARIABLES
        raw_rwrf = rwrf_vars or rwrf_config_vars or DEFAULT_RWRF_VARIABLES
        invariants = invariant_vars or invariant_config_vars or DEFAULT_INVARIANTS
        self.invariant_variables = list(dict.fromkeys(invariants))
        self.rwrf_variables = [var for var in raw_rwrf if var not in self.invariant_variables]

        self.pres_idx: Dict[int, int] = DEFAULT_PRES_IDX

        self.resample_mode = str(config.get("resample", "downsample")).lower()
        self.split_mode = str(config.get("split", "combined")).lower()
        self.skip_lowres = bool(config.get("skip-era5") or config.get("skip-lowres"))
        self.skip_highres = bool(config.get("skip-rwrf") or config.get("skip-highres"))
        self.overwrite = bool(config.get("overwrite", False))
        logger.info("overwrite is : %s", self.overwrite)
        self.compute_stats = bool(config.get("compute-stats", True))

        # multiprocessing settings
        max_workers_cfg = config.get("max-workers")
        if max_workers_cfg is None:
            self.max_workers = max(1, multiprocessing.cpu_count() // 2) # default to half of CPU cores
        else:
            self.max_workers = max(1, int(max_workers_cfg)) 

        self._era5_index: Dict[str, Dict[str, pathlib.Path]] = {}
        self._era5_slices: Dict[str, Tuple[slice, slice, np.ndarray, np.ndarray]] = {}
        self._rwrf_index: Dict[str, pathlib.Path] = {}
        self._qpepre_index: Dict[str, pathlib.Path] = {}
        self._rwrf_slice_cache: Optional[Tuple[slice, slice, np.ndarray, np.ndarray]] = None
        self._invariants_written = False

        logger.info(
            "Pipeline init | domain %s | lat[%s,%s] lon[%s,%s]",
            self.domain_size,
            self.lat_min,
            self.lat_max,
            self.lon_min,
            self.lon_max,
        )
        logger.info("ERA5 vars=%d | RWRF vars=%d | invariants=%d", len(self.era5_variables), len(self.rwrf_variables), len(self.invariant_variables))

    # ------------------------------------------------------------------
    # Date helpers
    # ------------------------------------------------------------------

    def _determine_splits(
        self,
        combined: Optional[List[Tuple[str, str]]],
        train: Optional[List[Tuple[str, str]]],
        valid: Optional[List[Tuple[str, str]]],
    ) -> Dict[str, List[Tuple[str, str]]]:
        """
        Determine which data splits to process based on the split mode configuration.
        
        This function organizes date ranges into appropriate splits (train/valid/combined)
        based on the pipeline's split_mode setting and available date range inputs.
        
        Args:
            combined: Optional list of date range tuples for combined processing
            train: Optional list of date range tuples for training data
            valid: Optional list of date range tuples for validation data
            
        Returns:
            Dictionary mapping split names to their corresponding date ranges
            
        Split modes:
            - "combined": Uses combined ranges if provided, otherwise merges train+valid
            - "train": Processes only training data
            - "valid": Processes only validation data  
            - "both": Processes both training and validation data separately
            
        Raises:
            ValueError: If required date ranges are missing for the specified split mode
        """
        mode = self.split_mode
        splits: Dict[str, List[Tuple[str, str]]] = {}
        if mode == "combined":
            if not combined:
                if train and valid:
                    combined = train + valid
                elif train:
                    combined = train
                elif valid:
                    combined = valid
            if not combined:
                raise ValueError("No date ranges supplied for combined split")
            splits["combined"] = combined
            return splits
        if mode in {"train", "both"}:
            if not train:
                raise ValueError("No train-ranges supplied")
            splits["train"] = train
        if mode in {"valid", "both"}:
            if not valid:
                raise ValueError("No valid-ranges supplied")
            splits["valid"] = valid
        if not splits:
            raise ValueError(f"Unsupported split mode: {mode}")
        return splits
    
    def _generate_datetimes(self, ranges: Sequence[Tuple[str, str]]) -> List[datetime]:
        """Generate list of hourly datetime objects from date ranges."""
        hours: List[datetime] = []
        for start_str, end_str in ranges:
            start = datetime.strptime(start_str, "%Y/%m/%d")
            end = datetime.strptime(end_str, "%Y/%m/%d")
            current = start
            while current <= end:
                for hour in range(24):
                    hours.append(current.replace(hour=hour))
                current += timedelta(days=1)
        unique_sorted = sorted({ensure_timezone_naive(dt) for dt in hours})
        logger.info("Generated %d hourly timestamps", len(unique_sorted))
        return unique_sorted
    # ------------------------------------------------------------------
    # ERA5 helpers
    # ------------------------------------------------------------------

    def _build_era5_index(self) -> None:
        """Build an index of available ERA5 files for quick lookup."""
        if self._era5_index:
            return
        logger.debug(f"Scanning ERA5 directory: {self.era5_base}")
        index: Dict[str, Dict[str, pathlib.Path]] = {}
        processed_count = 0
        for nc_path in self.era5_base.rglob("*.nc"):
            processed_count += 1
            if processed_count % 500 == 0:
                logger.debug(f"Processed {processed_count} ERA5 files so far...")
            filename = nc_path.stem
            # Match patterns like: varname_YYYYMM, varname_YYYYMMDDHH, or varname_YYYYMMDDTHH
            var_match = re.match(r"([a-zA-Z0-9]+)_(\d+(?:T\d+)?)", filename)
            if not var_match:
                logger.debug(f"Skipping file with unrecognized pattern: {nc_path}")
                continue
            var_name = var_match.group(1)
            date_part = var_match.group(2)
            if var_name not in index:
                index[var_name] = {}
                logger.debug(f"Found new ERA5 variable: {var_name}")
            
            # Normalize date_part: convert YYYYMMDDTHH to YYYYMMDDHH
            if 'T' in date_part:
                date_part = date_part.replace('T', '')
            
            if len(date_part) in (6, 10):
                index[var_name][date_part] = nc_path
            else:
                logger.warning(f"Unrecognized date format in {nc_path}: {date_part}")
        self._era5_index = index
        total_files = sum(len(var_files) for var_files in index.values())
        logger.info(f"Indexed {total_files} ERA5 files for {len(index)} variables")
        # logger debug all indexed files and path
        # for var_name, files in index.items():
        #     for date_key, path in files.items():
        #         logger.debug(f"ERA5 index | var={var_name} | date={date_key} | path={path}")
        # if not index:
        #     logger.warning(f"No ERA5 files found in {self.era5_base}")
            
        # exit(0)

    def _resolve_era5_path(self, variable: str, dt: datetime) -> Optional[pathlib.Path]:
        """Resolve the ERA5 file path for the given variable and datetime."""
        if not self._era5_index:
            self._build_era5_index()
        cache = self._era5_index.get(variable, {})
        if not cache:
            logger.warning("ERA5 variable %s not found in index", variable)
            return None
        for key in (
            dt.strftime("%Y%m%d%H"),
            dt.strftime("%Y%m%d") + str(dt.hour),
            dt.strftime("%Y%m"),
        ):
            path = cache.get(key)
            if path is not None:
                return path
        logger.warning("No ERA5 file found for %s at %s", variable, dt)
        return None

    # Cache and return latitude/longitude slice indices and coordinates for spatial subsetting within domain bounds.
    def _era5_slice(
        self, variable: str, lat: np.ndarray, lon: np.ndarray
    ) -> Tuple[slice, slice, np.ndarray, np.ndarray]:
        """Get or compute ERA5 slice for variable within domain bounds."""
        cached = self._era5_slices.get(variable)
        if cached is not None:
            return cached
        lat_idx = search_bounds_1d(lat, self.lat_min, self.lat_max)
        lon_idx = search_bounds_1d(lon, self.lon_min, self.lon_max)
        lat_slice = slice(lat_idx[0], lat_idx[1] + 1)
        lon_slice = slice(lon_idx[0], lon_idx[1] + 1)
        lat_sel = lat[lat_slice]
        lon_sel = lon[lon_slice]
        cached = (lat_slice, lon_slice, lat_sel, lon_sel)
        self._era5_slices[variable] = cached
        logger.debug("ERA5 slice %s: lat[%d:%d] lon[%d:%d]", variable, lat_idx[0], lat_idx[1], lon_idx[0], lon_idx[1])
        return cached

    def _ensure_level_selection(self, ds: Dataset, var_name: str, logical: str) -> Tuple[Optional[int], Optional[str]]:
        """Ensure correct level selection for ERA5 variable if applicable."""
        var_obj = ds.variables[var_name]
        return _era5_level_selection(ds, var_obj, logical, self.pres_idx)

    def _resize_to_domain(
        self, data: np.ndarray, lon_grid: np.ndarray, lat_grid: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Resize data to the configured domain size using specified resampling mode."""
        return resize_to_domain(data, lon_grid, lat_grid, self.domain_size, self.resample_mode)

    def _process_era5_single(self, dt: datetime, variable: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Process a single ERA5 variable at the specified datetime."""
        path = self._resolve_era5_path(variable, dt)
        if path is None or not path.exists(): # check existence in case index is stale
            logger.warning("ERA5 missing: %s %s", variable, dt)
            return None
        cfg = VARIABLE_CONFIGS.get(variable)
        nc_var = cfg.nc_var_name if cfg else variable
        logger.debug("Processing ERA5 %s at %s from %s", variable, dt, path)
        with Dataset(path, "r") as ds: # open netCDF file
            if nc_var not in ds.variables:
                logger.warning("NC var %s missing in %s", nc_var, path)
                return None
            # get lat/lon variables (try common names)
            lat_var = ds.variables.get("latitude") or ds.variables.get("lat")
            lon_var = ds.variables.get("longitude") or ds.variables.get("lon")
            if lat_var is None or lon_var is None:
                logger.warning("Latitude/longitude missing in %s", path)
                return None
            lat_arr = np.asarray(lat_var[:])
            lon_arr = np.asarray(lon_var[:])
            lat_slice, lon_slice, lat_sel, lon_sel = self._era5_slice(variable, lat_arr, lon_arr)
            time_var = ds.variables.get("time") or ds.variables.get("valid_time")
            time_idx = _select_time_index(time_var, dt)
            var_obj = ds.variables[nc_var]
            level_idx, level_dim = self._ensure_level_selection(ds, nc_var, variable)
            slices = [slice(None)] * var_obj.ndim
            dims = var_obj.dimensions
            if time_var is not None:
                for name in ("time", "valid_time"):
                    if name in dims:
                        slices[dims.index(name)] = time_idx
                        break
            if level_idx is not None and level_dim is not None:
                slices[dims.index(level_dim)] = level_idx
            slices[-2] = lat_slice
            slices[-1] = lon_slice
            data = np.asarray(var_obj[tuple(slices)], dtype=np.float32)
            data = np.squeeze(data)
            lon_grid, lat_grid = np.meshgrid(lon_sel, lat_sel)
            data_rs, lon_rs, lat_rs = self._resize_to_domain(data, lon_grid, lat_grid)
            data_rs = np.squeeze(np.asarray(data_rs, dtype=np.float32))
            logger.debug("ERA5 %s at %s: original shape %s -> resized %s", variable, dt, data.shape, data_rs.shape)
            return data_rs, lon_rs, lat_rs
    # ------------------------------------------------------------------
    # RWRF helpers
    # ------------------------------------------------------------------

    def _build_rwrf_index(self) -> None:
        """Build an index of available RWRF files for quick lookup."""
        if self._rwrf_index:
            return
        logger.debug(f"Scanning RWRF directory: {self.rwrf_base}")
        index: Dict[str, pathlib.Path] = {}
        processed_count = 0
        for path in self.rwrf_base.rglob("wrfout_d01_*_interp"):
            folder_name = path.parent.name
            index[folder_name] = path
            processed_count += 1
            if processed_count % 100 == 0:
                logger.debug(f"Indexed {processed_count} RWRF files so far...")
        self._rwrf_index = index
        logger.info(f"Indexed {len(index)} RWRF files")
        if not index:
            logger.warning(f"No RWRF files found in {self.rwrf_base}")

    def _build_qpepre_index(self) -> None:
        if self._qpepre_index:
            return
        if not self.qpepre_base.exists():
            logger.warning("QPEPRE base path missing: %s", self.qpepre_base)
            return
        for path in self.qpepre_base.rglob("qpepre_*_1_h.txt"):
            keys = _extract_qpepre_keys(path.name)
            if not keys:
                continue
            start_key, end_key = keys
            self._qpepre_index.setdefault(start_key, path)
            self._qpepre_index.setdefault(end_key, path)
        logger.info("Indexed %d QPEPRE files", len(self._qpepre_index))

    def _resolve_rwrf_path(self, dt: datetime) -> Optional[pathlib.Path]:
        if not self._rwrf_index:
            self._build_rwrf_index()
        key = dt.strftime("%Y-%m-%d_%H")
        logger.debug("Resolving RWRF key: %s", key)
        path = self._rwrf_index.get(key)
        logger.debug("Resolved RWRF path: %s", path)
        if path is None:
            logger.warning("RWRF missing for %s", key)
            return None
        return pathlib.Path(path)

    def _rwrf_slice(self, ds: Dataset) -> Tuple[slice, slice, np.ndarray, np.ndarray]:
        if self._rwrf_slice_cache is not None:
            return self._rwrf_slice_cache
        lat = np.asarray(ds.variables["XLAT"][0, :, 0])
        lon = np.asarray(ds.variables["XLONG"][0, 0, :])
        lat_idx = search_bounds_1d(lat, self.lat_min, self.lat_max)
        lon_idx = search_bounds_1d(lon, self.lon_min, self.lon_max)
        lat_slice = slice(lat_idx[0], lat_idx[1] + 1)
        lon_slice = slice(lon_idx[0], lon_idx[1] + 1)
        lat_sel = lat[lat_slice]
        lon_sel = lon[lon_slice]
        self._rwrf_slice_cache = (lat_slice, lon_slice, lat_sel, lon_sel)
        return self._rwrf_slice_cache

    def _select_rwrf_time_index(self, ds: Dataset, target: datetime) -> int:
        times_var = ds.variables.get("Times")
        if times_var is None:
            return 0
        values = [b"".join(row).decode("utf-8").strip() for row in times_var[:]]
        dt_array = np.array([np.datetime64(val.replace("_", "T")) for val in values])
        target64 = np.datetime64(target.strftime("%Y-%m-%dT%H"))
        idx = int(np.argmin(np.abs(dt_array - target64)))
        return idx

    def _process_qpepre_txt(self, date_or_dt, hr_str: Optional[str] = None) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Process QPEPRE precipitation for the given timestamp."""
        if isinstance(date_or_dt, datetime):
            dt_end = date_or_dt
            date_str = dt_end.strftime("%Y/%m/%d")
            hr_str = dt_end.strftime("%H")
        else:
            date_str = str(date_or_dt)
            if hr_str is None:
                raise ValueError("hr_str is required when passing a date string")
            dt_end = datetime.strptime(f"{date_str} {hr_str}", "%Y/%m/%d %H")

        dt_start = dt_end - timedelta(hours=1)
        end_key = dt_end.strftime("%Y%m%d%H%M")
        start_key = dt_start.strftime("%Y%m%d%H%M")

        if not self._qpepre_index:
            self._build_qpepre_index()
        txt_path = self._qpepre_index.get(end_key) or self._qpepre_index.get(start_key)
        if txt_path is None:
            candidate = self.qpepre_base / f"qpepre_{start_key}-{end_key}_1_h.txt"
            if candidate.exists():
                txt_path = candidate
            else:
                logger.warning("QPEPRE file not found for %s/%s", date_str, hr_str)
                return None

        raw = np.loadtxt(txt_path, dtype=np.float32)
        logger.debug("Loaded QPEPRE data from %s with shape %s", txt_path, raw.shape)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        _, lon_vals, lat_vals, rain_vals = raw.T
        lat_unique = np.unique(lat_vals)
        lon_unique = np.unique(lon_vals)
        logger.debug("QPEPRE unique lat=%d lon=%d", lat_unique.size, lon_unique.size)
        val_grid = np.full((lat_unique.size, lon_unique.size), np.nan, dtype=np.float32)
        lat_to_idx = {v: i for i, v in enumerate(lat_unique)}
        lon_to_idx = {v: i for i, v in enumerate(lon_unique)}
        for lon_v, lat_v, rain in zip(lon_vals, lat_vals, rain_vals):
            val_grid[lat_to_idx[lat_v], lon_to_idx[lon_v]] = rain
        lon_grid, lat_grid = np.meshgrid(lon_unique, lat_unique)
        resized, lon_rs, lat_rs = self._resize_to_domain(val_grid, lon_grid, lat_grid)
        logger.debug("QPEPRE data resized to domain with shape %s", resized.shape)
        return (
            np.asarray(resized, dtype=np.float32),
            lon_rs.astype(np.float32),
            lat_rs.astype(np.float32),
        )


    def _process_rwrf_single(
        self, dt: datetime
    ) -> Optional[Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]]:
        path = self._resolve_rwrf_path(dt)
        if path is None or not path.exists():
            return None
        with Dataset(path, "r") as ds:
            lat_slice, lon_slice, lat_sel, lon_sel = self._rwrf_slice(ds)
            lon_grid, lat_grid = np.meshgrid(lon_sel, lat_sel)
            time_idx = self._select_rwrf_time_index(ds, dt)
            results: Dict[str, np.ndarray] = {}
            resized_lon_grid: Optional[np.ndarray] = None
            resized_lat_grid: Optional[np.ndarray] = None
            for logical in self.rwrf_variables:
                nc_name, modifier = resolve_variable(logical)
                if nc_name not in ds.variables:
                    continue
                var = ds.variables[nc_name]
                slices = [slice(None)] * var.ndim
                dims = var.dimensions
                if "Time" in dims:
                    slices[dims.index("Time")] = time_idx
                elif "time" in dims:
                    slices[dims.index("time")] = time_idx
                level_idx = self.pres_idx.get(int(logical[1:])) if logical[1:].isdigit() else None
                if level_idx is not None:
                    level_set = False
                    for dim_name in dims:
                        if dim_name.lower() in LEVEL_DIM_NAMES_LOWER:
                            dim_idx = dims.index(dim_name)
                            if level_idx < var.shape[dim_idx]:
                                slices[dim_idx] = level_idx
                            else:
                                logger.warning(
                                    "Level index %s out of bounds for %s (dim %s size %s)",
                                    level_idx,
                                    logical,
                                    dim_name,
                                    var.shape[dim_idx],
                                )
                            level_set = True
                            break
                    if not level_set:
                        logger.debug("No pressure dimension found for %s with dims %s", logical, dims)
                slices[-2] = lat_slice
                slices[-1] = lon_slice
                data = np.asarray(var[tuple(slices)], dtype=np.float32)
                data = np.squeeze(modifier(data))
                data_rs, lon_rs, lat_rs = self._resize_to_domain(data, lon_grid, lat_grid)
                data_rs = np.squeeze(np.asarray(data_rs, dtype=np.float32))
                results[logical] = data_rs
                resized_lon_grid = lon_rs.astype(np.float32)
                resized_lat_grid = lat_rs.astype(np.float32)
            if "qpepre" not in results:
                qpe_result = self._process_qpepre_txt(dt)
                if qpe_result is not None:
                    qpe_data, qpe_lon, qpe_lat = qpe_result
                    results["qpepre"] = qpe_data
                    if resized_lon_grid is None:
                        resized_lon_grid = qpe_lon
                        resized_lat_grid = qpe_lat
            if resized_lon_grid is None or resized_lat_grid is None:
                logger.warning("Failed to determine resized grid for %s", dt)
                return None
            return results, resized_lon_grid, resized_lat_grid

    # ------------------------------------------------------------------
    # Invariant helpers
    # ------------------------------------------------------------------

    def _extract_invariants(
        self,
        dt: datetime,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if not self.invariant_variables:
            return None
        path = self._resolve_rwrf_path(dt)
        if path is None or not path.exists():
            return None
        with Dataset(path, "r") as ds:
            lat_slice, lon_slice, lat_sel, lon_sel = self._rwrf_slice(ds)
            lon_grid, lat_grid = np.meshgrid(lon_sel, lat_sel)
            data_list: List[np.ndarray] = []
            lon_template: Optional[np.ndarray] = None
            lat_template: Optional[np.ndarray] = None
            for logical in self.invariant_variables:
                try:
                    nc_name, modifier = resolve_variable(logical)
                except Exception:  # pragma: no cover - depends on optional lexicon package
                    logger.warning("Invariant %s is not defined in the RWRF lexicon", logical)
                    return None
                if nc_name not in ds.variables:
                    logger.warning("Invariant %s (source %s) missing in %s", logical, nc_name, path)
                    return None
                var = ds.variables[nc_name]
                slices = [slice(None)] * var.ndim
                dims = var.dimensions
                for name in ("Time", "time"):
                    if name in dims:
                        time_idx = self._select_rwrf_time_index(ds, dt)
                        slices[dims.index(name)] = time_idx
                        break
                if logical[1:].isdigit() and "pres_bottom_top" in dims:
                    level_idx = self.pres_idx.get(int(logical[1:]))
                    if level_idx is not None:
                        slices[dims.index("pres_bottom_top")] = level_idx
                slices[-2] = lat_slice
                slices[-1] = lon_slice
                data = np.asarray(var[tuple(slices)], dtype=np.float32)
                data = np.squeeze(modifier(data))
                resized, lon_rs, lat_rs = self._resize_to_domain(data, lon_grid, lat_grid)
                data_arr = np.squeeze(np.asarray(resized, dtype=np.float32))
                data_list.append(data_arr)
                lon_template = lon_rs.astype(np.float32)
                lat_template = lat_rs.astype(np.float32)
            if len(data_list) != len(self.invariant_variables) or lon_template is None or lat_template is None:
                logger.warning("Failed to load all invariants at %s", dt)
                return None
            stacked = np.stack(data_list, axis=0)
            return stacked.astype(np.float32), lon_template, lat_template

    def _ensure_invariants(self, datetimes: Sequence[datetime]) -> None:
        if self._invariants_written or not self.invariant_variables:
            return
        logger.info("Building invariants...")   
        for dt in datetimes:
            result = self._extract_invariants(dt)
            if result is None:
                continue
            data, lon_grid, lat_grid = result
            self._write_invariants(data, lon_grid, lat_grid)
            self._invariants_written = True
            return
        logger.warning("Unable to build invariants: no timestamp contained all invariant variables")

    def _write_invariants(
        self,
        data: np.ndarray,
        lon_grid: np.ndarray,
        lat_grid: np.ndarray,
    ) -> None:
        if xr is None or zarr is None:
            missing = []
            if xr is None:
                missing.append(f"xarray ({_XR_IMPORT_ERROR})")
            if zarr is None:
                missing.append(f"zarr ({_ZARR_IMPORT_ERROR})")
            raise RuntimeError(f"Missing required dependency: {', '.join(missing)}")
        channels = np.array(self.invariant_variables, dtype="object")
        _, H, W = data.shape
        coords = {
            "channel": channels,
            "y": np.arange(H, dtype=np.int32),
            "x": np.arange(W, dtype=np.int32),
        }
        ds = xr.Dataset(
            {"HighRes_invariants": (("channel", "y", "x"), data.astype(np.float32))},
            coords=coords,
        ).assign_coords(
            latitude=(("y", "x"), lat_grid.astype(np.float32)),
            longitude=(("y", "x"), lon_grid.astype(np.float32)),
        )
        encoding = {"HighRes_invariants": {"dtype": "float32"}}
        store = self.invariants_dir / "invariants.zarr"
        if self._prepare_store(store):
            ds.to_zarr(store, mode="w", consolidated=True, encoding=encoding)
            zarr.consolidate_metadata(store)
            logger.info("Wrote invariants to %s", store)
        else:
            logger.info("Invariants already exist at %s; skipping (overwrite disabled)", store)

    # ------------------------------------------------------------------
    # Writing helpers
    # ------------------------------------------------------------------

    def _prepare_store(self, store: pathlib.Path) -> bool:
        if store.exists():
            if not self.overwrite:
                return False
            if store.is_dir():
                shutil.rmtree(store)
            else:
                store.unlink()
        store.parent.mkdir(parents=True, exist_ok=True)
        return True

    def _normalize_field(
        self,
        field: np.ndarray,
        height: int,
        width: int,
        name: str,
        split: str,
        dt: Optional[datetime] = None,
    ) -> Optional[np.ndarray]:
        arr = np.asarray(field, dtype=np.float32)
        arr = np.squeeze(arr)
        if arr.shape == (height, width):
            return arr
        if arr.ndim == 2 and arr.shape[::-1] == (height, width):
            logger.debug(
                "Transposing split=%s field=%s dt=%s from shape %s to (%d,%d)",
                split,
                name,
                dt or "unknown",
                arr.shape,
                height,
                width,
            )
            return arr.T
        size = arr.size
        expected = height * width
        if size == expected:
            logger.debug(
                "Reshaping split=%s field=%s dt=%s from shape %s to (%d,%d)",
                split,
                name,
                dt or "unknown",
                arr.shape,
                height,
                width,
            )
            return arr.reshape(height, width)
        logger.warning(
            "Skipping split=%s field=%s dt=%s: incompatible shape %s (size=%d) expected (%d,%d)",
            split,
            name,
            dt or "unknown",
            arr.shape,
            size,
            height,
            width,
        )
        return None

    def _write_zarr(
        self,
        store: pathlib.Path,
        data_var: str,
        data_cube: np.ndarray,
        time_coord: np.ndarray,
        channels: List[str],
        lon_grid: np.ndarray,
        lat_grid: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
    ) -> None:
        if xr is None or zarr is None:
            missing = []
            if xr is None:
                missing.append(f"xarray ({_XR_IMPORT_ERROR})")
            if zarr is None:
                missing.append(f"zarr ({_ZARR_IMPORT_ERROR})")
            raise RuntimeError(f"Missing required dependency: {', '.join(missing)}")
        coords = {
            "time": time_coord,
            "channel": np.array(channels, dtype="object"),
            "y": np.arange(data_cube.shape[2], dtype=np.int32),
            "x": np.arange(data_cube.shape[3], dtype=np.int32),
        }
        data_vars = {
            data_var: (("time", "channel", "y", "x"), data_cube),
        }
        if valid_mask is not None:
            valid_arr = np.asarray(valid_mask, dtype=bool).reshape(-1, 1)
            coords["flag"] = np.array(["valid"], dtype="object")
            data_vars["valid"] = (("time", "flag"), valid_arr)
        ds = xr.Dataset(
            data_vars,
            coords=coords,
        ).assign_coords(
            latitude=(("y", "x"), lat_grid.astype(np.float32)),
            longitude=(("y", "x"), lon_grid.astype(np.float32)),
        )
        encoding = {data_var: {"dtype": "float32"}}
        if valid_mask is not None:
            encoding["valid"] = {"dtype": "bool"}  # bool flag stored as separate axis
        if not self._prepare_store(store):
            logger.info("%s already exists at %s; skipping (overwrite disabled)", data_var, store)
            return
        ds.to_zarr(store, mode="w", consolidated=True, encoding=encoding)
        zarr.consolidate_metadata(store)

    def _write_stats(self, store: pathlib.Path, stats: StreamingStats, channels: List[str]) -> None:
        if not self.compute_stats:
            return
        means, stds = stats.finalize()
        stats_dir = store.parent / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        np.save(stats_dir / "means.npy", means)
        np.save(stats_dir / "stds.npy", stds)
        logger.info("Wrote stats to %s", stats_dir)
        with open(stats_dir / "channels.txt", "w", encoding="utf-8") as fh:
            for name in channels:
                fh.write(f"{name}\n")

    def _load_valid_metadata(self, store: pathlib.Path) -> Optional[Dict[str, np.ndarray]]:
        """Load the per-timestamp validity mask and time coordinate from a Zarr store."""
        if xr is None or store is None:
            return None
        if not store.exists():
            return None
        try:
            ds = xr.open_zarr(store, consolidated=True)
        except Exception as exc:  # pragma: no cover - logging only
            logger.warning("Unable to open %s for validity stats: %s", store, exc)
            return None
        try:
            if "valid" not in ds:
                logger.warning("Dataset %s lacks 'valid' variable; skipping validity stats", store)
                return None
            valid = np.asarray(ds["valid"].to_numpy(), dtype=bool)
            if valid.ndim > 1:
                valid = valid.reshape(valid.shape[0], -1)[:, 0]
            if "time" not in ds.coords:
                logger.warning("Dataset %s lacks 'time' coordinate; skipping validity stats", store)
                return None
            times = np.asarray(ds.coords["time"].to_numpy())
            return {"mask": valid, "time": times}
        finally:
            ds.close()

    def _compute_valid_overlap(
        self,
        lhs: Dict[str, np.ndarray],
        rhs: Dict[str, np.ndarray],
    ) -> Tuple[int, int]:
        """Return shared timestamp count and overlap of valid samples between two masks."""
        left_time = np.asarray(lhs["time"]).astype("datetime64[ns]")
        right_time = np.asarray(rhs["time"]).astype("datetime64[ns]")
        left_mask = np.asarray(lhs["mask"], dtype=bool)
        right_mask = np.asarray(rhs["mask"], dtype=bool)
        left_lookup = {ts: bool(mask) for ts, mask in zip(left_time, left_mask)}
        shared = 0
        overlap = 0
        for ts, mask in zip(right_time, right_mask):
            left_val = left_lookup.get(ts)
            if left_val is None:
                continue
            shared += 1
            if left_val and mask:
                overlap += 1
        return shared, overlap

    def _report_split_validity(
        self,
        split_name: str,
        lowres_store: pathlib.Path,
        highres_store: pathlib.Path,
    ) -> None:
        """Log missing counts and overlap information for a given split."""
        lowres_meta = self._load_valid_metadata(lowres_store)
        highres_meta = self._load_valid_metadata(highres_store)

        if lowres_meta is not None:
            total = int(lowres_meta["mask"].size)
            missing = int((~lowres_meta["mask"]).sum())
            logger.info(
                "Split %s LowRes valid timestamps: %d/%d (%d missing)",
                split_name,
                total - missing,
                total,
                missing,
            )
        else:
            logger.info("Split %s LowRes validity stats unavailable", split_name)

        if highres_meta is not None:
            total = int(highres_meta["mask"].size)
            missing = int((~highres_meta["mask"]).sum())
            logger.info(
                "Split %s HighRes valid timestamps: %d/%d (%d missing)",
                split_name,
                total - missing,
                total,
                missing,
            )
        else:
            logger.info("Split %s HighRes validity stats unavailable", split_name)

        if lowres_meta is None or highres_meta is None:
            return

        shared, overlap = self._compute_valid_overlap(lowres_meta, highres_meta)
        if shared == 0:
            logger.info(
                "Split %s LowRes/HighRes share no timestamps; cannot compute valid overlap",
                split_name,
            )
            return
        logger.info(
            "Split %s LowRes & HighRes both valid %d/%d shared timestamps",
            split_name,
            overlap,
            shared,
        )

    # ------------------------------------------------------------------
    # Validity precomputation
    # ------------------------------------------------------------------

    def _precompute_combined_validity(self, datetimes: Sequence[datetime]) -> np.ndarray:
        """Precompute which timesteps have ALL required files (ERA5, RWRF, QPEPRE).
        
        Since LowRes and HighRes data are paired for training, we need both to be valid.
        A timestep is only valid if:
        - All ERA5 variables are present
        - RWRF file is present
        - QPEPRE file is present (if qpepre is in variables)
        
        Returns:
            Boolean array where True means all required files exist for that timestep.
        """
        T = len(datetimes)
        validity = np.ones(T, dtype=bool)
        
        era5_missing_count = 0
        rwrf_missing_count = 0
        qpepre_missing_count = 0
        
        for ti, dt in enumerate(datetimes):
            # Check all ERA5 variables
            for var in self.era5_variables:
                path = self._resolve_era5_path(var, dt)
                if path is None or not path.exists():
                    logger.debug("ERA5 missing for validity check: %s at %s", var, dt)
                    validity[ti] = False
                    era5_missing_count += 1
                    break
            
            # Check RWRF file (only if ERA5 was valid, for efficiency)
            if validity[ti]:
                rwrf_path = self._resolve_rwrf_path(dt)
                if rwrf_path is None or not rwrf_path.exists():
                    logger.debug("RWRF missing for validity check at %s", dt)
                    validity[ti] = False
                    rwrf_missing_count += 1
            
            # Check QPEPRE file if needed (only if both ERA5 and RWRF were valid)
            if validity[ti] and "qpepre" in self.rwrf_variables:
                end_key = dt.strftime("%Y%m%d%H%M")
                start_key = (dt - timedelta(hours=1)).strftime("%Y%m%d%H%M")
                qpepre_path_str = self._qpepre_index.get(end_key) or self._qpepre_index.get(start_key)
                if qpepre_path_str is None:
                    logger.debug("QPEPRE missing for validity check at %s", dt)
                    validity[ti] = False
                    qpepre_missing_count += 1
                else:
                    qpepre_path = pathlib.Path(qpepre_path_str)
                    if not qpepre_path.exists():
                        logger.debug("QPEPRE file does not exist: %s", qpepre_path)
                        validity[ti] = False
                        qpepre_missing_count += 1
        
        logger.info("Validity check summary: %d timesteps with missing ERA5, %d with missing RWRF, %d with missing QPEPRE",
                    era5_missing_count, rwrf_missing_count, qpepre_missing_count)
        
        return validity

    # ------------------------------------------------------------------
    # Split processing
    # ------------------------------------------------------------------

    @time_function
    def process_lowres(self, split_name: str, datetimes: Sequence[datetime], store: pathlib.Path, 
                       valid_mask: Optional[np.ndarray] = None) -> None:
        """Process LowRes (ERA5) data for the given split and datetimes, writing to the specified store.
        
        Args:
            split_name: Name of the split (train/valid/combined)
            datetimes: Sequence of timesteps to process
            store: Path to output Zarr store
            valid_mask: Pre-computed validity mask. If None, will compute combined validity.
        """
        if store.exists() and not self.overwrite:
            logger.info("overwrite: %s", self.overwrite)
            logger.info("LowRes %s already exists, skipping", split_name)
            return
        channels = list(self.era5_variables)
        T = len(datetimes)
        C = len(channels)
        H, W = self.domain_size
        data_cube = np.full((T, C, H, W), np.nan, dtype=np.float32)
        stats = StreamingStats(C)
        
        # Use provided validity mask or compute combined validity
        if valid_mask is None:
            logger.info("LowRes %s: precomputing combined validity for %d timesteps", split_name, T)
            valid_mask = self._precompute_combined_validity(datetimes)
        
        invalid_count = (~valid_mask).sum()
        logger.info("LowRes %s: %d/%d timesteps have all required inputs, %d will be skipped",
                    split_name, valid_mask.sum(), T, invalid_count)
        
        lon_template: Optional[np.ndarray] = None
        lat_template: Optional[np.ndarray] = None
        time_lookup = {dt: idx for idx, dt in enumerate(datetimes)}
        channel_lookup = {name: idx for idx, name in enumerate(channels)}

        # Filter to only valid timesteps for processing
        valid_datetimes = [dt for dt, is_valid in zip(datetimes, valid_mask) if is_valid]
        
        use_pool = self.max_workers > 1 and len(valid_datetimes) > 0
        if use_pool:
            logger.info("LowRes %s: using %d worker processes for %d valid timesteps", 
                        split_name, self.max_workers, len(valid_datetimes))
            self._build_era5_index()
            state = ERA5WorkerState(
                era5_index={var: {k: str(path) for k, path in files.items()} for var, files in self._era5_index.items()},
                domain_size=self.domain_size,
                lon_bounds=(self.lon_min, self.lon_max),
                lat_bounds=(self.lat_min, self.lat_max),
                resample_mode=self.resample_mode,
                pres_idx=self.pres_idx,
            )
            ctx = get_fork_context()
            executor_kwargs = dict(
                max_workers=self.max_workers,
                initializer=_era5_worker_init,
                initargs=(state,),
            )
            if ctx is not None:
                executor_kwargs["mp_context"] = ctx
            with ProcessPoolExecutor(**executor_kwargs) as executor:
                futures = {
                    executor.submit(_era5_worker, dt, tuple(channels)): dt
                    for dt in valid_datetimes
                }
                completed = 0
                for future in as_completed(futures):
                    dt, samples, lon_grid, lat_grid = future.result()
                    ti = time_lookup[dt]
                    missing = False
                    sample_present = False
                    for name in channels:
                        data = samples.get(name)
                        if data is None:
                            missing = True
                            continue
                        sample_present = True
                        ci = channel_lookup[name]
                        arr = self._normalize_field(data, H, W, name, split_name)
                        if arr is None:
                            missing = True
                            continue
                        data_cube[ti, ci] = arr
                        stats.update(ci, arr)
                    if lon_template is None and lon_grid is not None and lat_grid is not None:
                        lon_template = lon_grid.astype(np.float32)
                        lat_template = lat_grid.astype(np.float32)
                    if not sample_present:
                        missing = True
                    if missing:
                        valid_mask[ti] = False
                    completed += 1
                    if completed % 50 == 0 or completed == len(valid_datetimes):
                        logger.info("LowRes %s progress %d/%d valid (total: %d)", 
                                    split_name, completed, len(valid_datetimes), T)
        else:
            logger.info("LowRes %s: processing sequentially for %d valid timesteps", 
                        split_name, len(valid_datetimes))
            processed = 0
            for ti, dt in enumerate(datetimes):
                # Skip invalid timesteps - they already have NaN and valid_mask=False
                if not valid_mask[ti]:
                    continue
                    
                missing = False
                sample_present = False
                for ci, var in enumerate(channels):
                    sample = self._process_era5_single(dt, var)
                    if sample is None:
                        missing = True
                        continue
                    sample_present = True
                    data, lon_grid, lat_grid = sample
                    arr = self._normalize_field(data, H, W, var, split_name)
                    if arr is None:
                        missing = True
                        continue
                    data_cube[ti, ci] = arr
                    stats.update(ci, arr)
                    if lon_template is None:
                        lon_template = lon_grid.astype(np.float32)
                        lat_template = lat_grid.astype(np.float32)
                if not sample_present:
                    missing = True
                if missing:
                    valid_mask[ti] = False
                processed += 1
                if processed % 100 == 0 or processed == len(valid_datetimes):
                    logger.info("LowRes %s progress %d/%d valid (total: %d)", 
                                split_name, processed, len(valid_datetimes), T)
        if lon_template is None or lat_template is None:
            logger.warning("No ERA5 data collected for split %s", split_name)
            return
        time_coord = np.array(datetimes, dtype="datetime64[h]")
        self._write_zarr(
            store,
            "LowRes",
            data_cube,
            time_coord,
            channels,
            lon_template,
            lat_template,
            valid_mask,
        )
        if split_name.lower() == "train":
            self._write_stats(store, stats, channels)

    @time_function
    def process_highres(self, split_name: str, datetimes: Sequence[datetime], store: pathlib.Path,
                        valid_mask: Optional[np.ndarray] = None) -> None:
        """Process HighRes (RWRF + QPEPRE) data for the given split and datetimes, writing to the specified store.
        
        Args:
            split_name: Name of the split (train/valid/combined)
            datetimes: Sequence of timesteps to process
            store: Path to output Zarr store
            valid_mask: Pre-computed validity mask. If None, will compute combined validity.
        """
        if store.exists() and not self.overwrite:
            logger.info("HighRes %s already exists, skipping", split_name)
            return
        channels = list(self.rwrf_variables)
        channel_index = {name: idx for idx, name in enumerate(channels)}
        T = len(datetimes)
        C = len(channels)
        H, W = self.domain_size
        data_cube = np.full((T, C, H, W), np.nan, dtype=np.float32)
        stats = StreamingStats(C)
        
        # Use provided validity mask or compute combined validity
        if valid_mask is None:
            logger.info("HighRes %s: precomputing combined validity for %d timesteps", split_name, T)
            valid_mask = self._precompute_combined_validity(datetimes)
        
        invalid_count = (~valid_mask).sum()
        logger.info("HighRes %s: %d/%d timesteps have all required inputs, %d will be skipped",
                    split_name, valid_mask.sum(), T, invalid_count)
        
        lon_template: Optional[np.ndarray] = None
        lat_template: Optional[np.ndarray] = None
        time_lookup = {dt: idx for idx, dt in enumerate(datetimes)}

        # Filter to only valid timesteps for processing
        valid_datetimes = [dt for dt, is_valid in zip(datetimes, valid_mask) if is_valid]
        
        use_pool = self.max_workers > 1 and len(valid_datetimes) > 0
        if use_pool:
            logger.info("HighRes %s: using %d worker processes for %d valid timesteps", 
                        split_name, self.max_workers, len(valid_datetimes))
            self._build_rwrf_index()
            self._build_qpepre_index()
            state = RWRFWorkerState(
                rwrf_index={k: str(v) for k, v in self._rwrf_index.items()},
                qpepre_index={k: str(v) for k, v in self._qpepre_index.items()},
                domain_size=self.domain_size,
                lon_min=self.lon_min,
                lon_max=self.lon_max,
                lat_min=self.lat_min,
                lat_max=self.lat_max,
                resample_mode=self.resample_mode,
                pres_idx=self.pres_idx,
                rwrf_variables=tuple(channels),
            )
            ctx = get_fork_context()
            executor_kwargs = dict(
                max_workers=self.max_workers,
                initializer=_rwrf_worker_init,
                initargs=(state,),
            )
            if ctx is not None:
                executor_kwargs["mp_context"] = ctx
            with ProcessPoolExecutor(**executor_kwargs) as executor:
                futures = {
                    executor.submit(_rwrf_worker, dt): dt
                    for dt in valid_datetimes
                }
                completed = 0
                for future in as_completed(futures):
                    dt, variables, lon_grid, lat_grid = future.result()
                    ti = time_lookup[dt]
                    missing = False
                    sample_present = False
                    for name in channels:
                        data = variables.get(name)
                        if data is None:
                            missing = True
                            continue
                        sample_present = True
                        ci = channel_index[name]
                        arr = self._normalize_field(data, H, W, name, split_name, dt)
                        if arr is None:
                            missing = True
                            continue
                        data_cube[ti, ci] = arr
                        stats.update(ci, arr)
                    if lon_template is None and lon_grid is not None and lat_grid is not None:
                        lon_template = lon_grid.astype(np.float32)
                        lat_template = lat_grid.astype(np.float32)
                    if not sample_present:
                        missing = True
                    if missing:
                        valid_mask[ti] = False
                    completed += 1
                    if completed % 25 == 0 or completed == len(valid_datetimes):
                        logger.info("HighRes %s progress %d/%d valid (total: %d)", 
                                    split_name, completed, len(valid_datetimes), T)
        else:
            logger.info("HighRes %s: processing sequentially for %d valid timesteps", 
                        split_name, len(valid_datetimes))
            processed = 0
            for ti, dt in enumerate(datetimes):
                # Skip invalid timesteps - they already have NaN and valid_mask=False
                if not valid_mask[ti]:
                    continue
                    
                sample = self._process_rwrf_single(dt)
                missing = False
                sample_present = False
                if sample is None:
                    valid_mask[ti] = False
                    processed += 1
                    continue
                variables, lon_grid, lat_grid = sample
                for name in channels:
                    data = variables.get(name)
                    if data is None:
                        missing = True
                        continue
                    sample_present = True
                    ci = channel_index[name]
                    arr = self._normalize_field(data, H, W, name, split_name, dt)
                    if arr is None:
                        missing = True
                        continue
                    data_cube[ti, ci] = arr
                    stats.update(ci, arr)
                if lon_template is None and variables:
                    lon_template = lon_grid.astype(np.float32)
                    lat_template = lat_grid.astype(np.float32)
                if not sample_present:
                    missing = True
                if missing:
                    valid_mask[ti] = False
                processed += 1
                if processed % 50 == 0 or processed == len(valid_datetimes):
                    logger.info("HighRes %s progress %d/%d valid (total: %d)", 
                                split_name, processed, len(valid_datetimes), T)
        if lon_template is None or lat_template is None:
            logger.warning("No RWRF data collected for split %s", split_name)
            return
        time_coord = np.array(datetimes, dtype="datetime64[h]")
        self._write_zarr(
            store,
            "HighRes",
            data_cube,
            time_coord,
            channels,
            lon_template,
            lat_template,
            valid_mask,
        )
        self._ensure_invariants(datetimes)
        if split_name.lower() == "train":
            self._write_stats(store, stats, channels)

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(
        self,
        combined: Optional[List[Tuple[str, str]]],
        train: Optional[List[Tuple[str, str]]],
        valid: Optional[List[Tuple[str, str]]],
    ) -> None:
        splits = self._determine_splits(combined, train, valid)

        # initialize indices
        if not self.skip_lowres:
            self._build_era5_index()
        if not self.skip_highres:
            self._build_rwrf_index()
            self._build_qpepre_index()

        # process each split 
        for split_name, ranges in splits.items():
            datetimes = self._generate_datetimes(ranges)
            if not datetimes:
                logger.warning("No timestamps for split %s", split_name)
                continue
            
            # Precompute combined validity ONCE for both LowRes and HighRes
            # This ensures we only process timesteps where ALL required files exist
            logger.info("Split %s: precomputing combined validity for %d timesteps", split_name, len(datetimes))
            valid_mask = self._precompute_combined_validity(datetimes)
            valid_count = valid_mask.sum()
            invalid_count = (~valid_mask).sum()
            logger.info("Split %s: %d/%d timesteps are valid (both LowRes and HighRes complete), %d will be skipped",
                        split_name, valid_count, len(datetimes), invalid_count)
            
            lowres_store = self.output_base / "LowRes" / f"{split_name}_era5.zarr"
            highres_store = self.output_base / "HighRes" / f"{split_name}_rwrf.zarr"
            
            # Pass the same validity mask to both processing methods
            if not self.skip_lowres:
                self.process_lowres(split_name, datetimes, lowres_store, valid_mask=valid_mask)
            if not self.skip_highres:
                self.process_highres(split_name, datetimes, highres_store, valid_mask=valid_mask)
            self._report_split_validity(split_name, lowres_store, highres_store)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def setup_logging(config: Dict) -> None:
    level = str(config.get("log-level", "INFO")).upper()
    log_file = config.get("log-file")
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Direct NC to Zarr converter")
    parser.add_argument("--config", "-c", help="Path to YAML configuration file")
    parser.add_argument("--era5-path")
    parser.add_argument("--rwrf-path")
    parser.add_argument("--qpepre-path")
    parser.add_argument("--output-path")
    parser.add_argument("--domain-size", help="Domain size H,W")
    parser.add_argument("--lon-bounds", help="Longitude bounds min,max")
    parser.add_argument("--lat-bounds", help="Latitude bounds min,max")
    parser.add_argument("--era5-vars", help="Comma-separated ERA5 variables")
    parser.add_argument("--rwrf-vars", help="Comma-separated RWRF variables")
    parser.add_argument("--invariant-vars", help="Comma-separated invariant variables")
    parser.add_argument("--date-ranges", help="Combined date ranges")
    parser.add_argument("--train-ranges", help="Training date ranges")
    parser.add_argument("--valid-ranges", help="Validation date ranges")
    parser.add_argument("--split", choices=["combined", "train", "valid", "both"])
    parser.add_argument("--resample", choices=["downsample", "interp"])
    parser.add_argument("--skip-era5", action="store_true", default=None)
    parser.add_argument("--skip-rwrf", action="store_true", default=None)
    parser.add_argument("--compute-stats", action="store_true", default=None)
    parser.add_argument("--overwrite", action="store_true", default=None)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--log-file")
    return parser

@time_function
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    cfg: Dict = {}
    if args.config:
        cfg = load_config(args.config)
    elif pathlib.Path("config.yaml").exists():
        cfg = load_config("config.yaml")

    merged = merge_config_with_args(cfg, args)

    setup_logging(merged)

    missing = []
    if xr is None:
        missing.append(f"xarray ({XR_IMPORT_ERROR})")
    if zarr is None:
        missing.append(f"zarr ({ZARR_IMPORT_ERROR})")
    if missing:
        parser.error(f"Missing required Python modules: {', '.join(missing)}")

    required = ["era5-path", "rwrf-path", "qpepre-path", "output-path"]
    missing = [key for key in required if not merged.get(key)]
    if missing:
        parser.error(f"Missing required parameters: {', '.join(missing)}")

    combined = parse_date_ranges(merged.get("date-ranges"))
    train = parse_date_ranges(merged.get("train-ranges"))
    valid = parse_date_ranges(merged.get("valid-ranges"))

    if not (combined or train or valid):
        parser.error("At least one of date-ranges/train-ranges/valid-ranges must be provided")

    pipeline = NCToZarrPipeline(merged)
    pipeline.run(combined, train, valid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
