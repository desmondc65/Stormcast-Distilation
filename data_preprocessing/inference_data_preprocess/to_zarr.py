#!/usr/bin/env python3
"""
Lightweight tool to convert GRIB + RWRF + QPEPRE data to Zarr format for inference.

This tool processes a single hour of data from:
- Low-resolution GRIB files (replacing ERA5 NC files)
- High-resolution RWRF NetCDF files  
- QPEPRE precipitation text files

And outputs zarr stores compatible with the StormCast inference pipeline.

Usage:
    python to_zarr.py --config inference_data_preprocess_config.yaml
    python to_zarr.py --grib-path /path/to/file.grb --rwrf-path /path/to/wrfout --qpepre-path /path/to/qpepre.txt --output /path/to/output
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pygrib
    HAS_PYGRIB = True
except ImportError:
    HAS_PYGRIB = False
    pygrib = None

try:
    from netCDF4 import Dataset
    HAS_NETCDF4 = True
except ImportError:
    HAS_NETCDF4 = False
    Dataset = None

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False
    xr = None

try:
    import zarr
    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False
    zarr = None

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("inference_to_zarr")


# ---------------------------------------------------------------------------
# GRIB variable mapping (shortName + level -> logical name)
# ---------------------------------------------------------------------------

# Mapping from GRIB shortName to logical variable base name
GRIB_SHORTNAME_MAP = {
    "z": "z",       # Geopotential
    "q": "q",       # Specific humidity
    "t": "t",       # Temperature  
    "u": "u",       # U-wind
    "v": "v",       # V-wind
    "10u": "u10",   # 10m U-wind
    "10v": "v10",   # 10m V-wind
    "2t": "t2m",    # 2m temperature
    "msl": "mslp",  # Mean sea level pressure
    "sp": "sp",     # Surface pressure
    "tcwv": "tcwv", # Total column water vapor
}

# Pressure levels available in GRIB (from log)
GRIB_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]


# ---------------------------------------------------------------------------
# RWRF variable mapping
# Maps logical variable names to WRF NetCDF variable names
# ---------------------------------------------------------------------------

RWRF_VARIABLE_MAP = {
    # Surface variables - use actual WRF variable names
    "u10": "U10",       # 10m U-wind
    "v10": "V10",       # 10m V-wind
    "t2m": "T2",        # 2m temperature
    "sp": "PSFC",       # Surface pressure
    "qpepre": "qpepre", # QPEPRE precipitation (handled separately)
}

# Note: msl (sea level pressure) and tcwv (total column water vapor) are not 
# directly available in WRF output and would need to be computed.
# For now, we skip them if not available.

# Pressure level index mapping for RWRF (WRF model levels to pressure levels)
# This maps pressure level (hPa) to WRF bottom_top index
DEFAULT_PRES_IDX: Dict[int, int] = {
    1000: 0, 925: 3, 850: 6, 700: 11, 600: 13, 500: 15,
    400: 17, 300: 19, 250: 20, 200: 22, 150: 24, 100: 26, 70: 27, 50: 28,
}

LEVEL_DIM_NAMES_LOWER = (
    "pres_bottom_top", "pres_bottom_top_stag", "bottom_top", "bottom_top_stag",
    "level", "lev", "isobaricinhpa", "pressure_level",
)

# Build RWRF variable map for pressure level variables
# Note: WRF stores U/V on staggered grids, T as perturbation potential temp,
# and doesn't have direct pressure-level interpolated variables.
# These would need vertical interpolation from model levels.
# For now, map to raw WRF 3D variables (will extract specific levels)
for prefix, nc_name in [("u", "U"), ("v", "V"), ("t", "T"), ("q", "QVAPOR")]:
    for level in [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]:
        RWRF_VARIABLE_MAP[f"{prefix}{level}"] = nc_name

# Geopotential height needs to be computed from PH + PHB
for level in [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]:
    RWRF_VARIABLE_MAP[f"z{level}"] = "PH"  # Will compute geopotential from PH+PHB


# ---------------------------------------------------------------------------
# Default variable lists (matching training data)
# ---------------------------------------------------------------------------

# LowRes channels must match: ['mslp', 't2m', 'u10', 'v10', 'q1000', 'q850', 'q500', 'q250', 
#                              't1000', 't850', 't500', 't250', 'u1000', 'u850', 'u500', 'u250', 
#                              'v1000', 'v850', 'v500', 'v250', 'z1000', 'z850', 'z500', 'z250']
DEFAULT_LOWRES_VARIABLES = [
    "mslp", "t2m", "u10", "v10",
    "q1000", "q850", "q500", "q250",
    "t1000", "t850", "t500", "t250",
    "u1000", "u850", "u500", "u250",
    "v1000", "v850", "v500", "v250",
    "z1000", "z850", "z500", "z250",
]

# HighRes channels must match: ['t2m', 'u10', 'v10', 'qpepre']
DEFAULT_HIGHRES_VARIABLES = [
    "t2m", "u10", "v10", "qpepre",
]

DEFAULT_INVARIANT_VARIABLES = ["lsm", "orog"]


# ---------------------------------------------------------------------------
# Timestamp extraction helpers
# ---------------------------------------------------------------------------

def extract_timestamp_from_path(path: pathlib.Path) -> Optional[datetime]:
    """
    Extract timestamp from file path.
    
    Supports formats:
    - EC-pangu_2025120300-0.grb (YYYYMMDDHH)
    - wrfout_d02_2025-12-03_00:00:00 (YYYY-MM-DD_HH:MM:SS)
    - qpepre_202512030000-202512030100_1_h.txt (YYYYMMDDHHMM)
    - Directory names like 2025120300
    
    Returns:
        datetime object if found, None otherwise
    """
    path_str = str(path)
    
    # Pattern 1.1: EC-pangu Forecast format with offset (Base Time + Offset Hours)
    # e.g., EC-pangu_2025120300-0.grb, EC-pangu_2025120300-6.grb, EC-pangu_2025120300-100.grb
    # Regex breakdown:
    #   EC-pangu_       : Literal prefix
    #   (\d{4})         : Year
    #   (\d{2})         : Month
    #   (\d{2})         : Day
    #   (\d{2})         : Hour (Base)
    #   -               : Separator
    #   (\d+)           : Offset hours (1 or more digits)
    match = re.search(r'EC-pangu_(\d{4})(\d{2})(\d{2})(\d{2})-(\d+)', path_str)
    if match:
        year, month, day, hour, offset_hours = map(int, match.groups())
        base_time = datetime(year, month, day, hour)
        return base_time + timedelta(hours=offset_hours)
    
    # Pattern 1.2: WRF output format - wrfout_d02_2025-12-03_00:00:00
    match = re.search(r'(\d{4})-(\d{2})-(\d{2})_(\d{2}):(\d{2}):(\d{2})', path_str)
    if match:
        year, month, day, hour, minute, second = map(int, match.groups())
        return datetime(year, month, day, hour, minute, second)
    
    # Pattern 2: EC-pangu format - 2025120300 (YYYYMMDDHH)
    match = re.search(r'(\d{4})(\d{2})(\d{2})(\d{2})(?:-|\.|_|/)', path_str)
    if match:
        year, month, day, hour = map(int, match.groups())
        return datetime(year, month, day, hour)
    
    # Pattern 3: QPEPRE format - 202512030000 (YYYYMMDDHHMM)
    match = re.search(r'qpepre_(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})', path_str)
    if match:
        year, month, day, hour, minute = map(int, match.groups())
        return datetime(year, month, day, hour, minute)
    
    # Pattern 4: Directory format - /2025120300/
    match = re.search(r'/(\d{4})(\d{2})(\d{2})(\d{2})/', path_str)
    if match:
        year, month, day, hour = map(int, match.groups())
        return datetime(year, month, day, hour)
    
    return None


def infer_timestamp_from_config(config: "InferenceConfig") -> Optional[datetime]:
    """
    Infer timestamp from config file paths.
    
    Tries to extract timestamp from GRIB, RWRF, or QPEPRE paths.
    """
    # Try each path in order of reliability
    for path in [config.rwrf_path, config.grib_path, config.qpepre_path]:
        if path:
            ts = extract_timestamp_from_path(path)
            if ts:
                logger.info("Inferred timestamp %s from path: %s", ts, path)
                return ts
    return None


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class InferenceConfig:
    """Configuration for inference data preprocessing."""
    # Input paths
    grib_folder: Optional[pathlib.Path] = None  # Folder containing multiple GRIB files
    rwrf_path: Optional[pathlib.Path] = None
    qpepre_path: Optional[pathlib.Path] = None
    
    # Output path
    output_path: pathlib.Path = field(default_factory=lambda: pathlib.Path("./output"))
    
    # Inference settings
    n_steps: int = 12  # Number of forecast timesteps to prepare data for
    dt_hours: int = 1  # Hours between each inference timestep
    grib_base_time: Optional[datetime] = None  # GRIB forecast base time (if different from valid time)
    
    # Domain configuration
    domain_size: Tuple[int, int] = (224, 128)  # (lat, lon)
    lon_bounds: Tuple[float, float] = (119.75, 122.25)
    lat_bounds: Tuple[float, float] = (21.6, 25.6)
    
    # Variables
    lowres_variables: List[str] = field(default_factory=lambda: DEFAULT_LOWRES_VARIABLES.copy())
    highres_variables: List[str] = field(default_factory=lambda: DEFAULT_HIGHRES_VARIABLES.copy())
    invariant_variables: List[str] = field(default_factory=lambda: DEFAULT_INVARIANT_VARIABLES.copy())
    
    # Processing options
    overwrite: bool = True
    resample_mode: str = "interpolate"
    
    @classmethod
    def from_yaml(cls, path: str) -> "InferenceConfig":
        """Load config from YAML file."""
        if not HAS_YAML:
            raise RuntimeError("PyYAML is required to load config from file")
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        
        # Parse grib_base_time if provided
        grib_base_time = None
        if cfg.get("grib-base-time"):
            grib_base_time = datetime.fromisoformat(cfg["grib-base-time"])
        
        return cls(
            grib_folder=pathlib.Path(cfg["grib-folder"]) if cfg.get("grib-folder") else None,
            rwrf_path=pathlib.Path(cfg["rwrf-path"]) if cfg.get("rwrf-path") else None,
            qpepre_path=pathlib.Path(cfg["qpepre-path"]) if cfg.get("qpepre-path") else None,
            output_path=pathlib.Path(cfg.get("output-path", "./output")),
            n_steps=cfg.get("n-steps", 12),
            dt_hours=cfg.get("dt-hours", 1),
            grib_base_time=grib_base_time,
            domain_size=tuple(cfg.get("domain-size", [224, 128])),
            lon_bounds=tuple(cfg.get("lon-bounds", [119.75, 122.25])),
            lat_bounds=tuple(cfg.get("lat-bounds", [21.6, 25.6])),
            lowres_variables=cfg.get("lowres-variables", DEFAULT_LOWRES_VARIABLES.copy()),
            highres_variables=cfg.get("highres-variables", DEFAULT_HIGHRES_VARIABLES.copy()),
            invariant_variables=cfg.get("invariant-variables", DEFAULT_INVARIANT_VARIABLES.copy()),
            overwrite=cfg.get("overwrite", True),
            resample_mode=cfg.get("resample-mode", "interpolate"),
        )


# ---------------------------------------------------------------------------
# Interpolation helpers
# ---------------------------------------------------------------------------

def search_bounds_1d(arr: np.ndarray, vmin: float, vmax: float) -> Tuple[int, int]:
    """Find indices that bound [vmin, vmax] in a 1D array."""
    arr = np.asarray(arr)
    if arr[0] <= arr[-1]:
        lo = int(np.searchsorted(arr, vmin, side="left"))
        hi = int(np.searchsorted(arr, vmax, side="right") - 1)
    else:
        arr_rev = arr[::-1]
        lo_r = int(np.searchsorted(arr_rev, vmax, side="left"))
        hi_r = int(np.searchsorted(arr_rev, vmin, side="right") - 1)
        n = arr.size
        lo = n - 1 - hi_r
        hi = n - 1 - lo_r
    lo = max(0, min(lo, arr.size - 1))
    hi = max(0, min(hi, arr.size - 1))
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def resize_to_domain(
    data: np.ndarray,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    domain_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resize data to target domain size using bilinear interpolation."""
    from scipy.interpolate import RegularGridInterpolator
    
    target_h, target_w = domain_size
    src_h, src_w = data.shape[-2:]
    
    if src_h == target_h and src_w == target_w:
        return data, lon_grid, lat_grid
    
    lat1d = lat_grid[:, 0] if lat_grid.ndim == 2 else lat_grid
    lon1d = lon_grid[0, :] if lon_grid.ndim == 2 else lon_grid
    
    # Create new coordinate grids
    lat_new = np.linspace(lat1d.min(), lat1d.max(), target_h)
    lon_new = np.linspace(lon1d.min(), lon1d.max(), target_w)
    lon_new_grid, lat_new_grid = np.meshgrid(lon_new, lat_new)
    
    # Create interpolation points
    pts = np.stack([lat_new_grid.ravel(), lon_new_grid.ravel()], axis=-1)
    
    # Handle 2D and higher-dimensional arrays
    if data.ndim == 2:
        interpolator = RegularGridInterpolator(
            (lat1d, lon1d), data, bounds_error=False, fill_value=np.nan
        )
        result = interpolator(pts).reshape(target_h, target_w)
    else:
        orig_shape = data.shape[:-2]
        data_flat = data.reshape(-1, src_h, src_w)
        results = []
        for i in range(data_flat.shape[0]):
            interpolator = RegularGridInterpolator(
                (lat1d, lon1d), data_flat[i], bounds_error=False, fill_value=np.nan
            )
            results.append(interpolator(pts).reshape(target_h, target_w))
        result = np.array(results).reshape(*orig_shape, target_h, target_w)
    
    return result.astype(np.float32), lon_new_grid, lat_new_grid


# ---------------------------------------------------------------------------
# GRIB processing
# ---------------------------------------------------------------------------

def parse_grib_variable(shortName: str, level: int, typeOfLevel: str) -> Optional[str]:
    """Convert GRIB shortName + level to logical variable name."""
    if shortName in ["10u", "10v", "2t", "msl", "sp", "tcwv"]:
        return GRIB_SHORTNAME_MAP.get(shortName)
    
    if shortName in ["z", "q", "t", "u", "v"] and typeOfLevel == "isobaricInhPa":
        base = GRIB_SHORTNAME_MAP.get(shortName)
        if base:
            return f"{base}{level}"
    
    return None


def process_grib_file(
    grib_path: pathlib.Path,
    config: InferenceConfig,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Process GRIB file and extract requested low-resolution variables.
    
    Returns:
        Tuple of (variables_dict, lon_grid, lat_grid)
    """
    if not HAS_PYGRIB:
        raise RuntimeError("pygrib is required to process GRIB files")
    
    logger.info("Processing GRIB file: %s", grib_path)
    
    grbs = pygrib.open(str(grib_path))
    messages = list(grbs)
    
    results: Dict[str, np.ndarray] = {}
    lon_template: Optional[np.ndarray] = None
    lat_template: Optional[np.ndarray] = None
    
    # Build index of available variables
    available: Dict[str, pygrib.gribmessage] = {}
    for grb in messages:
        logical = parse_grib_variable(grb.shortName, grb.level, grb.typeOfLevel)
        if logical:
            available[logical] = grb
    
    logger.info("Available GRIB variables: %s", sorted(available.keys()))
    
    # Extract lat/lon bounds
    lon_min, lon_max = config.lon_bounds
    lat_min, lat_max = config.lat_bounds
    
    for var_name in config.lowres_variables:
        if var_name not in available:
            logger.warning("Variable %s not found in GRIB file", var_name)
            continue
        
        grb = available[var_name]
        data = grb.values
        lats, lons = grb.latlons()
        
        # Get 1D lat/lon arrays (GRIB data is usually regular grid)
        if lats.ndim == 2:
            lat1d = lats[:, 0]
            lon1d = lons[0, :]
        else:
            lat1d = np.unique(lats)
            lon1d = np.unique(lons)
        
        # Handle longitude wrapping (0-360 vs -180-180)
        if lon_min < 0 and lon1d.min() >= 0:
            # Need to convert lon bounds to 0-360
            lon_min_adj = lon_min + 360 if lon_min < 0 else lon_min
            lon_max_adj = lon_max + 360 if lon_max < 0 else lon_max
        else:
            lon_min_adj, lon_max_adj = lon_min, lon_max
        
        # Find subset indices
        lat_lo, lat_hi = search_bounds_1d(lat1d, lat_min, lat_max)
        lon_lo, lon_hi = search_bounds_1d(lon1d, lon_min_adj, lon_max_adj)
        
        # Extract subset
        data_sub = data[lat_lo:lat_hi+1, lon_lo:lon_hi+1]
        lat_sub = lat1d[lat_lo:lat_hi+1]
        lon_sub = lon1d[lon_lo:lon_hi+1]
        
        # Create 2D coordinate grids
        lon_grid, lat_grid = np.meshgrid(lon_sub, lat_sub)
        
        # Resize to domain
        data_rs, lon_rs, lat_rs = resize_to_domain(
            data_sub, lon_grid, lat_grid, config.domain_size
        )
        
        results[var_name] = np.asarray(data_rs, dtype=np.float32)
        
        if lon_template is None:
            lon_template = lon_rs.astype(np.float32)
            lat_template = lat_rs.astype(np.float32)
    
    grbs.close()
    
    logger.info("Extracted %d variables from GRIB", len(results))
    return results, lon_template, lat_template


# ---------------------------------------------------------------------------
# RWRF processing
# ---------------------------------------------------------------------------

def process_rwrf_file(
    rwrf_path: pathlib.Path,
    config: InferenceConfig,
    target_time: Optional[datetime] = None,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Process RWRF NetCDF file and extract requested high-resolution variables.
    
    Returns:
        Tuple of (variables_dict, lon_grid, lat_grid)
    """
    if not HAS_NETCDF4:
        raise RuntimeError("netCDF4 is required to process RWRF files")
    
    logger.info("Processing RWRF file: %s", rwrf_path)
    
    results: Dict[str, np.ndarray] = {}
    lon_template: Optional[np.ndarray] = None
    lat_template: Optional[np.ndarray] = None
    
    lon_min, lon_max = config.lon_bounds
    lat_min, lat_max = config.lat_bounds
    
    with Dataset(str(rwrf_path), "r") as ds:
        # Get lat/lon arrays
        lat = np.asarray(ds.variables["XLAT"][0, :, 0])
        lon = np.asarray(ds.variables["XLONG"][0, 0, :])
        
        # Find subset indices
        lat_lo, lat_hi = search_bounds_1d(lat, lat_min, lat_max)
        lon_lo, lon_hi = search_bounds_1d(lon, lon_min, lon_max)
        lat_slice = slice(lat_lo, lat_hi + 1)
        lon_slice = slice(lon_lo, lon_hi + 1)
        
        lat_sel = lat[lat_slice]
        lon_sel = lon[lon_slice]
        lon_grid, lat_grid = np.meshgrid(lon_sel, lat_sel)
        
        # Find time index
        time_idx = 0
        if target_time is not None:
            times_var = ds.variables.get("Times")
            if times_var is not None:
                values = [b"".join(row).decode("utf-8").strip() for row in times_var[:]]
                dt_array = np.array([np.datetime64(val.replace("_", "T")) for val in values])
                target64 = np.datetime64(target_time.strftime("%Y-%m-%dT%H"))
                time_idx = int(np.argmin(np.abs(dt_array - target64)))
        
        for logical in config.highres_variables:
            if logical == "qpepre":
                continue  # Handle separately
            
            nc_name = RWRF_VARIABLE_MAP.get(logical)
            if nc_name is None:
                logger.warning("Unknown RWRF variable mapping: %s", logical)
                continue
            
            if nc_name not in ds.variables:
                logger.debug("Variable %s (%s) not in RWRF file", logical, nc_name)
                continue
            
            var = ds.variables[nc_name]
            slices = [slice(None)] * var.ndim
            dims = var.dimensions
            
            # Handle time dimension
            if "Time" in dims:
                slices[dims.index("Time")] = time_idx
            elif "time" in dims:
                slices[dims.index("time")] = time_idx
            
            # Handle pressure level if applicable
            level_match = re.match(r"([a-zA-Z]+)(\d+)$", logical)
            if level_match:
                level = int(level_match.group(2))
                level_idx = DEFAULT_PRES_IDX.get(level)
                if level_idx is not None:
                    for dim_name in dims:
                        if dim_name.lower() in LEVEL_DIM_NAMES_LOWER:
                            dim_idx = dims.index(dim_name)
                            if level_idx < var.shape[dim_idx]:
                                slices[dim_idx] = level_idx
                            break
            
            # Apply spatial slices
            slices[-2] = lat_slice
            slices[-1] = lon_slice
            
            data = np.asarray(var[tuple(slices)], dtype=np.float32)
            data = np.squeeze(data)
            
            # Resize to domain
            data_rs, lon_rs, lat_rs = resize_to_domain(
                data, lon_grid, lat_grid, config.domain_size
            )
            
            results[logical] = np.asarray(data_rs, dtype=np.float32)
            
            if lon_template is None:
                lon_template = lon_rs.astype(np.float32)
                lat_template = lat_rs.astype(np.float32)
    
    logger.info("Extracted %d variables from RWRF", len(results))
    return results, lon_template, lat_template


# ---------------------------------------------------------------------------
# QPEPRE processing
# ---------------------------------------------------------------------------

def process_qpepre_file(
    qpepre_path: pathlib.Path,
    config: InferenceConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process QPEPRE precipitation text file.
    
    Returns:
        Tuple of (data_array, lon_grid, lat_grid)
    """
    logger.info("Processing QPEPRE file: %s", qpepre_path)
    
    raw = np.loadtxt(qpepre_path, dtype=np.float32)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    
    _, lon_vals, lat_vals, rain_vals = raw.T
    
    lat_unique = np.unique(lat_vals)
    lon_unique = np.unique(lon_vals)
    
    # Build grid
    val_grid = np.full((lat_unique.size, lon_unique.size), np.nan, dtype=np.float32)
    lat_to_idx = {v: i for i, v in enumerate(lat_unique)}
    lon_to_idx = {v: i for i, v in enumerate(lon_unique)}
    
    for lon_v, lat_v, rain in zip(lon_vals, lat_vals, rain_vals):
        val_grid[lat_to_idx[lat_v], lon_to_idx[lon_v]] = rain
    
    lon_grid, lat_grid = np.meshgrid(lon_unique, lat_unique)
    
    # Resize to domain
    data_rs, lon_rs, lat_rs = resize_to_domain(
        val_grid, lon_grid, lat_grid, config.domain_size
    )
    
    logger.info("QPEPRE data shape: %s -> %s", val_grid.shape, data_rs.shape)
    return np.asarray(data_rs, dtype=np.float32), lon_rs, lat_rs


# ---------------------------------------------------------------------------
# Zarr writing
# ---------------------------------------------------------------------------

def write_zarr_store(
    store_path: pathlib.Path,
    data_var: str,
    data_cube: np.ndarray,
    time_coord: np.ndarray,
    channels: List[str],
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    overwrite: bool = True,
) -> None:
    """Write data to Zarr store in StormCast format."""
    if not HAS_XARRAY or not HAS_ZARR:
        raise RuntimeError("xarray and zarr are required to write Zarr stores")
    
    if store_path.exists():
        if overwrite:
            shutil.rmtree(store_path)
        else:
            logger.info("Zarr store exists at %s; skipping", store_path)
            return
    
    store_path.parent.mkdir(parents=True, exist_ok=True)
    
    coords = {
        "time": time_coord,
        "channel": np.array(channels, dtype="object"),
        "y": np.arange(data_cube.shape[2], dtype=np.int32),
        "x": np.arange(data_cube.shape[3], dtype=np.int32),
    }
    
    # Add valid mask (all valid for inference)
    valid_arr = np.ones((data_cube.shape[0], 1), dtype=bool)
    coords["flag"] = np.array(["valid"], dtype="object")
    
    ds = xr.Dataset(
        {
            data_var: (("time", "channel", "y", "x"), data_cube),
            "valid": (("time", "flag"), valid_arr),
        },
        coords=coords,
    ).assign_coords(
        latitude=(("y", "x"), lat_grid.astype(np.float32)),
        longitude=(("y", "x"), lon_grid.astype(np.float32)),
    )
    
    encoding = {
        data_var: {"dtype": "float32"},
        "valid": {"dtype": "bool"},
    }
    
    ds.to_zarr(store_path, mode="w", consolidated=True, encoding=encoding)
    zarr.consolidate_metadata(store_path)
    logger.info("Wrote %s to %s", data_var, store_path)


def write_stats(
    stats_dir: pathlib.Path,
    means: np.ndarray,
    stds: np.ndarray,
    channels: List[str],
) -> None:
    """Write normalization statistics."""
    stats_dir.mkdir(parents=True, exist_ok=True)
    np.save(stats_dir / "means.npy", means)
    np.save(stats_dir / "stds.npy", stds)
    with open(stats_dir / "channels.txt", "w", encoding="utf-8") as f:
        for ch in channels:
            f.write(f"{ch}\n")
    logger.info("Wrote stats to %s", stats_dir)


def write_invariants(
    invariants_dir: pathlib.Path,
    data: np.ndarray,
    channels: List[str],
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    overwrite: bool = True,
) -> None:
    """Write invariants to Zarr store."""
    if not HAS_XARRAY or not HAS_ZARR:
        raise RuntimeError("xarray and zarr are required to write Zarr stores")
    
    store = invariants_dir / "invariants.zarr"
    if store.exists():
        if overwrite:
            shutil.rmtree(store)
        else:
            logger.info("Invariants exist at %s; skipping", store)
            return
    
    invariants_dir.mkdir(parents=True, exist_ok=True)
    
    _, H, W = data.shape
    coords = {
        "channel": np.array(channels, dtype="object"),
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
    ds.to_zarr(store, mode="w", consolidated=True, encoding=encoding)
    zarr.consolidate_metadata(store)
    logger.info("Wrote invariants to %s", store)


# ---------------------------------------------------------------------------
# GRIB file discovery and sorting
# ---------------------------------------------------------------------------

def discover_grib_files(
    grib_folder: pathlib.Path, 
    base_time: Optional[datetime] = None
) -> List[Tuple[pathlib.Path, datetime]]:
    """
    Discover all .grb files in a folder and extract their timestamps.
    
    Args:
        grib_folder: Folder containing GRIB files
        base_time: Optional base time for forecast files. If provided, will extract
                   offset from filename and calculate valid time as base_time + offset.
    
    Returns:
        List of (file_path, timestamp) tuples, sorted by timestamp
    """
    grib_files = []
    
    # Search for all .grb files recursively
    for grb_file in grib_folder.rglob("*.grb"):
        if base_time is not None:
            # Extract offset from filename (e.g., EC-pangu_2025120300-12.grb -> 12)
            match = re.search(r'-(\d+)\.grb$', grb_file.name)
            if match:
                offset_hours = int(match.group(1))
                timestamp = base_time + timedelta(hours=offset_hours)
                logger.debug(f"  {grb_file.name} -> base={base_time} + {offset_hours}h = {timestamp}")
            else:
                # Fallback to extracting from path
                timestamp = extract_timestamp_from_path(grb_file)
                if not timestamp:
                    logger.warning(f"Could not extract timestamp from {grb_file}")
                    continue
        else:
            timestamp = extract_timestamp_from_path(grb_file)
            if not timestamp:
                logger.warning(f"Could not extract timestamp from {grb_file}, skipping")
                continue
        
        grib_files.append((grb_file, timestamp))
    
    # Sort by timestamp
    grib_files.sort(key=lambda x: x[1])
    
    logger.info(f"Found {len(grib_files)} GRIB files in {grib_folder}")
    if base_time:
        logger.info(f"Using base time: {base_time}")
    for grb_file, ts in grib_files:
        logger.info(f"  {ts}: {grb_file.name}")
    
    return grib_files


# ---------------------------------------------------------------------------
# Main processing function
# ---------------------------------------------------------------------------

def process_single_hour(
    config: InferenceConfig,
    timestamp: Optional[datetime] = None,
    output_name: str = "inference",
) -> None:
    """
    Process a single hour of data from GRIB + RWRF + QPEPRE sources.
    
    Args:
        config: Processing configuration
        timestamp: Target timestamp (used for naming and RWRF time selection).
                   Priority: 1) CLI --timestamp, 2) inferred from file paths, 3) current time
        output_name: Base name for output zarr files
    """
    if timestamp is not None:
        # CLI timestamp has highest priority
        logger.info("Using timestamp from command line: %s", timestamp)
    else:
        # Try to infer timestamp from file paths
        timestamp = infer_timestamp_from_config(config)
        if timestamp is None:
            logger.warning(
                "Could not infer timestamp from file paths, using current time. "
                "Consider specifying --timestamp explicitly."
            )
            timestamp = datetime.now().replace(minute=0, second=0, microsecond=0)
            logger.info("Using current timestamp: %s", timestamp)
    
    output_base = config.output_path
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Time coordinate
    time_coord = np.array([np.datetime64(timestamp)], dtype="datetime64[ns]")
    
    # Process low-resolution (GRIB)
    lowres_data = {}
    lowres_lon = None
    lowres_lat = None
    
    if config.grib_path and config.grib_path.exists():
        lowres_data, lowres_lon, lowres_lat = process_grib_file(config.grib_path, config)
    else:
        logger.warning("GRIB path not specified or does not exist: %s", config.grib_path)
    
    # Process high-resolution (RWRF)
    highres_data = {}
    highres_lon = None
    highres_lat = None
    
    if config.rwrf_path and config.rwrf_path.exists():
        highres_data, highres_lon, highres_lat = process_rwrf_file(
            config.rwrf_path, config, target_time=timestamp
        )
    else:
        logger.warning("RWRF path not specified or does not exist: %s", config.rwrf_path)
    
    # Process QPEPRE
    if config.qpepre_path and config.qpepre_path.exists():
        qpepre_data, qpepre_lon, qpepre_lat = process_qpepre_file(config.qpepre_path, config)
        highres_data["qpepre"] = qpepre_data
        if highres_lon is None:
            highres_lon = qpepre_lon
            highres_lat = qpepre_lat
    else:
        logger.warning("QPEPRE path not specified or does not exist: %s", config.qpepre_path)
    
    # Build data cubes
    H, W = config.domain_size
    
    # LowRes cube
    if lowres_data and lowres_lon is not None:
        lowres_channels = [v for v in config.lowres_variables if v in lowres_data]
        lowres_cube = np.zeros((1, len(lowres_channels), H, W), dtype=np.float32)
        for i, ch in enumerate(lowres_channels):
            lowres_cube[0, i] = lowres_data[ch]
        
        # Write LowRes zarr
        lowres_dir = output_base / "LowRes"
        lowres_dir.mkdir(parents=True, exist_ok=True)
        write_zarr_store(
            lowres_dir / f"{output_name}.zarr",
            "LowRes",
            lowres_cube,
            time_coord,
            lowres_channels,
            lowres_lon,
            lowres_lat,
            overwrite=config.overwrite,
        )
        
        # Compute and write stats
        means = np.nanmean(lowres_cube, axis=(0, 2, 3))
        stds = np.nanstd(lowres_cube, axis=(0, 2, 3))
        stds = np.where(stds == 0, 1.0, stds)  # Avoid division by zero
        write_stats(lowres_dir / "stats", means, stds, lowres_channels)
    
    # HighRes cube
    if highres_data and highres_lon is not None:
        highres_channels = [v for v in config.highres_variables if v in highres_data]
        highres_cube = np.zeros((1, len(highres_channels), H, W), dtype=np.float32)
        for i, ch in enumerate(highres_channels):
            highres_cube[0, i] = highres_data[ch]
        
        # Write HighRes zarr
        highres_dir = output_base / "HighRes"
        highres_dir.mkdir(parents=True, exist_ok=True)
        write_zarr_store(
            highres_dir / f"{output_name}.zarr",
            "HighRes",
            highres_cube,
            time_coord,
            highres_channels,
            highres_lon,
            highres_lat,
            overwrite=config.overwrite,
        )
        
        # Compute and write stats
        means = np.nanmean(highres_cube, axis=(0, 2, 3))
        stds = np.nanstd(highres_cube, axis=(0, 2, 3))
        stds = np.where(stds == 0, 1.0, stds)
        write_stats(highres_dir / "stats", means, stds, highres_channels)
    
    # Extract and write invariants from RWRF if available
    if config.rwrf_path and config.rwrf_path.exists() and highres_lon is not None:
        invariant_data = extract_invariants(config.rwrf_path, config)
        if invariant_data:
            inv_array, inv_lon, inv_lat = invariant_data
            write_invariants(
                output_base / "invariants",
                inv_array,
                config.invariant_variables,
                inv_lon,
                inv_lat,
                overwrite=config.overwrite,
            )
    
    logger.info("Processing complete. Output written to: %s", output_base)


def process_multi_timestep(
    config: InferenceConfig,
    timestamp: Optional[datetime] = None,
) -> None:
    """
    Process multiple GRIB files for multi-timestep inference.
    
    Creates a LowRes zarr with multiple timesteps (one per 6 hours),
    HighRes zarr with single timestep (initial condition),
    and invariants zarr.
    
    Args:
        config: Processing configuration with grib_folder, n_steps, dt_hours
        timestamp: Initial condition timestamp for RWRF/QPEPRE data. If None, uses first GRIB timestamp.
    """
    if not config.grib_folder or not config.grib_folder.exists():
        raise ValueError(f"GRIB folder does not exist: {config.grib_folder}")
    
    # Discover all GRIB files in folder
    grib_files = discover_grib_files(config.grib_folder, base_time=config.grib_base_time)
    if not grib_files:
        raise ValueError(f"No GRIB files found in {config.grib_folder}")
    
    # Calculate how many LowRes timesteps we need
    # Inference outputs are at hours 1, 2, 3, ..., n_steps (starting from +1)
    # We need LowRes data up to and including hour n_steps
    # For n_steps with dt_hours=1, we need one LowRes per 6 hours
    # e.g., n_steps=6, dt=1 -> outputs at hours 1-6, need 2 LowRes files (0-5h, 6-11h)
    # e.g., n_steps=12, dt=1 -> outputs at hours 1-12, need 3 LowRes files (0-5h, 6-11h, 12-17h)
    lowres_dt = 6  # Low-resolution data timestep (6 hours)
    max_forecast_hour = config.n_steps * config.dt_hours
    # Add 1 because we need data for the transition at hour 6, 12, 18, etc.
    num_lowres_needed = (max_forecast_hour // lowres_dt) + 1
    
    logger.info(f"Processing for {config.n_steps} inference steps with dt={config.dt_hours}h")
    logger.info(f"Need {num_lowres_needed} LowRes timesteps (one per {lowres_dt}h)")
    
    if len(grib_files) < num_lowres_needed:
        logger.warning(
            f"Only {len(grib_files)} GRIB files found, but need {num_lowres_needed}. "
            f"Will process available files only."
        )
        num_lowres_needed = len(grib_files)
    
    # Determine base timestamp for RWRF/QPEPRE
    if timestamp is not None:
        base_timestamp = timestamp
        logger.info(f"Using provided timestamp as initial condition: {base_timestamp}")
        
        # Find the starting GRIB file index that matches or is closest to base_timestamp
        start_idx = 0
        for idx, (grib_path, grib_ts) in enumerate(grib_files):
            if grib_ts >= base_timestamp:
                start_idx = idx
                break
        
        if grib_files[start_idx][1] != base_timestamp:
            logger.warning(
                f"Exact match for timestamp {base_timestamp} not found. "
                f"Using closest GRIB file: {grib_files[start_idx][0].name} at {grib_files[start_idx][1]}"
            )
    else:
        start_idx = 0
        base_timestamp = grib_files[0][1]
        logger.info(f"Using first GRIB timestamp as base: {base_timestamp}")
    
    # Check if we have enough GRIB files from start_idx
    available_from_start = len(grib_files) - start_idx
    if available_from_start < num_lowres_needed:
        logger.warning(
            f"Only {available_from_start} GRIB files available from {base_timestamp}, "
            f"but need {num_lowres_needed}. Will process available files only."
        )
        num_lowres_needed = available_from_start
    
    logger.info(f"Starting from GRIB file index {start_idx}: {grib_files[start_idx][0].name}")
    
    output_base = config.output_path
    output_base.mkdir(parents=True, exist_ok=True)
    
    H, W = config.domain_size
    
    # Process all needed GRIB files for LowRes
    logger.info("Processing LowRes data from GRIB files...")
    lowres_channels = None
    lowres_lon = None
    lowres_lat = None
    lowres_cube_list = []
    time_coords = []
    
    for i in range(num_lowres_needed):
        grib_idx = start_idx + i
        grib_path, grib_timestamp = grib_files[grib_idx]
        logger.info(f"Processing LowRes timestep {i}: {grib_timestamp} from {grib_path.name}")
        
        lowres_data, lon_grid, lat_grid = process_grib_file(grib_path, config)
        
        if lowres_channels is None:
            lowres_channels = [v for v in config.lowres_variables if v in lowres_data]
            lowres_lon = lon_grid
            lowres_lat = lat_grid
        
        # Build data cube for this timestep
        lowres_slice = np.zeros((len(lowres_channels), H, W), dtype=np.float32)
        for j, ch in enumerate(lowres_channels):
            if ch in lowres_data:
                lowres_slice[j] = lowres_data[ch]
        
        lowres_cube_list.append(lowres_slice)
        time_coords.append(np.datetime64(grib_timestamp))
    
    # Stack all LowRes timesteps
    lowres_cube = np.stack(lowres_cube_list, axis=0)  # [T, C, H, W]
    time_coord = np.array(time_coords, dtype="datetime64[ns]")
    
    logger.info(f"LowRes cube shape: {lowres_cube.shape}")
    
    # Write LowRes zarr
    lowres_dir = output_base / "LowRes"
    lowres_dir.mkdir(parents=True, exist_ok=True)
    write_zarr_store(
        lowres_dir / "inference.zarr",
        "LowRes",
        lowres_cube,
        time_coord,
        lowres_channels,
        lowres_lon,
        lowres_lat,
        overwrite=config.overwrite,
    )
    
    # Compute and write LowRes stats
    means = np.nanmean(lowres_cube, axis=(0, 2, 3))
    stds = np.nanstd(lowres_cube, axis=(0, 2, 3))
    stds = np.where(stds == 0, 1.0, stds)
    write_stats(lowres_dir / "stats", means, stds, lowres_channels)
    
    # Process HighRes data (initial condition only)
    if config.rwrf_path and config.rwrf_path.exists():
        logger.info("Processing HighRes initial condition from RWRF...")
        highres_data, highres_lon, highres_lat = process_rwrf_file(
            config.rwrf_path, config, target_time=base_timestamp
        )
        
        # Process QPEPRE if available
        if config.qpepre_path and config.qpepre_path.exists():
            qpepre_data, qpepre_lon, qpepre_lat = process_qpepre_file(config.qpepre_path, config)
            highres_data["qpepre"] = qpepre_data
            if highres_lon is None:
                highres_lon = qpepre_lon
                highres_lat = qpepre_lat
        
        # Build HighRes cube (single timestep)
        highres_channels = [v for v in config.highres_variables if v in highres_data]
        highres_cube = np.zeros((1, len(highres_channels), H, W), dtype=np.float32)
        for i, ch in enumerate(highres_channels):
            highres_cube[0, i] = highres_data[ch]
        
        # Write HighRes zarr
        highres_dir = output_base / "HighRes"
        highres_dir.mkdir(parents=True, exist_ok=True)
        write_zarr_store(
            highres_dir / "inference.zarr",
            "HighRes",
            highres_cube,
            np.array([np.datetime64(base_timestamp)], dtype="datetime64[ns]"),
            highres_channels,
            highres_lon,
            highres_lat,
            overwrite=config.overwrite,
        )
        
        # Compute and write HighRes stats
        means = np.nanmean(highres_cube, axis=(0, 2, 3))
        stds = np.nanstd(highres_cube, axis=(0, 2, 3))
        stds = np.where(stds == 0, 1.0, stds)
        write_stats(highres_dir / "stats", means, stds, highres_channels)
        
        # Extract and write invariants
        invariant_data = extract_invariants(config.rwrf_path, config)
        if invariant_data:
            inv_array, inv_lon, inv_lat = invariant_data
            write_invariants(
                output_base / "invariants",
                inv_array,
                config.invariant_variables,
                inv_lon,
                inv_lat,
                overwrite=config.overwrite,
            )
    else:
        logger.warning("RWRF path not specified, skipping HighRes and invariants")
    
    logger.info("Multi-timestep processing complete. Output written to: %s", output_base)
    logger.info(f"LowRes: {num_lowres_needed} timesteps covering {num_lowres_needed * lowres_dt} hours")
    logger.info(f"HighRes: 1 timestep (initial condition)")


def extract_invariants(
    rwrf_path: pathlib.Path,
    config: InferenceConfig,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Extract invariant fields (lsm, orog) from RWRF file."""
    if not HAS_NETCDF4:
        return None
    
    inv_map = {
        "lsm": "LANDMASK",
        "orog": "HGT",
    }
    
    results = []
    lon_template = None
    lat_template = None
    
    lon_min, lon_max = config.lon_bounds
    lat_min, lat_max = config.lat_bounds
    
    with Dataset(str(rwrf_path), "r") as ds:
        lat = np.asarray(ds.variables["XLAT"][0, :, 0])
        lon = np.asarray(ds.variables["XLONG"][0, 0, :])
        
        lat_lo, lat_hi = search_bounds_1d(lat, lat_min, lat_max)
        lon_lo, lon_hi = search_bounds_1d(lon, lon_min, lon_max)
        lat_slice = slice(lat_lo, lat_hi + 1)
        lon_slice = slice(lon_lo, lon_hi + 1)
        
        lat_sel = lat[lat_slice]
        lon_sel = lon[lon_slice]
        lon_grid, lat_grid = np.meshgrid(lon_sel, lat_sel)
        
        for inv_name in config.invariant_variables:
            nc_name = inv_map.get(inv_name)
            if nc_name is None or nc_name not in ds.variables:
                logger.warning("Invariant %s not found in RWRF", inv_name)
                # Fill with zeros/ones as placeholder
                placeholder = np.zeros(config.domain_size, dtype=np.float32)
                results.append(placeholder)
                continue
            
            var = ds.variables[nc_name]
            if var.ndim >= 3:
                data = np.asarray(var[0, lat_slice, lon_slice], dtype=np.float32)
            else:
                data = np.asarray(var[lat_slice, lon_slice], dtype=np.float32)
            
            data_rs, lon_rs, lat_rs = resize_to_domain(
                data, lon_grid, lat_grid, config.domain_size
            )
            
            results.append(np.asarray(data_rs, dtype=np.float32))
            
            if lon_template is None:
                lon_template = lon_rs.astype(np.float32)
                lat_template = lat_rs.astype(np.float32)
    
    if not results or lon_template is None:
        return None
    
    return np.stack(results, axis=0), lon_template, lat_template


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert GRIB + RWRF + QPEPRE data to Zarr for inference"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--grib-folder",
        type=str,
        help="Path to folder containing GRIB files (low-resolution input)",
    )
    parser.add_argument(
        "--rwrf-path",
        type=str,
        help="Path to RWRF NetCDF file (high-resolution input)",
    )
    parser.add_argument(
        "--qpepre-path",
        type=str,
        help="Path to QPEPRE text file (precipitation)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for Zarr stores (if not specified, uses value from config file)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="inference",
        help="Base name for output Zarr files",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=12,
        help="Number of forecast timesteps to prepare data for (default: 12)",
    )
    parser.add_argument(
        "--dt-hours",
        type=int,
        default=1,
        help="Hours between each inference timestep (default: 1)",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        help="Target timestamp in ISO format (e.g., 2025-12-03T00:00:00). "
             "Has highest priority - overrides timestamp inferred from file paths.",
    )
    parser.add_argument(
        "--grib-base-time",
        type=str,
        help="GRIB forecast base time in ISO format (e.g., 2025-12-03T00:00:00). "
             "Use this when GRIB files are named with base time + offset (e.g., EC-pangu_2025120300-12.grb). "
             "The script will look for files matching: basename-{offset}.grb where offset = (valid_time - base_time) / dt_hours. "
             "Example: If base time is 2025-12-03T00:00:00 and you need data for 2025-12-03T12:00:00, "
             "it will look for EC-pangu_2025120300-12.grb",
    )
    parser.add_argument(
        "--domain-size",
        type=int,
        nargs=2,
        default=[224, 128],
        help="Domain size as [height, width]",
    )
    parser.add_argument(
        "--lon-bounds",
        type=float,
        nargs=2,
        default=[119.75, 122.25],
        help="Longitude bounds as [min, max]",
    )
    parser.add_argument(
        "--lat-bounds",
        type=float,
        nargs=2,
        default=[21.6, 25.6],
        help="Latitude bounds as [min, max]",
    )
    parser.add_argument(
        "--lowres-variables",
        type=str,
        help="Comma-separated list of low-resolution variables (e.g., 'mslp,t2m,u10,v10,...'). "
             "If not specified, uses default set matching training data.",
    )
    parser.add_argument(
        "--highres-variables",
        type=str,
        help="Comma-separated list of high-resolution variables (e.g., 't2m,u10,v10,qpepre'). "
             "If not specified, uses default set matching training data.",
    )
    parser.add_argument(
        "--invariant-variables",
        type=str,
        help="Comma-separated list of invariant variables (e.g., 'lsm,orog'). "
             "If not specified, uses default set.",
    )
    parser.add_argument(
        "--resample-mode",
        type=str,
        default="interpolate",
        choices=["interpolate", "nearest"],
        help="Resample mode for regridding (default: interpolate for bilinear interpolation)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=True,
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    # Load config from file or CLI args
    if args.config:
        config = InferenceConfig.from_yaml(args.config)
    else:
        config = InferenceConfig()
    
    # Override with CLI args
    if args.grib_folder:
        config.grib_folder = pathlib.Path(args.grib_folder)
    if args.rwrf_path:
        config.rwrf_path = pathlib.Path(args.rwrf_path)
    if args.qpepre_path:
        config.qpepre_path = pathlib.Path(args.qpepre_path)
    if args.output is not None:
        config.output_path = pathlib.Path(args.output)
    if args.n_steps:
        config.n_steps = args.n_steps
    if args.dt_hours:
        config.dt_hours = args.dt_hours
    if hasattr(args, 'grib_base_time') and args.grib_base_time:
        config.grib_base_time = datetime.fromisoformat(args.grib_base_time)
    if args.domain_size:
        config.domain_size = tuple(args.domain_size)
    if args.lon_bounds:
        config.lon_bounds = tuple(args.lon_bounds)
    if args.lat_bounds:
        config.lat_bounds = tuple(args.lat_bounds)
    
    # Parse variable lists from comma-separated strings
    if args.lowres_variables:
        config.lowres_variables = [v.strip() for v in args.lowres_variables.split(',') if v.strip()]
    if args.highres_variables:
        config.highres_variables = [v.strip() for v in args.highres_variables.split(',') if v.strip()]
    if args.invariant_variables:
        config.invariant_variables = [v.strip() for v in args.invariant_variables.split(',') if v.strip()]
    
    # Set resample mode
    if args.resample_mode:
        config.resample_mode = args.resample_mode
    
    config.overwrite = args.overwrite
    
    # Parse timestamp (used for RWRF/QPEPRE initial condition in multi-timestep mode)
    timestamp = None
    if args.timestamp:
        timestamp = datetime.fromisoformat(args.timestamp)
    
    # Decide whether to use multi-timestep processing
    if config.grib_folder:
        logger.info("Using multi-timestep processing mode")
        process_multi_timestep(config, timestamp=timestamp)
    else:
        logger.info("Using single-timestep processing mode (legacy)")
        
        # Process single timestep
        process_single_hour(
            config,
            timestamp=timestamp,
            output_name=args.output_name,
        )


if __name__ == "__main__":
    main()
