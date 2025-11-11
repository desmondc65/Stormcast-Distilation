from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from netCDF4 import Dataset, num2date

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(pathlib.Path(__file__).resolve().parent))
    from common import (
        get_logger,
        resize_to_domain,
        ensure_timezone_naive,
        search_bounds_1d,
    )
else:
    from .common import (
        get_logger,
        resize_to_domain,
        ensure_timezone_naive,
        search_bounds_1d,
    )

logger = get_logger()

@dataclass(frozen=True)
class VariableConfig:
    nc_var_name: str
    file_prefix: str


VARIABLE_CONFIGS: Dict[str, VariableConfig] = {
    "mslp": VariableConfig("mslp", "mslp"),
    "sp": VariableConfig("sp", "sp"),
    "t2m": VariableConfig("t2m", "t2m"),
    "u10": VariableConfig("u10", "u10"),
    "v10": VariableConfig("v10", "v10"),
    "tcwv": VariableConfig("tcwv", "tcwv"),
    "tp": VariableConfig("tp", "tp"),
    "q1000": VariableConfig("q1000", "q1000"),
    "q925": VariableConfig("q925", "q925"),
    "q850": VariableConfig("q850", "q850"),
    "q700": VariableConfig("q700", "q700"),
    "q500": VariableConfig("q500", "q500"),
    "q400": VariableConfig("q400", "q400"),
    "q300": VariableConfig("q300", "q300"),
    "q250": VariableConfig("q250", "q250"),
    "q200": VariableConfig("q200", "q200"),
    "q150": VariableConfig("q150", "q150"),
    "q100": VariableConfig("q100", "q100"),
    "q70": VariableConfig("q70", "q70"),
    "q50": VariableConfig("q50", "q50"),
    "t1000": VariableConfig("t1000", "t1000"),
    "t925": VariableConfig("t925", "t925"),
    "t850": VariableConfig("t850", "t850"),
    "t700": VariableConfig("t700", "t700"),
    "t500": VariableConfig("t500", "t500"),
    "t400": VariableConfig("t400", "t400"),
    "t300": VariableConfig("t300", "t300"),
    "t250": VariableConfig("t250", "t250"),
    "t200": VariableConfig("t200", "t200"),
    "t150": VariableConfig("t150", "t150"),
    "t100": VariableConfig("t100", "t100"),
    "t70": VariableConfig("t70", "t70"),
    "t50": VariableConfig("t50", "t50"),
    "u1000": VariableConfig("u1000", "u1000"),
    "u925": VariableConfig("u925", "u925"),
    "u850": VariableConfig("u850", "u850"),
    "u700": VariableConfig("u700", "u700"),
    "u500": VariableConfig("u500", "u500"),
    "u400": VariableConfig("u400", "u400"),
    "u300": VariableConfig("u300", "u300"),
    "u250": VariableConfig("u250", "u250"),
    "u200": VariableConfig("u200", "u200"),
    "u150": VariableConfig("u150", "u150"),
    "u100": VariableConfig("u100", "u100"),
    "u70": VariableConfig("u70", "u70"),
    "u50": VariableConfig("u50", "u50"),
    "v1000": VariableConfig("v1000", "v1000"),
    "v925": VariableConfig("v925", "v925"),
    "v850": VariableConfig("v850", "v850"),
    "v700": VariableConfig("v700", "v700"),
    "v500": VariableConfig("v500", "v500"),
    "v400": VariableConfig("v400", "v400"),
    "v300": VariableConfig("v300", "v300"),
    "v250": VariableConfig("v250", "v250"),
    "v200": VariableConfig("v200", "v200"),
    "v150": VariableConfig("v150", "v150"),
    "v100": VariableConfig("v100", "v100"),
    "v70": VariableConfig("v70", "v70"),
    "v50": VariableConfig("v50", "v50"),
    "z1000": VariableConfig("z1000", "z1000"),
    "z925": VariableConfig("z925", "z925"),
    "z850": VariableConfig("z850", "z850"),
    "z700": VariableConfig("z700", "z700"),
    "z500": VariableConfig("z500", "z500"),
    "z400": VariableConfig("z400", "z400"),
    "z300": VariableConfig("z300", "z300"),
    "z250": VariableConfig("z250", "z250"),
    "z200": VariableConfig("z200", "z200"),
    "z150": VariableConfig("z150", "z150"),
    "z100": VariableConfig("z100", "z100"),
    "z70": VariableConfig("z70", "z70"),
    "z50": VariableConfig("z50", "z50"),
}

DEFAULT_ERA5_VARIABLES: List[str] = [
    "mslp", "sp", "t2m", "u10", "v10", "tcwv",
    "q1000", "q850", "q500", "q250",
    "t1000", "t850", "t500", "t250",
    "u1000", "u850", "u500", "u250",
    "v1000", "v850", "v500", "v250",
    "z1000", "z850", "z500", "z250",
]

LEVEL_PATTERN = re.compile(r"([a-zA-Z]+)(\d+)$")
LEVEL_DIM_NAMES = (
    "pres_bottom_top",
    "pres_bottom_top_stag",
    "bottom_top",
    "bottom_top_stag",
    "level",
    "lev",
    "isobaricinpa",
    "isobaricInhPa",
    "pressure_level",
)
LEVEL_DIM_NAMES_LOWER = tuple(name.lower() for name in LEVEL_DIM_NAMES)


def _parse_era5_timestamp(key: str) -> Optional[datetime]:
    try:
        if len(key) == 10:
            return datetime.strptime(key, "%Y%m%d%H")
        if len(key) == 9:  # hour without leading zero
            return datetime.strptime(key[:8] + key[8:].zfill(2), "%Y%m%d%H")
        if len(key) == 6:
            return datetime.strptime(key + "01", "%Y%m%d")
    except Exception:
        return None
    return None


@dataclass(frozen=True)
class ERA5WorkerState:
    era5_index: Dict[str, Dict[str, str]]
    domain_size: Tuple[int, int]
    lon_bounds: Tuple[float, float]
    lat_bounds: Tuple[float, float]
    resample_mode: str
    pres_idx: Dict[int, int]


_ERA5_WORKER_STATE: ERA5WorkerState | None = None


def _era5_worker_init(state: ERA5WorkerState) -> None:
    global _ERA5_WORKER_STATE
    _ERA5_WORKER_STATE = state


def _era5_worker(dt: datetime, channels: Sequence[str]):
    if _ERA5_WORKER_STATE is None:
        raise RuntimeError("ERA5 worker state not initialized")
    results: Dict[str, np.ndarray] = {}
    lon_template: Optional[np.ndarray] = None
    lat_template: Optional[np.ndarray] = None
    for var in channels:
        sample = _era5_worker_single(_ERA5_WORKER_STATE, dt, var)
        if sample is None:
            continue
        data, lon_grid, lat_grid = sample
        results[var] = data
        if lon_template is None:
            lon_template = lon_grid
            lat_template = lat_grid
    return dt, results, lon_template, lat_template


def _era5_worker_single(state: ERA5WorkerState, dt: datetime, variable: str):
    path = _era5_worker_resolve_path(state, variable, dt)
    if path is None or not path.exists():
        logger.warning("ERA5 missing: %s %s", variable, dt)
        return None
    cfg = VARIABLE_CONFIGS.get(variable)
    nc_var = cfg.nc_var_name if cfg else variable
    logger.info("Processing ERA5 %s at %s from %s", variable, dt, path)
    with Dataset(path, "r") as ds:
        lat_var = ds.variables.get("latitude") or ds.variables.get("lat")
        lon_var = ds.variables.get("longitude") or ds.variables.get("lon")
        if lat_var is None or lon_var is None:
            logger.warning("Latitude/longitude missing in %s", path)
            return None
        lat_arr = np.asarray(lat_var[:])
        lon_arr = np.asarray(lon_var[:])
        lat_slice, lon_slice, lat_sel, lon_sel = _era5_worker_slice(state, lat_arr, lon_arr)
        if lat_sel.size == 0 or lon_sel.size == 0:
            logger.warning("No ERA5 data within bounds for %s at %s", variable, dt)
            return None
        time_var = ds.variables.get("time") or ds.variables.get("valid_time")
        time_idx = _select_time_index(time_var, dt)
        var_obj = ds.variables[nc_var]
        level_idx, level_dim = _era5_level_selection(ds, var_obj, variable, state.pres_idx)
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
        data_rs, lon_rs, lat_rs = resize_to_domain(data, lon_grid, lat_grid, state.domain_size, state.resample_mode)
        data_rs = np.squeeze(np.asarray(data_rs, dtype=np.float32))
        return data_rs, lon_rs.astype(np.float32), lat_rs.astype(np.float32)


def _era5_worker_resolve_path(state: ERA5WorkerState, variable: str, dt: datetime) -> Optional[pathlib.Path]:
    cache = state.era5_index.get(variable, {})
    for key in (
        dt.strftime("%Y%m%d%H"),
        dt.strftime("%Y%m%d") + str(dt.hour),
        dt.strftime("%Y%m"),
    ):
        path_str = cache.get(key)
        if path_str:
            return pathlib.Path(path_str)
    return None


def _era5_worker_slice(state: ERA5WorkerState, lat: np.ndarray, lon: np.ndarray):
    lat_slice_idx = search_bounds_1d(lat, state.lat_bounds[0], state.lat_bounds[1])
    lon_slice_idx = search_bounds_1d(lon, state.lon_bounds[0], state.lon_bounds[1])
    lat_slice = slice(lat_slice_idx[0], lat_slice_idx[1] + 1)
    lon_slice = slice(lon_slice_idx[0], lon_slice_idx[1] + 1)
    lat_sel = lat[lat_slice]
    lon_sel = lon[lon_slice]
    return lat_slice, lon_slice, lat_sel, lon_sel

def _parse_pressure_level(logical: str) -> Optional[int]:
    match = LEVEL_PATTERN.match(logical)
    if not match:
        return None
    base, lvl = match.groups()
    if base.lower() not in {"u", "v", "t", "q", "z"}:
        return None
    return int(lvl)


def _find_level_index(
    ds: Dataset,
    var_obj,
    level_dim: str,
    target_level: int,
    pres_idx: Dict[int, int],
) -> Optional[int]:
    dim_index = var_obj.dimensions.index(level_dim)
    dim_size = var_obj.shape[dim_index]
    if dim_size == 1:
        return 0

    coord_var = ds.variables.get(level_dim)
    if coord_var is not None:
        levels = np.asarray(coord_var[:], dtype=np.float64)
        if levels.size:
            idx = int(np.argmin(np.abs(levels - target_level)))
            val = float(levels[idx])
            if np.isclose(val, target_level, atol=1e-3):
                return idx
            # Handle Pa vs hPa if necessary
            if np.isclose(val / 100.0, target_level, atol=1e-2):
                return idx
            if np.isclose(val * 100.0, target_level, atol=1.0):
                return idx

    fallback = pres_idx.get(target_level)
    if fallback is not None and fallback < dim_size:
        return fallback

    return None


def _era5_level_selection(
    ds: Dataset,
    var_obj,
    logical: str,
    pres_idx: Dict[int, int],
) -> Tuple[Optional[int], Optional[str]]:
    target_level = _parse_pressure_level(logical)
    if target_level is None:
        return None, None
    for dim in var_obj.dimensions:
        if dim.lower() in LEVEL_DIM_NAMES_LOWER:
            idx = _find_level_index(ds, var_obj, dim, target_level, pres_idx)
            if idx is None:
                logger.warning(
                    "Unable to resolve level %s for %s; available dimension %s (size=%s)",
                    target_level,
                    logical,
                    dim,
                    var_obj.shape[var_obj.dimensions.index(dim)],
                )
                return None, None
            return idx, dim
    return None, None


def _select_time_index(time_var, target: datetime) -> int:
    if time_var is None:
        return 0
    values = np.asarray(time_var[:])
    if values.size == 0:
        return 0
    if hasattr(time_var, "units"):
        try:
            converted = num2date(values, units=time_var.units)
            dt_array = np.array([np.datetime64(ensure_timezone_naive(dt)) for dt in converted])
        except Exception:  # pragma: no cover
            dt_array = values
    elif values.dtype.kind in "OSU":
        dt_array = np.array([
            np.datetime64((v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v)))
            for v in values
        ])
    else:
        dt_array = values.astype("datetime64[h]")
    target64 = np.datetime64(ensure_timezone_naive(target)).astype("datetime64[h]")
    idx = int(np.argmin(np.abs(dt_array.astype("datetime64[h]") - target64)))
    return idx
