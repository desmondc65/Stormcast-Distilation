from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from netCDF4 import Dataset

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(pathlib.Path(__file__).resolve().parent))
    from common import (
        get_logger,
        resize_to_domain,
        search_bounds_1d,
    )
else:
    from .common import (
        get_logger,
        resize_to_domain,
        search_bounds_1d,
    )

logger = get_logger()

DEFAULT_RWRF_VARIABLES: List[str] = [
    "u10", "v10", "t2m", "sp", "msl", "tcwv",
    "u50", "u100", "u150", "u200", "u250", "u300", "u400", "u500", "u600", "u700", "u850", "u925", "u1000",
    "v50", "v100", "v150", "v200", "v250", "v300", "v400", "v500", "v600", "v700", "v850", "v925", "v1000",
    "z50", "z100", "z150", "z200", "z250", "z300", "z400", "z500", "z600", "z700", "z850", "z925", "z1000",
    "t50", "t100", "t150", "t200", "t250", "t300", "t400", "t500", "t600", "t700", "t850", "t925", "t1000",
    "q50", "q100", "q150", "q200", "q250", "q300", "q400", "q500", "q600", "q700", "q850", "q925", "q1000",
    "qpepre",
]

DEFAULT_INVARIANTS: List[str] = ["lsm", "orog"]

DEFAULT_PRES_IDX: Dict[int, int] = {
    1000: 0,
    925: 3,
    850: 6,
    700: 11,
    600: 13,
    500: 15,
    400: 17,
    300: 19,
    250: 20,
    200: 22,
    150: 24,
    100: 26,
    70: 27,
    50: 28,
}


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


def _build_variable_map() -> Dict[str, str]:
    variables = {
        "u10": "umet10",
        "v10": "vmet10",
        "t2m": "T2",
        "sp": "PSFC",
        "msl": "slp",
        "tcwv": "pw",
        "qpepre": "qpepre",
    }
    invariants = {"lsm": "LANDMASK", "orog": "HGT"}
    prs_levels = [
        50,
        100,
        150,
        200,
        250,
        300,
        400,
        500,
        600,
        700,
        850,
        925,
        1000,
    ]
    prs_names_to_id = {
        "u": "umet_p",
        "v": "vmet_p",
        "z": "z_p",
        "t": "tk_p",
        "q": "QVAPOR_p",
    }
    prs_variables: Dict[str, str] = {}
    for prefix, nc_name in prs_names_to_id.items():
        for level in prs_levels:
            prs_variables[f"{prefix}{level}"] = nc_name
    return {**variables, **invariants, **prs_variables}


RWRF_VARIABLE_MAP = _build_variable_map()


def resolve_variable(logical: str) -> Tuple[str, Callable[[np.ndarray], np.ndarray]]:
    try:
        nc_name = RWRF_VARIABLE_MAP[logical]
    except KeyError as exc:  # pragma: no cover - validated earlier
        raise KeyError(f"Unknown RWRF variable: {logical}") from exc

    def _identity(field: np.ndarray) -> np.ndarray:
        return field

    return nc_name, _identity


_QPEPRE_PATTERN = re.compile(r"qpepre_(\d{12})-(\d{12})_")


def _extract_qpepre_keys(name: str) -> Optional[Tuple[str, str]]:
    match = _QPEPRE_PATTERN.search(name)
    if not match:
        return None
    return match.group(1), match.group(2)


def _parse_rwrf_timestamp(key: str) -> Optional[datetime]:
    try:
        return datetime.strptime(key, "%Y-%m-%d_%H")
    except Exception:
        return None


def _parse_qpepre_timestamp(key: str) -> Optional[datetime]:
    try:
        return datetime.strptime(key, "%Y%m%d%H%M")
    except Exception:
        return None


@dataclass(frozen=True)
class RWRFWorkerState:
    rwrf_index: Dict[str, str]
    qpepre_index: Dict[str, str]
    domain_size: Tuple[int, int]
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    resample_mode: str
    pres_idx: Dict[int, int]
    rwrf_variables: Sequence[str]


@dataclass(frozen=True)
class RWRFSharedIndexWorkerState:
    """Lightweight worker state that references a shared index file instead of copying data."""
    index_file_path: str  # Path to JSON index file - only ~50 bytes to serialize
    domain_size: Tuple[int, int]
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    resample_mode: str
    pres_idx: Dict[int, int]
    rwrf_variables: Sequence[str]


_RWRF_WORKER_STATE: RWRFWorkerState | None = None
_RWRF_SHARED_INDEX_CACHE: Dict[str, Dict[str, str]] | None = None
_RWRF_SHARED_INDEX_STATE: RWRFSharedIndexWorkerState | None = None


def _rwrf_worker_init(state: RWRFWorkerState) -> None:
    global _RWRF_WORKER_STATE
    _RWRF_WORKER_STATE = state


def _rwrf_shared_index_worker_init(state: RWRFSharedIndexWorkerState) -> None:
    """Initialize worker with shared index state (loads index from file on first use)."""
    global _RWRF_SHARED_INDEX_STATE, _RWRF_SHARED_INDEX_CACHE
    _RWRF_SHARED_INDEX_STATE = state
    _RWRF_SHARED_INDEX_CACHE = None  # Will be lazily loaded


def _load_rwrf_shared_index() -> Dict[str, Dict[str, str]]:
    """Load RWRF/QPEPRE index from shared JSON file (cached after first load)."""
    global _RWRF_SHARED_INDEX_CACHE
    if _RWRF_SHARED_INDEX_CACHE is not None:
        return _RWRF_SHARED_INDEX_CACHE
    if _RWRF_SHARED_INDEX_STATE is None:
        raise RuntimeError("RWRF shared index state not initialized")
    import json
    with open(_RWRF_SHARED_INDEX_STATE.index_file_path, "r") as f:
        data = json.load(f)
    _RWRF_SHARED_INDEX_CACHE = data
    return _RWRF_SHARED_INDEX_CACHE


def _rwrf_shared_index_worker(dt: datetime):
    """RWRF worker using shared index file instead of copied dictionary."""
    if _RWRF_SHARED_INDEX_STATE is None:
        raise RuntimeError("RWRF shared index state not initialized")
    
    index_data = _load_rwrf_shared_index()
    rwrf_index = index_data.get("rwrf", {})
    qpepre_index = index_data.get("qpepre", {})
    state = _RWRF_SHARED_INDEX_STATE
    
    # Resolve RWRF path
    key = dt.strftime("%Y-%m-%d_%H")
    path_str = rwrf_index.get(key)
    
    if path_str is None:
        return dt, {}, None, None
    
    path = pathlib.Path(path_str)
    if not path.exists():
        return dt, {}, None, None
    
    results: Dict[str, np.ndarray] = {}
    lon_template: Optional[np.ndarray] = None
    lat_template: Optional[np.ndarray] = None
    
    with Dataset(path, "r") as ds:
        # Get slices
        lat = np.asarray(ds.variables["XLAT"][0, :, 0])
        lon = np.asarray(ds.variables["XLONG"][0, 0, :])
        lat_idx = search_bounds_1d(lat, state.lat_min, state.lat_max)
        lon_idx = search_bounds_1d(lon, state.lon_min, state.lon_max)
        lat_slice = slice(lat_idx[0], lat_idx[1] + 1)
        lon_slice = slice(lon_idx[0], lon_idx[1] + 1)
        lat_sel = lat[lat_slice]
        lon_sel = lon[lon_slice]
        lon_grid, lat_grid = np.meshgrid(lon_sel, lat_sel)
        
        # Get time index
        times_var = ds.variables.get("Times")
        time_idx = 0
        if times_var is not None:
            values = [b"".join(row).decode("utf-8").strip() for row in times_var[:]]
            dt_array = np.array([np.datetime64(val.replace("_", "T")) for val in values])
            target64 = np.datetime64(dt.strftime("%Y-%m-%dT%H"))
            time_idx = int(np.argmin(np.abs(dt_array - target64)))
        
        for logical in state.rwrf_variables:
            if logical == "qpepre":
                continue  # Handle separately
            
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
            
            level_idx = state.pres_idx.get(int(logical[1:])) if logical[1:].isdigit() else None
            if level_idx is not None:
                for dim_name in dims:
                    if dim_name.lower() in LEVEL_DIM_NAMES_LOWER:
                        dim_idx = dims.index(dim_name)
                        if level_idx < var.shape[dim_idx]:
                            slices[dim_idx] = level_idx
                        break
            
            slices[-2] = lat_slice
            slices[-1] = lon_slice
            data = np.asarray(var[tuple(slices)], dtype=np.float32)
            data = np.squeeze(modifier(data))
            data_rs, lon_rs, lat_rs = resize_to_domain(data, lon_grid, lat_grid, state.domain_size, state.resample_mode)
            data_rs = np.squeeze(np.asarray(data_rs, dtype=np.float32))
            results[logical] = data_rs
            lon_template = lon_rs.astype(np.float32)
            lat_template = lat_rs.astype(np.float32)
    
    # Handle QPEPRE
    if "qpepre" in state.rwrf_variables and "qpepre" not in results:
        end_key = dt.strftime("%Y%m%d%H%M")
        start_key = (dt - timedelta(hours=1)).strftime("%Y%m%d%H%M")
        qpe_path_str = qpepre_index.get(end_key) or qpepre_index.get(start_key)
        if qpe_path_str:
            qpe_path = pathlib.Path(qpe_path_str)
            if qpe_path.exists():
                raw = np.loadtxt(qpe_path, dtype=np.float32)
                if raw.ndim == 1:
                    raw = raw.reshape(1, -1)
                _, lon_vals, lat_vals, rain_vals = raw.T
                lat_unique = np.unique(lat_vals)
                lon_unique = np.unique(lon_vals)
                val_grid = np.full((lat_unique.size, lon_unique.size), np.nan, dtype=np.float32)
                lat_to_idx = {v: i for i, v in enumerate(lat_unique)}
                lon_to_idx = {v: i for i, v in enumerate(lon_unique)}
                for lon_v, lat_v, rain in zip(lon_vals, lat_vals, rain_vals):
                    val_grid[lat_to_idx[lat_v], lon_to_idx[lon_v]] = rain
                qpe_lon_grid, qpe_lat_grid = np.meshgrid(lon_unique, lat_unique)
                data_rs, lon_rs, lat_rs = resize_to_domain(val_grid, qpe_lon_grid, qpe_lat_grid, state.domain_size, state.resample_mode)
                results["qpepre"] = np.asarray(data_rs, dtype=np.float32)
                if lon_template is None:
                    lon_template = lon_rs.astype(np.float32)
                    lat_template = lat_rs.astype(np.float32)
    
    return dt, results, lon_template, lat_template


def _rwrf_worker(dt: datetime):
    if _RWRF_WORKER_STATE is None:
        raise RuntimeError("RWRF worker state not initialized")
    return _rwrf_worker_single(_RWRF_WORKER_STATE, dt)


def _rwrf_worker_single(state: RWRFWorkerState, dt: datetime):
    path = _rwrf_worker_resolve_path(state, dt)
    if path is None or not path.exists():
        return dt, {}, None, None
    results: Dict[str, np.ndarray] = {}
    lon_template: Optional[np.ndarray] = None
    lat_template: Optional[np.ndarray] = None
    logger.info("Processing RWRF file: %s for %s", path, dt)
    with Dataset(path, "r") as ds:
        lat_slice, lon_slice, lat_sel, lon_sel = _rwrf_worker_slice(state, ds)
        lon_grid, lat_grid = np.meshgrid(lon_sel, lat_sel)
        time_idx = _rwrf_worker_select_time_index(ds, dt)
        for logical in state.rwrf_variables:
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
            level_idx = state.pres_idx.get(int(logical[1:])) if logical[1:].isdigit() else None
            if level_idx is not None:
                level_set = False
                for dim_name in dims:
                    if dim_name.lower() in LEVEL_DIM_NAMES_LOWER:
                        dim_idx = dims.index(dim_name)
                        if level_idx < var.shape[dim_idx]:
                            slices[dim_idx] = level_idx
                        else:
                            logger.warning(
                                "Worker level index %s out of bounds for %s (dim %s size %s)",
                                level_idx,
                                logical,
                                dim_name,
                                var.shape[dim_idx],
                            )
                        level_set = True
                        break
                if not level_set:
                    logger.debug("Worker: no pressure dimension found for %s with dims %s", logical, dims)
            slices[-2] = lat_slice
            slices[-1] = lon_slice
            data = np.asarray(var[tuple(slices)], dtype=np.float32)
            data = np.squeeze(modifier(data))
            data_rs, lon_rs, lat_rs = resize_to_domain(data, lon_grid, lat_grid, state.domain_size, state.resample_mode)
            data_rs = np.squeeze(np.asarray(data_rs, dtype=np.float32))
            results[logical] = data_rs
            lon_template = lon_rs.astype(np.float32)
            lat_template = lat_rs.astype(np.float32)
    if "qpepre" in state.rwrf_variables and "qpepre" not in results:
        qpe = _rwrf_worker_qpepre(state, dt)
        if qpe is not None:
            data_qpe, lon_qpe, lat_qpe = qpe
            results["qpepre"] = data_qpe
            if lon_template is None:
                lon_template = lon_qpe.astype(np.float32)
                lat_template = lat_qpe.astype(np.float32)
    return dt, results, lon_template, lat_template


def _rwrf_worker_resolve_path(state: RWRFWorkerState, dt: datetime) -> Optional[pathlib.Path]:
    cache = state.rwrf_index
    key = dt.strftime("%Y-%m-%d_%H")
    path_str = cache.get(key)
    if path_str:
        return pathlib.Path(path_str)
    return None


def _rwrf_worker_slice(state: RWRFWorkerState, ds: Dataset):
    lat = np.asarray(ds.variables["XLAT"][0, :, 0])
    lon = np.asarray(ds.variables["XLONG"][0, 0, :])
    lat_idx = search_bounds_1d(lat, state.lat_min, state.lat_max)
    lon_idx = search_bounds_1d(lon, state.lon_min, state.lon_max)
    lat_slice = slice(lat_idx[0], lat_idx[1] + 1)
    lon_slice = slice(lon_idx[0], lon_idx[1] + 1)
    lat_sel = lat[lat_slice]
    lon_sel = lon[lon_slice]
    return lat_slice, lon_slice, lat_sel, lon_sel


def _rwrf_worker_select_time_index(ds: Dataset, target: datetime) -> int:
    times_var = ds.variables.get("Times")
    if times_var is None:
        return 0
    values = [b"".join(row).decode("utf-8").strip() for row in times_var[:]]
    dt_array = np.array([np.datetime64(val.replace("_", "T")) for val in values])
    target64 = np.datetime64(target.strftime("%Y-%m-%dT%H"))
    idx = int(np.argmin(np.abs(dt_array - target64)))
    return idx


def _rwrf_worker_qpepre(state: RWRFWorkerState, dt: datetime) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    target = dt
    index = state.qpepre_index
    end_key = dt.strftime("%Y%m%d%H%M")
    start_key = (dt - timedelta(hours=1)).strftime("%Y%m%d%H%M")
    path_str = index.get(end_key) or index.get(start_key)
    if path_str is None:
        return None
    else:
        path = pathlib.Path(path_str)
    raw = np.loadtxt(path, dtype=np.float32)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    _, lon_vals, lat_vals, rain_vals = raw.T
    lat_unique = np.unique(lat_vals)
    lon_unique = np.unique(lon_vals)
    val_grid = np.full((lat_unique.size, lon_unique.size), np.nan, dtype=np.float32)
    lat_to_idx = {v: i for i, v in enumerate(lat_unique)}
    lon_to_idx = {v: i for i, v in enumerate(lon_unique)}
    for lon_v, lat_v, rain in zip(lon_vals, lat_vals, rain_vals):
        val_grid[lat_to_idx[lat_v], lon_to_idx[lon_v]] = rain
    lon_grid, lat_grid = np.meshgrid(lon_unique, lat_unique)
    data_rs, lon_rs, lat_rs = resize_to_domain(val_grid, lon_grid, lat_grid, state.domain_size, state.resample_mode)
    return (
        np.asarray(data_rs, dtype=np.float32),
        lon_rs.astype(np.float32),
        lat_rs.astype(np.float32),
    )
