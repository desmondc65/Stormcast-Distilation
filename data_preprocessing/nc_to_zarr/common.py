from __future__ import annotations

import logging
import multiprocessing
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np

try:  # optional dependency
    import xarray as xr
    XR_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover
    xr = None  # type: ignore[assignment]
    XR_IMPORT_ERROR = exc

try:
    import zarr  # type: ignore[import]
    ZARR_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover
    zarr = None  # type: ignore[assignment]
    ZARR_IMPORT_ERROR = exc

try:
    from scipy.interpolate import griddata  # type: ignore[import]
    HAS_SCIPY = True
except Exception:  # pragma: no cover
    griddata = None  # type: ignore[assignment]
    HAS_SCIPY = False

LOGGER_NAME = "nc_to_zarr"


def get_logger() -> logging.Logger:
    """Return shared module logger."""

    return logging.getLogger(LOGGER_NAME)


def time_function(func):
    """Simple timing decorator."""

    def wrapper(*args, **kwargs):
        start = datetime.now()
        try:
            return func(*args, **kwargs)
        finally:
            duration = (datetime.now() - start).total_seconds()
            get_logger().debug("%s finished in %.2fs", func.__qualname__, duration)

    return wrapper


def get_fork_context():
    """Return multiprocessing fork context when available."""

    try:
        return multiprocessing.get_context("fork")
    except (ValueError, AttributeError):  # pragma: no cover
        return None


def ensure_timezone_naive(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(tz=None).replace(tzinfo=None)


def search_bounds_1d(arr: np.ndarray, vmin: float, vmax: float) -> Tuple[int, int]:
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


def get_equidistant_indices(length: int, target: int) -> np.ndarray:
    if target >= length:
        return np.arange(length, dtype=np.int64)
    idx = np.round(np.linspace(0, length - 1, target)).astype(np.int64)
    return np.unique(idx)


class StreamingStats:
    """Track per-channel mean/std using streaming accumulators."""

    def __init__(self, channels: int) -> None:
        self.sum = np.zeros(channels, dtype=np.float64)
        self.sumsq = np.zeros(channels, dtype=np.float64)
        self.count = np.zeros(channels, dtype=np.float64)

    def update(self, channel_idx: int, field: np.ndarray) -> None:
        data = np.asarray(field, dtype=np.float32)
        mask = np.isfinite(data)
        if not mask.any():
            return
        vals = np.where(mask, data, 0.0)
        self.count[channel_idx] += mask.sum(dtype=np.float64)
        self.sum[channel_idx] += vals.sum(dtype=np.float64)
        self.sumsq[channel_idx] += (vals * vals).sum(dtype=np.float64)

    def finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        safe = np.maximum(self.count, 1.0)
        means = np.divide(self.sum, safe, out=np.zeros_like(self.sum), where=self.count > 0)
        ex2 = np.divide(self.sumsq, safe, out=np.zeros_like(self.sumsq), where=self.count > 0)
        var = np.maximum(ex2 - means ** 2, 0.0)
        return means.astype(np.float32), np.sqrt(var).astype(np.float32)


def resize_to_domain(
    data: np.ndarray,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    domain_size: Tuple[int, int],
    resample_mode: str,
):
    """Resize data to the requested domain."""

    logger = get_logger()
    target_h, target_w = domain_size
    src_h, src_w = data.shape[-2:]
    logger.debug("Resizing field from %sx%s to %sx%s", src_h, src_w, target_h, target_w)
    if src_h == target_h and src_w == target_w:
        logger.debug("Source grid already matches target domain; skipping resize")
        return data, lon_grid, lat_grid

    use_downsample = (
        resample_mode == "downsample"
        and src_h >= target_h
        and src_w >= target_w
    )
    if use_downsample:
        yi = get_equidistant_indices(src_h, target_h)
        xi = get_equidistant_indices(src_w, target_w)
        logger.debug(
            "Applying downsample with %d/%d y-indices and %d/%d x-indices",
            yi.size,
            src_h,
            xi.size,
            src_w,
        )
        data_ds = np.take(data, yi, axis=-2)
        data_ds = np.take(data_ds, xi, axis=-1)
        lon_ds = np.take(np.take(lon_grid, yi, axis=0), xi, axis=1)
        lat_ds = np.take(np.take(lat_grid, yi, axis=0), xi, axis=1)
        return data_ds, lon_ds, lat_ds

    logger.debug("Falling back to interpolation (mode=%s, SciPy=%s)", resample_mode, HAS_SCIPY)
    lat_new = np.linspace(lat_grid.min(), lat_grid.max(), target_h)
    lon_new = np.linspace(lon_grid.min(), lon_grid.max(), target_w)
    lon_new_grid, lat_new_grid = np.meshgrid(lon_new, lat_new)
    lat1d = lat_grid[:, 0]
    lon1d = lon_grid[0, :]

    def _interp_slice(arr2d: np.ndarray) -> np.ndarray:
        arr2d = np.asarray(arr2d, dtype=np.float32)
        if HAS_SCIPY:
            from scipy.interpolate import RegularGridInterpolator  # type: ignore

            interpolator = RegularGridInterpolator(
                (lat1d, lon1d),
                arr2d,
                bounds_error=False,
                fill_value=np.nan,
            )
            pts = np.column_stack((lat_new_grid.ravel(), lon_new_grid.ravel()))
            return interpolator(pts).reshape(target_h, target_w)
        if xr is None:
            raise RuntimeError("Interpolation requires SciPy or xarray")
        da = xr.DataArray(
            arr2d,
            coords={"lat": lat1d, "lon": lon1d},
            dims=("lat", "lon"),
        )
        da_interp = da.interp(lat=lat_new, lon=lon_new, method="linear", kwargs={"fill_value": None})
        return np.asarray(da_interp.to_numpy(), dtype=np.float32)

    if data.ndim == 2:
        logger.debug("Interpolating single 2D slice")
        data_interp = _interp_slice(data)
    else:
        total_slices = int(np.prod(data.shape[:-2]))
        logger.debug("Interpolating %d slices along leading dimension", total_slices)
        data_interp = np.empty(data.shape[:-2] + (target_h, target_w), dtype=np.float32)
        for idx in np.ndindex(data.shape[:-2]):
            data_interp[idx] = _interp_slice(data[idx])

    return data_interp, lon_new_grid, lat_new_grid


def find_nearest_key(mapping: Dict[str, str], target: datetime, parser) -> Optional[Tuple[str, str]]:
    """Locate the nearest key in a timestamp-indexed mapping."""

    best_key: Optional[str] = None
    best_path: Optional[str] = None
    best_diff: Optional[float] = None
    for key, path in mapping.items():
        dt_key = parser(key)
        if dt_key is None:
            continue
        diff = abs((dt_key - target).total_seconds())
        if best_diff is None or diff < best_diff:
            best_key = key
            best_path = path
            best_diff = diff
    if best_key is None or best_path is None:
        return None
    return best_key, best_path
