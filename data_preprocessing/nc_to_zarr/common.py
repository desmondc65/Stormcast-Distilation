from __future__ import annotations

import logging
import multiprocessing
from datetime import datetime
from functools import lru_cache
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


# ---------------------------------------------------------------------------
# Optimized interpolation helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=32)
def _interp_coords_cache(
    lat_start: float,
    lat_end: float,
    lat_n: int,
    lon_start: float,
    lon_end: float,
    lon_n: int,
    target_h: int,
    target_w: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cache interpolation coordinate grids for a given source/target configuration."""
    lat_new = np.linspace(lat_start, lat_end, target_h)
    lon_new = np.linspace(lon_start, lon_end, target_w)
    lon_new_grid, lat_new_grid = np.meshgrid(lon_new, lat_new)
    pts = np.column_stack((lat_new_grid.ravel(), lon_new_grid.ravel()))
    return lat_new, lon_new, lon_new_grid, lat_new_grid, pts


@lru_cache(maxsize=32)
def _bilinear_weights_cache(
    src_h: int,
    src_w: int,
    target_h: int,
    target_w: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute bilinear interpolation indices and weights."""
    # Compute fractional index arrays
    yy = np.linspace(0, src_h - 1, target_h)
    xx = np.linspace(0, src_w - 1, target_w)
    
    y0 = np.floor(yy).astype(int)
    y1 = np.clip(y0 + 1, 0, src_h - 1)
    x0 = np.floor(xx).astype(int)
    x1 = np.clip(x0 + 1, 0, src_w - 1)
    
    wy = (yy - y0).astype(np.float32)
    wx = (xx - x0).astype(np.float32)
    
    return y0, y1, x0, x1, wy, wx


def _bilinear_interp_fast(
    data: np.ndarray,
    y0: np.ndarray,
    y1: np.ndarray,
    x0: np.ndarray,
    x1: np.ndarray,
    wy: np.ndarray,
    wx: np.ndarray,
) -> np.ndarray:
    """Fast bilinear interpolation using precomputed indices and weights.
    
    Args:
        data: Input array with shape (..., src_h, src_w)
        y0, y1, x0, x1: Index arrays for the four corners
        wy, wx: Weight arrays for interpolation
    
    Returns:
        Interpolated array with shape (..., target_h, target_w)
    """
    # Create meshgrids for 2D indexing
    y0_grid, x0_grid = np.meshgrid(y0, x0, indexing='ij')
    y0_grid_r, x1_grid = np.meshgrid(y0, x1, indexing='ij')
    y1_grid, x0_grid_b = np.meshgrid(y1, x0, indexing='ij')
    y1_grid_r, x1_grid_b = np.meshgrid(y1, x1, indexing='ij')
    
    # Extract values at the four corners
    v00 = data[..., y0_grid, x0_grid]  # top-left
    v01 = data[..., y0_grid_r, x1_grid]  # top-right
    v10 = data[..., y1_grid, x0_grid_b]  # bottom-left
    v11 = data[..., y1_grid_r, x1_grid_b]  # bottom-right
    
    # Bilinear interpolation formula
    # Interpolate along x-axis first
    wy_grid = wy[:, np.newaxis]
    wx_grid = wx[np.newaxis, :]
    
    v0 = v00 * (1 - wx_grid) + v01 * wx_grid  # top edge
    v1 = v10 * (1 - wx_grid) + v11 * wx_grid  # bottom edge
    
    # Interpolate along y-axis
    result = v0 * (1 - wy_grid) + v1 * wy_grid
    
    return result.astype(np.float32)


def _interp_tensor_scipy(
    data: np.ndarray,
    lat1d: np.ndarray,
    lon1d: np.ndarray,
    lat_new: np.ndarray,
    lon_new: np.ndarray,
    pts: np.ndarray,
    target_h: int,
    target_w: int,
    lon_new_grid: np.ndarray,
    lat_new_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized interpolation for all slices at once using SciPy."""
    from scipy.interpolate import RegularGridInterpolator  # type: ignore
    
    # Move lat/lon to front so grid dims are first
    value = np.moveaxis(data, (-2, -1), (0, 1))  # (H, W, ...)
    
    interpolator = RegularGridInterpolator(
        (lat1d, lon1d),
        value,
        bounds_error=False,
        fill_value=np.nan,
    )
    
    out = interpolator(pts).reshape(target_h, target_w, *value.shape[2:])
    
    # Move lat/lon back to the last two dims
    out = np.moveaxis(out, (0, 1), (-2, -1))  # (..., H_new, W_new)
    
    return out.astype(np.float32), lon_new_grid, lat_new_grid


def resize_to_domain(
    data: np.ndarray,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    domain_size: Tuple[int, int],
    resample_mode: str,
):
    """Resize data to the requested domain using optimized interpolation.
    
    This function has been optimized for performance:
    - Uses cached coordinate grids to avoid repeated computation
    - Vectorized interpolation processes all slices in one call
    - Fast bilinear interpolation for regular grids
    """
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

    # Use fast bilinear interpolation for regular grids when possible
    lat1d = lat_grid[:, 0]
    lon1d = lon_grid[0, :]
    
    # Check if grid is regular (uniform spacing)
    lat_diffs = np.diff(lat1d)
    lon_diffs = np.diff(lon1d)
    is_regular_grid = (
        np.allclose(lat_diffs, lat_diffs[0], rtol=1e-5) and
        np.allclose(lon_diffs, lon_diffs[0], rtol=1e-5)
    )
    
    logger.debug(
        "Using interpolation: mode=%s, SciPy=%s, regular_grid=%s",
        resample_mode,
        HAS_SCIPY,
        is_regular_grid,
    )
    
    # Try fast bilinear interpolation first if grid is regular
    if is_regular_grid:
        try:
            y0, y1, x0, x1, wy, wx = _bilinear_weights_cache(src_h, src_w, target_h, target_w)
            
            # Handle both 2D and higher-dimensional arrays
            if data.ndim == 2:
                data_interp = _bilinear_interp_fast(data[np.newaxis, ...], y0, y1, x0, x1, wy, wx)[0]
            else:
                # Reshape to (N, H, W) for vectorized processing
                orig_shape = data.shape[:-2]
                data_flat = data.reshape(-1, src_h, src_w)
                data_interp = _bilinear_interp_fast(data_flat, y0, y1, x0, x1, wy, wx)
                data_interp = data_interp.reshape(*orig_shape, target_h, target_w)
            
            # Compute new coordinate grids
            lat_new = np.linspace(lat1d[0], lat1d[-1], target_h)
            lon_new = np.linspace(lon1d[0], lon1d[-1], target_w)
            lon_new_grid, lat_new_grid = np.meshgrid(lon_new, lat_new)
            
            logger.debug("Used fast bilinear interpolation")
            return data_interp.astype(np.float32), lon_new_grid, lat_new_grid
        except Exception as e:  # pragma: no cover
            logger.warning("Fast bilinear interpolation failed: %s, falling back to SciPy", e)
    
    # Fall back to SciPy/xarray interpolation
    if HAS_SCIPY:
        # Use cached coordinate grids
        cache_key = (
            float(lat1d[0]),
            float(lat1d[-1]),
            int(lat1d.size),
            float(lon1d[0]),
            float(lon1d[-1]),
            int(lon1d.size),
            int(target_h),
            int(target_w),
        )
        lat_new, lon_new, lon_new_grid, lat_new_grid, pts = _interp_coords_cache(*cache_key)
        
        # Handle both 2D and higher-dimensional arrays
        if data.ndim == 2:
            data_input = data[np.newaxis, ...]
            data_interp, _, _ = _interp_tensor_scipy(
                data_input, lat1d, lon1d, lat_new, lon_new, pts, target_h, target_w,
                lon_new_grid, lat_new_grid
            )
            data_interp = data_interp[0]
        else:
            data_interp, _, _ = _interp_tensor_scipy(
                data, lat1d, lon1d, lat_new, lon_new, pts, target_h, target_w,
                lon_new_grid, lat_new_grid
            )
        
        logger.debug("Used vectorized SciPy interpolation")
        return data_interp, lon_new_grid, lat_new_grid
    
    # Final fallback to xarray (slowest method)
    if xr is None:
        raise RuntimeError("Interpolation requires SciPy or xarray")
    
    logger.debug("Falling back to xarray interpolation (slowest method)")
    lat_new = np.linspace(lat1d.min(), lat1d.max(), target_h)
    lon_new = np.linspace(lon1d.min(), lon1d.max(), target_w)
    lon_new_grid, lat_new_grid = np.meshgrid(lon_new, lat_new)
    
    def _interp_slice_xr(arr2d: np.ndarray) -> np.ndarray:
        arr2d = np.asarray(arr2d, dtype=np.float32)
        da = xr.DataArray(
            arr2d,
            coords={"lat": lat1d, "lon": lon1d},
            dims=("lat", "lon"),
        )
        da_interp = da.interp(lat=lat_new, lon=lon_new, method="linear", kwargs={"fill_value": None})
        return np.asarray(da_interp.to_numpy(), dtype=np.float32)
    
    if data.ndim == 2:
        data_interp = _interp_slice_xr(data)
    else:
        total_slices = int(np.prod(data.shape[:-2]))
        logger.debug("Interpolating %d slices with xarray", total_slices)
        data_interp = np.empty(data.shape[:-2] + (target_h, target_w), dtype=np.float32)
        for idx in np.ndindex(data.shape[:-2]):
            data_interp[idx] = _interp_slice_xr(data[idx])
    
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
