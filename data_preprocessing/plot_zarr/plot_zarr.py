#!/usr/bin/env python3
"""
Script to plot ERA5/RWRF or StormCast (HighRes/LowRes) variables from a Zarr dataset.

Usage:
    python plot_zarr.py <zarr_path>.zarr --source era5 --variable "u@500" --time 0
    python plot_zarr.py <zarr_path>.zarr --source cwb --variable "tk_p@850" --time 5
    python plot_zarr.py <zarr_path>.zarr --source HighRes --variable "u10" --time 12
    python plot_zarr.py <zarr_path>.zarr --list-variables
    python plot_zarr.py <zarr_path>.zarr --list-times
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    ccrs = None
    cfeature = None
    CARTOPY_AVAILABLE = False

# Set up logger
LOGGER = logging.getLogger(__name__)


@dataclass
class SourceSpec:
    """Metadata describing a data variable and its channel coordinate."""

    name: str
    data_var: str
    channel_coord: str
    valid_var: Optional[str] = None
    var_name_var: Optional[str] = None
    pressure_var: Optional[str] = None


def detect_data_sources(ds: xr.Dataset) -> Dict[str, SourceSpec]:
    """Return available data sources for the dataset."""
    specs: Dict[str, SourceSpec] = {}

    if 'era5' in ds and 'era5_channel' in ds.coords:
        specs['era5'] = SourceSpec(
            name='era5',
            data_var='era5',
            channel_coord='era5_channel',
            valid_var='era5_valid' if 'era5_valid' in ds else None,
            var_name_var='era5_variable' if 'era5_variable' in ds else None,
            pressure_var='era5_pressure' if 'era5_pressure' in ds else None,
        )

    if 'cwb' in ds and 'cwb_channel' in ds.coords:
        specs['cwb'] = SourceSpec(
            name='cwb',
            data_var='cwb',
            channel_coord='cwb_channel',
            valid_var='cwb_valid' if 'cwb_valid' in ds else None,
            var_name_var='cwb_variable' if 'cwb_variable' in ds else None,
            pressure_var='cwb_pressure' if 'cwb_pressure' in ds else None,
        )

    if 'HighRes' in ds and 'channel' in ds.coords:
        specs['HighRes'] = SourceSpec(
            name='HighRes',
            data_var='HighRes',
            channel_coord='channel',
            valid_var='valid' if 'valid' in ds else None,
        )

    if 'LowRes' in ds and 'channel' in ds.coords:
        specs['LowRes'] = SourceSpec(
            name='LowRes',
            data_var='LowRes',
            channel_coord='channel',
            valid_var='valid' if 'valid' in ds else None,
        )

    if 'HighRes_invariants' in ds and 'channel' in ds.coords:
        specs['HighRes_invariants'] = SourceSpec(
            name='HighRes_invariants',
            data_var='HighRes_invariants',
            channel_coord='channel',
        )

    return specs


def _get_array(ds: xr.Dataset, names) -> Optional[np.ndarray]:
    """Return the first matching coordinate/data array."""
    for name in names:
        if name in ds:
            return ds[name].values
        if name in ds.coords:
            return ds.coords[name].values
    return None


def get_lat_lon(ds: xr.Dataset) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Fetch latitude and longitude grids if present."""
    lat = _get_array(ds, ['XLAT', 'latitude', 'lat', 'LAT'])
    lon = _get_array(ds, ['XLON', 'longitude', 'lon', 'LON'])
    return lat, lon


def get_valid_mask(ds: xr.Dataset, spec: SourceSpec) -> Optional[np.ndarray]:
    """Return boolean mask of valid timesteps for the source, if available."""
    if not spec.valid_var or spec.valid_var not in ds:
        return None
    valid = ds[spec.valid_var]
    if 'time' not in valid.dims:
        return None
    reduce_dims = [dim for dim in valid.dims if dim != 'time']
    valid_bool = valid.astype(bool)
    if reduce_dims:
        valid_bool = valid_bool.all(dim=reduce_dims)
    return valid_bool.values


def resolve_source(specs: Dict[str, SourceSpec], requested: Optional[str]) -> SourceSpec:
    """Resolve which data source to use."""
    if not specs:
        raise ValueError("No plottable data variables found in dataset")

    if requested is None:
        if len(specs) == 1:
            return next(iter(specs.values()))
        available = ', '.join(specs.keys())
        raise ValueError(f"--source is required. Available options: {available}")

    for name, spec in specs.items():
        if requested == name or requested.lower() == name.lower():
            return spec

    available = ', '.join(specs.keys())
    raise ValueError(f"Unknown source '{requested}'. Available options: {available}")


def resolve_channel(channels: np.ndarray, variable: str) -> Optional[object]:
    """Match a variable name (case-insensitive) to a channel entry."""
    for ch in channels:
        if variable == ch or variable == str(ch):
            return ch
    variable_lower = variable.lower()
    for ch in channels:
        if str(ch).lower() == variable_lower:
            return ch
    return None


def list_variables(zarr_path: Path) -> None:
    """List all available variables in the Zarr dataset."""
    LOGGER.info("Opening Zarr dataset: %s", zarr_path)
    print(f"\nOpening Zarr dataset: {zarr_path}")
    ds = xr.open_zarr(zarr_path, consolidated=True)
    LOGGER.info("Successfully opened Zarr dataset")

    print("\n" + "=" * 80)
    print("AVAILABLE VARIABLES")
    print("=" * 80)

    specs = detect_data_sources(ds)
    if not specs:
        print("No channelized data variables found in this dataset.")
        print("\n" + "=" * 80)
        ds.close()
        return

    for spec in specs.values():
        channel_values = ds.coords[spec.channel_coord].values
        print(f"\n{spec.name} Variables ({len(channel_values)} channels):")
        print("-" * 80)
        data_var = ds[spec.data_var]
        var_names = None
        pressures = None

        if spec.var_name_var and spec.var_name_var in ds:
            var_names = ds[spec.var_name_var].values
        if spec.pressure_var and spec.pressure_var in ds:
            pressures = ds[spec.pressure_var].values

        for idx, channel in enumerate(channel_values):
            channel_str = str(channel)
            label = channel_str
            pressure_text = ""
            if var_names is not None and idx < len(var_names):
                label = str(var_names[idx])
            if pressures is not None and idx < len(pressures):
                pressure = pressures[idx]
                if not np.isnan(pressure):
                    pressure_text = f" ({int(pressure)} hPa)"
                else:
                    pressure_text = " (surface)"

            size_str = "N/A"
            try:
                selection = {spec.channel_coord: channel}
                arr = data_var.sel(selection)
                if 'time' in arr.dims and arr.sizes.get('time', 0) > 0:
                    arr = arr.isel(time=0)
                vals = arr.values
                if hasattr(vals, 'shape') and len(vals.shape) >= 2:
                    nlat, nlon = vals.shape[-2], vals.shape[-1]
                    size_str = f"x: {nlon}, y: {nlat}"
                elif hasattr(vals, 'shape') and len(vals.shape) == 1:
                    size_str = f"x: {vals.shape[0]}, y: 1"
            except Exception:
                size_str = "N/A"

            print(f"  {channel_str:20s} - {label}{pressure_text} - size ({size_str})")

    print("\n" + "=" * 80)
    ds.close()


def list_times(zarr_path: Path) -> None:
    """List all available timesteps in the Zarr dataset."""
    LOGGER.info("Opening Zarr dataset: %s", zarr_path)
    print(f"\nOpening Zarr dataset: {zarr_path}")
    ds = xr.open_zarr(zarr_path, consolidated=True)
    LOGGER.info("Successfully opened Zarr dataset")

    print("\n" + "=" * 80)
    print("AVAILABLE TIMESTEPS")
    print("=" * 80)

    if 'time' not in ds.coords:
        print("No time dimension found in dataset")
        print("\n" + "=" * 80)
        ds.close()
        return

    times = ds.coords['time'].values
    specs = detect_data_sources(ds)
    valid_masks = {name: mask for name, spec in specs.items() if (mask := get_valid_mask(ds, spec)) is not None}

    print(f"\nTotal timesteps: {len(times)}")
    print("-" * 80)
    for i, time_val in enumerate(times):
        timestamp_ns = int(time_val.astype('datetime64[ns]').astype('int64'))
        time_dt = datetime.fromtimestamp(timestamp_ns / 1e9, tz=timezone.utc)
        status_parts = []
        for name in specs:
            mask = valid_masks.get(name)
            if mask is None:
                continue
            status_parts.append(f"{name}: {'✓' if mask[i] else '✗'}")
        status_text = " | ".join(status_parts) if status_parts else "No validity flags"
        print(f"  [{i:3d}] {time_dt.strftime('%Y-%m-%d %H:%M:%S')} | {status_text}")

    print("-" * 80)
    print("\nValidity Summary:")
    print(f"  Total timesteps:              {len(times)}")

    for name, mask in valid_masks.items():
        valid_count = int(mask.sum())
        percent = valid_count / len(times) * 100
        print(f"  Total valid {name}:           {valid_count} ({percent:.1f}%)")

    # Special-case overlap reporting for ERA5/CWB legacy datasets
    if 'era5' in valid_masks and 'cwb' in valid_masks:
        era5_mask = valid_masks['era5']
        cwb_mask = valid_masks['cwb']
        both_valid = int((era5_mask & cwb_mask).sum())
        era5_only = int((era5_mask & ~cwb_mask).sum())
        cwb_only = int((~era5_mask & cwb_mask).sum())
        print(f"  Valid ERA5 only:              {era5_only}")
        print(f"  Valid RWRF only:              {cwb_only}")
        print(f"  Valid both (overlapped):      {both_valid}")
        print(f"  Overlapped valid / total:     {both_valid} / {len(times)} ({both_valid/len(times)*100:.1f}%)")

    print("\n" + "=" * 80)
    ds.close()


def plot_variable(
    zarr_path: Path,
    source: Optional[str],
    variable: str,
    time_index: int,
    output_file: Optional[str] = None,
    show_plot: bool = True,
    cmap: str = 'viridis',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """
    Plot a specific variable from the Zarr dataset.
    
    Args:
        zarr_path: Path to the Zarr dataset
        source: Name of data variable (e.g., 'era5', 'cwb', 'HighRes'). If None, auto-detect when possible.
        variable: Variable name (e.g., 'u@500', 't@850', 'T2@surface')
        time_index: Index of the timestep to plot
        output_file: If provided, save the plot to this file
        show_plot: If True, display the plot interactively
        cmap: Colormap name for the plot
        vmin: Minimum value for colorbar
        vmax: Maximum value for colorbar
    """
    LOGGER.info("Starting plot generation")
    LOGGER.info("Opening Zarr dataset: %s", zarr_path)
    print(f"\nOpening Zarr dataset: {zarr_path}")
    ds = xr.open_zarr(zarr_path, consolidated=True)
    LOGGER.info("Successfully opened Zarr dataset")

    specs = detect_data_sources(ds)
    try:
        spec = resolve_source(specs, source)
    except ValueError as exc:
        LOGGER.error("Source resolution error: %s", exc)
        print(f"Error: {exc}")
        ds.close()
        sys.exit(1)

    data_var_name = spec.data_var
    channel_coord = spec.channel_coord

    if data_var_name not in ds:
        LOGGER.error("Data variable '%s' not found in dataset", data_var_name)
        print(f"Error: '{data_var_name}' not found in dataset")
        ds.close()
        sys.exit(1)

    if channel_coord not in ds.coords:
        LOGGER.error("Channel coordinate '%s' not found in dataset", channel_coord)
        print(f"Error: '{channel_coord}' coordinate not found in dataset")
        ds.close()
        sys.exit(1)

    channels = ds.coords[channel_coord].values
    LOGGER.info("Found %d channels in %s", len(channels), spec.name)

    LOGGER.info("Searching for variable: %s", variable)
    channel_value = resolve_channel(channels, variable)
    if channel_value is None:
        LOGGER.error("Variable '%s' not found in %s channels", variable, spec.name)
        available = ', '.join(str(ch) for ch in channels)
        print(f"Error: Variable '{variable}' not found in {spec.name} channels")
        print(f"Available channels: {available}")
        ds.close()
        sys.exit(1)
    LOGGER.info("Variable '%s' found in channels", variable)
    
    # Get time information
    LOGGER.info("Validating time index: %d", time_index)
    if 'time' not in ds.coords:
        LOGGER.error("Dataset does not contain a 'time' coordinate")
        print("Error: Dataset does not contain a 'time' coordinate")
        ds.close()
        sys.exit(1)

    times = ds.coords['time'].values
    if time_index < 0 or time_index >= len(times):
        LOGGER.error("Time index %d out of range [0, %d]", time_index, len(times)-1)
        print(f"Error: time_index {time_index} out of range [0, {len(times)-1}]")
        ds.close()
        sys.exit(1)
    
    # Convert numpy datetime64 to Python datetime
    time_val = times[time_index]
    timestamp_ns = int(time_val.astype('datetime64[ns]').astype('int64'))
    time_dt = datetime.fromtimestamp(timestamp_ns / 1e9, tz=timezone.utc)
    LOGGER.info("Selected timestep: %s", time_dt.strftime('%Y-%m-%d %H:%M:%S'))
    
    # Select data
    LOGGER.info("Extracting data for variable '%s' at time index %d", variable, time_index)
    data = ds[data_var_name].isel(time=time_index).sel({channel_coord: channel_value}).values
    LOGGER.info("Data shape: %s", data.shape)
    
    # Get coordinates
    LOGGER.info("Loading spatial coordinates")
    lat, lon = get_lat_lon(ds)
    if lat is not None and lon is not None:
        LOGGER.info("Using latitude/longitude coordinates: lat range [%.2f, %.2f], lon range [%.2f, %.2f]", 
                   lat.min(), lat.max(), lon.min(), lon.max())
    else:
        LOGGER.warning("Latitude/longitude not found, using indices")
        print("Warning: latitude/longitude not found, using indices")
        lat = np.arange(data.shape[0])
        lon = np.arange(data.shape[1])
    
    # Get variable metadata
    LOGGER.info("Retrieving variable metadata")
    var_label = str(channel_value)
    if spec.var_name_var and spec.var_name_var in ds:
        channel_idx = np.where(np.asarray(channels) == channel_value)[0][0]
        var_label = str(ds[spec.var_name_var].values[channel_idx])
        if spec.pressure_var and spec.pressure_var in ds:
            pressure = ds[spec.pressure_var].values[channel_idx]
            if not np.isnan(pressure):
                var_label = f"{var_label} ({int(pressure)} hPa)"
                LOGGER.info("Variable label: %s at %d hPa", var_label, int(pressure))
            else:
                LOGGER.info("Variable label: %s (surface level)", var_label)
    
    # Create figure
    LOGGER.info("Creating figure")
    fig = plt.figure(figsize=(12, 8))
    
    # Use PlateCarree projection if we have real lat/lon
    use_cartopy = (
        CARTOPY_AVAILABLE
        and lat is not None
        and lon is not None
        and np.nanmin(lat) >= -90
        and np.nanmax(lat) <= 90
    )

    if use_cartopy:
        LOGGER.info("Using Cartopy PlateCarree projection")
        ax = plt.axes(projection=ccrs.PlateCarree())

        LOGGER.info("Adding map features (coastlines, borders, land)")
        # ax.coastlines(resolution='10m', linewidth=0.5)
        # ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        # ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)

        LOGGER.info("Plotting data with colormap '%s'", cmap)
        im = ax.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(),
                           cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')

        LOGGER.info("Adding gridlines")
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
    else:
        LOGGER.info("Using simple matplotlib plot (no cartopy)")
        if lat is not None and lon is not None and not CARTOPY_AVAILABLE:
            LOGGER.warning("Cartopy not installed; falling back to plain axes")
            print("Warning: cartopy not installed; using plain matplotlib axes")
        ax = plt.axes()
        im = ax.pcolormesh(lon, lat, data, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
        ax.set_xlabel('Longitude Index' if lon is None or np.nanmax(lon) > 360 else 'Longitude')
        ax.set_ylabel('Latitude Index' if lat is None or np.nanmax(lat) > 90 else 'Latitude')
    
    # Add colorbar
    LOGGER.info("Adding colorbar")
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, aspect=40)
    cbar.set_label(var_label, fontsize=10)
    
    # Check validity
    LOGGER.info("Checking data validity flags")
    valid_flag = True
    mask = get_valid_mask(ds, spec)
    if mask is not None:
        valid_flag = bool(mask[time_index])
    
    validity_str = "✓ Valid" if valid_flag else "✗ Invalid (missing/error)"
    LOGGER.info("Data validity: %s", validity_str)
    
    # Add title
    title = f"{spec.name.upper()}: {var_label}\n{time_dt.strftime('%Y-%m-%d %H:%M:%S UTC')} | {validity_str}"
    plt.title(title, fontsize=12, fontweight='bold')
    
    # Add statistics
    LOGGER.info("Computing data statistics")
    valid_data = data[np.isfinite(data)]
    if len(valid_data) > 0:
        stats_text = f"Min: {valid_data.min():.2f} | Mean: {valid_data.mean():.2f} | Max: {valid_data.max():.2f}"
        plt.figtext(0.5, 0.02, stats_text, ha='center', fontsize=9, style='italic')
        LOGGER.info("Statistics - Min: %.2f, Mean: %.2f, Max: %.2f", 
                   valid_data.min(), valid_data.mean(), valid_data.max())
    else:
        LOGGER.warning("No valid data points found (all NaN/Inf)")
    
    plt.tight_layout()
    
    # Save if requested
    if output_file:
        LOGGER.info("Saving plot to: %s", output_file)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
        LOGGER.info("Plot saved successfully")
    
    # Show if requested
    if show_plot:
        LOGGER.info("Displaying plot")
        plt.show()
    else:
        LOGGER.info("Closing plot without display")
        plt.close()
    
    ds.close()
    LOGGER.info("Plot generation completed")


def main():
    parser = argparse.ArgumentParser(
        description="Plot ERA5 or RWRF variables from a Zarr dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available variables
  python plot_zarr.py <zarr_path>.zarr --list-variables
  
  # List all available timesteps
  python plot_zarr.py <zarr_path>.zarr --list-times
  
  # Plot ERA5 u-wind at 500 hPa at time index 0
  python plot_zarr.py <zarr_path>.zarr --source era5 --variable "u@500" --time 0
  
  # Plot RWRF temperature at 850 hPa at time index 5, save to file
  python plot_zarr.py <zarr_path>.zarr --source cwb --variable "tk_p@850" --time 5 --output plot.png
  
  # Plot with custom colormap and value range
  python plot_zarr.py <zarr_path>.zarr --source era5 --variable "t@850" --time 0 --cmap coolwarm --vmin 250 --vmax 300

  # Plot StormCast HighRes u10 at time index 12
  python plot_zarr.py /path/to/HighRes/train_rwrf.zarr --source HighRes --variable "u10" --time 12
        """
    )
    
    parser.add_argument('zarr_path', type=Path, help='Path to the Zarr dataset')
    parser.add_argument('--list-variables', action='store_true', 
                       help='List all available variables and exit')
    parser.add_argument('--list-times', action='store_true',
                       help='List all available timesteps and exit')
    parser.add_argument('--source', type=str,
                       help='Data variable to plot (e.g., era5, cwb, HighRes, LowRes)')
    parser.add_argument('--variable', type=str,
                       help='Variable name (e.g., "u@500", "T2@surface")')
    parser.add_argument('--time', type=int,
                       help='Time index to plot')
    parser.add_argument('--output', type=str,
                       help='Output file path (e.g., plot.png)')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display the plot interactively')
    parser.add_argument('--cmap', type=str, default='viridis',
                       help='Colormap name (default: viridis)')
    parser.add_argument('--vmin', type=float,
                       help='Minimum value for colorbar')
    parser.add_argument('--vmax', type=float,
                       help='Maximum value for colorbar')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    LOGGER.info("=" * 80)
    LOGGER.info("Starting Zarr Plot Utility")
    LOGGER.info("=" * 80)
    
    # Validate zarr path
    LOGGER.info("Validating Zarr path: %s", args.zarr_path)
    if not args.zarr_path.exists():
        LOGGER.error("Zarr dataset not found: %s", args.zarr_path)
        print(f"Error: Zarr dataset not found: {args.zarr_path}")
        sys.exit(1)
    LOGGER.info("Zarr path validated successfully")
    LOGGER.info("Zarr path validated successfully")
    
    # Handle list operations
    if args.list_variables:
        LOGGER.info("Listing variables")
        list_variables(args.zarr_path)
        LOGGER.info("Completed listing variables")
        return
    
    if args.list_times:
        LOGGER.info("Listing timesteps")
        list_times(args.zarr_path)
        LOGGER.info("Completed listing timesteps")
        return
    
    # Validate plot arguments
    if not args.variable:
        LOGGER.error("Missing required argument: --variable")
        print("Error: --variable is required for plotting")
        print("Use --list-variables to see available variables")
        sys.exit(1)
    
    if args.time is None:
        LOGGER.error("Missing required argument: --time")
        print("Error: --time is required for plotting")
        print("Use --list-times to see available timesteps")
        sys.exit(1)
    
    LOGGER.info("All required arguments validated")
    LOGGER.info("Plot parameters: source=%s, variable=%s, time=%d, output=%s", 
               args.source, args.variable, args.time, args.output or "None")
    
    # Plot the variable
    plot_variable(
        zarr_path=args.zarr_path,
        source=args.source,
        variable=args.variable,
        time_index=args.time,
        output_file=args.output,
        show_plot=not args.no_show,
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
    )
    
    LOGGER.info("=" * 80)
    LOGGER.info("Zarr Plot Utility completed successfully")
    LOGGER.info("=" * 80)


if __name__ == '__main__':
    main()
