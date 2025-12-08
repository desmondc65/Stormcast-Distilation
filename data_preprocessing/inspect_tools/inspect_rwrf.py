#!/usr/bin/env python3
"""
Inspect RWRF (WRF) NetCDF file variables.

Usage:
    python inspect_rwrf.py /path/to/wrfout_file
    python inspect_rwrf.py /path/to/wrfout_file --verbose
    python inspect_rwrf.py /path/to/wrfout_file --filter T2,U10,V10
"""

import argparse
import sys
from pathlib import Path

try:
    from netCDF4 import Dataset
    import numpy as np
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Install with: pip install netCDF4 numpy")
    sys.exit(1)


def inspect_rwrf(filepath: str, verbose: bool = False, filter_vars: list = None):
    """Inspect all variables in a RWRF/WRF NetCDF file."""
    
    path = Path(filepath)
    if not path.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    
    print(f"{'='*70}")
    print(f"RWRF/WRF File: {filepath}")
    print(f"{'='*70}\n")
    
    with Dataset(str(path), 'r') as ds:
        # Global attributes
        print("=== Global Attributes ===")
        for attr in ds.ncattrs():
            val = getattr(ds, attr)
            if isinstance(val, str) and len(val) > 80:
                val = val[:77] + "..."
            print(f"  {attr}: {val}")
        print()
        
        # Dimensions
        print("=== Dimensions ===")
        for name, dim in ds.dimensions.items():
            print(f"  {name}: {len(dim)} {'(unlimited)' if dim.isunlimited() else ''}")
        print()
        
        # Print vertical level information
        print("=== Vertical Levels ===")
        
        # Check for pres_levels variable (interpolated RWRF files)
        if 'pres_levels' in ds.variables:
            pres_levels = np.asarray(ds.variables['pres_levels'][:])
            print(f"  pres_levels (pressure levels): {len(pres_levels)} levels")
            print(f"    Values (hPa): {np.array2string(pres_levels, precision=1, separator=', ', max_line_width=70)}")
        
        # ZNU: eta values on half (mass) levels
        if 'ZNU' in ds.variables:
            znu = ds.variables['ZNU']
            if znu.ndim >= 2:
                znu_data = znu[0, :]
            else:
                znu_data = znu[:]
            print(f"  ZNU (eta on mass levels): {len(znu_data)} levels")
            print(f"    Values: {np.array2string(np.asarray(znu_data), precision=4, separator=', ', max_line_width=70)}")
        
        # ZNW: eta values on full (w) levels
        if 'ZNW' in ds.variables:
            znw = ds.variables['ZNW']
            if znw.ndim >= 2:
                znw_data = znw[0, :]
            else:
                znw_data = znw[:]
            print(f"  ZNW (eta on w levels): {len(znw_data)} levels")
            print(f"    Values: {np.array2string(np.asarray(znw_data), precision=4, separator=', ', max_line_width=70)}")
        
        # Print approximate pressure levels if P and PB available (raw WRF output)
        if 'P' in ds.variables and 'PB' in ds.variables:
            # Total pressure = perturbation + base state
            p = ds.variables['P'][0, :, :, :]  # (bottom_top, south_north, west_east)
            pb = ds.variables['PB'][0, :, :, :]
            total_p = (p + pb) / 100.0  # Convert to hPa
            # Get mean pressure at each level (averaged over domain)
            mean_p_levels = np.mean(total_p, axis=(1, 2))
            print(f"  Approximate pressure levels (hPa, domain-averaged):")
            print(f"    {len(mean_p_levels)} levels from surface to top:")
            for i, p_val in enumerate(mean_p_levels):
                if i < 10 or i >= len(mean_p_levels) - 5 or i % 5 == 0:
                    print(f"      Level {i:2d}: {p_val:7.1f} hPa")
                elif i == 10:
                    print(f"      ...")
        
        # If no vertical level info found
        if ('pres_levels' not in ds.variables and 
            'ZNU' not in ds.variables and 
            'ZNW' not in ds.variables and
            not ('P' in ds.variables and 'PB' in ds.variables)):
            print("  No vertical level information found.")
        
        print()
        
        # Variables summary
        print(f"=== Variables ({len(ds.variables)} total) ===\n")
        
        # Categorize variables by dimensions
        surface_vars = []
        level_vars = []
        other_vars = []
        
        for name, var in sorted(ds.variables.items()):
            dims = var.dimensions
            
            if filter_vars and name not in filter_vars:
                continue
            
            # Categorize
            if 'bottom_top' in dims or 'bottom_top_stag' in dims or 'pres_bottom_top' in dims:
                level_vars.append((name, var))
            elif 'south_north' in dims and 'west_east' in dims:
                surface_vars.append((name, var))
            else:
                other_vars.append((name, var))
        
        def print_var_info(name, var, show_stats=False):
            dims = var.dimensions
            shape = var.shape
            dtype = var.dtype
            
            # Get variable attributes
            units = getattr(var, 'units', 'N/A')
            description = getattr(var, 'description', getattr(var, 'long_name', 'N/A'))
            
            print(f"  {name}")
            print(f"    Shape: {shape}")
            print(f"    Dims:  {dims}")
            print(f"    Type:  {dtype}")
            print(f"    Units: {units}")
            if description != 'N/A':
                desc_str = str(description)
                if len(desc_str) > 60:
                    desc_str = desc_str[:57] + "..."
                print(f"    Desc:  {desc_str}")
            
            if show_stats and var.ndim > 0:
                try:
                    data = var[:]
                    if hasattr(data, 'compressed'):
                        data = data.compressed()
                    data = np.asarray(data).flatten()
                    if data.size > 0 and np.issubdtype(data.dtype, np.number):
                        print(f"    Min:   {np.nanmin(data):.6g}")
                        print(f"    Max:   {np.nanmax(data):.6g}")
                        print(f"    Mean:  {np.nanmean(data):.6g}")
                        print(f"    Std:   {np.nanstd(data):.6g}")
                except Exception as e:
                    print(f"    Stats: Error reading data - {e}")
            print()
        
        # Print surface variables
        if surface_vars:
            print(f"--- Surface Variables ({len(surface_vars)}) ---\n")
            for name, var in surface_vars:
                print_var_info(name, var, show_stats=verbose)
        
        # Print 3D/level variables
        if level_vars:
            print(f"--- 3D/Level Variables ({len(level_vars)}) ---\n")
            for name, var in level_vars:
                print_var_info(name, var, show_stats=verbose)
        
        # Print other variables
        if other_vars:
            print(f"--- Other Variables ({len(other_vars)}) ---\n")
            for name, var in other_vars:
                print_var_info(name, var, show_stats=verbose)
        
        # Quick reference table
        if not filter_vars:
            print("\n=== Quick Reference (Surface Variables) ===")
            print(f"{'Name':<15} {'Shape':<25} {'Description'}")
            print("-" * 70)
            for name, var in surface_vars[:30]:  # Limit to first 30
                desc = getattr(var, 'description', getattr(var, 'long_name', ''))
                if len(str(desc)) > 35:
                    desc = str(desc)[:32] + "..."
                print(f"{name:<15} {str(var.shape):<25} {desc}")
            
            if len(surface_vars) > 30:
                print(f"... and {len(surface_vars) - 30} more surface variables")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect RWRF/WRF NetCDF file variables"
    )
    parser.add_argument(
        "filepath",
        type=str,
        help="Path to RWRF/WRF NetCDF file"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show statistics (min, max, mean, std) for each variable"
    )
    parser.add_argument(
        "-f", "--filter",
        type=str,
        default=None,
        help="Comma-separated list of variable names to inspect (e.g., T2,U10,V10)"
    )
    
    args = parser.parse_args()
    
    filter_vars = None
    if args.filter:
        filter_vars = [v.strip() for v in args.filter.split(',')]
    
    inspect_rwrf(args.filepath, verbose=args.verbose, filter_vars=filter_vars)


if __name__ == "__main__":
    main()
