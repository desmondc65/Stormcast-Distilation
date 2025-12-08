"""
Prediction Plotting Script

Plots inference output from StormCast NCDR.
Creates one figure per variable with subplots for each timestep.

Usage:
    python prediction_plotting.py --output_dir /path/to/inference/output
    python prediction_plotting.py --output_dir /path/to/output --save_dir /path/to/save/plots
"""

import os
import glob
import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import List, Dict, Optional

# Domain configuration (from inference_data_preprocess_config.yaml)
DOMAIN_SIZE = [224, 128]  # [height, width] (lat, lon)
LON_BOUNDS = [119.75, 122.25]  # [min, max]
LAT_BOUNDS = [21.6, 25.6]  # [min, max]


def load_prediction_files(output_dir: str) -> tuple:
    """
    Load all prediction files from output directory.
    
    Args:
        output_dir: Directory containing prediction files
        
    Returns:
        Tuple of (list of xarray datasets sorted by step, list of step numbers)
    """
    # Find all prediction nc files (excluding input reference)
    nc_files = sorted(glob.glob(os.path.join(output_dir, "step_*.nc")))
    
    # Separate input and prediction files
    input_files = [f for f in nc_files if "input" in os.path.basename(f)]
    pred_files = [f for f in nc_files if "input" not in os.path.basename(f)]
    
    # Sort prediction files by step number
    def get_step_num(filepath):
        basename = os.path.basename(filepath)
        # Format: step_01_2024010101.nc
        return int(basename.split("_")[1])
    
    pred_files = sorted(pred_files, key=get_step_num)
    
    # Load datasets
    datasets = []
    step_nums = []
    for f in pred_files:
        ds = xr.open_dataset(f)
        datasets.append(ds)
        step_nums.append(get_step_num(f))
    
    # Load input if exists
    input_ds = None
    if input_files:
        input_ds = xr.open_dataset(input_files[0])
    
    return datasets, step_nums, input_ds


def get_variable_names(ds: xr.Dataset) -> List[str]:
    """Get list of variable names from dataset (excluding coordinates)."""
    coord_names = set(ds.coords.keys())
    var_names = [v for v in ds.data_vars if v not in coord_names]
    return var_names


def plot_variable_timesteps(
    datasets: List[xr.Dataset],
    step_nums: List[int],
    variable: str,
    input_ds: Optional[xr.Dataset] = None,
    save_path: Optional[str] = None,
    figsize_per_subplot: tuple = (4, 5),
    cmap: str = "viridis",
    include_input: bool = False,
):
    """
    Plot all timesteps of a single variable in one figure with geographic coordinates.
    
    Args:
        datasets: List of xarray datasets for each timestep
        step_nums: List of step numbers
        variable: Variable name to plot
        input_ds: Optional input dataset (step 0)
        save_path: Path to save figure (if None, displays)
        figsize_per_subplot: Size of each subplot
        cmap: Colormap name
        include_input: Whether to include input (step 0) in the plot
    """
    n_steps = len(datasets)
    
    # Include input as step 0 if provided
    if include_input and input_ds is not None and variable in input_ds:
        all_data = [input_ds[variable].values[0]] + [ds[variable].values[0] for ds in datasets]
        all_labels = ["Input (t=0)"] + [f"Step {s} (t+{s}h)" for s in step_nums]
        all_times = [input_ds.time.values[0]] + [ds.time.values[0] for ds in datasets]
    else:
        all_data = [ds[variable].values[0] for ds in datasets]
        all_labels = [f"Step {s} (t+{s}h)" for s in step_nums]
        all_times = [ds.time.values[0] for ds in datasets]
    
    n_plots = len(all_data)
    
    # Determine grid layout (prefer wider than tall)
    if n_plots <= 3:
        nrows, ncols = 1, n_plots
    elif n_plots <= 6:
        nrows, ncols = 2, (n_plots + 1) // 2
    elif n_plots <= 9:
        nrows, ncols = 3, (n_plots + 2) // 3
    else:
        ncols = 4
        nrows = (n_plots + ncols - 1) // ncols
    
    # Create figure
    figsize = (figsize_per_subplot[0] * ncols, figsize_per_subplot[1] * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    # Find global min/max for consistent colorbar
    vmin = min(d.min() for d in all_data)
    vmax = max(d.max() for d in all_data)
    
    # Special handling for specific variables
    if variable == "qpepre":
        # Precipitation: use log scale or specific range
        vmin = max(0, vmin)
        cmap = "Blues"
    elif variable in ["t2m"]:
        cmap = "RdYlBu_r"
    elif variable in ["u10", "v10"]:
        # Wind: symmetric around zero
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max
        cmap = "RdBu_r"
    
    # Geographic extent [lon_min, lon_max, lat_min, lat_max]
    extent = [LON_BOUNDS[0], LON_BOUNDS[1], LAT_BOUNDS[0], LAT_BOUNDS[1]]
    
    # Plot each timestep
    for idx, (data, label, time) in enumerate(zip(all_data, all_labels, all_times)):
        ax = axes[idx]
        
        # Plot with geographic extent
        im = ax.imshow(
            data, 
            origin="lower", 
            cmap=cmap, 
            vmin=vmin, 
            vmax=vmax, 
            aspect="auto",
            extent=extent,
        )
        
        # Add colorbar to each subplot (like plots.py)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Format timestamp
        time_str = np.datetime_as_string(time, unit='h')
        ax.set_title(f"{label}\n{time_str}", fontsize=10)
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        
        # Set tick labels
        ax.set_xticks(np.linspace(LON_BOUNDS[0], LON_BOUNDS[1], 5))
        ax.set_yticks(np.linspace(LAT_BOUNDS[0], LAT_BOUNDS[1], 5))
    
    # Hide empty subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis("off")
    
    # Title
    fig.suptitle(f"Variable: {variable} - {n_plots} Timesteps", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
        plt.close()
    else:
        plt.show()


def plot_all_variables(
    output_dir: str,
    save_dir: Optional[str] = None,
    include_input: bool = False,
):
    """
    Plot all variables from inference output.
    
    Args:
        output_dir: Directory containing prediction files
        save_dir: Directory to save plots (if None, displays)
        include_input: Whether to include input (step 0) in plots
    """
    print(f"Loading predictions from: {output_dir}")
    datasets, step_nums, input_ds = load_prediction_files(output_dir)
    
    if not datasets:
        print("No prediction files found!")
        return
    
    print(f"Found {len(datasets)} prediction timesteps: steps {step_nums}")
    if input_ds is not None:
        print("Found input reference file")
    
    # Get variable names from first dataset
    variables = get_variable_names(datasets[0])
    print(f"Variables to plot: {variables}")
    
    # Create save directory if needed
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Plot each variable
    for var in variables:
        print(f"Plotting variable: {var}")
        
        save_path = None
        if save_dir:
            save_path = os.path.join(save_dir, f"{var}_timesteps.png")
        
        plot_variable_timesteps(
            datasets=datasets,
            step_nums=step_nums,
            variable=var,
            input_ds=input_ds,
            save_path=save_path,
            include_input=include_input,
        )
    
    # Close all datasets
    for ds in datasets:
        ds.close()
    if input_ds is not None:
        input_ds.close()
    
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Plot StormCast NCDR inference output")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/master/13/dczy/code/stormcast-ncdr/data/stormcast_nano5_inference/inference_output/stormcast-ncdr-inference/0",
        help="Directory containing prediction files",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/home/master/13/dczy/code/stormcast-ncdr/data/stormcast_nano5_inference/inference_output/stormcast-ncdr-inference/0/plots",
        help="Directory to save plots (if not specified, displays interactively)",
    )
    parser.add_argument(
        "--include_input",
        action="store_true",
        help="Include input (step 0) in plots (default: exclude)",
    )
    
    args = parser.parse_args()
    
    # If no save_dir specified, save to output_dir/plots
    save_dir = args.save_dir
    if save_dir is None:
        save_dir = os.path.join(args.output_dir, "plots")
    
    plot_all_variables(
        output_dir=args.output_dir,
        save_dir=save_dir,
        include_input=args.include_input,
    )


if __name__ == "__main__":
    main()
