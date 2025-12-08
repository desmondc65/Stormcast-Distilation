# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
StormCast NCDR Inference Script

Multi-timestep autoregressive inference for NCDR data:
- Loads preprocessed zarr data (from to_zarr.py output)
- Runs regression + diffusion model to predict t+1, t+2, ... t+n
- For n_steps > 1, uses predicted HighRes as input for next timestep
- Saves output to both zarr and netcdf formats

Usage:
    python inference_ncdr.py
    python inference_ncdr.py inference.n_steps=6
    python inference_ncdr.py inference.rundir=/path/to/output
    python inference_ncdr.py dataset.location=/path/to/zarr/data
"""

import os
import hydra
import torch
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from typing import List, Optional
from omegaconf import DictConfig, OmegaConf
from physicsnemo.distributed import DistributedManager
from physicsnemo.models import Module
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper

from datasets import dataset_classes
from utils.nn import build_network_condition_and_target, diffusion_model_forward

logger = PythonLogger("inference_ncdr")


def save_multistep_output(
    output_dir: str,
    predictions: List[np.ndarray],
    channels: list,
    latitude: np.ndarray,
    longitude: np.ndarray,
    timestamps: List[np.datetime64],
    input_state: Optional[np.ndarray] = None,
    input_timestamp: Optional[np.datetime64] = None,
):
    """
    Save multi-step inference output to separate zarr and NetCDF files for each timestep.
    
    Args:
        output_dir: Directory to save output
        predictions: List of predicted state arrays, each [C, H, W]
        channels: List of channel names
        latitude: Latitude coordinates [H, W]
        longitude: Longitude coordinates [H, W]
        timestamps: List of output timestamps
        input_state: Optional input state for reference [C, H, W]
        input_timestamp: Optional input timestamp
        
    Returns:
        Tuple of (list of zarr paths, list of nc paths)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    n_steps = len(predictions)
    height, width = predictions[0].shape[1], predictions[0].shape[2]
    
    zarr_paths = []
    nc_paths = []
    
    # Save each timestep to separate files
    for step_idx, (prediction, timestamp) in enumerate(zip(predictions, timestamps)):
        step_num = step_idx + 1
        
        # Format timestamp for filename (e.g., 2024010112 for 2024-01-01 12:00)
        ts_str = np.datetime_as_string(timestamp, unit='h').replace('-', '').replace('T', '').replace(':', '')[:10]
        
        # Create dataset for this timestep
        ds = xr.Dataset(
            coords={
                "time": [timestamp],
                "y": np.arange(height),
                "x": np.arange(width),
                "latitude": (["y", "x"], latitude),
                "longitude": (["y", "x"], longitude),
            },
            attrs={
                "description": f"StormCast NCDR inference output - step {step_num}",
                "step": step_num,
                "created": datetime.now().isoformat(),
            }
        )
        
        # Add per-channel variables
        for i, ch in enumerate(channels):
            ds[ch] = (["time", "y", "x"], prediction[np.newaxis, i, :, :])
            ds[ch].attrs["channel_index"] = i
        
        # Save to zarr
        zarr_path = os.path.join(output_dir, f"step_{step_num:02d}_{ts_str}.zarr")
        ds.to_zarr(zarr_path, mode="w", consolidated=True)
        zarr_paths.append(zarr_path)
        
        # Save to NetCDF
        nc_path = os.path.join(output_dir, f"step_{step_num:02d}_{ts_str}.nc")
        ds.to_netcdf(nc_path, format="NETCDF4")
        nc_paths.append(nc_path)
    
    # Save input reference separately
    if input_state is not None and input_timestamp is not None:
        ts_str = np.datetime_as_string(input_timestamp, unit='h').replace('-', '').replace('T', '').replace(':', '')[:10]
        
        ds_input = xr.Dataset(
            coords={
                "time": [input_timestamp],
                "y": np.arange(height),
                "x": np.arange(width),
                "latitude": (["y", "x"], latitude),
                "longitude": (["y", "x"], longitude),
            },
            attrs={
                "description": "Input state at t=0",
            }
        )
        for i, ch in enumerate(channels):
            ds_input[ch] = (["time", "y", "x"], input_state[np.newaxis, i, :, :])
        
        ds_input.to_netcdf(os.path.join(output_dir, f"step_00_input_{ts_str}.nc"))
        ds_input.to_zarr(os.path.join(output_dir, f"step_00_input_{ts_str}.zarr"), mode="w", consolidated=True)
    
    return zarr_paths, nc_paths


def run_single_step_inference(
    background: torch.Tensor,
    state: torch.Tensor,
    invariant_tensor: Optional[torch.Tensor],
    regression_model: Optional[Module],
    diffusion_model: Module,
    condition_list: list,
    regression_condition_list: list,
    sampler_args: dict,
) -> torch.Tensor:
    """
    Run a single step of inference.
    
    Args:
        background: LowRes background tensor [B, C, H, W]
        state: Current HighRes state tensor [B, C, H, W]
        invariant_tensor: Invariant tensor or None
        regression_model: Regression model or None
        diffusion_model: Diffusion model
        condition_list: List of conditions for diffusion model
        regression_condition_list: List of conditions for regression model
        sampler_args: Sampler arguments for diffusion
        
    Returns:
        Predicted state tensor [B, C, H, W] (normalized)
    """
    with torch.no_grad():
        # Build condition tensor and run regression model
        (condition, _, regression_output) = build_network_condition_and_target(
            background,
            [state, state],  # state tuple: [input_state, target_state placeholder]
            invariant_tensor,
            regression_net=regression_model,
            condition_list=condition_list,
            regression_condition_list=regression_condition_list,
        )
        
        # If no regression model, start from zeros
        if regression_output is None:
            regression_output = torch.zeros_like(state)
        
        # Run diffusion model
        diffusion_output = diffusion_model_forward(
            diffusion_model,
            condition,
            state.shape,
            sampler_args=sampler_args,
        )
        
        # Combine regression and diffusion outputs
        prediction = regression_output + diffusion_output
        
    return prediction


@hydra.main(version_base=None, config_path="config", config_name="inference_ncdr")
def main(cfg: DictConfig) -> None:
    """
    Run multi-step autoregressive inference using StormCast regression + diffusion models.
    
    For n_steps > 1, uses the predicted HighRes state as input for the next timestep,
    while keeping the same LowRes background.
    """
    # Initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()
    logger0 = RankZeroLoggingWrapper(logger, dist)
    
    # Settings
    n_steps = getattr(cfg.inference, 'n_steps', 1)
    dt_hours = getattr(cfg.inference, 'dt_hours', 1)  # Hours per timestep
    
    # Use GPU
    device = dist.device
    torch.cuda.empty_cache()
    logger0.info(f"Using GPU: {device}")
    
    # Print configuration
    if dist.rank == 0:
        logger0.info("=" * 60)
        logger0.info("StormCast NCDR Multi-Step Inference")
        logger0.info("=" * 60)
        logger0.info(f"Number of forecast steps: {n_steps}")
        logger0.info(f"Hours per step: {dt_hours}")
        logger0.info("Configuration:")
        logger0.info(OmegaConf.to_yaml(cfg))
    
    # Create output directory
    os.makedirs(cfg.inference.rundir, exist_ok=True)
    
    # Load dataset
    logger0.info(f"Loading dataset from: {cfg.dataset.location}")
    dataset_cls = dataset_classes[cfg.dataset.name]
    dataset = dataset_cls(cfg.dataset, train=False)
    
    # Get metadata
    background_channels = dataset.background_channels()
    state_channels = dataset.state_channels()
    
    logger0.info(f"Background channels ({len(background_channels)}): {background_channels}")
    logger0.info(f"State channels ({len(state_channels)}): {state_channels}")
    logger0.info(f"Image shape: {dataset.image_shape()}")
    
    # Get input data
    input_data = dataset.get_input_data()
    background = input_data["background"].to(device=device, dtype=torch.float32).unsqueeze(0)
    state = input_data["state"].to(device=device, dtype=torch.float32).unsqueeze(0)
    input_timestamp = input_data["timestamp"]
    
    # Convert timestamp to numpy datetime64 if needed
    if not isinstance(input_timestamp, np.datetime64):
        input_timestamp = np.datetime64(input_timestamp)
    
    logger0.info(f"Input timestamp: {input_timestamp}")
    logger0.info(f"Background shape: {background.shape}")
    logger0.info(f"State shape: {state.shape}")
    
    # Load invariants
    invariant_array = dataset.get_invariants()
    if invariant_array is not None:
        invariant_tensor = torch.from_numpy(invariant_array).to(device=device, dtype=torch.float32).unsqueeze(0)
        logger0.info(f"Invariants shape: {invariant_tensor.shape}")
    else:
        invariant_tensor = None
        logger0.info("No invariants loaded")
    
    # Get coordinate data
    latitude = dataset.latitude()
    longitude = dataset.longitude()
    
    # Save input state for reference (denormalized)
    input_state_denorm = dataset.denormalize_state(state.cpu().numpy()[0].copy())
    
    # Load pretrained models
    logger0.info("Loading pretrained models...")
    
    regression_model = None
    if "regression" in cfg.model.diffusion_conditions:
        logger0.info(f"Loading regression model from: {cfg.inference.regression_checkpoint}")
        regression_model = Module.from_checkpoint(cfg.inference.regression_checkpoint)
        regression_model = regression_model.to(device)
        regression_model.eval()
    
    logger0.info(f"Loading diffusion model from: {cfg.inference.diffusion_checkpoint}")
    diffusion_model = Module.from_checkpoint(cfg.inference.diffusion_checkpoint)
    diffusion_model = diffusion_model.to(device)
    diffusion_model.eval()
    
    # Get sampler args
    sampler_args = dict(cfg.sampler.args) if hasattr(cfg, 'sampler') else {}
    
    # Run multi-step inference
    logger0.info(f"Running {n_steps}-step autoregressive inference...")
    
    predictions = []
    timestamps = []
    current_state = state  # Start with initial state (normalized)
    
    for step in range(n_steps):
        step_num = step + 1
        logger0.info(f"Step {step_num}/{n_steps}...")
        
        # Run single step inference
        predicted_state = run_single_step_inference(
            background=background,  # Same LowRes for all steps
            state=current_state,    # Use current HighRes state
            invariant_tensor=invariant_tensor,
            regression_model=regression_model,
            diffusion_model=diffusion_model,
            condition_list=cfg.model.diffusion_conditions,
            regression_condition_list=cfg.model.regression_conditions,
            sampler_args=sampler_args,
        )
        
        # Calculate output timestamp
        output_timestamp = input_timestamp + np.timedelta64(step_num * dt_hours, 'h')
        timestamps.append(output_timestamp)
        
        # Denormalize and store prediction
        prediction_denorm = dataset.denormalize_state(
            predicted_state.cpu().numpy()[0].copy()
        )
        predictions.append(prediction_denorm)
        
        logger0.info(f"  Step {step_num} output timestamp: {output_timestamp}")
        logger0.info(f"  Prediction range: [{prediction_denorm.min():.4f}, {prediction_denorm.max():.4f}]")
        
        # Update current state for next iteration (keep normalized)
        current_state = predicted_state
        
        # Clear cache between steps
        torch.cuda.empty_cache()
    
    # Free model memory
    del regression_model, diffusion_model
    torch.cuda.empty_cache()
    
    # Save outputs (separate zarr and nc for each timestep)
    logger0.info("Saving outputs to separate zarr and NetCDF files for each timestep...")
    
    zarr_paths, nc_paths = save_multistep_output(
        output_dir=cfg.inference.rundir,
        predictions=predictions,
        channels=state_channels,
        latitude=latitude,
        longitude=longitude,
        timestamps=timestamps,
        input_state=input_state_denorm,
        input_timestamp=input_timestamp,
    )
    
    logger0.info(f"Saved {len(zarr_paths)} zarr files and {len(nc_paths)} NetCDF files")
    for zp, ncp in zip(zarr_paths, nc_paths):
        logger0.info(f"  {os.path.basename(zp)}")
    
    # Print summary
    logger0.info("=" * 60)
    logger0.info("Inference Summary")
    logger0.info("=" * 60)
    logger0.info(f"Input timestamp: {input_timestamp}")
    logger0.info(f"Forecast steps: {n_steps}")
    logger0.info(f"Hours per step: {dt_hours}")
    logger0.info(f"Total forecast hours: {n_steps * dt_hours}")
    logger0.info(f"Output timestamps:")
    for i, ts in enumerate(timestamps):
        logger0.info(f"  Step {i+1}: {ts}")
    logger0.info(f"Output directory: {cfg.inference.rundir}")
    logger0.info("=" * 60)
    logger0.info("Inference complete!")
    logger0.info("=" * 60)


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
