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
Simplified inference dataloader for single-timestep zarr data.
This dataloader is designed for inference only - it loads a single timestep
of LowRes/HighRes data from a zarr store and provides it for model inference.

Expected zarr structure:
    output/
    ├── LowRes/
    │   ├── inference.zarr/
    │   └── stats/
    │       ├── means.npy
    │       └── stds.npy
    ├── HighRes/
    │   ├── inference.zarr/
    │   └── stats/
    │       ├── means.npy
    │       └── stds.npy
    └── invariants/
        └── invariants.zarr/
"""

import os
import glob
import torch
import numpy as np
import xarray as xr
import dask
from datetime import datetime
from typing import Optional

from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.distributed import DistributedManager

from .dataset import StormCastDataset

logger = PythonLogger("inference_dataset")


class InferenceDataset(StormCastDataset):
    """
    Single-timestep dataset for inference.
    Loads one timestep of LowRes and HighRes data from zarr stores.
    No target data (t+1) is needed - only input data for t=0.
    """

    def __init__(self, params, train=False):
        """
        Initialize inference dataset.
        
        Args:
            params: Configuration object with:
                - location: Path to zarr output directory
                - HighRes_img_size: Image dimensions [H, W]
                - invariants: List of invariant channel names
                - kept_LowRes_channels: 'all' or list of channels
                - kept_HighRes_channels: 'all' or list of channels
            train: Ignored for inference (always False behavior)
        """
        dist = DistributedManager()
        self.logger0 = RankZeroLoggingWrapper(logger, dist)
        
        dask.config.set(scheduler="synchronous")
        
        self.params = params
        self.location = self.params.location
        self.train = False  # Always inference mode
        self.normalize = True
        
        # Load data and stats
        self._load_zarr_data()
        
        # Channel selection
        self.kept_LowRes_channels = (
            self.LowRes_channels
            if params.kept_LowRes_channels == "all"
            else params.kept_LowRes_channels
        )
        self.kept_HighRes_channels = (
            self.HighRes_channels
            if params.kept_HighRes_channels == "all"
            else params.kept_HighRes_channels
        )
        
        kept_LowRes_idx = [self.LowRes_channels.index(c) for c in self.kept_LowRes_channels]
        kept_HighRes_idx = [self.HighRes_channels.index(c) for c in self.kept_HighRes_channels]
        
        # Load normalization stats
        self.means_HighRes = np.load(
            os.path.join(self.location, "HighRes", "stats", "means.npy")
        )[kept_HighRes_idx, None, None]
        self.stds_HighRes = np.load(
            os.path.join(self.location, "HighRes", "stats", "stds.npy")
        )[kept_HighRes_idx, None, None]
        self.means_LowRes = np.load(
            os.path.join(self.location, "LowRes", "stats", "means.npy")
        )[kept_LowRes_idx, None, None]
        self.stds_LowRes = np.load(
            os.path.join(self.location, "LowRes", "stats", "stds.npy")
        )[kept_LowRes_idx, None, None]
        
        self.invariants = params.invariants
        
        self.logger0.info(f"Loaded inference dataset from {self.location}")
        self.logger0.info(f"LowRes channels: {self.LowRes_channels}")
        self.logger0.info(f"HighRes channels: {self.HighRes_channels}")
        self.logger0.info(f"Kept LowRes channels: {self.kept_LowRes_channels}")
        self.logger0.info(f"Kept HighRes channels: {self.kept_HighRes_channels}")
        self.logger0.info(f"Timestamp: {self.timestamp}")

    def _load_zarr_data(self):
        """Load zarr stores and extract metadata."""
        # Find LowRes zarr
        lowres_zarr_path = os.path.join(self.location, "LowRes", "inference.zarr")
        if not os.path.exists(lowres_zarr_path):
            # Try to find any zarr in LowRes directory
            lowres_zarrs = glob.glob(
                os.path.join(self.location, "LowRes", "**", "*.zarr"), recursive=True
            )
            if lowres_zarrs:
                lowres_zarr_path = lowres_zarrs[0]
            else:
                raise FileNotFoundError(f"No LowRes zarr found in {self.location}/LowRes")
        
        # Find HighRes zarr
        highres_zarr_path = os.path.join(self.location, "HighRes", "inference.zarr")
        if not os.path.exists(highres_zarr_path):
            highres_zarrs = glob.glob(
                os.path.join(self.location, "HighRes", "**", "*.zarr"), recursive=True
            )
            if highres_zarrs:
                highres_zarr_path = highres_zarrs[0]
            else:
                raise FileNotFoundError(f"No HighRes zarr found in {self.location}/HighRes")
        
        # Support for multi-timestep LowRes data
        self.multi_timestep_lowres = False
        
        self.logger0.info(f"Loading LowRes from: {lowres_zarr_path}")
        self.logger0.info(f"Loading HighRes from: {highres_zarr_path}")
        
        # Open datasets
        self.ds_LowRes = xr.open_zarr(lowres_zarr_path, consolidated=True)
        self.ds_HighRes = xr.open_zarr(highres_zarr_path, consolidated=True)
        
        # Extract channel info
        self.LowRes_channels = list(self.ds_LowRes.channel.values)
        self.HighRes_channels = list(self.ds_HighRes.channel.values)
        
        # Extract coordinates
        self.LowRes_lat = self.ds_LowRes.latitude
        self.LowRes_lon = self.ds_LowRes.longitude
        self.HighRes_lat = self.ds_HighRes.latitude
        self.HighRes_lon = self.ds_HighRes.longitude
        
        # Extract timestamps
        self.LowRes_timestamps = self.ds_LowRes.time.values
        self.HighRes_timestamps = self.ds_HighRes.time.values
        
        # Check if we have multi-timestep LowRes data
        if len(self.LowRes_timestamps) > 1:
            self.multi_timestep_lowres = True
            self.logger0.info(f"Multi-timestep LowRes data detected: {len(self.LowRes_timestamps)} timesteps")
            self.logger0.info(f"LowRes timesteps: {self.LowRes_timestamps}")
        else:
            self.logger0.info("Single-timestep LowRes data")
        
        # Base timestamp (from HighRes, which is always single timestep initial condition)
        if len(self.HighRes_timestamps) != 1:
            self.logger0.warning(f"Expected 1 HighRes timestamp, got {len(self.HighRes_timestamps)}. Using first.")
        self.timestamp = self.HighRes_timestamps[0]

    def background_channels(self):
        """Metadata for the background channels (LowRes)."""
        return self.kept_LowRes_channels

    def state_channels(self):
        """Metadata for the state channels (HighRes)."""
        return self.kept_HighRes_channels

    def image_shape(self):
        """Get the (height, width) of the data."""
        return tuple(self.params.HighRes_img_size)

    def latitude(self):
        """Return latitude coordinates."""
        return np.asarray(self.HighRes_lat.values)

    def longitude(self):
        """Return longitude coordinates."""
        return np.asarray(self.HighRes_lon.values)

    def get_invariants(self):
        """Return invariants used for inference, or None if no invariants are used."""
        invariants_path = os.path.join(self.location, "invariants", "invariants.zarr")
        
        if not os.path.exists(invariants_path):
            self.logger0.warning(f"Invariants not found at {invariants_path}")
            return None
        
        invariants = xr.open_zarr(invariants_path)
        invariant_channels_in_dataset = list(invariants.channel.values)
        
        for invariant in self.invariants:
            if invariant not in invariant_channels_in_dataset:
                raise ValueError(
                    f"Requested invariant {invariant} not in dataset. "
                    f"Available: {invariant_channels_in_dataset}"
                )
        
        invariant_array = (
            invariants["HighRes_invariants"].sel(channel=self.invariants).values
        )
        
        return invariant_array

    def __len__(self):
        """Return 1 since this is single-timestep inference."""
        return 1

    def normalize_background(self, x: np.ndarray) -> np.ndarray:
        """Convert background from physical units to normalized data."""
        if self.normalize:
            x = x - self.means_LowRes
            x = x / self.stds_LowRes
        return x

    def denormalize_background(self, x: np.ndarray) -> np.ndarray:
        """Convert background from normalized data to physical units."""
        if self.normalize:
            x = x * self.stds_LowRes
            x = x + self.means_LowRes
        return x

    def normalize_state(self, x: np.ndarray) -> np.ndarray:
        """Convert state from physical units to normalized data."""
        if self.normalize:
            x = x - self.means_HighRes
            x = x / self.stds_HighRes
        return x

    def denormalize_state(self, x: np.ndarray) -> np.ndarray:
        """Convert state from normalized data to physical units."""
        if self.normalize:
            x = x * self.stds_HighRes
            x = x + self.means_HighRes
        return x

    def _get_LowRes(self, time_idx: int = 0):
        """Get normalized LowRes data at given time index."""
        inp_field = self.ds_LowRes.sel(
            time=self.LowRes_timestamps[time_idx], 
            channel=self.kept_LowRes_channels
        ).LowRes.values.copy()
        
        inp = self.normalize_background(inp_field)
        return torch.as_tensor(inp, dtype=torch.float32)
    
    def get_LowRes_for_timestep(self, forecast_step: int, dt_hours: int = 1):
        """
        Get LowRes data for a specific forecast timestep.
        
        For multi-timestep LowRes data, selects the appropriate LowRes based on
        6-hour intervals. Inference loop uses step=0,1,2,... which produces outputs
        at hours 1,2,3,... (step+1). For example with dt_hours=1:
        - forecast_step 0-5 (hours 1-6) -> LowRes timestep 0 (covers 0-5h) until hour 6
        - forecast_step 6-11 (hours 7-12) -> LowRes timestep 1 (covers 6-11h)
        - etc.
        
        Args:
            forecast_step: The forecast step number from inference loop (0, 1, 2, ...)
            dt_hours: Hours per forecast step (default: 1)
            
        Returns:
            Normalized LowRes tensor [C, H, W]
        """
        if not self.multi_timestep_lowres:
            # Single timestep mode - always return the same LowRes
            return self._get_LowRes(0)
        
        # Calculate which LowRes timestep to use (one per 6 hours)
        lowres_dt = 6  # LowRes data available every 6 hours
        forecast_hour = forecast_step * dt_hours
        lowres_idx = forecast_hour // lowres_dt
        
        # Clamp to available range
        lowres_idx = min(lowres_idx, len(self.LowRes_timestamps) - 1)
        
        return self._get_LowRes(lowres_idx)
    
    def get_LowRes_timestep_info(self, forecast_step: int, dt_hours: int = 1):
        """
        Get information about which LowRes timestep will be used.
        
        Returns:
            Tuple of (lowres_idx, lowres_timestamp_str) or (0, 'single') for single-timestep mode
        """
        if not self.multi_timestep_lowres:
            return 0, 'single-timestep'
        
        lowres_dt = 6
        forecast_hour = forecast_step * dt_hours
        lowres_idx = forecast_hour // lowres_dt
        lowres_idx = min(lowres_idx, len(self.LowRes_timestamps) - 1)
        
        timestamp_str = str(self.LowRes_timestamps[lowres_idx])
        return lowres_idx, timestamp_str

    def _get_HighRes(self, time_idx: int = 0):
        """Get normalized HighRes data at given time index."""
        # For inference, we only have one HighRes timestep (initial condition)
        # self.timestamp is a scalar, not an array
        inp_field = self.ds_HighRes.sel(
            time=self.timestamp,
            channel=self.kept_HighRes_channels
        ).HighRes.values.copy()
        
        inp = self.normalize_state(inp_field)
        return torch.as_tensor(inp, dtype=torch.float32)

    def __getitem__(self, idx):
        """
        Return data for inference.
        
        Returns:
            dict with:
                - background: LowRes data (normalized)
                - state: HighRes data (normalized) - only input, no target
                - timestamp: datetime of the data
        """
        # For inference, we only have one HighRes timestep (initial condition)
        # Always use index 0 for HighRes, but LowRes might have multiple timesteps
        time_idx = 0
        
        lowres = self._get_LowRes(time_idx)
        highres = self._get_HighRes(time_idx)
        
        return {
            "background": lowres,
            "state": highres,  # Single tensor, not tuple (no target for inference)
            "timestamp": self.timestamp,
        }

    def get_input_data(self):
        """
        Convenience method to get all input data for inference.
        
        Returns:
            dict with:
                - background: LowRes tensor [C, H, W]
                - state: HighRes tensor [C, H, W]
                - invariants: Invariant tensor [C, H, W] or None
                - timestamp: datetime
        """
        data = self[0]
        invariants = self.get_invariants()
        
        return {
            "background": data["background"],
            "state": data["state"],
            "invariants": torch.from_numpy(invariants) if invariants is not None else None,
            "timestamp": self.timestamp,
            "background_channels": self.kept_LowRes_channels,
            "state_channels": self.kept_HighRes_channels,
        }


# Alias for dataset discovery
Dataset = InferenceDataset
