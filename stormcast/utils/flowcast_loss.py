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

"""Conditional Flow Matching loss (Ribeiro & Pucer 2025, FlowCast Algorithm 1).

Adapted to StormCast's residual pipeline: ``images`` is expected to be the
*raw* target residual R_t = X_t - M_t. Internally the loss standardizes by
``sigma_data`` so the flow operates on a roughly-unit-variance manifold, and
supports the same per-channel weighting + qpepre log-PSD regularizer that the
Consistency Distillation loss already exposes.
"""

from typing import Dict, Optional, Sequence

import torch
from torch import Tensor

from physicsnemo.experimental.metrics.diffusion.consistency_loss import (
    _radial_log_psd,
)


class FlowCastLoss:
    """Independent Conditional Flow Matching loss (I-CFM).

    Given a target residual ``x_1`` and a noise sample ``x_0``, draws a time
    ``t ~ U(0, 1)`` and an interpolated state

        x_t = (1 - t) * x_0 + t * x_1 + sigma_path * eps,   eps ~ N(0, I)

    then asks the student network to regress the straight-line vector field
    u_t = x_1 - x_0 (Lipman et al. 2023; Tong et al. 2024). The FlowCast paper
    uses ``sigma_path = 0.01`` which "thickens" the otherwise-singular
    conditional path and stabilizes high-dimensional training.

    Parameters
    ----------
    sigma_data : float
        Standard deviation of the raw target residual. Inputs are divided by
        this before constructing x_t and the predicted velocity is interpreted
        in the same standardized space.
    sigma_path : float
        The I-CFM probability-path standard deviation (FlowCast uses 0.01).
    t_eps : float
        Minimum time offset to avoid exactly-zero / exactly-one time samples.
    channel_weights : sequence of float, optional
        Per-channel weights on the MSE term (same convention as CD loss).
    spectral_channels : sequence of int, optional
        Target-channel indices on which to add a radial log-PSD L1 regularizer
        between the predicted-clean and target-clean fields (default: off).
    spectral_weight : float
        Weight on the log-PSD term.
    """

    def __init__(
        self,
        sigma_data: float = 0.5,
        sigma_path: float = 0.01,
        t_eps: float = 1e-5,
        channel_weights: Optional[Sequence[float]] = None,
        spectral_channels: Optional[Sequence[int]] = None,
        spectral_weight: float = 0.0,
    ):
        self.sigma_data = float(sigma_data)
        self.sigma_path = float(sigma_path)
        self.t_eps = float(t_eps)
        self.channel_weights = (
            None
            if channel_weights is None
            else tuple(float(w) for w in channel_weights)
        )
        self.spectral_channels = (
            tuple(int(i) for i in spectral_channels) if spectral_channels else ()
        )
        self.spectral_weight = float(spectral_weight)

    def __call__(
        self,
        student: torch.nn.Module,
        images: Tensor,
        condition: Tensor,
    ) -> Dict[str, Tensor]:
        """Compute the I-CFM regression loss.

        Parameters
        ----------
        student : torch.nn.Module
            FlowCastPrecond (or DDP-wrapped version). Must implement the
            signature ``student(x, t, condition=...)`` and return a tensor of
            the same shape as ``x``.
        images : Tensor
            Raw target residual R_t of shape (B, C, H, W).
        condition : Tensor
            Conditioning tensor of shape (B, C_cond, H, W).

        Returns
        -------
        dict
            ``{"loss": scalar total, "pointwise": scalar MSE,
               "spectral": scalar log-PSD, "sigma_path": float}``.
        """
        B = images.shape[0]
        device = images.device
        dtype = images.dtype

        # Standardize target residual to unit-variance working space.
        x1 = images / self.sigma_data
        x0 = torch.randn_like(x1)

        # Uniform time in [t_eps, 1 - t_eps].
        t = torch.rand(B, device=device, dtype=dtype)
        t = t * (1.0 - 2.0 * self.t_eps) + self.t_eps
        t_view = t.view(-1, 1, 1, 1)

        # Gaussian probability path with a small constant std.
        eps = torch.randn_like(x1) if self.sigma_path > 0 else None
        xt = (1.0 - t_view) * x0 + t_view * x1
        if self.sigma_path > 0:
            xt = xt + self.sigma_path * eps

        # Straight-line target field.
        u_target = x1 - x0

        # Student predicts the velocity in standardized space.
        v_pred = student(xt, t, condition=condition)

        # --- Pointwise MSE, with optional per-channel β ---
        pw = (v_pred - u_target) ** 2  # (B, C, H, W)
        if self.channel_weights is not None:
            assert len(self.channel_weights) == pw.shape[1], (
                f"channel_weights length {len(self.channel_weights)} "
                f"!= num channels {pw.shape[1]}"
            )
            beta = torch.tensor(
                self.channel_weights, device=pw.device, dtype=pw.dtype
            ).view(1, -1, 1, 1)
            pw = pw * beta
        pointwise_loss = pw.mean()

        # --- Spectral regularizer on qpepre-style channels ---
        if self.spectral_channels and self.spectral_weight > 0:
            # Reconstruct the implied clean targets at this step:
            #   x1_pred = xt + (1 - t) * v_pred
            # and compare its log-PSD to the ground-truth x1. Operating on
            # clean fields (not velocities) keeps the spectrum interpretable.
            x1_pred = xt + (1.0 - t_view) * v_pred
            spec_terms = []
            for cidx in self.spectral_channels:
                log_ps_s = _radial_log_psd(x1_pred[:, cidx])
                log_ps_t = _radial_log_psd(x1[:, cidx])
                spec_terms.append((log_ps_s - log_ps_t).abs().mean())
            spectral_loss = self.spectral_weight * torch.stack(spec_terms).mean()
        else:
            spectral_loss = pointwise_loss.new_zeros(())

        total = pointwise_loss + spectral_loss
        return {
            "loss": total,
            "pointwise": pointwise_loss.detach(),
            "spectral": spectral_loss.detach(),
            "sigma_path": self.sigma_path,
        }
