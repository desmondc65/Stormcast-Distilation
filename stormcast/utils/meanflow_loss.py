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

"""MeanFlow loss (Geng et al. 2025, "Mean Flows for One-step Generative Modeling").

Adapted to StormCast's residual pipeline: ``images`` is expected to be the
*raw* target residual r_{t+1} = M_{t+1} - mu_{t+1}. Internally the loss
standardizes by ``sigma_data`` so the flow operates on a roughly-unit-variance
manifold (same convention as FlowCastLoss), and supports the same per-channel
weighting + qpepre log-PSD regularizer.

In the repo's flow convention (t=0 noise -> t=1 data, linear path
z_t = (1-t) x0 + t x1, instantaneous velocity v = x1 - x0), the average
velocity u(z_r, r, t) over [r, t] satisfies the MeanFlow identity

    u(z_r, r, t) = v + (t - r) * d/dr u(z_r, r, t) ,

where d/dr is the total derivative along the trajectory
(dz_r/dr = v, dt/dr = 0). The d/dr term is evaluated with a single
forward-mode JVP and the resulting target is treated as a constant
(stop-gradient), so the network bootstraps its own average velocity from the
ordinary conditional flow-matching signal. At r == t the identity collapses
to plain I-CFM, which is what the majority of the batch trains on.
"""

from typing import Dict, Optional, Sequence

import torch
from torch import Tensor

from .flowcast_loss import _radial_log_psd


class MeanFlowLoss:
    """MeanFlow average-velocity matching loss.

    Each batch element draws two times (sorted so that r <= t). A fraction
    ``mf_ratio`` of the batch keeps r < t and trains on the JVP-bootstrapped
    MeanFlow identity; the rest is collapsed to r = t and trains on the plain
    I-CFM target v = x1 - x0. The prediction pass runs once over the full
    batch, so DDP sees a single forward per micro-batch.

    Parameters
    ----------
    sigma_data : float
        Standard deviation of the raw target residual. Inputs are divided by
        this before constructing z_r; the predicted average velocity lives in
        the same standardized space.
    t_eps : float
        Clamp for the time samples, t ~ U(t_eps, 1 - t_eps).
    mf_ratio : float
        Fraction of the batch trained on the r < t MeanFlow identity
        (paper default: 0.25; the rest reduces to I-CFM).
    adaptive_p : float
        Exponent of the adaptive loss weight w = 1 / (mse + adaptive_eps)^p
        applied per sample with stop-gradient (paper default p=1.0).
        Set to 0 to disable and recover a plain (weighted) MSE.
    adaptive_eps : float
        Stabilizer inside the adaptive weight denominator.
    channel_weights : sequence of float, optional
        Per-channel weights on the squared error (same convention as
        FlowCastLoss).
    spectral_channels : sequence of int, optional
        Target-channel indices on which to add a radial log-PSD L1
        regularizer between the implied clean prediction and the target.
        Applied only on the r == t (pure flow-matching) part of the batch,
        where the one-step extrapolation x1_pred = z_r + (1 - r) u is exact
        in expectation.
    spectral_weight : float
        Weight on the log-PSD term.
    """

    def __init__(
        self,
        sigma_data: float = 0.5,
        t_eps: float = 1e-5,
        mf_ratio: float = 0.25,
        adaptive_p: float = 1.0,
        adaptive_eps: float = 1e-3,
        channel_weights: Optional[Sequence[float]] = None,
        spectral_channels: Optional[Sequence[int]] = None,
        spectral_weight: float = 0.0,
    ):
        self.sigma_data = float(sigma_data)
        self.t_eps = float(t_eps)
        self.mf_ratio = float(mf_ratio)
        self.adaptive_p = float(adaptive_p)
        self.adaptive_eps = float(adaptive_eps)
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
        """Compute the MeanFlow loss.

        Parameters
        ----------
        student : torch.nn.Module
            MeanFlowPrecond (possibly DDP-wrapped). Must implement the
            signature ``student(x, r, t, condition=...)`` and return a tensor
            of the same shape as ``x``. The JVP target pass uses the
            unwrapped module (``student.module`` if present) under no_grad so
            DDP only ever sees the single gradient-carrying forward.
        images : Tensor
            Raw target residual r_{t+1} of shape (B, C, H, W).
        condition : Tensor
            Conditioning tensor of shape (B, C_cond, H, W).

        Returns
        -------
        dict
            ``{"loss": scalar total, "pointwise": scalar raw weighted MSE,
               "spectral": scalar log-PSD, "mf_fraction": float}``.
        """
        B = images.shape[0]
        device = images.device
        dtype = images.dtype

        # The JVP pass needs the bare module: forward-mode AD through the
        # DDP wrapper is unsupported, and the pass carries no gradients.
        raw_net = getattr(student, "module", student)

        # Standardize target residual to unit-variance working space.
        x1 = images / self.sigma_data
        x0 = torch.randn_like(x1)
        v = x1 - x0

        # Two times per sample, sorted so r <= t; collapse t to r on the
        # pure flow-matching part of the batch.
        t_pair = torch.rand(B, 2, device=device, dtype=dtype)
        t_pair = t_pair * (1.0 - 2.0 * self.t_eps) + self.t_eps
        r, t = t_pair.min(dim=1).values, t_pair.max(dim=1).values
        mf_mask = torch.rand(B, device=device) < self.mf_ratio
        t = torch.where(mf_mask, t, r)

        r_view = r.view(-1, 1, 1, 1)
        z_r = (1.0 - r_view) * x0 + r_view * x1

        # I-CFM target everywhere; replaced by the MeanFlow identity on the
        # mf_mask subset below.
        u_target = v.clone()

        if mf_mask.any():
            idx = mf_mask.nonzero(as_tuple=True)[0]
            z_m, r_m, t_m, v_m = z_r[idx], r[idx], t[idx], v[idx]
            cond_m = condition[idx]

            def _u_fn(z_in, r_in, t_in):
                return raw_net(z_in, r_in, t_in, condition=cond_m)

            # Total derivative along the trajectory: tangent (v, 1, 0) for
            # (z, r, t). Forward-mode AD works under no_grad, and the target
            # is stop-gradient anyway.
            with torch.no_grad():
                _, du_dr = torch.func.jvp(
                    _u_fn,
                    (z_m, r_m, t_m),
                    (v_m, torch.ones_like(r_m), torch.zeros_like(t_m)),
                )
            gap = (t_m - r_m).view(-1, 1, 1, 1)
            u_target[idx] = v_m + gap * du_dr

        # Single gradient-carrying forward over the full batch.
        u_pred = student(z_r, r, t, condition=condition)

        # --- Pointwise squared error, with optional per-channel beta ---
        pw = (u_pred - u_target.detach()) ** 2  # (B, C, H, W)
        if self.channel_weights is not None:
            assert len(self.channel_weights) == pw.shape[1], (
                f"channel_weights length {len(self.channel_weights)} "
                f"!= num channels {pw.shape[1]}"
            )
            beta = torch.tensor(
                self.channel_weights, device=pw.device, dtype=pw.dtype
            ).view(1, -1, 1, 1)
            pw = pw * beta

        per_sample = pw.mean(dim=(1, 2, 3))  # (B,)
        raw_mse = per_sample.mean()

        # --- Adaptive weighting (paper: w = 1/(mse + eps)^p, stop-grad) ---
        if self.adaptive_p > 0:
            w = (per_sample.detach() + self.adaptive_eps) ** (-self.adaptive_p)
            pointwise_loss = (w * per_sample).mean()
        else:
            pointwise_loss = raw_mse

        # --- Spectral regularizer on qpepre-style channels (FM part only) ---
        fm_mask = ~mf_mask
        if self.spectral_channels and self.spectral_weight > 0 and fm_mask.any():
            # On the r == t subset u_pred is the instantaneous velocity, so
            # the implied clean field is x1_pred = z_r + (1 - r) * u_pred
            # (same construction as FlowCastLoss).
            fidx = fm_mask.nonzero(as_tuple=True)[0]
            x1_pred = z_r[fidx] + (1.0 - r[fidx].view(-1, 1, 1, 1)) * u_pred[fidx]
            x1_true = x1[fidx]
            spec_terms = []
            for cidx in self.spectral_channels:
                log_ps_s = _radial_log_psd(x1_pred[:, cidx])
                log_ps_t = _radial_log_psd(x1_true[:, cidx])
                spec_terms.append((log_ps_s - log_ps_t).abs().mean())
            spectral_loss = self.spectral_weight * torch.stack(spec_terms).mean()
        else:
            spectral_loss = pointwise_loss.new_zeros(())

        total = pointwise_loss + spectral_loss
        return {
            "loss": total,
            "pointwise": raw_mse.detach(),
            "spectral": spectral_loss.detach(),
            "mf_fraction": float(mf_mask.float().mean().item()),
        }
