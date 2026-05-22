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

"""Anchored Stochastic-Interpolant Bridge loss for the StormCast residual.

Reframes the FlowCast (Ribeiro & Pucer 2025) noise-to-residual flow as a
stochastic interpolant (Albergo & Vanden-Eijnden 2023) from the regression
mean to the truth: x_0 = mu_{t+1} + sigma_prior * eps, x_1 = M_{t+1}. The
network learns the bridge velocity field; at inference the ODE is integrated
from x_0 to x_1 directly, so mu is no longer just a conditioning channel but
the structural prior endpoint of the flow.

Schedule (default "quadratic"):
    alpha(t) = 1 - t,  beta(t) = t,  gamma(t) = sigma_max * t * (1 - t)
    x_t = alpha(t) * x_0 + beta(t) * x_1 + gamma(t) * z,   z ~ N(0, I)
    u_t = -x_0 + x_1 + sigma_max * (1 - 2t) * z
        = (x_1 - x_0) + sigma_max * (1 - 2t) * z

gamma(0) = gamma(1) = 0 so the marginals at the endpoints match the prior
and the data distributions exactly. The network is regressed against u_t;
the optimal regressor equals E[u_t | x_t], which is the drift of the
marginal probability-flow ODE (Albergo 2023, Prop. 2.6).

Optional knobs (kept on top of the bridge core so the same loss exposes
the qpepre-specific knobs that already exist in flowcast_loss.py):
    coupling           'iid' or 'ot'           OT-CFM-style minibatch coupling.
    prior_channel_std  per-channel sigma_prior heavy-tail / channel-aware prior.
    channel_weights    per-channel beta        same convention as FlowCastLoss.
    spectral_channels  qpepre, ...             radial log-PSD L1 regularizer.
"""

import math
from typing import Dict, Optional, Sequence

import torch
from torch import Tensor


def _radial_log_psd(x: Tensor, eps: float = 1e-12) -> Tensor:
    """Radially-averaged log power spectrum of a 2D field, (B, H, W) -> (B, K)."""
    B, H, W = x.shape
    X = torch.fft.rfft2(x, norm="ortho")
    P = X.real**2 + X.imag**2

    ky = torch.fft.fftfreq(H, device=x.device) * H
    kx = torch.fft.rfftfreq(W, device=x.device) * W
    kyy, kxx = torch.meshgrid(ky, kx, indexing="ij")
    k_int = torch.sqrt(kyy**2 + kxx**2).round().long()

    k_max = int(k_int.max().item()) + 1
    flat_idx = k_int.flatten()
    batch_idx = flat_idx.unsqueeze(0).expand(B, -1)

    Pk = torch.zeros(B, k_max, device=x.device, dtype=P.dtype)
    Pk.scatter_add_(1, batch_idx, P.flatten(1))

    counts = torch.zeros(k_max, device=x.device, dtype=P.dtype)
    counts.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=P.dtype))

    Pk = Pk / counts.clamp_min(1.0).unsqueeze(0)
    return torch.log(Pk.clamp_min(eps))


def _ot_permutation(x0: Tensor, x1: Tensor, downsample: int = 4) -> Tensor:
    """Solve min_pi sum_i ||x_0[pi(i)] - x_1[i]||^2 via the Hungarian algorithm.

    Computes the cost on a spatially-downsampled view to keep the B x B matrix
    cheap and robust to small-scale jitter (Tong et al. 2024 §5: OT coupling is
    robust to feature compression).

    Returns the permutation `pi` such that re-indexing x_0 by it pairs each
    x_0[pi(i)] with x_1[i] optimally.
    """
    from scipy.optimize import linear_sum_assignment

    if downsample > 1:
        x0d = torch.nn.functional.avg_pool2d(x0, downsample)
        x1d = torch.nn.functional.avg_pool2d(x1, downsample)
    else:
        x0d, x1d = x0, x1
    B = x0d.shape[0]
    a = x0d.reshape(B, -1)
    b = x1d.reshape(B, -1)
    cost = torch.cdist(a, b).pow(2).detach().cpu().numpy()
    _, col = linear_sum_assignment(cost)
    return torch.as_tensor(col, dtype=torch.long, device=x0.device)


class BridgeMatchingLoss:
    """Anchored stochastic-interpolant bridge loss: mu -> M.

    Operates in *raw* (physical / log1p-encoded) space, not the standardized
    space FlowCastLoss uses. The bridge does not need sigma_data scaling
    because both endpoints share the same physical scale; the regression mean
    already sits in the right metric.

    Parameters
    ----------
    sigma_prior : float
        Standard deviation of the Gaussian noise added to mu to form the prior
        endpoint x_0 = mu + sigma_prior * eps. Setting sigma_prior=0 collapses
        x_0 to mu deterministically; the bridge then only injects stochasticity
        via gamma(t)*z, which still provides generative capacity if sigma_max>0.
    sigma_max : float
        Peak magnitude of the bridge's interior noise term gamma(t)*z. Set to
        0 for a fully deterministic linear interpolant (equivalent to regressing
        the residual at every t -- useful as an ablation).
    schedule : {"quadratic", "trig"}
        Shape of gamma(t). Both vanish at t=0 and t=1 so the marginals match.
        "quadratic" gives gamma = sigma_max * t * (1 - t) (smooth, easy to
        verify by hand). "trig" gives gamma = sigma_max * sin(pi t) (peaks
        higher in the interior; similar to I^2SB's VP-style schedule).
    coupling : {"iid", "ot"}
        Prior-data coupling. 'iid' pairs the noise eps and target M randomly
        within a minibatch; 'ot' solves a Hungarian assignment so the bridge
        starts at the noise sample geometrically closest to its target. The
        latter is OT-CFM (Tong et al. 2024) applied to the bridge prior.
    t_eps : float
        Clamp t into [t_eps, 1 - t_eps] to avoid touching the endpoint
        marginals during training.
    channel_weights : sequence of float, optional
        Per-channel weight on the MSE term. Same convention as FlowCastLoss.
    spectral_channels : sequence of int, optional
        Target-channel indices on which to add a radial log-PSD L1 regularizer
        between an extrapolated-clean field and the truth. (Approximate in the
        stochastic-bridge case; exact in the deterministic limit.)
    spectral_weight : float
        Weight on the log-PSD term.
    prior_channel_std : sequence of float, optional
        Per-channel multiplier on the prior noise std. Lets you push the
        qpepre channel toward a heavier-tailed prior than the wind/temperature
        channels (Direction 2 of the improvement plan).
    """

    def __init__(
        self,
        sigma_prior: float = 0.05,
        sigma_max: float = 0.5,
        schedule: str = "quadratic",
        coupling: str = "iid",
        t_eps: float = 1e-5,
        channel_weights: Optional[Sequence[float]] = None,
        spectral_channels: Optional[Sequence[int]] = None,
        spectral_weight: float = 0.0,
        prior_channel_std: Optional[Sequence[float]] = None,
    ):
        if schedule not in ("quadratic", "trig"):
            raise ValueError(f"Unknown schedule: {schedule}")
        if coupling not in ("iid", "ot"):
            raise ValueError(f"Unknown coupling: {coupling}")
        self.sigma_prior = float(sigma_prior)
        self.sigma_max = float(sigma_max)
        self.schedule = schedule
        self.coupling = coupling
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
        self.prior_channel_std = (
            None
            if prior_channel_std is None
            else tuple(float(s) for s in prior_channel_std)
        )

    def _schedule_fn(self, t: Tensor):
        """Return (alpha, beta, gamma, d_alpha, d_beta, d_gamma) at t."""
        if self.schedule == "quadratic":
            alpha = 1.0 - t
            beta = t
            gamma = self.sigma_max * t * (1.0 - t)
            d_alpha = torch.full_like(t, -1.0)
            d_beta = torch.full_like(t, 1.0)
            d_gamma = self.sigma_max * (1.0 - 2.0 * t)
        else:  # "trig"
            pi = math.pi
            alpha = 1.0 - t
            beta = t
            gamma = self.sigma_max * torch.sin(pi * t)
            d_alpha = torch.full_like(t, -1.0)
            d_beta = torch.full_like(t, 1.0)
            d_gamma = self.sigma_max * pi * torch.cos(pi * t)
        return alpha, beta, gamma, d_alpha, d_beta, d_gamma

    def _prior_noise(self, shape, device, dtype) -> Tensor:
        eps = torch.randn(shape, device=device, dtype=dtype)
        if self.prior_channel_std is not None:
            assert len(self.prior_channel_std) == shape[1], (
                f"prior_channel_std length {len(self.prior_channel_std)} "
                f"!= num channels {shape[1]}"
            )
            s = torch.tensor(
                self.prior_channel_std, device=device, dtype=dtype
            ).view(1, -1, 1, 1)
            eps = eps * s
        return eps

    def __call__(
        self,
        student: torch.nn.Module,
        images: Tensor,
        mu: Tensor,
        condition: Tensor,
    ) -> Dict[str, Tensor]:
        """Compute the bridge-matching regression loss.

        Parameters
        ----------
        student : torch.nn.Module
            Velocity network (FlowCastPrecond / DDP wrapper). Called as
            ``student(x_t, t, condition=...)`` and must return a tensor with
            the same shape as ``x_t``.
        images : Tensor
            Ground-truth target ``M_{t+1}`` in raw physical (log1p-encoded
            for qpepre) units, shape (B, C, H, W). NOT the residual.
        mu : Tensor
            Frozen regression mean ``mu_{t+1} = F_theta(M_t, S_t, I)`` in the
            same units as ``images``. Shape (B, C, H, W).
        condition : Tensor
            Conditioning tensor of shape (B, C_cond, H, W) passed to the
            network. Typically still contains mu as one channel; that is fine
            and complementary (mu is used both as the prior endpoint and as
            input conditioning information).
        """
        B = images.shape[0]
        device = images.device
        dtype = images.dtype

        # --- Prior endpoint: x_0 = mu + sigma_prior * eps ---
        eps0 = self._prior_noise(images.shape, device, dtype)
        x0 = mu + self.sigma_prior * eps0
        x1 = images

        # --- Optional OT-CFM minibatch coupling on the prior noise. ---
        # We permute the eps0 *offsets* relative to mu, not mu itself, because
        # the per-sample pairing of (mu_i, M_i) is fixed by the regression
        # network and must be preserved. The Hungarian solve operates on the
        # full prior sample x_0 vs the target x_1 in spatially-downsampled
        # space, then we re-build x_0 with the permuted noise.
        if self.coupling == "ot" and B > 1:
            perm = _ot_permutation(x0, x1)
            eps0 = eps0[perm]
            x0 = mu + self.sigma_prior * eps0

        # --- Uniform time in [t_eps, 1 - t_eps] ---
        t = torch.rand(B, device=device, dtype=dtype)
        t = t * (1.0 - 2.0 * self.t_eps) + self.t_eps
        t_view = t.view(-1, 1, 1, 1)

        alpha, beta, gamma, d_alpha, d_beta, d_gamma = self._schedule_fn(t_view)

        # --- Stochastic interpolant ---
        if self.sigma_max > 0:
            z = torch.randn_like(x1)
            xt = alpha * x0 + beta * x1 + gamma * z
            u_target = d_alpha * x0 + d_beta * x1 + d_gamma * z
        else:
            xt = alpha * x0 + beta * x1
            u_target = d_alpha * x0 + d_beta * x1

        # --- Predicted velocity ---
        v_pred = student(xt, t, condition=condition)

        # --- Pointwise MSE with optional per-channel weights ---
        pw = (v_pred - u_target) ** 2
        if self.channel_weights is not None:
            assert len(self.channel_weights) == pw.shape[1], (
                f"channel_weights length {len(self.channel_weights)} "
                f"!= num channels {pw.shape[1]}"
            )
            beta_w = torch.tensor(
                self.channel_weights, device=pw.device, dtype=pw.dtype
            ).view(1, -1, 1, 1)
            pw = pw * beta_w
        pointwise_loss = pw.mean()

        # --- Spectral regularizer on qpepre-style channels ---
        # First-order extrapolation x1_pred = xt + (1 - t) * v_pred is exact
        # in the deterministic (sigma_max=0) limit and a biased-but-usable
        # surrogate otherwise -- the bias is the z-dependent term
        # [gamma(t) + (1-t) d_gamma(t)] z, which has zero mean per pixel and
        # acts as a stochastic regularizer on the log-PSD term.
        if self.spectral_channels and self.spectral_weight > 0:
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
            "sigma_prior": self.sigma_prior,
            "sigma_max": self.sigma_max,
        }
