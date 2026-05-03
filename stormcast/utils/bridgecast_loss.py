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

"""BridgeCast composite loss (new_method_plan.md §2.8).

Implements the five-term loss

    L = lambda_v * L_v
      + lambda_m * L_mask
      + lambda_e * L_ES
      + lambda_s * L_spec
      + lambda_d * L_div

where:

* ``L_v``: channel-weighted velocity-matching MSE on the mean-anchored
  stochastic interpolant (plan §2.5), with the asinh-compressed path on the
  qpepre channel (§2.4) and Lipman-style antithetic-pair variance reduction.
* ``L_mask``: focal BCE between the mask head and the ground-truth wet/dry
  indicator at a fixed precipitation threshold (§2.4).
* ``L_ES``: K-sample energy score evaluated via 1-step Euler from the bridge
  for the full field (qpepre at pixel-level, others on a 4×4 average pool)
  per §2.6. Uses the *fair* (K(K-1) corrected) estimator — strictly proper.
* ``L_spec``: radial log-PSD L1 on selected channels (re-uses
  ``physicsnemo.experimental.metrics.diffusion.consistency_loss._radial_log_psd``).
* ``L_div``: weak 2-D divergence penalty on the (u10, v10) prediction (§2.7).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from physicsnemo.experimental.metrics.diffusion.consistency_loss import (
    _radial_log_psd,
)

from .qpepre_transform import (
    asinh_compress,
    asinh_decompress,
    focal_bce_with_logits,
    _replace_channel,
)


def _avg_pool(x: Tensor, k: int = 4) -> Tensor:
    """Channel-wise 2D average pool with kernel k (for pooled energy score)."""
    if k <= 1:
        return x
    return torch.nn.functional.avg_pool2d(x, kernel_size=k, stride=k)


def _energy_score_fair(samples: Tensor, target: Tensor) -> Tensor:
    """Fair empirical energy score (Gneiting & Raftery 2007).

    Parameters
    ----------
    samples : Tensor, shape (K, B, C, H, W)
        Ensemble of K predictions per batch element.
    target : Tensor, shape (B, C, H, W)

    Returns
    -------
    Scalar tensor — mean over batch of the strictly-proper energy score.
    """
    K = samples.shape[0]
    if K < 2:
        # ES degenerates to MAE (a strictly proper score for K=1, but with
        # zero spread term); just return the L2 distance to truth.
        return (samples[0] - target).pow(2).mean()

    # Term 1: E[ ||X̂ - X|| ] over members. ``+1e-8`` inside the sqrt avoids
    # the inf-gradient of ``sqrt(0)`` at the (rare) coincident sample case.
    diff_to_truth = samples - target.unsqueeze(0)  # (K, B, C, H, W)
    norm_truth = (diff_to_truth.flatten(2).pow(2).sum(dim=-1) + 1e-8).sqrt()  # (K, B)
    term1 = norm_truth.mean(dim=0)  # (B,)

    # Term 2: 1/(2K(K-1)) * sum_{k != j} ||X̂_k - X̂_j||. The diagonal is
    # guaranteed 0 in value but ``sqrt(0)`` has +inf gradient — adding 1e-8
    # under the sqrt makes both forward and backward NaN-safe at negligible
    # bias (sqrt(1e-8) = 1e-4 per diagonal entry, divided by 2K(K-1)).
    pairwise = samples.unsqueeze(0) - samples.unsqueeze(1)  # (K, K, B, C, H, W)
    pairwise_norm = (pairwise.flatten(3).pow(2).sum(dim=-1) + 1e-8).sqrt()
    term2 = pairwise_norm.sum(dim=(0, 1)) / (2.0 * K * (K - 1))

    return (term1 - term2).mean()


def _divergence_penalty(u: Tensor, v: Tensor) -> Tensor:
    """L2 norm of the 2-D divergence ∂_x u + ∂_y v (centred finite diff)."""
    du_dx = u[:, :, :, 2:] - u[:, :, :, :-2]
    dv_dy = v[:, :, 2:, :] - v[:, :, :-2, :]
    # Crop to common interior so the two terms align.
    du_dx = du_dx[:, :, 1:-1, :]
    dv_dy = dv_dy[:, :, :, 1:-1]
    div = 0.5 * (du_dx + dv_dy)
    return div.pow(2).mean()


class BridgeCastLoss:
    """Composite BridgeCast loss with antithetic-pair velocity matching.

    The interpolant lives in *standardized residual* space:
        R_norm = (X_t - M_t) / sigma_data
    on the linear channels, and on the qpepre channel:
        R_norm_qpepre = asinh(R_qpepre / (sigma_data * kappa)) when qpepre_asinh
    so all four channels enter the bridge in roughly unit-variance coordinates.

    The bridge with anchor jitter ``sigma_a`` and bridge noise ``sigma_b`` is

        z_t = (1 - t) sigma_a eps + t R_norm + sigma_b sqrt(t (1-t)) eps

    using a single shared correlated noise tensor ``eps`` so the path is a
    Brownian bridge in the climatology-correlated covariance.

    Parameters
    ----------
    sigma_data : float
        Per-channel std for de-/standardization.
    sigma_b : float
        Bridge noise magnitude.
    sigma_a : float
        Anchor jitter (default 0 = deterministic anchor).
    t_eps : float
        Endpoint clip (gamma'(t) diverges at the corners).
    channel_weights : sequence of float, optional
        Per-channel β (length = num target channels).
    qpepre_idx : int
        Channel index of qpepre. ``< 0`` disables the qpepre-specific paths.
    qpepre_kappa : float
        asinh knee scale on the standardized qpepre residual.
    qpepre_asinh : bool
        Toggle the asinh-compressed bridge for qpepre.
    rain_threshold : float
        Wet/dry threshold (in physical mm/h) for the mask head.
    enable_mask : bool
        If False, ``L_mask`` is forced to zero (ablation #5).
    spectral_channels : sequence of int, optional
        Channel indices on which to add the radial log-PSD L1 penalty.
    spectral_weight : float
        Weight on ``L_spec``.
    divergence_channels : tuple of (int, int), optional
        ``(u_idx, v_idx)`` channel indices for the divergence penalty;
        ``None`` disables it.
    es_K : int
        Number of ensemble members for the energy-score branch (0 disables).
    es_pool : int
        Spatial pooling factor for non-qpepre channels in the ES term.
    lambda_v, lambda_m, lambda_e, lambda_s, lambda_d : float
        Term weights.
    """

    def __init__(
        self,
        sigma_data: float = 0.5,
        sigma_b: float = 0.15,
        sigma_a: float = 0.0,
        t_eps: float = 1e-3,
        channel_weights: Optional[Sequence[float]] = None,
        qpepre_idx: int = -1,
        qpepre_kappa: float = 1.0,
        qpepre_asinh: bool = True,
        rain_threshold: float = 0.1,
        enable_mask: bool = True,
        spectral_channels: Optional[Sequence[int]] = None,
        spectral_weight: float = 0.0,
        divergence_channels: Optional[Tuple[int, int]] = None,
        es_K: int = 0,
        es_pool: int = 4,
        lambda_v: float = 1.0,
        lambda_m: float = 0.1,
        lambda_e: float = 0.5,
        lambda_s: float = 0.1,
        lambda_d: float = 1e-3,
    ):
        self.sigma_data = float(sigma_data)
        self.sigma_b = float(sigma_b)
        self.sigma_a = float(sigma_a)
        self.t_eps = float(t_eps)
        self.channel_weights = (
            None
            if channel_weights is None
            else tuple(float(w) for w in channel_weights)
        )
        self.qpepre_idx = int(qpepre_idx)
        self.qpepre_kappa = float(qpepre_kappa)
        self.qpepre_asinh = bool(qpepre_asinh) and self.qpepre_idx >= 0
        self.rain_threshold = float(rain_threshold)
        self.enable_mask = bool(enable_mask) and self.qpepre_idx >= 0
        self.spectral_channels = (
            tuple(int(i) for i in spectral_channels) if spectral_channels else ()
        )
        self.spectral_weight = float(spectral_weight)
        self.divergence_channels = divergence_channels
        self.es_K = int(es_K)
        self.es_pool = int(es_pool)
        self.lambda_v = float(lambda_v)
        self.lambda_m = float(lambda_m)
        self.lambda_e = float(lambda_e)
        self.lambda_s = float(lambda_s)
        self.lambda_d = float(lambda_d)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_bridge_target(self, residual_raw: Tensor) -> Tensor:
        """Standardize the raw residual and apply the asinh transform on qpepre.

        Returns a tensor in the bridge's working coordinate system: linear
        unit-variance for non-qpepre channels, asinh-compressed unit-variance
        for qpepre when ``qpepre_asinh`` is on.
        """
        z1 = residual_raw / self.sigma_data
        if self.qpepre_asinh:
            new_q = asinh_compress(z1[:, self.qpepre_idx], kappa=self.qpepre_kappa)
            z1 = _replace_channel(z1, self.qpepre_idx, new_q)
        return z1

    def _from_bridge_state(self, z: Tensor) -> Tensor:
        """Inverse of :meth:`_to_bridge_target` — back to raw residual units."""
        if self.qpepre_asinh:
            new_q = asinh_decompress(z[:, self.qpepre_idx], kappa=self.qpepre_kappa)
            z = _replace_channel(z, self.qpepre_idx, new_q)
        return z * self.sigma_data

    # ------------------------------------------------------------------
    # Main __call__
    # ------------------------------------------------------------------

    def __call__(
        self,
        student: torch.nn.Module,
        residual: Tensor,
        regression_mean: Tensor,
        condition: Tensor,
        clim_noise_sampler,
        target_qpepre_raw: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Compute the BridgeCast composite loss.

        Parameters
        ----------
        student : nn.Module
            BridgeCastPrecond (or DDP wrapper). Must implement
            ``forward(x, t, condition=...) -> (v, m_logits)``.
        residual : Tensor, (B, C, H, W)
            Raw target residual R_t = X_t - M_t.
        regression_mean : Tensor, (B, C, H, W)
            M_t — used by the energy-score branch to reconstruct full states.
        condition : Tensor, (B, C_cond, H, W)
            Conditioning bundle.
        clim_noise_sampler : ClimNoiseSampler
            Climatology-correlated noise sampler (with or without colouring).
        target_qpepre_raw : Tensor, (B, H, W), optional
            Ground-truth qpepre in physical mm/h, used to define the wet/dry
            mask target. If ``None``, derived from ``residual + regression_mean``.

        Returns
        -------
        dict with keys ``loss``, ``pointwise``, ``mask``, ``energy``,
        ``spectral``, ``divergence`` (each a scalar tensor); plus
        ``loss_pointwise`` etc. detached scalars for logging.
        """
        B, C, H, W = residual.shape
        device = residual.device
        dtype = residual.dtype

        # ----- Build the interpolant -----
        z1 = self._to_bridge_target(residual)  # (B, C, H, W) in bridge-space

        # Sample t; antithetic pairs reduce variance of the velocity term so
        # we duplicate the batch with eps and -eps using the same t and z1.
        t = torch.rand(B, device=device, dtype=dtype)
        t = t * (1.0 - 2.0 * self.t_eps) + self.t_eps
        t_view = t.view(-1, 1, 1, 1)

        eps = clim_noise_sampler.sample_like(z1).to(dtype=dtype)

        # gamma(t) and gamma'(t) for the bridge noise schedule.
        gamma = self.sigma_b * torch.sqrt(t_view * (1.0 - t_view))
        # (1 - 2t) / (2 sqrt(t(1-t))), clamped numerically stable.
        gamma_prime = self.sigma_b * (1.0 - 2.0 * t_view) / (
            2.0 * torch.sqrt(t_view * (1.0 - t_view))
        )

        # x_0 = sigma_a * eps_a, but we're using a single eps and treat the
        # anchor jitter as a separate eps draw.  When sigma_a == 0 (default)
        # this whole branch collapses, and the interpolant reduces to
        # z_t = t * z1 + gamma(t) * eps.
        if self.sigma_a > 0.0:
            eps_anchor = clim_noise_sampler.sample_like(z1).to(dtype=dtype)
            z_anchor = self.sigma_a * eps_anchor
        else:
            eps_anchor = torch.zeros_like(eps)
            z_anchor = torch.zeros_like(z1)

        z_t = (1.0 - t_view) * z_anchor + t_view * z1 + gamma * eps

        # Bridge target velocity (Albergo et al. 2023, Theorem 2.6 + the
        # explicit gamma' term):
        #   dz_t/dt = (z1 - z_anchor) + gamma'(t) * eps
        u_target = (z1 - z_anchor) + gamma_prime * eps

        # ----- Antithetic pair: same z1, t but eps -> -eps -----
        z_t_neg = (1.0 - t_view) * (-z_anchor) + t_view * z1 + gamma * (-eps)
        u_target_neg = (z1 - (-z_anchor)) + gamma_prime * (-eps)

        # Stack the two antithetic halves into a single batch.
        x_in = torch.cat([z_t, z_t_neg], dim=0)
        cond_in = torch.cat([condition, condition], dim=0)
        t_in = torch.cat([t, t], dim=0)
        u_tgt = torch.cat([u_target, u_target_neg], dim=0)

        v_pred, m_logits = student(x_in, t_in, condition=cond_in)

        # ----- L_v: channel-weighted velocity MSE -----
        pw = (v_pred - u_tgt) ** 2
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

        # ----- L_mask: focal BCE wet/dry on the qpepre channel -----
        if self.enable_mask:
            if target_qpepre_raw is None:
                full_target = residual + regression_mean
                qpepre_phys = full_target[:, self.qpepre_idx]
            else:
                qpepre_phys = target_qpepre_raw
            wet = (qpepre_phys > self.rain_threshold).to(dtype=v_pred.dtype)
            wet_pair = torch.cat([wet, wet], dim=0)  # match antithetic batch
            mask_loss = focal_bce_with_logits(
                m_logits[:, 0], wet_pair, gamma=2.0, alpha=0.5
            )
        else:
            mask_loss = pw.new_zeros(())

        # ----- L_spec: radial log-PSD on selected channels -----
        if self.spectral_channels and self.spectral_weight > 0:
            # Predicted clean residual implied by the velocity at this t:
            #   z1_hat = z_t + (1 - t) * v_pred
            # (Linear-interpolant tangent extrapolation; we only use the
            # antithetic +eps half to avoid double-counting.)
            v_pos = v_pred[: B]
            z1_hat = z_t + (1.0 - t_view) * v_pos
            spec_terms = []
            for cidx in self.spectral_channels:
                log_ps_s = _radial_log_psd(z1_hat[:, cidx])
                log_ps_t = _radial_log_psd(z1[:, cidx])
                spec_terms.append((log_ps_s - log_ps_t).abs().mean())
            spectral_loss = self.spectral_weight * torch.stack(spec_terms).mean()
        else:
            spectral_loss = pw.new_zeros(())

        # ----- L_ES: K-sample energy score from 1-step Euler -----
        if self.es_K > 1 and self.lambda_e > 0:
            es_members = []
            for _k in range(self.es_K):
                eps_k = clim_noise_sampler.sample_like(z1).to(dtype=dtype)
                # 1-step Euler from t=0:
                #   z̃_1 = z_anchor + v_theta(z_anchor + gamma(0+) eps_k, 0)
                # We use a small t_mid = 0.5 inside the velocity to keep the
                # gamma' term finite — Algorithm 1 of plan §3 uses midpoint
                # times for the same reason.
                t_es = torch.full(
                    [B], 0.5, device=device, dtype=dtype
                )
                t_es_view = t_es.view(-1, 1, 1, 1)
                gamma_mid = self.sigma_b * torch.sqrt(
                    t_es_view * (1.0 - t_es_view)
                )
                if self.sigma_a > 0:
                    eps_a_k = clim_noise_sampler.sample_like(z1).to(dtype=dtype)
                    z_anchor_k = self.sigma_a * eps_a_k
                else:
                    z_anchor_k = torch.zeros_like(z1)
                x_in_es = z_anchor_k + gamma_mid * eps_k
                v_es, _ = student(x_in_es, t_es, condition=condition)
                # Step from 0 to 1 using v at the midpoint as a 1st-order
                # approximation of the integral.
                z1_pred = z_anchor_k + v_es * 1.0
                # Decompress qpepre and de-standardize → back to raw R units.
                R_pred_raw = self._from_bridge_state(z1_pred)
                X_pred_raw = R_pred_raw + regression_mean
                # Compute pooled-or-not field for ES.
                if self.es_pool > 1:
                    field = X_pred_raw.clone()
                    pooled = _avg_pool(field, k=self.es_pool)
                    if self.qpepre_idx >= 0:
                        # Keep qpepre at full resolution — pool only other
                        # channels by replacing their pooled tensor with
                        # something of size pooled.shape, then concat.
                        # Simpler: build two ES terms (qpepre at full, others
                        # pooled) and average.
                        es_members.append(("split", X_pred_raw, pooled))
                    else:
                        es_members.append(("pool", pooled, None))
                else:
                    es_members.append(("full", X_pred_raw, None))

            mode = es_members[0][0]
            target_full = residual + regression_mean
            if mode == "full":
                samples = torch.stack([m[1] for m in es_members], dim=0)
                energy_loss = _energy_score_fair(samples, target_full)
            elif mode == "pool":
                samples_pooled = torch.stack([m[1] for m in es_members], dim=0)
                target_pool = _avg_pool(target_full, k=self.es_pool)
                energy_loss = _energy_score_fair(samples_pooled, target_pool)
            else:  # "split": qpepre at full resolution, others on pooled grid
                K_es = len(es_members)
                samples_qpepre = torch.stack(
                    [
                        es_members[k][1][:, self.qpepre_idx : self.qpepre_idx + 1]
                        for k in range(K_es)
                    ],
                    dim=0,
                )
                target_qpepre = target_full[
                    :, self.qpepre_idx : self.qpepre_idx + 1
                ]
                es_qpepre = _energy_score_fair(samples_qpepre, target_qpepre)

                non_q = [i for i in range(C) if i != self.qpepre_idx]
                samples_others = torch.stack(
                    [
                        _avg_pool(es_members[k][1][:, non_q], k=self.es_pool)
                        for k in range(K_es)
                    ],
                    dim=0,
                )
                target_others = _avg_pool(target_full[:, non_q], k=self.es_pool)
                es_others = _energy_score_fair(samples_others, target_others)

                energy_loss = 0.5 * (es_qpepre + es_others)
        else:
            energy_loss = pw.new_zeros(())

        # ----- L_div: divergence penalty on (u10, v10) -----
        if self.divergence_channels is not None and self.lambda_d > 0:
            u_idx, v_idx = self.divergence_channels
            # Use the predicted clean state from the antithetic positive
            # half.  Decompressed back to raw units.
            v_pos = v_pred[: B]
            z1_hat = z_t + (1.0 - t_view) * v_pos
            R_pred_raw = self._from_bridge_state(z1_hat)
            X_pred_raw = R_pred_raw + regression_mean
            div_loss = _divergence_penalty(
                X_pred_raw[:, u_idx : u_idx + 1],
                X_pred_raw[:, v_idx : v_idx + 1],
            )
        else:
            div_loss = pw.new_zeros(())

        total = (
            self.lambda_v * pointwise_loss
            + self.lambda_m * mask_loss
            + self.lambda_e * energy_loss
            + spectral_loss  # already pre-multiplied by spectral_weight
            + self.lambda_d * div_loss
        )

        return {
            "loss": total,
            "pointwise": pointwise_loss.detach(),
            "mask": mask_loss.detach(),
            "energy": energy_loss.detach(),
            "spectral": spectral_loss.detach(),
            "divergence": div_loss.detach(),
        }
