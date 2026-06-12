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

"""Gated-Spectral MeanFlow loss (GS-MeanFlow MVP).

Generalizes ``MeanFlowLoss`` with the two MVP pillars:

* **Pillar 1 - group-decoupled adaptive weighting.** The MeanFlow adaptive
  weight ``w = 1/(mse + eps)^p`` is computed *per channel group* rather than
  once over all channels. The qpw ablation (CLAUDE.md S6.5) shows no single
  channel weight wins all four variables -- winds prefer qpw=1.4 while t2m and
  qpepre prefer qpw=2.0 -- because one shared weight forces a compromise and a
  sample with hard qpepre down-weights its own easy wind channels through the
  shared ``w``. Splitting the smooth group (t2m, u10, v10) from the precip
  channel (qpepre) lets each be re-weighted independently, removing that
  cross-channel gradient interference.

* **Pillar 2 - occurrence (hurdle) gate.** A focal-BCE term trains the
  preconditioner's gate head to predict where qpepre is wet at t+1, decoupling
  the Bernoulli dry/wet mass from the continuous intensity the flow transports.
  This directly attacks the "always slightly wet" failure (CLAUDE.md S8): at
  inference the gate forces confidently-dry pixels to exact zero.

The average-velocity matching itself (two times per sample, JVP-bootstrapped
MeanFlow identity on the ``mf_ratio`` subset, plain I-CFM on the rest) and the
qpepre log-PSD spectral regularizer are carried over unchanged from
``MeanFlowLoss`` so GS-MeanFlow vs MeanFlow is a clean A/B on the two additions.
"""

from typing import Dict, Optional, Sequence

import torch
from torch import Tensor

from utils.flowcast_loss import _radial_log_psd


class GSMeanFlowLoss:
    """MeanFlow average-velocity matching + group decoupling + occurrence gate.

    Parameters
    ----------
    sigma_data, t_eps, mf_ratio, adaptive_p, adaptive_eps, channel_weights,
    spectral_channels, spectral_weight :
        Same meaning as ``MeanFlowLoss``.
    gate_channel_index : int, optional
        Target-channel index of qpepre. Enables both the group-decoupled
        adaptive weighting (this channel forms its own group) and the
        occurrence-gate loss. If None, the loss reduces to plain MeanFlow with a
        single global adaptive weight and no gate term.
    gate_wet_threshold : float
        Wet/dry decision boundary on the *standardized* qpepre field (i.e. the
        dataset-normalized value of the physical precip threshold, e.g. 0.1
        mm/h). A target pixel counts as wet when its standardized qpepre exceeds
        this.
    gate_weight : float
        Weight on the focal-BCE occurrence term in the total loss.
    gate_focal_gamma : float
        Focal-loss focusing exponent (0 recovers weighted BCE).
    gate_focal_alpha : float
        Positive-class (wet) weighting in [0, 1]; > 0.5 emphasizes the sparse
        wet pixels.
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
        gate_channel_index: Optional[int] = None,
        gate_wet_threshold: float = 0.0,
        gate_weight: float = 1.0,
        gate_focal_gamma: float = 2.0,
        gate_focal_alpha: float = 0.75,
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
        self.gate_channel_index = (
            None if gate_channel_index is None else int(gate_channel_index)
        )
        self.gate_wet_threshold = float(gate_wet_threshold)
        self.gate_weight = float(gate_weight)
        self.gate_focal_gamma = float(gate_focal_gamma)
        self.gate_focal_alpha = float(gate_focal_alpha)

    # ------------------------------------------------------------------
    def _adaptive(self, per_sample: Tensor) -> Tensor:
        """Per-sample adaptive weight w = 1/(mse + eps)^p (stop-grad)."""
        if self.adaptive_p > 0:
            return (per_sample.detach() + self.adaptive_eps) ** (-self.adaptive_p)
        return torch.ones_like(per_sample)

    def _gate_loss(self, logits: Tensor, wet: Tensor) -> Tensor:
        """Focal binary cross-entropy for the occurrence gate."""
        ce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, wet, reduction="none"
        )
        if self.gate_focal_gamma > 0 or self.gate_focal_alpha != 0.5:
            p = torch.sigmoid(logits)
            p_t = p * wet + (1.0 - p) * (1.0 - wet)
            mod = (1.0 - p_t).clamp_min(1e-6) ** self.gate_focal_gamma
            alpha_t = self.gate_focal_alpha * wet + (1.0 - self.gate_focal_alpha) * (
                1.0 - wet
            )
            ce = alpha_t * mod * ce
        return ce.mean()

    # ------------------------------------------------------------------
    def __call__(
        self,
        student: torch.nn.Module,
        images: Tensor,
        condition: Tensor,
        target_field: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Compute the GS-MeanFlow loss.

        Parameters
        ----------
        student : torch.nn.Module
            ``GSMeanFlowPrecond`` (possibly DDP-wrapped). The gradient-carrying
            forward is called with ``return_gate=True`` so a single backward
            exercises both the flow and the gate parameters; the JVP target pass
            uses the unwrapped module's flow-only path.
        images : Tensor
            Raw target residual r_{t+1} = M_{t+1} - mu_{t+1}, shape (B, C, H, W).
        condition : Tensor
            Conditioning bundle, shape (B, C_cond, H, W).
        target_field : Tensor, optional
            Absolute standardized target M_{t+1}, shape (B, C, H, W). Required
            for the occurrence-gate term (its qpepre channel defines the wet
            mask). If None, the gate term is skipped.
        """
        B = images.shape[0]
        device = images.device
        dtype = images.dtype
        q_idx = self.gate_channel_index

        raw_net = getattr(student, "module", student)

        # Standardize target residual to unit-variance working space.
        x1 = images / self.sigma_data
        x0 = torch.randn_like(x1)
        v = x1 - x0

        # Two times per sample, sorted so r <= t; collapse t to r on the pure
        # flow-matching part of the batch.
        t_pair = torch.rand(B, 2, device=device, dtype=dtype)
        t_pair = t_pair * (1.0 - 2.0 * self.t_eps) + self.t_eps
        r, t = t_pair.min(dim=1).values, t_pair.max(dim=1).values
        mf_mask = torch.rand(B, device=device) < self.mf_ratio
        t = torch.where(mf_mask, t, r)

        r_view = r.view(-1, 1, 1, 1)
        z_r = (1.0 - r_view) * x0 + r_view * x1

        u_target = v.clone()
        if mf_mask.any():
            idx = mf_mask.nonzero(as_tuple=True)[0]
            z_m, r_m, t_m, v_m = z_r[idx], r[idx], t[idx], v[idx]
            cond_m = condition[idx]

            def _u_fn(z_in, r_in, t_in):
                # Flow-only path (return_gate=False) so forward-mode AD never
                # touches the gate head.
                return raw_net(z_in, r_in, t_in, condition=cond_m)

            with torch.no_grad():
                _, du_dr = torch.func.jvp(
                    _u_fn,
                    (z_m, r_m, t_m),
                    (v_m, torch.ones_like(r_m), torch.zeros_like(t_m)),
                )
            gap = (t_m - r_m).view(-1, 1, 1, 1)
            u_target[idx] = v_m + gap * du_dr

        # Single gradient-carrying forward: velocity + occurrence logits.
        u_pred, gate_logits = student(z_r, r, t, condition=condition, return_gate=True)

        # --- Pointwise squared error with optional per-channel beta ---
        pw = (u_pred - u_target.detach()) ** 2  # (B, C, H, W)
        C = pw.shape[1]
        if self.channel_weights is not None:
            assert len(self.channel_weights) == C, (
                f"channel_weights length {len(self.channel_weights)} "
                f"!= num channels {C}"
            )
            beta = torch.tensor(
                self.channel_weights, device=pw.device, dtype=pw.dtype
            ).view(1, -1, 1, 1)
            pw = pw * beta

        raw_mse = pw.mean().detach()

        # --- Pillar 1: group-decoupled adaptive weighting ---
        if q_idx is not None and C > 1:
            smooth_idx = [c for c in range(C) if c != q_idx]
            mse_smooth = pw[:, smooth_idx].mean(dim=(1, 2, 3))  # (B,)
            mse_precip = pw[:, q_idx].mean(dim=(1, 2))           # (B,)
            w_s = self._adaptive(mse_smooth)
            w_p = self._adaptive(mse_precip)
            pointwise_loss = (w_s * mse_smooth + w_p * mse_precip).mean()
        else:
            per_sample = pw.mean(dim=(1, 2, 3))
            pointwise_loss = (self._adaptive(per_sample) * per_sample).mean()

        # --- Spectral regularizer on qpepre-style channels (FM part only) ---
        fm_mask = ~mf_mask
        if self.spectral_channels and self.spectral_weight > 0 and fm_mask.any():
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

        # --- Pillar 2: occurrence (hurdle) gate ---
        if q_idx is not None and target_field is not None and self.gate_weight > 0:
            wet = (
                target_field[:, q_idx : q_idx + 1] > self.gate_wet_threshold
            ).to(gate_logits.dtype)
            gate_raw = self._gate_loss(gate_logits, wet)
            gate_term = self.gate_weight * gate_raw
            wet_frac = wet.mean().detach()
            pred_wet_frac = (gate_logits > 0).float().mean().detach()
        else:
            gate_raw = pointwise_loss.new_zeros(())
            gate_term = pointwise_loss.new_zeros(())
            wet_frac = pointwise_loss.new_zeros(())
            pred_wet_frac = pointwise_loss.new_zeros(())

        total = pointwise_loss + spectral_loss + gate_term
        return {
            "loss": total,
            "pointwise": raw_mse,
            "spectral": spectral_loss.detach(),
            "gate": gate_raw.detach(),
            "wet_fraction": float(wet_frac.item()),
            "pred_wet_fraction": float(pred_wet_frac.item()),
            "mf_fraction": float(mf_mask.float().mean().item()),
        }
