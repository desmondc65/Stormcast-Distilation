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

"""Consistency Distillation loss function (Song et al., 2023)."""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor


def _radial_log_psd(x: Tensor, eps: float = 1e-12) -> Tensor:
    """Radially-averaged log power spectrum of a 2D field.

    Parameters
    ----------
    x : Tensor
        Input of shape (B, H, W).
    eps : float
        Floor for the log.

    Returns
    -------
    Tensor
        Log power spectrum of shape (B, K), where K is the number of radial bins.
    """
    B, H, W = x.shape
    X = torch.fft.rfft2(x, norm="ortho")
    P = X.real**2 + X.imag**2  # (B, H, W//2+1)

    ky = torch.fft.fftfreq(H, device=x.device) * H
    kx = torch.fft.rfftfreq(W, device=x.device) * W
    kyy, kxx = torch.meshgrid(ky, kx, indexing="ij")
    k_int = torch.sqrt(kyy**2 + kxx**2).round().long()  # (H, W//2+1)

    k_max = int(k_int.max().item()) + 1
    flat_idx = k_int.flatten()
    batch_idx = flat_idx.unsqueeze(0).expand(B, -1)

    Pk = torch.zeros(B, k_max, device=x.device, dtype=P.dtype)
    Pk.scatter_add_(1, batch_idx, P.flatten(1))

    counts = torch.zeros(k_max, device=x.device, dtype=P.dtype)
    counts.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=P.dtype))

    Pk = Pk / counts.clamp_min(1.0).unsqueeze(0)
    return torch.log(Pk.clamp_min(eps))


class ConsistencyDistillationLoss:
    """Loss function for Consistency Distillation.

    Distills a pre-trained EDM diffusion teacher into a consistency model
    that can generate samples in 1 or 2 steps. The loss enforces that the
    student maps any point on the same PF-ODE trajectory to the same
    clean output, using the teacher to provide one-step ODE guidance.

    Per the CD spec (stormcast_distillation.md §2), the distance supports:
    - Pseudo-Huber (or MSE) on all target channels with per-channel weights β_k.
    - A radial log-PSD regularizer on user-selected channels (e.g. qpepre)
      to preserve spatial structure of sparse, heavy-tailed fields.
    - Per-step weighting λ(σ_n) = 1 / (σ_{n+1} - σ_n) on the Karras ρ=7 grid.
    - Heun teacher step (Φ), matching the teacher's inference sampler.

    Parameters
    ----------
    sigma_min : float
        Minimum noise level (epsilon for the boundary condition).
    sigma_max : float
        Maximum noise level.
    sigma_data : float
        Expected standard deviation of the training data.
    rho : float
        Exponent for the Karras noise schedule.
    N_0 : int
        Initial number of discretization steps.
    N_total : int
        Final number of discretization steps.
    total_train_steps : int
        Total number of training steps (for the N(k) schedule).
    huber_c : float or None
        Pseudo-Huber loss parameter. If None or 0, uses MSE.
    channel_weights : sequence of float or None
        Per-channel β_k weights (length = target_channels). None → uniform.
    spectral_channels : sequence of int or None
        Indices of target channels on which to add the radial log-PSD L1
        penalty. None or empty disables the spectral term.
    spectral_weight : float
        α_spec: weight of the log-PSD term.

    Note
    ----
    Reference: Song, Y., Dhariwal, P., Chen, M. and Sutskever, I., 2023.
    Consistency Models. ICML 2023.
    """

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 0.5,
        rho: float = 7.0,
        N_0: int = 2,
        N_total: int = 150,
        total_train_steps: int = 400000,
        huber_c: Optional[float] = None,
        channel_weights: Optional[Sequence[float]] = None,
        spectral_channels: Optional[Sequence[int]] = None,
        spectral_weight: float = 0.0,
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.N_0 = N_0
        self.N_total = N_total
        self.total_train_steps = total_train_steps
        self.huber_c = huber_c
        self.channel_weights = (
            None if channel_weights is None else tuple(float(w) for w in channel_weights)
        )
        self.spectral_channels = (
            tuple(int(i) for i in spectral_channels) if spectral_channels else ()
        )
        self.spectral_weight = float(spectral_weight)

    def get_num_steps(self, current_step: int) -> int:
        """Compute the adaptive number of discretization steps N(k).

        Uses a sqrt schedule to increase from N_0 to N_total over training.
        """
        k = min(current_step, self.total_train_steps)
        K = self.total_train_steps
        N = math.ceil(
            math.sqrt(
                (k / K) * ((self.N_total + 1) ** 2 - self.N_0**2) + self.N_0**2
            )
            - 1
        ) + 1
        return max(N, self.N_0)

    def get_discretization(self, N: int, device: torch.device) -> Tensor:
        """Compute the Karras noise schedule for N steps.

        Returns a descending sequence: t[0] = sigma_max, t[N] = sigma_min.
        """
        indices = torch.arange(N + 1, dtype=torch.float64, device=device)
        t = (
            self.sigma_max ** (1 / self.rho)
            + indices
            / N
            * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
        ) ** self.rho
        return t

    @torch.no_grad()
    def heun_step(
        self,
        teacher: torch.nn.Module,
        x: Tensor,
        t_cur: Tensor,
        t_next: Tensor,
        condition: Tensor,
    ) -> Tensor:
        """Take a single Heun (2nd-order) ODE step using the teacher model.

        Solves the EDM probability flow ODE dx/dt = (x - D(x,t)) / t with the
        Heun correction: an Euler predictor followed by a trapezoidal corrector
        that re-evaluates the drift at t_next. Matches the teacher's inference
        sampler and the Φ operator defined in the CD spec.
        """
        denoised = teacher(x, t_cur.flatten(), condition=condition)
        d_cur = (x - denoised) / t_cur
        x_euler = x + (t_next - t_cur) * d_cur

        denoised_next = teacher(x_euler, t_next.flatten(), condition=condition)
        d_next = (x_euler - denoised_next) / t_next
        x_next = x + (t_next - t_cur) * 0.5 * (d_cur + d_next)
        return x_next

    def _pointwise_distance(self, a: Tensor, b: Tensor) -> Tensor:
        if self.huber_c is not None and self.huber_c > 0:
            diff = a - b
            return torch.sqrt(diff**2 + self.huber_c**2) - self.huber_c
        return (a - b) ** 2

    def __call__(
        self,
        student: torch.nn.Module,
        teacher: torch.nn.Module,
        ema_student: torch.nn.Module,
        images: Tensor,
        condition: Tensor,
        current_step: int,
    ) -> Dict[str, Tensor]:
        """Compute the Consistency Distillation loss.

        Parameters
        ----------
        student : torch.nn.Module
            Online student consistency model (DDP-wrapped, trainable).
        teacher : torch.nn.Module
            Frozen EDM teacher model.
        ema_student : torch.nn.Module
            EMA target consistency model (no grad).
        images : Tensor
            Clean target images of shape (B, C, H, W).
        condition : Tensor
            Conditioning tensor of shape (B, C_cond, H, W).
        current_step : int
            Current training step (for the adaptive schedule).

        Returns
        -------
        dict
            {"loss": scalar total loss (backprop this),
             "pointwise": scalar weighted Huber/MSE component,
             "spectral": scalar log-PSD component (0 if disabled),
             "N": current N(k),
             "t_n1": noisier σ per-sample, "t_n": less-noisy σ per-sample}
        """
        N = self.get_num_steps(current_step)
        t_schedule = self.get_discretization(N, device=images.device)

        batch_size = images.shape[0]
        n = torch.randint(1, N + 1, (batch_size,), device=images.device)

        t_n1 = t_schedule[n - 1].to(torch.float32).view(-1, 1, 1, 1)  # noisier
        t_n = t_schedule[n].to(torch.float32).view(-1, 1, 1, 1)  # less noisy

        noise = torch.randn_like(images)
        x_noisy = images + t_n1 * noise

        with torch.no_grad():
            x_hat = self.heun_step(teacher, x_noisy, t_n1, t_n, condition)

        student_out = student(x_noisy, t_n1.flatten(), condition=condition)

        with torch.no_grad():
            target_out = ema_student(x_hat, t_n.flatten(), condition=condition)

        # --- Pointwise distance (Huber or MSE), per channel ---
        pw = self._pointwise_distance(student_out, target_out)  # (B, C, H, W)

        if self.channel_weights is not None:
            assert len(self.channel_weights) == pw.shape[1], (
                f"channel_weights length {len(self.channel_weights)} "
                f"!= num channels {pw.shape[1]}"
            )
            beta = torch.tensor(
                self.channel_weights, device=pw.device, dtype=pw.dtype
            ).view(1, -1, 1, 1)
            pw = pw * beta

        # Per-step weight λ(σ_n) = 1 / (σ_{n+1} - σ_n) on Karras ρ=7 grid.
        # Note: this spec-compliant weighting (and the mean reduction below)
        # changes the gradient scale vs. the previous 1/N + sum/C convention,
        # so LR / huber_c / clip_grad_norm may need to be re-tuned.
        lam = 1.0 / (t_n1 - t_n).clamp_min(1e-8)
        pointwise_loss = (pw * lam).mean()

        # --- Spectral (log-PSD) term on selected channels ---
        if self.spectral_channels and self.spectral_weight > 0:
            spec_terms = []
            for cidx in self.spectral_channels:
                s_field = student_out[:, cidx]  # (B, H, W)
                t_field = target_out[:, cidx]
                log_ps_s = _radial_log_psd(s_field)
                log_ps_t = _radial_log_psd(t_field)
                spec_terms.append((log_ps_s - log_ps_t).abs().mean())
            spectral_loss = self.spectral_weight * torch.stack(spec_terms).mean()
        else:
            spectral_loss = pointwise_loss.new_zeros(())

        total = pointwise_loss + spectral_loss

        return {
            "loss": total,
            "pointwise": pointwise_loss.detach(),
            "spectral": spectral_loss.detach(),
            "N": N,
        }
