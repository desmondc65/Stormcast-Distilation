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

import torch
from torch import Tensor


class ConsistencyDistillationLoss:
    """Loss function for Consistency Distillation.

    Distills a pre-trained EDM diffusion teacher into a consistency model
    that can generate samples in 1 or 2 steps. The loss enforces that the
    student maps any point on the same PF-ODE trajectory to the same
    clean output, using the teacher to provide one-step ODE guidance.

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
        Pseudo-Huber loss parameter. If None, uses MSE.

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
        huber_c: float = None,
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.N_0 = N_0
        self.N_total = N_total
        self.total_train_steps = total_train_steps
        self.huber_c = huber_c

    def get_num_steps(self, current_step: int) -> int:
        """Compute the adaptive number of discretization steps N(k).

        Uses a sqrt schedule to increase from N_0 to N_total over training.

        Parameters
        ----------
        current_step : int
            Current training step.

        Returns
        -------
        int
            Number of discretization steps at the current step.
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

        Parameters
        ----------
        N : int
            Number of discretization steps.
        device : torch.device
            Device for the output tensor.

        Returns
        -------
        Tensor
            Noise levels of shape (N + 1,).
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
    def euler_step(
        self,
        teacher: torch.nn.Module,
        x: Tensor,
        t_cur: Tensor,
        t_next: Tensor,
        condition: Tensor,
    ) -> Tensor:
        """Take a single Euler ODE step using the teacher model.

        Solves the EDM probability flow ODE: dx/dt = (x - D(x,t)) / t

        Parameters
        ----------
        teacher : torch.nn.Module
            Frozen EDM teacher model.
        x : Tensor
            Noisy input of shape (B, C, H, W).
        t_cur : Tensor
            Current noise levels of shape (B, 1, 1, 1).
        t_next : Tensor
            Target noise levels of shape (B, 1, 1, 1).
        condition : Tensor
            Conditioning tensor.

        Returns
        -------
        Tensor
            Denoised estimate at noise level t_next.
        """
        denoised = teacher(x, t_cur.flatten(), condition=condition)
        d = (x - denoised) / t_cur
        x_next = x + (t_next - t_cur) * d
        return x_next

    def __call__(
        self,
        student: torch.nn.Module,
        teacher: torch.nn.Module,
        ema_student: torch.nn.Module,
        images: Tensor,
        condition: Tensor,
        current_step: int,
    ) -> Tensor:
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
        Tensor
            Loss tensor of shape (B, C, H, W) (unreduced).
        """
        N = self.get_num_steps(current_step)
        t_schedule = self.get_discretization(N, device=images.device)
        # t_schedule: [sigma_max, ..., sigma_min], length N+1
        # t_schedule[0] = sigma_max (noisiest)
        # t_schedule[N] = sigma_min (cleanest)

        batch_size = images.shape[0]

        # Sample random index n in [1, N] (inclusive).
        # t_schedule[n-1] is the noisier level, t_schedule[n] is the less noisy level.
        n = torch.randint(1, N + 1, (batch_size,), device=images.device)

        t_n1 = t_schedule[n - 1].to(torch.float32)  # noisier: t_{n+1} in paper notation
        t_n = t_schedule[n].to(torch.float32)  # less noisy: t_n in paper notation

        t_n1 = t_n1.view(-1, 1, 1, 1)
        t_n = t_n.view(-1, 1, 1, 1)

        # Create noisy sample at the noisier level
        noise = torch.randn_like(images)
        x_noisy = images + t_n1 * noise

        # Teacher Euler step: from t_{n+1} down to t_n
        with torch.no_grad():
            x_hat = self.euler_step(teacher, x_noisy, t_n1, t_n, condition)

        # Student prediction at (x_noisy, t_{n+1})
        student_out = student(x_noisy, t_n1.flatten(), condition=condition)

        # EMA target prediction at (x_hat, t_n) — stop gradient
        with torch.no_grad():
            target_out = ema_student(x_hat, t_n.flatten(), condition=condition)

        # Compute loss
        if self.huber_c is not None and self.huber_c > 0:
            # Pseudo-Huber loss
            diff = student_out - target_out
            loss = torch.sqrt(diff**2 + self.huber_c**2) - self.huber_c
        else:
            loss = (student_out - target_out) ** 2

        # Weight by 1/N to stabilize across schedule progression
        loss = loss / N

        return loss
