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

"""Progressive Distillation loss function (Salimans & Ho, 2022).

Trains a student diffusion model to match two teacher DDIM/Euler steps with
a single student step, enabling iterative halving of sampling cost.

Reference: Salimans, T. and Ho, J., 2022. Progressive Distillation for Fast
Sampling of Diffusion Models. ICLR 2022.
"""

import torch
from torch import Tensor


class ProgressiveDistillationLoss:
    """Loss function for Progressive Distillation of EDM diffusion models.

    Given a frozen teacher and a trainable student (both EDMPrecond), this loss:
    1. Constructs a 2N-step Karras noise schedule (teacher resolution).
    2. Samples a random student step index i, identifying three consecutive
       teacher sigma values: sigma_{2i}, sigma_{2i+1}, sigma_{2i+2}.
    3. Creates a noisy sample x at sigma_{2i} from clean data.
    4. Runs the teacher for TWO Euler PF-ODE steps:
       sigma_{2i} -> sigma_{2i+1} -> sigma_{2i+2} to produce x_target.
    5. Runs the student for ONE Euler PF-ODE step:
       sigma_{2i} -> sigma_{2i+2} to produce x_student.
    6. Returns MSE(x_student, x_target), optionally with EDM weighting.

    The student's effective inference schedule has N+1 sigma values (N steps),
    corresponding to the even-indexed points of the teacher's 2N+1 schedule.

    Parameters
    ----------
    sigma_min : float
        Minimum noise level (lower bound of the Karras schedule).
    sigma_max : float
        Maximum noise level (upper bound of the Karras schedule).
    sigma_data : float
        Expected standard deviation of the training data.
    rho : float
        Exponent for the Karras noise schedule.
    num_steps : int
        Current number of student sampling steps (N). The teacher operates
        on a 2N-step schedule. Updated via ``set_num_steps`` between phases.
    loss_weighting : str
        Loss weighting strategy. 'uniform' applies no per-sample weighting.
        'edm' applies the EDM weighting w(sigma) = (sigma^2 + sigma_data^2)
        / (sigma * sigma_data)^2 based on the starting noise level.

    Note
    ----
    Both teacher and student must be EDMPrecond instances. The student is
    initialized from the teacher's weights at the start of each phase.
    After training, the student becomes the teacher for the next phase
    (with N halved).
    """

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 0.5,
        rho: float = 7.0,
        num_steps: int = 512,
        loss_weighting: str = "uniform",
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.num_steps = num_steps
        self.loss_weighting = loss_weighting

    def set_num_steps(self, num_steps: int):
        """Update student step count when entering a new distillation phase.

        Parameters
        ----------
        num_steps : int
            New number of student sampling steps.
        """
        self.num_steps = num_steps

    def get_schedule(self, N: int, device: torch.device) -> Tensor:
        """Compute the Karras noise schedule for N steps.

        Returns a descending sequence of N+1 sigma values:
        schedule[0] = sigma_max, schedule[N] = sigma_min.

        Parameters
        ----------
        N : int
            Number of sampling steps.
        device : torch.device
            Device for the output tensor.

        Returns
        -------
        Tensor
            Noise levels of shape (N + 1,) in float64 precision.
        """
        indices = torch.arange(N + 1, dtype=torch.float64, device=device)
        schedule = (
            self.sigma_max ** (1 / self.rho)
            + indices
            / N
            * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
        ) ** self.rho
        return schedule

    @torch.no_grad()
    def _teacher_two_steps(
        self,
        teacher: torch.nn.Module,
        x: Tensor,
        sigma_start: Tensor,
        sigma_mid: Tensor,
        sigma_end: Tensor,
        condition: Tensor,
    ) -> Tensor:
        """Execute two Euler PF-ODE steps with the frozen teacher.

        Integrates the probability flow ODE dx/dsigma = (x - D(x,sigma))/sigma
        from sigma_start through sigma_mid to sigma_end.

        Parameters
        ----------
        teacher : torch.nn.Module
            Frozen EDMPrecond teacher model.
        x : Tensor
            Noisy input at sigma_start, shape (B, C, H, W).
        sigma_start : Tensor
            Starting noise level, shape (B, 1, 1, 1).
        sigma_mid : Tensor
            Intermediate noise level, shape (B, 1, 1, 1).
        sigma_end : Tensor
            Target noise level, shape (B, 1, 1, 1).
        condition : Tensor
            Conditioning tensor, shape (B, C_cond, H, W).

        Returns
        -------
        Tensor
            Result of two Euler steps at sigma_end, shape (B, C, H, W).
        """
        # Step 1: sigma_start -> sigma_mid
        denoised_1 = teacher(x, sigma_start.flatten(), condition=condition)
        d_1 = (x - denoised_1) / sigma_start
        x_mid = x + (sigma_mid - sigma_start) * d_1

        # Step 2: sigma_mid -> sigma_end
        denoised_2 = teacher(x_mid, sigma_mid.flatten(), condition=condition)
        d_2 = (x_mid - denoised_2) / sigma_mid
        x_end = x_mid + (sigma_end - sigma_mid) * d_2

        return x_end

    def _student_one_step(
        self,
        student: torch.nn.Module,
        x: Tensor,
        sigma_start: Tensor,
        sigma_end: Tensor,
        condition: Tensor,
    ) -> Tensor:
        """Execute one Euler PF-ODE step with the student model.

        Parameters
        ----------
        student : torch.nn.Module
            Trainable student EDMPrecond model (may be DDP-wrapped).
        x : Tensor
            Noisy input at sigma_start, shape (B, C, H, W).
        sigma_start : Tensor
            Starting noise level, shape (B, 1, 1, 1).
        sigma_end : Tensor
            Target noise level, shape (B, 1, 1, 1).
        condition : Tensor
            Conditioning tensor, shape (B, C_cond, H, W).

        Returns
        -------
        Tensor
            Result of one Euler step at sigma_end, shape (B, C, H, W).
        """
        denoised = student(x, sigma_start.flatten(), condition=condition)
        d = (x - denoised) / sigma_start
        x_end = x + (sigma_end - sigma_start) * d
        return x_end

    def __call__(
        self,
        student: torch.nn.Module,
        teacher: torch.nn.Module,
        images: Tensor,
        condition: Tensor,
    ) -> Tensor:
        """Compute the Progressive Distillation loss.

        For each sample in the batch, samples a random student step index,
        then compares the student's 1-step output against the teacher's
        2-step output starting from the same noisy input.

        Parameters
        ----------
        student : torch.nn.Module
            Online student model (DDP-wrapped, trainable).
        teacher : torch.nn.Module
            Frozen teacher model.
        images : Tensor
            Clean target images of shape (B, C, H, W).
        condition : Tensor
            Conditioning tensor of shape (B, C_cond, H, W).

        Returns
        -------
        Tensor
            Unreduced loss tensor of shape (B, C, H, W).
        """
        N = self.num_steps

        # Teacher schedule: 2N steps => 2N+1 sigma values
        teacher_schedule = self.get_schedule(2 * N, device=images.device)

        batch_size = images.shape[0]

        # Sample random student step indices i in [0, N-1].
        # Student step i maps to teacher indices 2i, 2i+1, 2i+2.
        i = torch.randint(0, N, (batch_size,), device=images.device)

        sigma_start = teacher_schedule[2 * i].to(torch.float32).view(-1, 1, 1, 1)
        sigma_mid = teacher_schedule[2 * i + 1].to(torch.float32).view(-1, 1, 1, 1)
        sigma_end = teacher_schedule[2 * i + 2].to(torch.float32).view(-1, 1, 1, 1)

        # Create noisy samples: x = clean + sigma_start * noise
        noise = torch.randn_like(images)
        x_noisy = images + sigma_start * noise

        # Teacher: 2 Euler steps (no grad)
        x_target = self._teacher_two_steps(
            teacher, x_noisy, sigma_start, sigma_mid, sigma_end, condition
        )

        # Student: 1 Euler step (grad flows through student parameters)
        x_student = self._student_one_step(
            student, x_noisy, sigma_start, sigma_end, condition
        )

        # MSE loss in trajectory space
        loss = (x_student - x_target.detach()) ** 2

        # Optional EDM-style per-sample weighting
        if self.loss_weighting == "edm":
            sigma_flat = sigma_start.squeeze()
            weight = (sigma_flat**2 + self.sigma_data**2) / (
                sigma_flat * self.sigma_data
            ) ** 2
            loss = weight.view(-1, 1, 1, 1) * loss

        return loss
