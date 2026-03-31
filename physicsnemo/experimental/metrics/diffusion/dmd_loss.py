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

"""Distribution Matching Distillation loss (Yin et al., 2024).

Distills a pre-trained EDM diffusion teacher into a 1-step generator by
minimising the KL divergence between student and teacher output distributions,
estimated via the teacher's score function.

References
----------
- Yin et al., 2024. "One-step Diffusion with Distribution Matching Distillation"
- Yin et al., 2024. "Improved Distribution Matching Distillation" (DMD2)
- Karras et al., 2022. "Elucidating the Design Space of Diffusion-Based
  Generative Models" (EDM)
"""

import torch
from torch import Tensor

from physicsnemo.utils.diffusion import deterministic_sampler


class DMDLoss:
    """Loss function for Distribution Matching Distillation.

    Combines two terms:

    1. **Distribution matching loss** — uses the difference between the
       teacher's score evaluated on student samples (real score) and the
       score of the student's own distribution (fake score) to provide
       a gradient direction that minimises KL(q_student || p_teacher).

    2. **Regression loss** — MSE between the student's output and a
       teacher-generated sample from the same latent noise, providing
       a stable learning signal especially early in training.

    The regression weight is linearly annealed from ``lambda_reg`` to
    ``lambda_reg_final`` over ``reg_anneal_steps``.

    Parameters
    ----------
    sigma_min : float
        Minimum noise level for the teacher ODE schedule.
    sigma_max : float
        Maximum noise level / student denoising level.
    sigma_data : float
        Expected standard deviation of the training data.
    P_mean : float
        Mean of the log-normal distribution for score evaluation sigma.
    P_std : float
        Std of the log-normal distribution for score evaluation sigma.
    rho : float
        Exponent for the Karras noise schedule (teacher ODE).
    lambda_reg : float
        Initial regression loss weight.
    lambda_reg_final : float
        Final regression loss weight (after annealing).
    reg_anneal_steps : int
        Number of steps over which to anneal lambda_reg.
    num_teacher_steps : int
        Number of ODE steps for generating teacher regression targets.
    score_sigma_min : float
        Minimum sigma for score evaluation (clamp).
    score_sigma_max : float
        Maximum sigma for score evaluation (clamp).
    use_dmd2 : bool
        If True, use DMD2 approach (EMA student for fake score).
        If False, use teacher score on real data as proxy for fake score.
    """

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 0.5,
        P_mean: float = -1.2,
        P_std: float = 1.2,
        rho: float = 7.0,
        lambda_reg: float = 1.0,
        lambda_reg_final: float = 0.0,
        reg_anneal_steps: int = 200000,
        num_teacher_steps: int = 18,
        score_sigma_min: float = 0.05,
        score_sigma_max: float = 10.0,
        use_dmd2: bool = True,
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.P_mean = P_mean
        self.P_std = P_std
        self.rho = rho
        self.lambda_reg = lambda_reg
        self.lambda_reg_final = lambda_reg_final
        self.reg_anneal_steps = reg_anneal_steps
        self.num_teacher_steps = num_teacher_steps
        self.score_sigma_min = score_sigma_min
        self.score_sigma_max = score_sigma_max
        self.use_dmd2 = use_dmd2

    def _get_regression_weight(self, current_step: int) -> float:
        """Linearly anneal lambda_reg from initial to final value.

        Parameters
        ----------
        current_step : int
            Current training step.

        Returns
        -------
        float
            Regression loss weight at the current step.
        """
        if self.reg_anneal_steps <= 0:
            return self.lambda_reg_final
        t = min(current_step / self.reg_anneal_steps, 1.0)
        return self.lambda_reg * (1.0 - t) + self.lambda_reg_final * t

    @torch.no_grad()
    def _generate_teacher_sample(
        self,
        teacher: torch.nn.Module,
        latents: Tensor,
        condition: Tensor,
    ) -> Tensor:
        """Generate a teacher sample via deterministic ODE integration.

        Uses the same latent noise as the student so that the regression
        loss provides a paired signal.

        Parameters
        ----------
        teacher : torch.nn.Module
            Frozen EDM teacher model.
        latents : Tensor
            Initial noise of shape (B, C, H, W), already scaled by sigma_max.
        condition : Tensor
            Conditioning tensor of shape (B, C_cond, H, W).

        Returns
        -------
        Tensor
            Teacher-generated samples of shape (B, C, H, W).
        """
        return deterministic_sampler(
            net=teacher,
            latents=latents / self.sigma_max,  # deterministic_sampler rescales internally
            img_lr=condition,
            num_steps=self.num_teacher_steps,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            rho=self.rho,
            solver="euler",
        )

    def _compute_score(
        self,
        teacher: torch.nn.Module,
        x: Tensor,
        sigma: Tensor,
        condition: Tensor,
    ) -> Tensor:
        """Compute the score s(x, sigma) using the teacher denoiser.

        The score of the smoothed distribution at noise level sigma is:
            s(x, sigma) = (D(x, sigma) - x) / sigma^2

        where D is the denoiser (teacher) output.

        Parameters
        ----------
        teacher : torch.nn.Module
            Frozen EDM teacher model.
        x : Tensor
            Noisy input of shape (B, C, H, W).
        sigma : Tensor
            Noise levels of shape (B,) or (B, 1, 1, 1).
        condition : Tensor
            Conditioning tensor of shape (B, C_cond, H, W).

        Returns
        -------
        Tensor
            Score estimate of shape (B, C, H, W).
        """
        sigma_flat = sigma.flatten()
        sigma_4d = sigma.view(-1, 1, 1, 1) if sigma.dim() == 1 else sigma
        denoised = teacher(x, sigma_flat, condition=condition)
        return (denoised - x) / sigma_4d**2

    def __call__(
        self,
        student: torch.nn.Module,
        teacher: torch.nn.Module,
        ema_student: torch.nn.Module,
        images: Tensor,
        condition: Tensor,
        current_step: int,
    ) -> Tensor:
        """Compute the combined DMD loss.

        Parameters
        ----------
        student : torch.nn.Module
            Online student model (DDP-wrapped, trainable).
        teacher : torch.nn.Module
            Frozen EDM teacher model.
        ema_student : torch.nn.Module
            EMA copy of the student (no grad).
        images : Tensor
            Clean target images of shape (B, C, H, W). Not directly used
            in the loss but kept for API consistency with other losses.
        condition : Tensor
            Conditioning tensor of shape (B, C_cond, H, W).
        current_step : int
            Current training step (for regression weight annealing).

        Returns
        -------
        Tensor
            Unreduced loss tensor of shape (B, C, H, W).
        """
        B, C, H, W = images.shape
        device = images.device

        # --- Step 1: Sample shared latent noise ---
        z = torch.randn_like(images) * self.sigma_max
        sigma_gen = torch.full([B], self.sigma_max, device=device, dtype=images.dtype)

        # --- Step 2: Student forward pass (gradients flow) ---
        x_student = student(z, sigma_gen, condition=condition)

        # --- Step 3: Regression loss ---
        with torch.no_grad():
            x_teacher = self._generate_teacher_sample(teacher, z, condition)
            x_teacher = x_teacher.to(images.dtype)

        loss_reg = (x_student - x_teacher) ** 2

        # --- Step 4: Distribution matching loss ---
        # Sample score evaluation noise level from log-normal
        rnd_normal = torch.randn([B, 1, 1, 1], device=device, dtype=images.dtype)
        sigma_score = (rnd_normal * self.P_std + self.P_mean).exp()
        sigma_score = sigma_score.clamp(self.score_sigma_min, self.score_sigma_max)

        # Shared noise for score evaluation
        eps = torch.randn_like(x_student)

        # Noisy student output for real score evaluation
        x_noisy = x_student.detach() + sigma_score * eps

        with torch.no_grad():
            # Real score: teacher score on noisy student output
            s_real = self._compute_score(teacher, x_noisy, sigma_score, condition)

            if self.use_dmd2:
                # DMD2: fake score via EMA student output with same noise
                x_ema = ema_student(z, sigma_gen, condition=condition)
                x_noisy_ema = x_ema + sigma_score * eps
                s_fake = self._compute_score(
                    teacher, x_noisy_ema, sigma_score, condition
                )
            else:
                # Fallback: use teacher score on clean data as proxy
                # This is a simplified approximation
                x_noisy_real = images + sigma_score * eps
                s_fake = self._compute_score(
                    teacher, x_noisy_real, sigma_score, condition
                )

        # Score difference (detached — acts as pseudo-gradient direction)
        score_diff = (s_fake - s_real).detach()

        # DMD loss: autograd gives grad_theta L = score_diff * grad_theta(x_student)
        loss_dm = score_diff * x_student

        # --- Step 5: Combine losses ---
        lam = self._get_regression_weight(current_step)
        loss = loss_dm + lam * loss_reg

        return loss
