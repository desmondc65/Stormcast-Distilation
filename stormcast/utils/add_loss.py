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

"""Adversarial Diffusion Distillation (ADD) loss functions.

Implements the two-part ADD objective:
  1. Adversarial loss (hinge) with R1 gradient penalty on the discriminator.
  2. Score distillation loss using the frozen DM-Teacher, with optional
     Noise-Free Score Distillation (NFSD) weighting.

Reference: Sauer et al., "Adversarial Diffusion Distillation", 2023.
"""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd


# ---------------------------------------------------------------------------
# Hinge adversarial loss
# ---------------------------------------------------------------------------

def hinge_loss_discriminator(real_logits: list[torch.Tensor],
                             fake_logits: list[torch.Tensor]) -> torch.Tensor:
    """Multi-scale hinge loss for the discriminator.

    Args:
        real_logits: List of per-scale logits for real samples.
        fake_logits: List of per-scale logits for fake (student) samples.

    Returns:
        Scalar discriminator loss.
    """
    loss = 0.0
    for r, f in zip(real_logits, fake_logits):
        loss = loss + F.relu(1.0 - r).mean() + F.relu(1.0 + f).mean()
    return loss / len(real_logits)


def hinge_loss_generator(fake_logits: list[torch.Tensor]) -> torch.Tensor:
    """Multi-scale hinge loss for the generator (student).

    Args:
        fake_logits: List of per-scale logits for generated samples.

    Returns:
        Scalar generator adversarial loss.
    """
    loss = 0.0
    for f in fake_logits:
        loss = loss - f.mean()
    return loss / len(fake_logits)


# ---------------------------------------------------------------------------
# R1 gradient penalty
# ---------------------------------------------------------------------------

def r1_gradient_penalty(
    discriminator: nn.Module,
    real: torch.Tensor,
    condition: torch.Tensor | None = None,
) -> torch.Tensor:
    """R1 gradient penalty computed on discriminator head inputs (feature maps).

    Following ADD, the penalty is applied at the input to each discriminator
    head (intermediate feature maps after the backbone), not on the raw
    multi-channel input.  This is more stable for high-dimensional inputs
    like 99-channel atmospheric state and matches ADD's design where the
    frozen backbone does not receive R1 gradients.

    Args:
        discriminator: The multi-scale discriminator (may be DDP-wrapped).
        real: Real residual tensor [B, C, H, W].
        condition: Optional conditioning tensor.

    Returns:
        Scalar R1 penalty (averaged over scales).
    """
    # Unwrap DDP if needed
    disc = discriminator.module if hasattr(discriminator, 'module') else discriminator

    # Build combined input (matching discriminator.forward logic)
    x = real.detach()
    if condition is not None:
        x = torch.cat([x, condition.detach()], dim=1)

    # Forward through backbone (stem + stages) without gradient tracking
    with torch.no_grad():
        h = disc.stem(x)

    penalty = 0.0
    for stage, head in zip(disc.stages, disc.heads):
        with torch.no_grad():
            h = stage(h)

        # Detach feature map and enable gradients for head-input R1
        feat = h.detach().requires_grad_(True)
        logit = head(feat)

        (grad_feat,) = autograd.grad(
            outputs=logit.sum(),
            inputs=feat,
            create_graph=True,
        )
        penalty = penalty + grad_feat.pow(2).flatten(1).sum(1).mean()

    return penalty / len(disc.heads)


# ---------------------------------------------------------------------------
# Score distillation loss
# ---------------------------------------------------------------------------

class ScoreDistillationLoss(nn.Module):
    """Score distillation from a frozen DM-Teacher.

    The student's output is diffused under the teacher's noise schedule,
    then the teacher predicts the denoised sample. The distillation loss
    is the L2 distance between the student sample and the teacher target
    (with stop-gradient on the teacher output).

    Supports standard SDS weighting and Noise-Free Score Distillation (NFSD).

    Parameters
    ----------
    sigma_min : float
        Minimum noise level for diffusing the student output.
    sigma_max : float
        Maximum noise level for diffusing the student output.
    sigma_data : float
        EDM data standard deviation (for SNR weighting).
    P_mean : float
        Mean of the log-normal sigma sampling distribution.
    P_std : float
        Std of the log-normal sigma sampling distribution.
    use_nfsd : bool
        If True, subtract the unconditional (noise-only) score estimate
        to reduce artifacts (Noise-Free Score Distillation).
    """

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 0.5,
        P_mean: float = -1.2,
        P_std: float = 1.2,
        use_nfsd: bool = False,
    ):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.P_mean = P_mean
        self.P_std = P_std
        self.use_nfsd = use_nfsd
        if use_nfsd:
            warnings.warn(
                "use_nfsd=True but Noise-Free Score Distillation is not yet "
                "implemented. Falling back to standard score distillation. "
                "NFSD requires an unconditional teacher forward pass to "
                "subtract the noise-only score estimate.",
                stacklevel=2,
            )

    def forward(
        self,
        student_output: torch.Tensor,
        teacher: nn.Module,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """Compute score distillation loss.

        Args:
            student_output: Student's generated residual [B, C, H, W].
            teacher: Frozen DM-Teacher (EDMPrecond).
            condition: Conditioning tensor [B, C_cond, H, W].

        Returns:
            Scalar distillation loss.
        """
        B = student_output.shape[0]
        device = student_output.device

        # Sample sigma from log-normal distribution (EDM convention)
        rnd = torch.randn(B, device=device)
        sigma = (rnd * self.P_std + self.P_mean).exp()
        sigma = sigma.clamp(self.sigma_min, self.sigma_max)
        sigma_view = sigma.view(B, 1, 1, 1)

        # Diffuse the student output under the teacher's schedule
        noise = torch.randn_like(student_output)
        noised_student = student_output + sigma_view * noise

        # Teacher denoises the noised student output (stop-gradient)
        with torch.no_grad():
            teacher_denoised = teacher(noised_student, sigma, condition=condition)

        # SNR weighting: w(sigma) = 1 / (sigma^2 + sigma_data^2)
        weight = 1.0 / (sigma_view ** 2 + self.sigma_data ** 2)

        # L2 distillation loss with SNR weighting
        loss = weight * (student_output - teacher_denoised.detach()) ** 2
        return loss.mean()


# ---------------------------------------------------------------------------
# Combined ADD loss
# ---------------------------------------------------------------------------

class ADDLoss(nn.Module):
    """Combined Adversarial Diffusion Distillation loss.

    L = L_adv^G + lambda_distill * L_distill

    Parameters
    ----------
    sigma_min : float
        Minimum sigma for score distillation.
    sigma_max : float
        Maximum sigma for score distillation.
    sigma_data : float
        EDM data standard deviation.
    P_mean : float
        Mean of log-normal sigma sampling.
    P_std : float
        Std of log-normal sigma sampling.
    lambda_distill : float
        Weight for the score distillation loss.
    r1_gamma : float
        Weight for the R1 gradient penalty on the discriminator.
    use_nfsd : bool
        Use Noise-Free Score Distillation.
    """

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 0.5,
        P_mean: float = -1.2,
        P_std: float = 1.2,
        lambda_distill: float = 2.5,
        r1_gamma: float = 1e-5,
        use_nfsd: bool = False,
    ):
        super().__init__()
        self.lambda_distill = lambda_distill
        self.r1_gamma = r1_gamma

        self.distill_loss = ScoreDistillationLoss(
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            sigma_data=sigma_data,
            P_mean=P_mean,
            P_std=P_std,
            use_nfsd=use_nfsd,
        )

    def generator_loss(
        self,
        student_output: torch.Tensor,
        teacher: nn.Module,
        discriminator: nn.Module,
        condition: torch.Tensor,
        disc_condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the combined generator (student) loss.

        Args:
            student_output: Student's predicted residual [B, C, H, W].
            teacher: Frozen DM-Teacher.
            discriminator: The multi-scale discriminator.
            condition: Conditioning for the teacher.
            disc_condition: Conditioning for the discriminator (can differ).

        Returns:
            loss: Scalar combined loss for the student.
            log_dict: Dictionary of individual loss components for logging.
        """
        # Adversarial loss
        fake_logits = discriminator(student_output, disc_condition)
        loss_adv = hinge_loss_generator(fake_logits)

        # Score distillation loss
        loss_distill = self.distill_loss(student_output, teacher, condition)

        loss = loss_adv + self.lambda_distill * loss_distill

        log_dict = {
            "loss_adv_G": loss_adv.item(),
            "loss_distill": loss_distill.item(),
            "loss_G_total": loss.item(),
        }
        return loss, log_dict

    def discriminator_loss(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
        discriminator: nn.Module,
        disc_condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the discriminator loss with R1 penalty.

        Args:
            real: Real residual [B, C, H, W].
            fake: Student-generated residual [B, C, H, W] (detached).
            discriminator: The multi-scale discriminator.
            disc_condition: Optional conditioning for the discriminator.

        Returns:
            loss: Scalar discriminator loss.
            log_dict: Dictionary of individual loss components for logging.
        """
        real_logits = discriminator(real, disc_condition)
        fake_logits = discriminator(fake.detach(), disc_condition)
        loss_d = hinge_loss_discriminator(real_logits, fake_logits)

        # R1 gradient penalty
        r1 = r1_gradient_penalty(discriminator, real, disc_condition)
        loss = loss_d + self.r1_gamma * r1

        log_dict = {
            "loss_D_hinge": loss_d.item(),
            "loss_D_r1": r1.item(),
            "loss_D_total": loss.item(),
        }
        return loss, log_dict
