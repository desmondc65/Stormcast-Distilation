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

from collections.abc import Iterable

import torch
from physicsnemo.models import Module
from physicsnemo.models.diffusion import ConsistencyPrecond, EDMPrecond, StormCastUNet
from physicsnemo.utils.diffusion import deterministic_sampler


def get_preconditioned_architecture(
    name: str,
    target_channels: int,
    conditional_channels: int = 0,
    spatial_embedding: bool = True,
    img_resolution: tuple = (512, 640),
    attn_resolutions: list = [],
) -> EDMPrecond | StormCastUNet:
    """

    Args:
        name: 'regression' or 'diffusion' to select between either model type
        target_channels: The number of channels in the target
        conditional_channels: The number of channels in the conditioning
        spatial_embedding: whether or not to use the additive spatial embedding in the U-Net
        img_resolution: resolution of the data (U-Net inputs/outputs)
        attn_resolutions: resolution of internal U-Net stages to use self-attention
    Returns:
        EDMPrecond or StormCastUNet: a wrapped torch module net(x+n, sigma, condition, class_labels) -> x
    """
    if name == "diffusion":
        return EDMPrecond(
            img_resolution=img_resolution,
            img_channels=target_channels + conditional_channels,
            img_out_channels=target_channels,
            model_type="SongUNet",
            channel_mult=[1, 2, 2, 2, 2],
            attn_resolutions=attn_resolutions,
            additive_pos_embed=spatial_embedding,
        )

    elif name == "consistency":
        return ConsistencyPrecond(
            img_resolution=img_resolution,
            img_channels=target_channels + conditional_channels,
            img_out_channels=target_channels,
            model_type="SongUNet",
            channel_mult=[1, 2, 2, 2, 2],
            attn_resolutions=attn_resolutions,
            additive_pos_embed=spatial_embedding,
        )

    elif name == "dmd":
        return EDMPrecond(
            img_resolution=img_resolution,
            img_channels=target_channels + conditional_channels,
            img_out_channels=target_channels,
            model_type="SongUNet",
            channel_mult=[1, 2, 2, 2, 2],
            attn_resolutions=attn_resolutions,
            additive_pos_embed=spatial_embedding,
        )

    elif name == "add":
        return EDMPrecond(
            img_resolution=img_resolution,
            img_channels=target_channels + conditional_channels,
            img_out_channels=target_channels,
            model_type="SongUNet",
            channel_mult=[1, 2, 2, 2, 2],
            attn_resolutions=attn_resolutions,
            additive_pos_embed=spatial_embedding,
        )

    elif name == "regression":
        return StormCastUNet(
            img_resolution=img_resolution,
            img_in_channels=conditional_channels,
            img_out_channels=target_channels,
            model_type="SongUNet",
            embedding_type="zero",
            channel_mult=[1, 2, 2, 2, 2],
            attn_resolutions=attn_resolutions,
            additive_pos_embed=spatial_embedding,
        )


def build_network_condition_and_target(
    background: torch.Tensor,
    state: tuple[torch.Tensor, torch.Tensor],
    invariant_tensor: torch.Tensor | None,
    regression_net: Module | None = None,
    condition_list: Iterable[str] = ("state", "background"),
    regression_condition_list: Iterable[str] = ("state", "background"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Build the condition and target tensors for the network.

    Args:
        background: background tensor
        state: tuple of previous state and target state
        invariant_tensor: invariant tensor or None if no invariant is used
        regression_net: regression model, can be None if 'regression' is not in condition_list
        condition_list: list of conditions to include, may include 'state', 'background', 'regression' and 'invariant'
        regression_condition_list: list of conditions for the regression network, may include 'state', 'background', and 'invariant'
            This is only used if regression_net is set.
    Returns:
        A tuple of tensors: (
            condition: model condition concatenated from conditions specified in condition_list,
            target: training target,
            regression: regression model output
        ). The regression model output will be None if 'regression' is not in condition_list.
    """
    if ("regression" in condition_list) and (regression_net is None):
        raise ValueError(
            "regression_net must be provided if 'regression' is in condition_list"
        )
    target = state[1]

    condition_tensors = {
        "state": state[0],
        "background": background,
        "invariant": invariant_tensor,
        "regression": None,
    }

    with torch.no_grad():
        if "regression" in condition_list:
            # Inference regression model
            condition_tensors["regression"] = regression_model_forward(
                regression_net,
                state[0],
                background,
                invariant_tensor,
                condition_list=regression_condition_list,
            )
            target = target - condition_tensors["regression"]

        condition = [
            y for c in condition_list if (y := condition_tensors[c]) is not None
        ]
        condition = torch.cat(condition, dim=1)

    return (condition, target, condition_tensors["regression"])


def diffusion_model_forward(model, condition, shape, sampler_args={}):
    """Helper function to run diffusion model sampling"""

    latents = torch.randn(*shape, device=condition.device, dtype=condition.dtype)

    return deterministic_sampler(
        model, latents=latents, img_lr=condition, **sampler_args
    )


def regression_model_forward(
    model, state, background, invariant_tensor, condition_list=("state", "background")
):
    """Helper function to run regression model forward pass in inference"""

    (x, _, _) = build_network_condition_and_target(
        background, (state, None), invariant_tensor, condition_list=condition_list
    )

    return model(x)


def consistency_model_forward(
    model,
    condition,
    shape,
    sigma_max=80.0,
    sigma_min=0.002,
    rho=7.0,
    num_steps=1,
    intermediate_sigmas=None,
):
    """Multi-step generation using a consistency model.

    Implements the Consistency Models multi-step sampler (Song et al. 2023,
    Algorithm 1): evaluate f once at sigma_max, then for each intermediate
    sigma re-noise the clean estimate by sqrt(sigma^2 - sigma_min^2) and
    re-evaluate f. This preserves stochasticity between steps and is the
    knob the CD spec wants exposed for num_steps ∈ {1, 2, 4}.

    Args:
        model: ConsistencyPrecond model.
        condition: conditioning tensor [B, C_cond, H, W].
        shape: shape of the output tensor [B, C, H, W].
        sigma_max: maximum noise level (start of trajectory).
        sigma_min: minimum noise level (boundary of f).
        rho: Karras schedule exponent (only used if intermediate_sigmas is None).
        num_steps: number of function evaluations. 1 = one-shot.
        intermediate_sigmas: optional explicit list of intermediate σ values
            (length num_steps - 1), descending, each in (sigma_min, sigma_max).
            If None, chosen from the Karras ρ-schedule.

    Returns:
        Generated samples [B, C, H, W].
    """
    device = condition.device
    dtype = condition.dtype
    B = shape[0]

    x = torch.randn(*shape, device=device, dtype=dtype) * sigma_max
    sigma = torch.full([B], sigma_max, device=device, dtype=dtype)
    x0 = model(x, sigma, condition=condition)

    if num_steps <= 1:
        return x0

    if intermediate_sigmas is None:
        # Pick num_steps+1 points on the Karras grid between sigma_max and
        # sigma_min, drop the endpoints → num_steps-1 intermediate σ values.
        idx = torch.arange(num_steps + 1, dtype=torch.float64, device=device)
        t = (
            sigma_max ** (1 / rho)
            + idx / num_steps * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        intermediate_sigmas = t[1:-1].to(dtype).tolist()

    for s in intermediate_sigmas:
        s = float(s)
        z = torch.randn_like(x0)
        noise_scale = (max(s * s - sigma_min * sigma_min, 0.0)) ** 0.5
        x_noisy = x0 + noise_scale * z
        sigma = torch.full([B], s, device=device, dtype=dtype)
        x0 = model(x_noisy, sigma, condition=condition)

    return x0


def dmd_model_forward(model, condition, shape, sigma_max=80.0):
    """1-step generation using a DMD-distilled EDMPrecond model.

    Samples z ~ N(0, sigma_max^2 I) and evaluates the denoiser D(z, sigma_max)
    in a single forward pass. The model has been trained via Distribution
    Matching Distillation to produce high-quality samples in one step.

    Args:
        model: EDMPrecond model (DMD-distilled)
        condition: conditioning tensor [B, C_cond, H, W]
        shape: shape of the output tensor [B, C, H, W]
        sigma_max: maximum noise level

    Returns:
        Generated samples [B, C, H, W]
    """
    z = torch.randn(*shape, device=condition.device, dtype=condition.dtype) * sigma_max
    sigma = torch.full([z.shape[0]], sigma_max, device=z.device, dtype=z.dtype)
    return model(z, sigma, condition=condition)


def progressive_distilled_forward(
    model, condition, shape, num_steps=4, sigma_min=0.002, sigma_max=80.0, rho=7.0
):
    """Inference with a progressively distilled EDM model.

    Uses the standard deterministic sampler with reduced step count. The model
    is a regular EDMPrecond whose weights were trained via progressive
    distillation to produce high-quality samples in fewer steps.

    Args:
        model: EDMPrecond model (distilled student)
        condition: conditioning tensor [B, C_cond, H, W]
        shape: shape of the output tensor [B, C, H, W]
        num_steps: number of sampling steps (should match the distillation target)
        sigma_min: minimum noise level
        sigma_max: maximum noise level
        rho: Karras schedule exponent

    Returns:
        Generated samples [B, C, H, W]
    """
    sampler_args = dict(
        num_steps=num_steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
        solver="euler",
    )
    return diffusion_model_forward(model, condition, shape, sampler_args)


def add_model_forward(model, condition, shape, sigma_max=80.0, num_steps=1,
                      sigma_min=0.002, rho=7.0):
    """Inference with an ADD-distilled EDMPrecond model.

    For 1-step: sample z ~ N(0, sigma_max^2 I), denoise once.
    For N>1 steps: use deterministic Euler schedule.

    Args:
        model: EDMPrecond model (ADD-distilled student).
        condition: conditioning tensor [B, C_cond, H, W].
        shape: shape of the output tensor [B, C, H, W].
        sigma_max: maximum noise level.
        num_steps: number of denoising steps (1-4).
        sigma_min: minimum noise level.
        rho: Karras schedule exponent.

    Returns:
        Generated samples [B, C, H, W].
    """
    device = condition.device
    dtype = condition.dtype
    B = shape[0]

    if num_steps == 1:
        z = torch.randn(*shape, device=device, dtype=dtype) * sigma_max
        sigma = torch.full([B], sigma_max, device=device, dtype=dtype)
        return model(z, sigma, condition=condition)

    # Multi-step Euler sampling with Karras schedule
    step_indices = torch.arange(num_steps, device=device, dtype=dtype)
    sigma_max_inv = sigma_max ** (1 / rho)
    sigma_min_inv = sigma_min ** (1 / rho)
    t_steps = (sigma_max_inv + step_indices / (num_steps - 1) * (sigma_min_inv - sigma_max_inv)) ** rho
    t_steps = torch.cat([t_steps, torch.zeros(1, device=device)])

    x = torch.randn(*shape, device=device, dtype=dtype) * t_steps[0]
    for i in range(num_steps):
        sigma = torch.full([B], t_steps[i].item(), device=device, dtype=dtype)
        denoised = model(x, sigma, condition=condition)
        d = (x - denoised) / t_steps[i]
        x = x + d * (t_steps[i + 1] - t_steps[i])

    return x


def regression_loss_fn(
    net: Module,
    images,
    condition,
    class_labels=None,
    augment_pipe=None,
    return_model_outputs=False,
):
    """Helper function for training the StormCast regression model, so that it has a similar call signature as
    the EDMLoss and the same training loop can be used to train both regression and diffusion models

    Args:
        net: physicsnemo.models.diffusion.StormCastUNet
        images: Target data, shape [batch_size, target_channels, w, h]
        condition: input to the model, shape=[batch_size, condition_channel, w, h]
        class_labels: unused (applied to match EDMLoss signature)
        augment_pipe: optional data augmentation pipe
        return_model_outputs: If True, will return the generated outputs
    Returns:
        out: loss function with shape [batch_size, target_channels, w, h]
            This should be averaged to get the mean loss for gradient descent.
    """

    y, augment_labels = (
        augment_pipe(images) if augment_pipe is not None else (images, None)
    )

    D_yn = net(x=condition)
    loss = (D_yn - y) ** 2
    if return_model_outputs:
        return loss, D_yn
    else:
        return loss
