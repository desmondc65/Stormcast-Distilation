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
from physicsnemo.models.diffusion import EDMPrecond, StormCastUNet
from physicsnemo.utils.diffusion import deterministic_sampler

from .flowcast_precond import FlowCastPrecond


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

    elif name == "flowcast":
        return FlowCastPrecond(
            img_resolution=img_resolution,
            img_channels=target_channels + conditional_channels,
            img_in_channels=target_channels + conditional_channels,
            img_out_channels=target_channels,
            model_type="SongUNet",
            channel_mult=[1, 2, 2, 2, 2],
            attn_resolutions=attn_resolutions,
            additive_pos_embed=spatial_embedding,
        )

    elif name == "bridge":
        # Anchored Stochastic-Interpolant Bridge: shares the FlowCast SongUNet
        # backbone (same in/out shape contract: predict a velocity field with
        # the same channel count as the target). The only difference vs.
        # 'flowcast' is at the loss / sampler level -- mu enters as the prior
        # endpoint of the flow rather than as a target shift.
        return FlowCastPrecond(
            img_resolution=img_resolution,
            img_channels=target_channels + conditional_channels,
            img_in_channels=target_channels + conditional_channels,
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
    subtract_regression: bool = True,
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
        subtract_regression: if True (legacy EDM / FlowCast behaviour) the
            returned target is the residual ``M_{t+1} - mu_{t+1}``. If False,
            the returned target is the raw ``M_{t+1}`` -- used by the
            anchored bridge loop, which needs mu as the *prior endpoint* of
            the flow rather than a target shift. The regression output is
            still returned separately as the third tuple element either way.
    Returns:
        A tuple of tensors: (
            condition: model condition concatenated from conditions specified in condition_list,
            target: training target (residual if subtract_regression else raw M_{t+1}),
            regression: regression model output (mu_{t+1}), or None
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
            if subtract_regression:
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


def flowcast_model_forward(
    model,
    condition,
    shape,
    num_steps: int = 10,
    sigma_data: float = 0.5,
    t_start: float = 0.0,
    t_end: float = 1.0,
    solver: str = "euler",
):
    """Sample the FlowCast residual with a fixed-step ODE solver.

    Integrates ``dz/dt = v_theta(z, t, condition)`` from t_start to t_end
    starting at ``z(0) ~ N(0, I)`` (standardized space) and returns the
    de-normalized residual ``z(1) * sigma_data``. Used as the generative step
    on top of the regression mean mu_{t+1}; the final prediction is
    ``mu_{t+1} + r_{t+1}``.

    Args:
        model: FlowCastPrecond model (or an EMA shadow of one).
        condition: conditioning tensor [B, C_cond, H, W].
        shape: shape of the output tensor [B, C_target, H, W].
        num_steps: number of ODE steps (FlowCast paper default: 10).
        sigma_data: standard deviation used for standardization at train time.
        t_start, t_end: integration bounds along the flow axis.
        solver: 'euler' or 'midpoint'. Midpoint is a 2nd-order Runge–Kutta
            variant that doubles the NFE per step.

    Returns:
        Predicted residual R_hat of shape ``shape`` (in raw, un-standardized
        units), ready to add to the regression mean.
    """
    device = condition.device
    dtype = condition.dtype

    z = torch.randn(*shape, device=device, dtype=dtype)
    dt = (t_end - t_start) / num_steps

    for i in range(num_steps):
        t_i = t_start + i * dt
        t_vec = torch.full([shape[0]], t_i, device=device, dtype=dtype)
        if solver == "midpoint":
            v_half = model(z, t_vec, condition=condition)
            t_half = t_vec + 0.5 * dt
            z_half = z + v_half * (0.5 * dt)
            v = model(z_half, t_half, condition=condition)
        else:  # euler
            v = model(z, t_vec, condition=condition)
        z = z + v * dt

    return z * sigma_data


def bridge_model_forward(
    model,
    condition,
    mu,
    num_steps: int = 10,
    sigma_prior: float = 0.05,
    solver: str = "euler",
    t_start: float = 0.0,
    t_end: float = 1.0,
    prior_channel_std=None,
):
    """Sample the next-state field via the anchored Stochastic-Interpolant
    Bridge ODE.

    Integrates ``dz/dt = v_theta(z, t, condition)`` from ``t_start`` to
    ``t_end`` starting at ``z(0) = mu + sigma_prior * eps`` (so the prior
    endpoint is the frozen regression mean, perturbed). Returns ``z(t_end)``
    *in raw physical units* -- the full prediction ``M_{t+1}``, NOT a
    residual. Callers should NOT add ``mu`` back.

    Differences from :func:`flowcast_model_forward`:
        * the integrator starts at ``mu + sigma_prior * eps`` instead of
          standardized Gaussian noise;
        * no ``* sigma_data`` scaling at the end, because the bridge already
          operates in raw space;
        * the returned tensor is ``M_{t+1}`` directly.

    Args:
        model: FlowCastPrecond (or its EMA shadow) trained with the
            BridgeMatchingLoss objective.
        condition: conditioning tensor ``[B, C_cond, H, W]``.
        mu: frozen regression mean ``[B, C_target, H, W]`` -- the bridge
            prior endpoint anchor.
        num_steps: number of ODE steps (default 10, matching FlowCast paper).
        sigma_prior: scalar std of the additive Gaussian perturbation on the
            prior endpoint.
        solver: 'euler' or 'midpoint'.
        t_start, t_end: integration bounds along the flow axis. Default
            ``[0, 1]`` matches training.
        prior_channel_std: optional per-channel std multipliers applied
            elementwise to the prior noise (use to push the qpepre channel to
            a heavier or lighter prior).

    Returns:
        ``M_pred`` of shape ``mu.shape`` in raw physical units, ready to use
        as the next state directly.
    """
    device = mu.device
    dtype = mu.dtype
    shape = mu.shape

    eps = torch.randn(shape, device=device, dtype=dtype)
    if prior_channel_std is not None:
        s = torch.tensor(prior_channel_std, device=device, dtype=dtype).view(
            1, -1, 1, 1
        )
        eps = eps * s
    z = mu + sigma_prior * eps

    dt = (t_end - t_start) / num_steps

    for i in range(num_steps):
        t_i = t_start + i * dt
        t_vec = torch.full([shape[0]], t_i, device=device, dtype=dtype)
        if solver == "midpoint":
            v_half = model(z, t_vec, condition=condition)
            t_half = t_vec + 0.5 * dt
            z_half = z + v_half * (0.5 * dt)
            v = model(z_half, t_half, condition=condition)
        else:  # euler
            v = model(z, t_vec, condition=condition)
        z = z + v * dt

    return z


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
