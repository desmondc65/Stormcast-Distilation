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

"""MeanFlow preconditioner.

Wraps a SongUNet to predict the *average* velocity field u_theta(z_r, r, t, c)
of a flow-matching trajectory (Geng et al. 2025, "Mean Flows for One-step
Generative Modeling"), defined as

    u(z_r, r, t) = 1/(t - r) * int_r^t v(z_tau, tau) dtau ,

so a single network evaluation transports the state across the whole
interval: z_t = z_r + (t - r) * u(z_r, r, t). At r == t the average velocity
collapses to the instantaneous one, making this a strict generalization of
the FlowCast vector field on the same backbone.

Adapted to the StormCast residual pipeline exactly like FlowCastPrecond: the
target is the per-step residual r_{t+1} = M_{t+1} - mu_{t+1} of the frozen
regression model, the conditioning c bundles (M_t, mu_{t+1}, I), and the flow
runs in standardized residual space (z = R / sigma_data) from t=0 (noise) to
t=1 (data).

The two times are injected into the SongUNet as follows: the state time r
goes through the usual noise-label positional embedding (scaled by
``time_scale`` like FlowCast), and the interval length (t - r) enters via the
SongUNet's ``augment_labels`` pathway as fixed sin/cos Fourier features.
"""

import importlib
import math
from dataclasses import dataclass

import torch

from physicsnemo.models.meta import ModelMetaData
from physicsnemo.models.module import Module

network_module = importlib.import_module("physicsnemo.models.diffusion")


@dataclass
class MeanFlowPrecondMetaData(ModelMetaData):
    """MeanFlowPrecond meta data"""

    name: str = "MeanFlowPrecond"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    # Data type
    bf16: bool = False
    # Inference
    onnx: bool = False
    # Physics informed
    func_torch: bool = False
    auto_grad: bool = False


class MeanFlowPrecond(Module):
    """Average-velocity predictor for MeanFlow on the StormCast residual.

    The inner network (default SongUNet) is invoked as
    ``u = model(cat[x, condition], r * time_scale, augment_labels=FF(t - r))``
    and the return value is the average velocity u_theta(x, r, t, c) over the
    interval [r, t]. Querying with t == r yields the instantaneous vector
    field, so the model can also be integrated with a many-step Euler solver
    like FlowCast.

    Parameters
    ----------
    img_resolution : int or tuple
        Image resolution passed to the SongUNet.
    img_channels : int
        Default channel count used when both ``img_in_channels`` and
        ``img_out_channels`` are unset. For this residual setup, prefer
        passing the two explicitly.
    label_dim : int
        Number of class labels, 0 = unconditional. By default 0.
    use_fp16 : bool
        Run the underlying SongUNet in FP16. By default False.
    sigma_data : float
        Standard deviation of the (raw) target residual. Used only as a
        reference by the loss / sampler to normalize inputs and de-normalize
        outputs; the preconditioner itself does not apply scaling.
    time_scale : float
        Scalar multiplier applied to r in [0, 1] before feeding it into the
        SongUNet's positional/noise embedding layer (same convention as
        FlowCastPrecond).
    gap_embed_dim : int
        Width of the sin/cos Fourier embedding of the interval length
        (t - r), injected through the SongUNet ``augment_labels`` pathway.
        Must be even.
    model_type : str
        Underlying UNet class name (must live in ``physicsnemo.models.diffusion``).
    img_in_channels : int, optional
        Override input channels (target + conditioning).
    img_out_channels : int, optional
        Override output channels (must equal target channels).
    **model_kwargs : dict
        Forwarded to the underlying UNet constructor.
    """

    def __init__(
        self,
        img_resolution,
        img_channels,
        label_dim: int = 0,
        use_fp16: bool = False,
        sigma_data: float = 0.5,
        time_scale: float = 1000.0,
        gap_embed_dim: int = 32,
        model_type: str = "SongUNet",
        img_in_channels: int | None = None,
        img_out_channels: int | None = None,
        **model_kwargs,
    ):
        super().__init__(meta=MeanFlowPrecondMetaData)
        self.img_resolution = img_resolution
        if img_in_channels is None:
            img_in_channels = img_channels
        if img_out_channels is None:
            img_out_channels = img_channels

        if gap_embed_dim % 2 != 0:
            raise ValueError(f"gap_embed_dim must be even, got {gap_embed_dim}")

        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_data = sigma_data
        self.time_scale = time_scale
        self.gap_embed_dim = gap_embed_dim

        # Fixed log-spaced frequencies for the (t - r) Fourier features. The
        # gap lives in [0, 1]; frequencies up to ~time_scale give the linear
        # map_augment layer enough resolution across the whole interval.
        n_freq = gap_embed_dim // 2
        freqs = torch.exp(
            torch.linspace(math.log(1.0), math.log(1000.0), n_freq)
        )
        self.register_buffer("gap_freqs", freqs)

        model_class = getattr(network_module, model_type)
        self.model = model_class(
            img_resolution=img_resolution,
            in_channels=img_in_channels,
            out_channels=img_out_channels,
            label_dim=label_dim,
            augment_dim=gap_embed_dim,
            **model_kwargs,
        )

    def forward(
        self,
        x: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor | None = None,
        class_labels=None,
        force_fp32: bool = False,
        **model_kwargs,
    ) -> torch.Tensor:
        """Predict the average velocity over the interval [r, t].

        Parameters
        ----------
        x : Tensor, shape (B, C_out, H, W)
            Current state z_r on the flow trajectory (standardized space).
        r : Tensor, shape (B,) or broadcastable
            Time of the current state, in [0, 1].
        t : Tensor, shape (B,) or broadcastable
            End time of the averaging interval, in [r, 1]. t == r recovers
            the instantaneous velocity.
        condition : Tensor or None, shape (B, C_cond, H, W)
            Conditioning tensor to concatenate channel-wise with x.
        """
        x = x.to(torch.float32)
        r = r.to(torch.float32).reshape(-1)
        t = t.to(torch.float32).reshape(-1)

        class_labels = (
            None
            if self.label_dim == 0
            else torch.zeros([1, self.label_dim], device=x.device)
            if class_labels is None
            else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        )
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        if condition is not None:
            arg = torch.cat([x, condition], dim=1)
        else:
            arg = x

        # State time r through the noise embedding (FlowCast convention);
        # interval length through sin/cos features into map_augment.
        r_embed = r * self.time_scale
        ang = (t - r).unsqueeze(1) * self.gap_freqs.unsqueeze(0)
        augment_labels = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)

        F_x = self.model(
            arg.to(dtype),
            r_embed,
            class_labels=class_labels,
            augment_labels=augment_labels.to(dtype),
            **model_kwargs,
        )

        if (F_x.dtype != dtype) and not torch.is_autocast_enabled():
            raise ValueError(
                f"Expected the dtype to be {dtype}, but got {F_x.dtype} instead."
            )
        return F_x.to(torch.float32)
