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

"""FlowCast preconditioner.

Wraps a SongUNet to predict a flow-matching vector field v_theta(x_t, t, c)
for use with Conditional Flow Matching (Lipman et al. 2023; Tong et al. 2024).

Adapted to the StormCast residual pipeline: the target is the per-step
residual R_t = X_t - M_t produced by the frozen regression model M, and the
conditioning c bundles (X_{t-1}, S_t, M_t, I). Unlike the EDMPrecond, there
are no c_skip / c_out scales; the model output IS the velocity estimate.
"""

import importlib
from dataclasses import dataclass

import torch

from physicsnemo.models.meta import ModelMetaData
from physicsnemo.models.module import Module

network_module = importlib.import_module("physicsnemo.models.diffusion")


@dataclass
class FlowCastPrecondMetaData(ModelMetaData):
    """FlowCastPrecond meta data"""

    name: str = "FlowCastPrecond"
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


class FlowCastPrecond(Module):
    """Vector-field predictor for Conditional Flow Matching.

    The inner network (default SongUNet) is invoked as
    ``v = model(cat[x, condition], t * time_scale)`` and the return value
    is the vector field estimate v_theta(x, t, c). The flow operates in the
    *standardized* target space: training divides the raw residual by
    sigma_data before feeding the loss, and the Euler sampler multiplies the
    integrated state by sigma_data before returning it (see
    :mod:`stormcast.utils.nn.flowcast_model_forward`). Keep sigma_data matched
    to the EDM teacher's sigma_data so that conditioning statistics are
    comparable across methods.

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
        Scalar multiplier applied to t ∈ [0, 1] before feeding it into the
        SongUNet's positional/noise embedding layer. A value in the
        diffusion-timestep range (~1000) produces embeddings that vary
        sufficiently across the flow interval.
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
        model_type: str = "SongUNet",
        img_in_channels: int | None = None,
        img_out_channels: int | None = None,
        **model_kwargs,
    ):
        super().__init__(meta=FlowCastPrecondMetaData)
        self.img_resolution = img_resolution
        if img_in_channels is None:
            img_in_channels = img_channels
        if img_out_channels is None:
            img_out_channels = img_channels

        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_data = sigma_data
        self.time_scale = time_scale

        model_class = getattr(network_module, model_type)
        self.model = model_class(
            img_resolution=img_resolution,
            in_channels=img_in_channels,
            out_channels=img_out_channels,
            label_dim=label_dim,
            **model_kwargs,
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor | None = None,
        class_labels=None,
        force_fp32: bool = False,
        **model_kwargs,
    ) -> torch.Tensor:
        """Predict the flow-matching vector field.

        Parameters
        ----------
        x : Tensor, shape (B, C_out, H, W)
            Current state on the flow trajectory (in standardized space).
        t : Tensor, shape (B,) or broadcastable
            Flow time in [0, 1].
        condition : Tensor or None, shape (B, C_cond, H, W)
            Conditioning tensor to concatenate channel-wise with x.
        """
        x = x.to(torch.float32)
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

        # Feed a scaled time into the SongUNet's positional embedding. The
        # positional-embedding layer's output frequencies are tuned to the
        # diffusion-timestep regime, so scaling t into that range preserves
        # the conditioning capacity the pretrained architecture was designed
        # for.
        t_embed = t * self.time_scale

        F_x = self.model(
            arg.to(dtype),
            t_embed,
            class_labels=class_labels,
            **model_kwargs,
        )

        if (F_x.dtype != dtype) and not torch.is_autocast_enabled():
            raise ValueError(
                f"Expected the dtype to be {dtype}, but got {F_x.dtype} instead."
            )
        return F_x.to(torch.float32)
