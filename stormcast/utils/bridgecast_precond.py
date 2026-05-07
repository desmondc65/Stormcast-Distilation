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

"""BridgeCast preconditioner — dual-head SongUNet (plan §2.5, §5.3).

A thin wrapper over the same SongUNet backbone FlowCast uses, but with
``out_channels = target_channels + 1``: the first ``target_channels`` are the
velocity field v_theta(x_t, t, c) and the last channel is the auxiliary
wet/dry mask logit m_theta(x_t, t, c) (plan §2.4). Same parameter count as
FlowCast modulo a single 1x1 output kernel column — ensures all comparisons
in §7.2 are FLOP-matched.

The forward pass returns a tuple ``(v_pred, m_logits)`` where:
- ``v_pred``: (B, target_channels, H, W) — bridge velocity in standardized space.
- ``m_logits``: (B, 1, H, W) — pre-sigmoid mask logits (always returned, even
  if the mask head is unused, since the SongUNet returns a single concatenated
  output tensor and we always pay for that channel).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

import torch

from physicsnemo.models.meta import ModelMetaData
from physicsnemo.models.module import Module

network_module = importlib.import_module("physicsnemo.models.diffusion")


@dataclass
class BridgeCastPrecondMetaData(ModelMetaData):
    """BridgeCastPrecond meta data."""

    name: str = "BridgeCastPrecond"
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    bf16: bool = False
    onnx: bool = False
    func_torch: bool = False
    auto_grad: bool = False


class BridgeCastPrecond(Module):
    """Dual-head velocity + mask predictor for the mean-anchored bridge.

    Operates on standardized residual space (raw / sigma_data). The bridge
    integration loop calls ``forward`` with the current trajectory state
    ``x_t``, the time ``t in [0, 1]``, and the conditioning bundle.

    Parameters
    ----------
    img_resolution : int or tuple
        Image resolution (passed through to the SongUNet).
    img_channels : int
        Total in/out channel count used for sizing default-fallback heads.
        Prefer setting ``img_in_channels`` / ``target_channels`` explicitly.
    target_channels : int
        Number of generated channels (4 for the Taiwan RWRF setup).
    label_dim : int
        Class-conditioning dim, 0 = unconditional.
    use_fp16 : bool
        Run the inner UNet in FP16.
    sigma_data : float
        Standard deviation of the raw target residual; the loss / sampler
        normalize by this so the bridge runs in unit-variance working space.
    time_scale : float
        Multiplier applied to ``t in [0, 1]`` before feeding it to the UNet's
        positional embedding (matches the FlowCast convention; values around
        1000 stay in the diffusion-timestep regime the UNet was tuned for).
    enable_mask_head : bool
        When False, the ``m_logits`` output is still returned (zeros) but is
        not learned — used for the §4 ablation #5 that disables the mask head.
    model_type : str
        Inner UNet class name (must live in ``physicsnemo.models.diffusion``).
    img_in_channels : int, optional
        Override input channels (target + conditioning).
    **model_kwargs
        Forwarded to the underlying UNet constructor.
    """

    def __init__(
        self,
        img_resolution,
        img_channels: int,
        target_channels: int,
        label_dim: int = 0,
        use_fp16: bool = False,
        sigma_data: float = 0.5,
        time_scale: float = 1000.0,
        enable_mask_head: bool = True,
        model_type: str = "SongUNet",
        img_in_channels: int | None = None,
        **model_kwargs,
    ):
        super().__init__(meta=BridgeCastPrecondMetaData)
        self.img_resolution = img_resolution
        if img_in_channels is None:
            img_in_channels = img_channels

        self.target_channels = int(target_channels)
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_data = sigma_data
        self.time_scale = time_scale
        self.enable_mask_head = bool(enable_mask_head)

        # Reserve one extra output channel for the mask logits.
        self.mask_channel_count = 1
        out_channels = self.target_channels + self.mask_channel_count

        model_class = getattr(network_module, model_type)
        self.model = model_class(
            img_resolution=img_resolution,
            in_channels=img_in_channels,
            out_channels=out_channels,
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the bridge velocity and the wet/dry mask logits.

        Parameters
        ----------
        x : Tensor, shape (B, target_channels, H, W)
            Current state on the bridge trajectory (in standardized space).
        t : Tensor, shape (B,) or broadcastable
            Bridge time in [0, 1].
        condition : Tensor or None, shape (B, C_cond, H, W)
            Conditioning bundle to concatenate channel-wise with x.

        Returns
        -------
        v_pred : Tensor, shape (B, target_channels, H, W)
            Predicted bridge velocity in standardized space.
        m_logits : Tensor, shape (B, 1, H, W)
            Pre-sigmoid mask logits. If ``enable_mask_head`` is False this
            tensor is detached (no gradient) — the loss should also skip its
            BCE term in that case.
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

        t_embed = t * self.time_scale

        F_x = self.model(
            arg.to(dtype),
            t_embed,
            class_labels=class_labels,
            **model_kwargs,
        )
        if (F_x.dtype != dtype) and not torch.is_autocast_enabled():
            raise ValueError(
                f"Expected dtype {dtype}, got {F_x.dtype}."
            )
        F_x = F_x.to(torch.float32)

        v_pred = F_x[:, : self.target_channels]
        m_logits = F_x[:, self.target_channels : self.target_channels + 1]

        if not self.enable_mask_head:
            m_logits = m_logits.detach()

        return v_pred, m_logits
