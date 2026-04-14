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

"""Multi-scale discriminator for Adversarial Diffusion Distillation (ADD).

Standard ADD uses a pretrained vision backbone (e.g. DINOv2) designed for
3-channel RGB images.  StormCast residuals have 99 channels of 3-D atmospheric
variables, so we build a custom multi-scale feature extractor and attach
lightweight discriminator heads at multiple resolutions.

The architecture:
  1. A convolutional feature backbone that progressively downsamples the input,
     producing feature maps at multiple scales.
  2. Lightweight 1x1 conv heads attached to each intermediate feature map that
     output a scalar realness score per spatial location (PatchGAN-style).
  3. Conditioning is injected by concatenating the conditioning tensor with the
     input along the channel dimension.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block with GroupNorm and SiLU activation."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + self.skip(x))


class DiscriminatorHead(nn.Module):
    """Lightweight 1x1 head producing per-patch realness logits."""

    def __init__(self, in_ch: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1),
            nn.SiLU(),
            nn.Conv2d(in_ch, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class MultiScaleDiscriminator(nn.Module):
    """Custom multi-scale PatchGAN discriminator for 99-channel atmospheric data.

    Parameters
    ----------
    in_channels : int
        Number of input channels (residual channels, e.g. 99).
    cond_channels : int
        Number of conditioning channels (concatenated with input).
    base_ch : int
        Base channel width of the feature backbone.
    num_scales : int
        Number of downsampling stages (each produces a discriminator head).
    """

    def __init__(
        self,
        in_channels: int = 99,
        cond_channels: int = 0,
        base_ch: int = 64,
        num_scales: int = 4,
    ):
        super().__init__()
        self.num_scales = num_scales

        total_in = in_channels + cond_channels

        # Initial projection from high-dimensional input to base_ch
        self.stem = nn.Sequential(
            nn.Conv2d(total_in, base_ch, 3, padding=1),
            nn.SiLU(),
        )

        # Build backbone stages — each doubles channels and halves resolution
        self.stages = nn.ModuleList()
        self.heads = nn.ModuleList()

        ch = base_ch
        for i in range(num_scales):
            out_ch = min(ch * 2, 512)
            stage = nn.Sequential(
                ResBlock(ch, out_ch),
                ResBlock(out_ch, out_ch),
                nn.AvgPool2d(2),
            )
            self.stages.append(stage)
            self.heads.append(DiscriminatorHead(out_ch))
            ch = out_ch

    def forward(
        self, x: torch.Tensor, condition: torch.Tensor | None = None
    ) -> list[torch.Tensor]:
        """Return per-patch logits from each scale head.

        Parameters
        ----------
        x : Tensor [B, C_in, H, W]
            Input residual (real or generated).
        condition : Tensor [B, C_cond, H, W], optional
            Conditioning tensor (e.g. mu_{t+1}, M_t).

        Returns
        -------
        list[Tensor]
            List of logit tensors, one per scale.
        """
        if condition is not None:
            x = torch.cat([x, condition], dim=1)

        h = self.stem(x)
        logits = []
        for stage, head in zip(self.stages, self.heads):
            h = stage(h)
            logits.append(head(h))
        return logits

    def get_features(
        self, x: torch.Tensor, condition: torch.Tensor | None = None
    ) -> list[torch.Tensor]:
        """Return intermediate features (before heads) for R1 penalty.

        Parameters
        ----------
        x : Tensor [B, C_in, H, W]
            Input residual.
        condition : Tensor [B, C_cond, H, W], optional
            Conditioning tensor.

        Returns
        -------
        list[Tensor]
            Feature maps at each scale.
        """
        if condition is not None:
            x = torch.cat([x, condition], dim=1)

        h = self.stem(x)
        features = []
        for stage in self.stages:
            h = stage(h)
            features.append(h)
        return features
