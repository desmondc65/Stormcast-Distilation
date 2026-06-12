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

"""Gated-Spectral MeanFlow preconditioner (GS-MeanFlow).

A drop-in successor to ``MeanFlowPrecond`` that adds the two MVP pillars of the
GS-MeanFlow design (CLAUDE.md, "novel meanflow architecture" thread):

1. **Average-velocity flow** -- identical to ``MeanFlowPrecond``: a SongUNet
   predicts the average velocity u_theta(z_r, r, t, c) of the residual flow, so
   one evaluation transports the whole interval (z_t = z_r + (t - r) u). This
   models the 4-channel residual r_{t+1} = M_{t+1} - mu_{t+1} exactly as the
   MeanFlow / FlowCast students do.

2. **Occurrence gate (hurdle head)** -- a small, deterministic U-Net that maps
   the conditioning bundle ``c`` (NOT the noisy flow state) to a single
   per-pixel logit, the probability that qpepre is *wet* at t+1. At inference
   the gate forces confidently-dry pixels to exact zero precipitation, which a
   continuous transport from N(0, I) structurally cannot do (it smears the
   dry/wet boundary -- the documented "always slightly wet" failure mode,
   CLAUDE.md S8). The gate costs one extra forward of a network ~50-100x
   smaller than the flow backbone, so sampling stays in the MeanFlow 1-2 NFE
   regime.

The channel-decoupling pillar lives in the *loss* (group-wise adaptive
weighting), not here, so this module stays a clean superset of MeanFlowPrecond:
calling ``forward(... , return_gate=False)`` reproduces the MeanFlow vector
field bit-for-bit, and the JVP target pass in the loss uses exactly that path.
"""

import importlib
import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from physicsnemo.models.meta import ModelMetaData
from physicsnemo.models.module import Module

network_module = importlib.import_module("physicsnemo.models.diffusion")


# ---------------------------------------------------------------------------
# Lightweight occurrence-gate backbone
# ---------------------------------------------------------------------------
class _ConvBlock(nn.Module):
    """Two 3x3 convs with GroupNorm + SiLU, residual when shapes match."""

    def __init__(self, in_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        g1 = math.gcd(groups, in_ch)
        g2 = math.gcd(groups, out_ch)
        self.norm1 = nn.GroupNorm(g1, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(g2, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = (
            nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)
        )

    def forward(self, x):
        h = self.conv1(torch.nn.functional.silu(self.norm1(x)))
        h = self.conv2(torch.nn.functional.silu(self.norm2(h)))
        return h + self.skip(x)


class _GateUNet(nn.Module):
    """Compact 2-level U-Net mapping conditioning -> single occurrence logit.

    Deliberately small (a few hundred K parameters): precip occurrence is a
    largely synoptic, low-frequency decision well within reach of a shallow
    encoder-decoder with enough receptive field, and the whole point of the
    gate is that it is nearly free relative to the SongUNet flow.
    """

    def __init__(self, in_ch: int, base: int = 48, out_ch: int = 1):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, base, 3, padding=1)
        self.enc1 = _ConvBlock(base, base)
        self.down1 = nn.Conv2d(base, base * 2, 3, stride=2, padding=1)
        self.enc2 = _ConvBlock(base * 2, base * 2)
        self.down2 = nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1)
        self.mid = _ConvBlock(base * 4, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _ConvBlock(base * 2, base)
        self.head = nn.Conv2d(base, out_ch, 1)
        # Bias the gate towards "dry" at init (precip is sparse) so early
        # training is not dominated by false-wet gradients.
        nn.init.constant_(self.head.bias, -2.0)

    @staticmethod
    def _match(x, ref):
        """Center-crop / pad x to ref's spatial size (odd dims after stride)."""
        dh = ref.shape[-2] - x.shape[-2]
        dw = ref.shape[-1] - x.shape[-1]
        if dh != 0 or dw != 0:
            x = torch.nn.functional.pad(x, [0, dw, 0, dh])
        return x

    def forward(self, c):
        s = self.stem(c)
        e1 = self.enc1(s)
        e2 = self.enc2(self.down1(e1))
        m = self.mid(self.down2(e2))
        d2 = self.up2(m)
        d2 = self.dec2(torch.cat([self._match(d2, e2), e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([self._match(d1, e1), e1], dim=1))
        return self.head(d1)


@dataclass
class GSMeanFlowPrecondMetaData(ModelMetaData):
    """GSMeanFlowPrecond meta data"""

    name: str = "GSMeanFlowPrecond"
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


class GSMeanFlowPrecond(Module):
    """Average-velocity predictor + occurrence gate on the StormCast residual.

    Parameters
    ----------
    img_resolution : int or tuple
        Image resolution passed to the SongUNet (and the gate U-Net).
    img_channels : int
        Default channel count used when both ``img_in_channels`` and
        ``img_out_channels`` are unset. Prefer passing the two explicitly.
    cond_channels : int
        Number of *conditioning* channels (the bundle ``c`` alone, without the
        flow state). This is the input width of the occurrence gate. If None it
        is inferred as ``img_in_channels - img_out_channels``.
    label_dim, use_fp16, sigma_data, time_scale, gap_embed_dim, model_type :
        Same meaning as ``MeanFlowPrecond``.
    gate_base_channels : int
        Base width of the occurrence-gate U-Net (its capacity knob).
    img_in_channels, img_out_channels : int, optional
        Override the flow's input (target + conditioning) / output (target)
        channels.
    **model_kwargs :
        Forwarded to the underlying SongUNet flow backbone.
    """

    def __init__(
        self,
        img_resolution,
        img_channels,
        cond_channels: int | None = None,
        label_dim: int = 0,
        use_fp16: bool = False,
        sigma_data: float = 0.5,
        time_scale: float = 1000.0,
        gap_embed_dim: int = 32,
        model_type: str = "SongUNet",
        gate_base_channels: int = 48,
        img_in_channels: int | None = None,
        img_out_channels: int | None = None,
        **model_kwargs,
    ):
        super().__init__(meta=GSMeanFlowPrecondMetaData)
        self.img_resolution = img_resolution
        if img_in_channels is None:
            img_in_channels = img_channels
        if img_out_channels is None:
            img_out_channels = img_channels
        if cond_channels is None:
            cond_channels = img_in_channels - img_out_channels

        if gap_embed_dim % 2 != 0:
            raise ValueError(f"gap_embed_dim must be even, got {gap_embed_dim}")
        if cond_channels <= 0:
            raise ValueError(
                f"cond_channels must be positive (got {cond_channels}); the "
                "occurrence gate needs the conditioning bundle as input."
            )

        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_data = sigma_data
        self.time_scale = time_scale
        self.gap_embed_dim = gap_embed_dim
        self.cond_channels = cond_channels
        self.img_out_channels = img_out_channels

        # Fixed log-spaced frequencies for the (t - r) Fourier features (same
        # convention as MeanFlowPrecond).
        n_freq = gap_embed_dim // 2
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), n_freq))
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

        # Occurrence gate: deterministic c -> 1 logit. Kept in float32; it is
        # tiny and the BCE/focal target benefits from full precision.
        self.gate = _GateUNet(cond_channels, base=gate_base_channels, out_ch=1)

    # ------------------------------------------------------------------
    def gate_logits(self, condition: torch.Tensor) -> torch.Tensor:
        """Per-pixel wet/dry logit for qpepre, shape (B, 1, H, W).

        Depends only on the conditioning bundle, never on the flow state, so it
        is a clean deterministic occurrence model (the "hurdle").
        """
        return self.gate(condition.to(torch.float32))

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor | None = None,
        class_labels=None,
        force_fp32: bool = False,
        return_gate: bool = False,
        **model_kwargs,
    ):
        """Predict the average velocity over [r, t] (and optionally the gate).

        With ``return_gate=False`` (the default, and what the loss's JVP pass
        uses) this is bit-for-bit the MeanFlowPrecond vector field. With
        ``return_gate=True`` it additionally returns the occurrence logits so a
        single DDP forward exercises both the flow and the gate parameters
        (keeping gradient all-reduce correct without find_unused_parameters).
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

        # State time r through the noise embedding; interval length (t - r)
        # through sin/cos features into map_augment.
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
        velocity = F_x.to(torch.float32)

        if return_gate:
            if condition is None:
                raise ValueError("return_gate=True requires a conditioning tensor.")
            return velocity, self.gate_logits(condition)
        return velocity
