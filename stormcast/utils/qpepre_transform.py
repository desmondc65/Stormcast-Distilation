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

"""Channel-adaptive precipitation transform for BridgeCast (plan §2.4).

Implements the asinh-compressed bridge target for the qpepre channel and a
rain-mask gating helper. asinh(q/kappa) is linear near 0 (preserving the large
mass of zero/light rain) and logarithmic in the tail (compressing extremes),
which keeps the bridge target velocity well-conditioned on a heavy-tailed
sparse field. The same scale kappa is used for forward and inverse so the
transform is exactly invertible.
"""

from __future__ import annotations

import torch


def asinh_compress(x: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
    """Forward transform: y = asinh(x / kappa).

    Behaves linearly with slope 1/kappa for |x| << kappa and logarithmically
    for |x| >> kappa. Preserves sign so it is safe to apply to a residual
    (which can be negative).
    """
    return torch.asinh(x / kappa)


def asinh_decompress(y: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
    """Inverse of :func:`asinh_compress`: x = kappa * sinh(y)."""
    return kappa * torch.sinh(y)


def _replace_channel(x: torch.Tensor, idx: int, new_channel: torch.Tensor) -> torch.Tensor:
    """Return a new (B, C, H, W) tensor with channel ``idx`` replaced.

    Built by stacking — autograd-safe (no in-place mutations on slices),
    so it composes cleanly inside the velocity-matching loss where multiple
    branches consume the predicted state.
    """
    channels = [x[:, c] if c != idx else new_channel for c in range(x.shape[1])]
    return torch.stack(channels, dim=1)


def channel_asinh_forward(
    x: torch.Tensor,
    qpepre_idx: int,
    kappa: float = 1.0,
) -> torch.Tensor:
    """Apply asinh forward only on the qpepre channel of a (B, C, H, W) tensor.

    Returns a NEW tensor without any in-place mutation; safe inside autograd
    graphs that reuse ``x`` elsewhere. ``qpepre_idx < 0`` is a no-op.
    """
    if qpepre_idx < 0:
        return x
    new_q = asinh_compress(x[:, qpepre_idx], kappa=kappa)
    return _replace_channel(x, qpepre_idx, new_q)


def channel_asinh_inverse(
    y: torch.Tensor,
    qpepre_idx: int,
    kappa: float = 1.0,
) -> torch.Tensor:
    """Apply asinh inverse only on the qpepre channel of a (B, C, H, W) tensor.

    Returns a NEW tensor without any in-place mutation. ``qpepre_idx < 0``
    is a no-op.
    """
    if qpepre_idx < 0:
        return y
    new_q = asinh_decompress(y[:, qpepre_idx], kappa=kappa)
    return _replace_channel(y, qpepre_idx, new_q)


def softplus_nonneg(x: torch.Tensor, beta: float = 5.0) -> torch.Tensor:
    """Smooth non-negativity gate.

    For training stability we use softplus rather than ``relu`` so gradients
    flow through the negative-input region during early training. As ``beta``
    grows the function approaches ``relu``. Default beta=5 gives a knee
    width of ~0.2 in unit-variance space.
    """
    return torch.nn.functional.softplus(x, beta=beta)


def apply_rain_mask(
    x: torch.Tensor,
    mask_logits: torch.Tensor,
    qpepre_idx: int,
    threshold: float = 0.5,
    hard: bool = True,
) -> torch.Tensor:
    """Gate qpepre channel by a per-pixel rain/no-rain mask.

    Parameters
    ----------
    x : Tensor, shape (B, C, H, W)
        Decoded prediction in physical units.
    mask_logits : Tensor, shape (B, 1, H, W) or (B, H, W)
        Output of the auxiliary mask head (pre-sigmoid).
    qpepre_idx : int
        Channel index of qpepre. ``< 0`` disables masking.
    threshold : float
        Probability above which a pixel is considered "rain".
    hard : bool
        Whether to apply a hard zero-or-passthrough gate (inference) or a
        soft sigmoid-multiplication (training, differentiable).

    Returns
    -------
    Tensor, same shape as ``x`` — qpepre channel multiplied by the mask;
    other channels left unchanged.
    """
    if qpepre_idx < 0:
        return x

    if mask_logits.dim() == 4 and mask_logits.shape[1] == 1:
        m = mask_logits[:, 0]
    elif mask_logits.dim() == 3:
        m = mask_logits
    else:
        raise ValueError(
            f"Expected mask_logits of shape (B,1,H,W) or (B,H,W), "
            f"got {mask_logits.shape}"
        )

    if hard:
        gate = (torch.sigmoid(m) > threshold).to(x.dtype)
    else:
        gate = torch.sigmoid(m).to(x.dtype)

    new_q = x[:, qpepre_idx] * gate
    return _replace_channel(x, qpepre_idx, new_q)


def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Focal binary cross-entropy with logits (Lin et al. 2017).

    The wet/dry split for qpepre on this dataset is heavily imbalanced
    (>90% dry pixels) — focal loss down-weights easy negatives so the
    network actually learns the rain edges. ``gamma=2`` is the original
    paper default; ``alpha`` balances positive vs negative class weight.
    """
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    p = torch.sigmoid(logits)
    pt = torch.where(targets > 0.5, p, 1.0 - p)
    alpha_t = torch.where(
        targets > 0.5,
        torch.full_like(targets, alpha),
        torch.full_like(targets, 1.0 - alpha),
    )
    return (alpha_t * (1.0 - pt).pow(gamma) * bce).mean()
