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

"""Factory + samplers for Gated-Spectral MeanFlow (GS-MeanFlow).

Kept out of ``utils/nn.py`` so the shared factory stays untouched; the GS
trainer / inference scripts import the builder and sampler from here while still
reusing ``utils.nn.build_network_condition_and_target`` for conditioning.
"""

import math

import torch

from gsmeanflow_precond import GSMeanFlowPrecond


def get_gsmeanflow_architecture(
    target_channels: int,
    conditional_channels: int = 0,
    spatial_embedding: bool = True,
    img_resolution: tuple = (512, 640),
    attn_resolutions: list = [],
    gate_base_channels: int = 48,
) -> GSMeanFlowPrecond:
    """Build a GSMeanFlowPrecond with the same SongUNet flow backbone the EDM /
    FlowCast / MeanFlow heads use, plus the occurrence gate.

    Args mirror ``utils.nn.get_preconditioned_architecture``; ``cond_channels``
    is passed explicitly so the gate knows the width of the conditioning bundle.
    """
    return GSMeanFlowPrecond(
        img_resolution=img_resolution,
        img_channels=target_channels + conditional_channels,
        img_in_channels=target_channels + conditional_channels,
        img_out_channels=target_channels,
        cond_channels=conditional_channels,
        gate_base_channels=gate_base_channels,
        model_type="SongUNet",
        channel_mult=[1, 2, 2, 2, 2],
        attn_resolutions=attn_resolutions,
        additive_pos_embed=spatial_embedding,
    )


def qpepre_standardized_levels(dataset, threshold_mm: float = 0.1):
    """Map physical qpepre levels into the dataset-standardized space.

    The gate decides wet/dry on the *standardized* field the network sees, so
    the physical wet threshold (e.g. 0.1 mm/h) and the dry value (0 mm/h) must
    be converted through the same log1p (optional) + standard-score pipeline the
    loader applies.

    Returns
    -------
    (q_idx, wet_threshold_std, dry_value_std) : (int, float, float)
        ``q_idx`` is the qpepre target-channel index; the two floats are the
        standardized wet threshold and the standardized value of zero precip.
    """
    state_channels = dataset.state_channels()
    q_idx = state_channels.index("qpepre")
    mean_q = float(dataset.means_HighRes[q_idx, 0, 0])
    std_q = float(dataset.stds_HighRes[q_idx, 0, 0])
    log1p = bool(getattr(dataset, "qpepre_log1p", True))

    def to_std(phys: float) -> float:
        val = math.log1p(phys) if log1p else phys
        return (val - mean_q) / std_q

    return q_idx, to_std(float(threshold_mm)), to_std(0.0)


def gsmeanflow_model_forward(
    model,
    condition,
    shape,
    num_steps: int = 2,
    sigma_data: float = 0.5,
    t_start: float = 0.0,
    t_end: float = 1.0,
):
    """Sample the GS-MeanFlow residual and the occurrence logits.

    The residual is produced exactly like ``meanflow_model_forward`` (one or a
    few average-velocity segments). The occurrence gate is evaluated once on the
    conditioning. Composing the gate into the final field (forcing dry pixels to
    exact zero) is done by ``apply_occurrence_gate`` on the *standardized*
    ``mu_{t+1} + residual`` so the subsequent ``denormalize_state`` maps dry
    pixels to 0 mm/h exactly.

    Returns
    -------
    (residual, gate_logits) :
        ``residual`` in raw (de-standardized by sigma_data) units, shape
        ``shape``; ``gate_logits`` shape ``(B, 1, H, W)``.
    """
    device = condition.device
    dtype = condition.dtype

    z = torch.randn(*shape, device=device, dtype=dtype)
    dt = (t_end - t_start) / num_steps

    for i in range(num_steps):
        r_i = t_start + i * dt
        t_i = t_start + (i + 1) * dt
        r_vec = torch.full([shape[0]], r_i, device=device, dtype=dtype)
        t_vec = torch.full([shape[0]], t_i, device=device, dtype=dtype)
        u = model(z, r_vec, t_vec, condition=condition)
        z = z + u * dt

    gate_logits = model.gate_logits(condition)
    return z * sigma_data, gate_logits


def apply_occurrence_gate(
    state_std,
    gate_logits,
    q_idx: int,
    dry_value: float,
    threshold: float = 0.5,
):
    """Force confidently-dry pixels of the qpepre channel to exact zero precip.

    Operates in place on the standardized state ``mu_{t+1} + residual`` and
    returns it along with the wet probability map. Pixels whose gate probability
    is below ``threshold`` are set to ``dry_value`` (the standardized value of
    0 mm/h) so ``denormalize_state`` yields exactly 0 mm/h there.

    Parameters
    ----------
    state_std : Tensor, shape (B, C, H, W)
        Standardized predicted field (modified in place on the qpepre channel).
    gate_logits : Tensor, shape (B, 1, H, W)
        Raw occurrence logits from the gate.
    q_idx : int
        qpepre channel index.
    dry_value : float
        Standardized value corresponding to 0 mm/h.
    threshold : float
        Wet-probability decision boundary (default 0.5).
    """
    prob = torch.sigmoid(gate_logits)[:, 0]  # (B, H, W)
    dry = prob < threshold
    chan = state_std[:, q_idx]
    chan[dry] = dry_value
    state_std[:, q_idx] = chan
    return state_std, prob
