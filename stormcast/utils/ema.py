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

"""Exponential Moving Average utility for Consistency Distillation."""

import math
import copy
import torch


class ExponentialMovingAverage:
    """Maintains an exponential moving average of model parameters.

    Shadow parameters are stored as plain tensors (not nn.Parameters)
    to avoid allocating optimizer state for them.

    Parameters
    ----------
    model : torch.nn.Module
        The model whose parameters to track.
    decay : float
        Initial EMA decay rate.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            name: param.data.clone()
            for name, param in model.named_parameters()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module, decay: float = None):
        """Update shadow parameters with exponential moving average.

        Parameters
        ----------
        model : torch.nn.Module
            The model with current parameters.
        decay : float, optional
            Override decay rate (used for adaptive schedule).
        """
        d = decay if decay is not None else self.decay
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].lerp_(param.data, 1.0 - d)

    @torch.no_grad()
    def apply_shadow(self, model: torch.nn.Module):
        """Load shadow parameters into a model for inference.

        Parameters
        ----------
        model : torch.nn.Module
            Target model to receive shadow parameters.
        """
        model_state = model.state_dict()
        for name in self.shadow:
            if name in model_state:
                model_state[name].copy_(self.shadow[name])

    def state_dict(self):
        """Return shadow parameters for checkpointing."""
        return {name: param.clone() for name, param in self.shadow.items()}

    def load_state_dict(self, state):
        """Restore shadow parameters from a checkpoint.

        Parameters
        ----------
        state : dict
            Dictionary mapping parameter names to tensors.
        """
        for name, param in state.items():
            if name in self.shadow:
                self.shadow[name].copy_(param)


def ema_decay_schedule(N_0: int, N_k: int, mu_0: float = 0.95) -> float:
    """Compute EMA decay mu(k) for Consistency Distillation.

    Per Song et al. (2023), Eq. 14:
        mu(k) = exp(s_0 * log(mu_0) / s_k)
    where s_k and s_0 depend on the number of discretization steps.

    In practice this simplifies to:
        mu(k) = mu_0 ** (N_0 / N_k)
    when using the standard schedule parameterization.

    Parameters
    ----------
    N_0 : int
        Initial number of discretization steps.
    N_k : int
        Current number of discretization steps at training step k.
    mu_0 : float
        Base EMA decay rate, by default 0.95.

    Returns
    -------
    float
        The EMA decay rate at the current training step.
    """
    if N_k <= 0:
        return mu_0
    return mu_0 ** (N_0 / N_k)
