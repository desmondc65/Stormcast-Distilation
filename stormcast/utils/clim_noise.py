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

"""Climatology-correlated noise prior for BridgeCast (new_method_plan.md §2.3).

Standard generative atmospheric models draw noise from white-isotropic
``N(0, I)``; the model then has to spend capacity inventing the spatial
autocorrelation that real residuals already have. We instead estimate the
radially-averaged power spectrum P_k(kappa) of the training residuals once,
build a per-channel Fourier amplitude filter sqrt(P_k), and sample noise as

    eps_k = IFFT[ sqrt(P_k(kappa)) * xi_k ],   xi_k ~ CN(0, I)

so eps is Gaussian with zero mean and a 2-point autocorrelation that matches
climatology. We normalize the filter so the per-channel pixel variance of the
output is exactly 1, i.e. eps is *unit-variance* coloured noise — drop-in for
``torch.randn_like`` in any sampler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import torch


def _radial_psd_estimate(field_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the radially-averaged PSD of a single (H, W) field.

    Uses an orthonormal 2-D rFFT so Parseval holds with no extra factor and
    the integrated PSD equals the field's pixel variance (after demeaning).
    """
    H, W = field_2d.shape
    f = field_2d - field_2d.mean()
    F = np.fft.rfft2(f, norm="ortho")
    P = (F.real ** 2 + F.imag ** 2)  # (H, W//2+1)

    ky = np.fft.fftfreq(H) * H
    kx = np.fft.rfftfreq(W) * W
    kyy, kxx = np.meshgrid(ky, kx, indexing="ij")
    k_int = np.sqrt(kyy ** 2 + kxx ** 2).round().astype(np.int64)

    k_max = int(k_int.max()) + 1
    Pk = np.bincount(k_int.flatten(), weights=P.flatten(), minlength=k_max)
    counts = np.bincount(k_int.flatten(), minlength=k_max).astype(np.float64)
    Pk = Pk / np.maximum(counts, 1.0)
    return np.arange(k_max), Pk


def fit_climatology_psd(
    residual_samples: np.ndarray,
    eps_floor: float = 1e-12,
) -> np.ndarray:
    """Estimate per-channel radial PSD from a stack of training residuals.

    Parameters
    ----------
    residual_samples : ndarray, shape (N, C, H, W)
        Stack of training residuals R_t = X_t - M_t. May be drawn from a
        random subset of training time steps; ``N >= 256`` is plenty for a
        smooth radial estimate at the 224x128 grid.
    eps_floor : float
        Lower clip applied to the PSD before taking sqrt — protects against
        zero-variance modes (the DC bin) creating NaN amplitudes.

    Returns
    -------
    Pk : ndarray, shape (C, k_max)
        Per-channel radially-averaged power spectrum, averaged over samples.
    """
    if residual_samples.ndim != 4:
        raise ValueError(
            f"Expected residual_samples shape (N, C, H, W), got {residual_samples.shape}"
        )
    N, C, H, W = residual_samples.shape
    Pk_list = []
    for c in range(C):
        per_channel = []
        for n in range(N):
            _, Pk_n = _radial_psd_estimate(residual_samples[n, c])
            per_channel.append(Pk_n)
        Pk_c = np.stack(per_channel, axis=0).mean(axis=0)
        Pk_list.append(np.maximum(Pk_c, eps_floor))
    return np.stack(Pk_list, axis=0)  # (C, k_max)


@dataclass
class _Buffers:
    amplitude: torch.Tensor  # (C, H, W//2+1) — FFT amplitude filter
    norm: torch.Tensor       # (C,) — per-channel scaling so output var == 1


class ClimNoiseSampler:
    """Per-channel climatology-correlated Gaussian noise sampler.

    Builds a Fourier amplitude filter ``sqrt(P_k)`` once per channel (either
    from a fitted radial PSD or from a pre-computed buffer) and provides a
    cheap ``__call__`` that draws zero-mean unit-variance correlated noise
    of any batch size.

    Parameters
    ----------
    image_shape : tuple (H, W)
        Spatial resolution of the residual fields.
    num_channels : int
        Number of channels (4 for the Taiwan RWRF setup).
    Pk_per_channel : ndarray, shape (C, k_max), optional
        Radial PSD (output of :func:`fit_climatology_psd`). If ``None`` the
        sampler falls back to white noise (sqrt(P_k) constant), which matches
        the FlowCast / EDM baselines and is useful for the §4 ablation #4.
    device : torch.device or str
        Device the buffers live on. Use ``cpu`` at construction time and
        call :meth:`to` to move once distributed init is done.
    dtype : torch.dtype
        Buffer dtype (defaults to ``float32``; the FFT operates in this
        dtype regardless of AMP).
    """

    def __init__(
        self,
        image_shape: tuple[int, int],
        num_channels: int,
        Pk_per_channel: Optional[np.ndarray] = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.image_shape = (int(image_shape[0]), int(image_shape[1]))
        self.num_channels = int(num_channels)
        self.dtype = dtype

        H, W = self.image_shape
        Wf = W // 2 + 1

        ky = torch.fft.fftfreq(H) * H
        kx = torch.fft.rfftfreq(W) * W
        kyy, kxx = torch.meshgrid(ky, kx, indexing="ij")
        k_int = torch.sqrt(kyy ** 2 + kxx ** 2).round().long()  # (H, Wf)
        self._k_int = k_int.to(device)

        if Pk_per_channel is None:
            # White noise fallback: sqrt(P_k) = 1.
            amp = torch.ones(self.num_channels, H, Wf, dtype=dtype, device=device)
        else:
            if Pk_per_channel.shape[0] != self.num_channels:
                raise ValueError(
                    f"Pk_per_channel.shape[0]={Pk_per_channel.shape[0]} "
                    f"!= num_channels={self.num_channels}"
                )
            k_max = Pk_per_channel.shape[1]
            k_clamp = torch.clamp(self._k_int, max=k_max - 1)
            amp_list = []
            for c in range(self.num_channels):
                Pk_c = torch.as_tensor(
                    Pk_per_channel[c], dtype=dtype, device=device
                )
                Pk_grid = Pk_c[k_clamp]  # (H, Wf), gather on radial bin
                amp_list.append(torch.sqrt(Pk_grid))
            amp = torch.stack(amp_list, dim=0)

        # Normalise so that pixel-variance of the sampled noise is exactly 1
        # per channel. With orthonormal IFFT, Parseval gives
        #   Var(eps) = (1/N_pix) * sum_modes |amp|^2 * (multiplicity)
        # where the rFFT half-plane convention multiplies non-Nyquist modes
        # by 2. We just compute the empirical pixel variance numerically with
        # a one-shot draw and divide — robust to convention drift in PyTorch.
        with torch.no_grad():
            test = self._raw_sample(batch=64, amplitude=amp)
            per_channel_std = test.flatten(2).std(dim=2).mean(dim=0)  # (C,)
            norm = 1.0 / per_channel_std.clamp_min(1e-8)
        self._buf = _Buffers(amplitude=amp, norm=norm.to(device=device, dtype=dtype))

    @staticmethod
    def _raw_sample(batch: int, amplitude: torch.Tensor) -> torch.Tensor:
        """Internal: sample without the per-channel unit-variance correction."""
        C, H, Wf = amplitude.shape
        W = (Wf - 1) * 2
        # Complex Gaussian frequency-domain noise.
        re = torch.randn(batch, C, H, Wf, dtype=amplitude.dtype, device=amplitude.device)
        im = torch.randn(batch, C, H, Wf, dtype=amplitude.dtype, device=amplitude.device)
        Z = torch.complex(re, im) * (1.0 / np.sqrt(2.0))
        # Apply per-channel amplitude filter (broadcast over batch).
        Z = Z * amplitude.unsqueeze(0).to(Z.real.dtype)
        eps = torch.fft.irfft2(Z, s=(H, W), norm="ortho")
        return eps  # (B, C, H, W)

    def to(self, device: torch.device | str) -> "ClimNoiseSampler":
        """Move all buffers to the given device, in-place."""
        self._buf = _Buffers(
            amplitude=self._buf.amplitude.to(device),
            norm=self._buf.norm.to(device),
        )
        self._k_int = self._k_int.to(device)
        return self

    @property
    def device(self) -> torch.device:
        return self._buf.amplitude.device

    def sample(self, batch: int) -> torch.Tensor:
        """Draw ``(batch, C, H, W)`` noise with per-channel unit pixel variance.

        Output is zero-mean by construction and channel-wise variance is
        exactly 1; the spatial autocorrelation matches the fitted PSD.
        """
        eps = self._raw_sample(batch=batch, amplitude=self._buf.amplitude)
        eps = eps * self._buf.norm.view(1, -1, 1, 1)
        return eps

    def sample_like(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: draw correlated noise matching ``x.shape`` and dtype."""
        if x.dim() != 4:
            raise ValueError(f"Expected x.dim()==4 (B,C,H,W), got {x.shape}")
        B, C, H, W = x.shape
        if (C, H, W) != (self.num_channels, *self.image_shape):
            raise ValueError(
                f"Sampler shape ({self.num_channels}, *{self.image_shape}) "
                f"does not match x ({C}, {H}, {W})."
            )
        eps = self.sample(batch=B)
        return eps.to(dtype=x.dtype)


def estimate_psd_from_dataloader(
    dataset,
    regression_net,
    invariant_tensor,
    build_condition_fn,
    condition_list,
    regression_condition_list,
    num_samples: int = 512,
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> np.ndarray:
    """One-pass estimator of per-channel residual PSD from a torch Dataset.

    Walks ``num_samples`` random indices of ``dataset``, runs each batch
    through the frozen regression net to compute ``R_t = X_t - M_t``, and
    averages the radial PSD. Returns ``Pk_per_channel`` of shape (C, k_max)
    suitable for :class:`ClimNoiseSampler`.

    The regression net is expected to be in eval mode and on ``device``;
    ``invariant_tensor`` should already be on ``device``.
    """
    rng = np.random.RandomState(seed)
    n = len(dataset)
    indices = rng.choice(n, size=min(num_samples, n), replace=False)

    residuals = []
    with torch.no_grad():
        for idx in indices:
            sample = dataset[int(idx)]
            background = sample["background"]
            state_prev, state_cur = sample["state"]
            background = torch.as_tensor(background, device=device).unsqueeze(0).float()
            state_prev = torch.as_tensor(state_prev, device=device).unsqueeze(0).float()
            state_cur = torch.as_tensor(state_cur, device=device).unsqueeze(0).float()

            inv = invariant_tensor[:1] if invariant_tensor is not None else None
            (_cond, _target, reg_out) = build_condition_fn(
                background,
                (state_prev, state_cur),
                inv,
                regression_net=regression_net,
                condition_list=condition_list,
                regression_condition_list=regression_condition_list,
            )
            if reg_out is None:
                # No regression conditioning → use the raw target itself,
                # which collapses to the (un-anchored) FlowCast prior.
                R = state_cur
            else:
                R = state_cur - reg_out
            residuals.append(R.detach().cpu().numpy()[0])

    residuals = np.stack(residuals, axis=0)  # (N, C, H, W)
    return fit_climatology_psd(residuals)
