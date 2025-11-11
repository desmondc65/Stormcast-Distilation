# spectrum.py

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

import torch
import numpy as np
import matplotlib.pyplot as plt
from physicsnemo.metrics.general.power_spectrum import power_spectrum


def ps1d_plots(generated: torch.Tensor,
               target: torch.Tensor,
               fields: list,
               diffusion_channels: list):
    """
    Compute and plot 1D power spectra for all channels in `fields`.

    Args
    ----
    generated : (C,H,W) tensor
    target    : (C,H,W) tensor
    fields    : list[str] of channel names to plot/log
    diffusion_channels : full ordered list of channel names (len == C)

    Returns
    -------
    figs : dict[str, matplotlib.figure.Figure]
        Figures keyed by "PS1D_{field}".
    ratios : dict[str, float]
        Mean ratio of Pk_gen / Pk_tar between k~[10, 320] for each field,
        keyed as "specratio_{field}".
    numeric : dict
        {
          "k": np.ndarray (L,),
          "Pk_gen": np.ndarray (C,L),
          "Pk_tar": np.ndarray (C,L),
          "ratio_all": np.ndarray (C,L),
        }
    """
    assert generated.shape == target.shape, "generated and target must have same shape"

    generated = generated.detach()
    target = target.detach()

    with torch.no_grad():
        k, Pk_gen = power_spectrum(generated)  # k: (L,), Pk_gen: (C,L)
        _, Pk_tar = power_spectrum(target)     # Pk_tar: (C,L)

        k = k.detach().cpu().numpy()
        Pk_gen = Pk_gen.detach().cpu().numpy()
        Pk_tar = Pk_tar.detach().cpu().numpy()

    figs = {}
    ratios = {}
    ratio_all = np.divide(Pk_gen, Pk_tar, out=np.zeros_like(Pk_gen), where=(Pk_tar != 0))

    lo = np.argmin(np.abs(k - 10.0))
    hi = np.argmin(np.abs(k - 320.0))
    hi = max(hi, lo + 1)

    for _f in fields:
        cidx = diffusion_channels.index(_f)
        f, (a0, a1) = plt.subplots(
            2, 1, gridspec_kw={"height_ratios": [2, 1], "hspace": 0}, figsize=(6, 4)
        )
        a0.plot(k, Pk_tar[cidx], "k-", label=_f)
        a0.plot(k, Pk_gen[cidx], "r-", label="prediction")
        a0.set_yscale("log")
        a0.set_xscale("log")
        a0.set_xlabel("Wavenumber")
        a0.set_ylabel("PS1D")
        a0.tick_params(axis="x", direction="in", labelbottom=False, which="both")
        a0.tick_params(axis="x", length=5, which="major")
        a0.tick_params(axis="x", length=3, which="minor")
        a0.legend()

        ratio = ratio_all[cidx]
        a1.plot(k, ratio, "r-")
        a1.plot(k, np.ones_like(k), "k--")
        a1.set_xlabel("Wavenumber")
        a1.set_ylabel("Ratio")
        a1.set_xscale("log")
        a1.set_ylim((0, 2))
        a1.minorticks_on()
        a1.tick_params(axis="x", top=True, direction="inout", labeltop=False, which="both")
        a1.tick_params(axis="x", length=5, which="major")
        a1.tick_params(axis="x", length=3, which="minor")

        figs["PS1D_" + _f] = f
        ratios["specratio_" + _f] = float(np.mean(ratio[lo:hi]))

    numeric = {
        "k": k,
        "Pk_gen": Pk_gen,
        "Pk_tar": Pk_tar,
        "ratio_all": ratio_all,
    }
    return figs, ratios, numeric
