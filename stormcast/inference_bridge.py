# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Autoregressive inference for the anchored Stochastic-Interpolant Bridge.

Mirrors ``inference_flowcast.py`` but swaps the noise-to-residual sampler
for the bridge sampler ``bridge_model_forward`` that integrates from the
regression mean directly to the next-state field::

    mu_{t+1} = F_theta(M_t, S_t, I)
    M_{t+1}  = BridgeSample(mu_{t+1}, condition)     # NOT mu + residual

The student weights are loaded from the EMA shadow saved during training
(``<rundir>/ema_state.pt``).

qpepre is automatically returned in mm/h: clean_zarr.py stores it as
``log1p(mm/h)`` and the data loader's ``denormalize_state`` applies
``expm1`` as part of de-normalisation. No special handling here.
"""

import matplotlib.pyplot as plt
import torch
from datetime import datetime
import pandas as pd
import hydra
from physicsnemo.distributed import DistributedManager
from omegaconf import DictConfig
from physicsnemo.models import Module

from datasets import dataset_classes
from utils.io import (
    init_inference_results_zarr,
    write_inference_results_zarr,
    save_inference_results_netcdf,
)
from utils.nn import (
    build_network_condition_and_target,
    bridge_model_forward,
    get_preconditioned_architecture,
)
from utils.plots import inference_plot


def _load_bridge_student(cfg, dataset, invariant_tensor, device):
    """Construct a FlowCastPrecond (used as the bridge backbone) and load the
    EMA shadow into it. Same loader contract as inference_flowcast.
    """
    state_channels = dataset.state_channels()
    background_channels = dataset.background_channels()

    condition_list = list(cfg.model.diffusion_conditions)
    if "regression" not in condition_list:
        raise ValueError(
            "Bridge inference requires 'regression' in model.diffusion_conditions."
        )

    num_condition_channels = {
        "state": len(state_channels),
        "background": len(background_channels),
        "regression": len(state_channels),
        "invariant": 0 if invariant_tensor is None else invariant_tensor.shape[1],
    }
    num_condition_channels = sum(num_condition_channels[c] for c in condition_list)

    arch_kwargs = dict(
        img_resolution=dataset.image_shape(),
        target_channels=len(state_channels),
        conditional_channels=num_condition_channels,
        spatial_embedding=cfg.model.spatial_pos_embed,
        attn_resolutions=list(cfg.model.attn_resolutions),
    )

    net = get_preconditioned_architecture(name="bridge", **arch_kwargs)
    net = net.to(device).eval().requires_grad_(False)
    net.sigma_data = float(cfg.model.sigma_data)
    net.time_scale = float(cfg.model.time_scale)

    ema_path = cfg.inference.bridge_ema_path
    print(f"[bridge] loading EMA shadow weights from {ema_path}")
    ema_state = torch.load(ema_path, map_location=device)
    if "shadow_params" in ema_state and "param_names" in ema_state:
        names = ema_state["param_names"]
        shadows = ema_state["shadow_params"]
        target_state = dict(net.state_dict())
        for i, name in enumerate(names):
            if name in target_state:
                target_state[name] = shadows[i].to(target_state[name].dtype)
        net.load_state_dict(target_state, strict=False)
    else:
        net.load_state_dict(ema_state, strict=False)

    return net


@hydra.main(version_base=None, config_path="config", config_name="bridge_inference")
def main(cfg: DictConfig):
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device

    initial_time = datetime.fromisoformat(cfg.inference.initial_time)
    n_steps = cfg.inference.n_steps

    dataset_cls = dataset_classes[cfg.dataset.name]
    dataset = dataset_cls(cfg.dataset, train=False)

    background_channels = dataset.background_channels()
    state_channels = dataset.state_channels()

    invariant_array = dataset.get_invariants()
    invariant_tensor = (
        torch.from_numpy(invariant_array).to(device).repeat(1, 1, 1, 1)
        if invariant_array is not None
        else None
    )

    if len(cfg.inference.output_state_channels) == 0:
        output_state_channels = state_channels.copy()
    else:
        output_state_channels = cfg.inference.output_state_channels

    vardict_state = {c: i for i, c in enumerate(state_channels)}
    vardict_background = {c: i for i, c in enumerate(background_channels)}

    hours_since_jan_01 = int(
        (initial_time - datetime(initial_time.year, 1, 1, 0, 0)).total_seconds() / 3600
    )

    net = Module.from_checkpoint(cfg.inference.regression_checkpoint)
    regression_model = net.to(device).eval()
    bridge_model = _load_bridge_student(cfg, dataset, invariant_tensor, device)

    prior_channel_std = cfg.inference.bridge.get("prior_channel_std", None)
    if prior_channel_std is not None:
        prior_channel_std = list(prior_channel_std)

    br_kwargs = dict(
        num_steps=int(cfg.inference.bridge.num_steps),
        sigma_prior=float(cfg.inference.bridge.sigma_prior),
        solver=str(cfg.inference.bridge.solver),
        prior_channel_std=prior_channel_std,
    )
    print(f"[bridge] sampler kwargs: {br_kwargs}")

    # Output zarr -- reuse the same writers as inference_flowcast.py. The
    # "edm" group holds the Bridge-sampled prediction, the "noedm" group
    # holds the bare regression mean (mu_{t+1} alone) for direct comparison.
    (
        group,
        target_group,
        edm_prediction_group,
        noedm_prediction_group,
    ) = init_inference_results_zarr(
        dataset, cfg.inference.rundir, output_state_channels, n_steps
    )

    with torch.no_grad():
        for i in range(n_steps):
            data = dataset[i + hours_since_jan_01]

            background = data["background"].to(device=device, dtype=torch.float32).unsqueeze(0)

            if i == 0:
                state_pred = data["state"][0].to(device=device, dtype=torch.float32).unsqueeze(0)
                state_pred_br = state_pred.clone()      # Bridge-corrected
                state_pred_reg = state_pred.clone()     # regression-only

            assert state_pred_br.shape == (1, len(state_channels)) + dataset.image_shape()
            assert state_pred_reg.shape == (1, len(state_channels)) + dataset.image_shape()

            write_inference_results_zarr(
                dataset.denormalize_state(state_pred_br.cpu().numpy())[0],
                dataset.denormalize_state(state_pred_reg.cpu().numpy())[0],
                dataset.denormalize_state(data["state"][0].cpu().numpy()),
                edm_prediction_group,
                noedm_prediction_group,
                target_group,
                output_state_channels,
                vardict_state,
                i,
            )

            # Build condition + run frozen regression -> mu_{t+1}.
            # NOTE: subtract_regression=False so 'target' here is just the
            # placeholder M_t (unused), and mu comes back separately.
            (condition, _, mu) = build_network_condition_and_target(
                background,
                [state_pred, state_pred],
                invariant_tensor,
                regression_net=regression_model,
                condition_list=cfg.model.diffusion_conditions,
                regression_condition_list=cfg.model.regression_conditions,
                subtract_regression=False,
            )

            state_pred_reg = mu.clone()

            # Bridge sample -> full M_{t+1} prediction, raw units.
            m_pred = bridge_model_forward(
                bridge_model,
                condition,
                mu,
                **br_kwargs,
            )

            state_pred = m_pred
            state_pred_br = state_pred.clone()

            varidx_state = vardict_state[cfg.inference.plot_var_state]
            varidx_background = vardict_background[cfg.inference.plot_var_background]

            background_arr = dataset.denormalize_background(background.cpu().numpy()[0])
            state_true_arr = dataset.denormalize_state(data["state"][1].cpu().numpy())
            state_pred_arr = dataset.denormalize_state(state_pred.cpu().numpy()[0])

            fig = inference_plot(
                background_arr[varidx_background],
                state_pred_arr[varidx_state],
                state_true_arr[varidx_state],
                cfg.inference.plot_var_background,
                cfg.inference.plot_var_state,
                initial_time,
                i,
            )
            fig.savefig(f"{cfg.inference.rundir}/out_{i}.png")
            plt.close(fig)

    initial_time_pd = pd.to_datetime(initial_time)
    val_times = [initial_time_pd + pd.Timedelta(hours=i) for i in range(n_steps)]

    save_inference_results_netcdf(
        ds_out_path=cfg.inference.rundir,
        zarr_group=group,
        vertical_vars=cfg.inference.save_vertical_vars,
        level_names=cfg.inference.save_vertical_levels,
        horizontal_vars=cfg.inference.save_horizontal_vars,
        val_times=val_times,
    )


if __name__ == "__main__":
    main()
