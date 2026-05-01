# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Autoregressive inference for FlowCast (Conditional Flow Matching distillate).

Mirrors ``inference.py`` but swaps the EDM diffusion sampler for the
fixed-step ODE sampler in ``flowcast_model_forward``. The pipeline is::

    M_t = F_xi(X_{t-1}, S_t, I)                  # frozen regression mean
    R_t = FlowCast(z, t, condition)              # learned residual
    X_t = M_t + R_t

The FlowCast student weights are loaded from the EMA shadow saved during
training (``<rundir>/ema_state.pt``); these, NOT the raw student
checkpoint, are what trainer_flowcast.py promotes to inference.

qpepre is automatically returned in mm/h: clean_zarr.py stores it as
``log1p(mm/h)`` and the data loader's ``denormalize_state`` applies
``expm1`` as part of de-normalisation. No special handling is needed here.
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
    flowcast_model_forward,
    get_preconditioned_architecture,
)
from utils.plots import inference_plot


def _load_flowcast_student(cfg, dataset, invariant_tensor, device):
    """Construct a FlowCastPrecond and load the EMA shadow into it.

    The EMA shadow lives at ``cfg.inference.flowcast_ema_path`` (or, by
    convention, ``<flowcast_rundir>/ema_state.pt``). We materialise a fresh
    FlowCastPrecond with the exact arch the trainer used, then overwrite its
    parameters with the shadow tensors. Returns the network in ``eval()`` mode.
    """
    state_channels = dataset.state_channels()
    background_channels = dataset.background_channels()

    condition_list = list(cfg.model.diffusion_conditions)
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

    net = get_preconditioned_architecture(name="flowcast", **arch_kwargs)
    net = net.to(device).eval().requires_grad_(False)
    net.sigma_data = float(cfg.model.sigma_data)
    net.time_scale = float(cfg.model.time_scale)

    # The training loop saves the EMA wrapper's state_dict (a flat dict of
    # shadow tensors keyed by parameter name). Apply it directly to ``net``.
    ema_path = cfg.inference.flowcast_ema_path
    print(f"[flowcast] loading EMA shadow weights from {ema_path}")
    ema_state = torch.load(ema_path, map_location=device)
    # ExponentialMovingAverage.state_dict stores shadows under "shadow_params"
    # (keyed by index) plus the parameter names list. Cope with both that
    # layout and a plain {param_name: tensor} dump.
    if "shadow_params" in ema_state and "param_names" in ema_state:
        names = ema_state["param_names"]
        shadows = ema_state["shadow_params"]
        target_state = dict(net.state_dict())
        for i, name in enumerate(names):
            if name in target_state:
                target_state[name] = shadows[i].to(target_state[name].dtype)
        net.load_state_dict(target_state, strict=False)
    else:
        # Fallback: assume it's already {param_name: tensor}.
        net.load_state_dict(ema_state, strict=False)

    return net


@hydra.main(version_base=None, config_path="config", config_name="flowcast_inference")
def main(cfg: DictConfig):
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device

    initial_time = datetime.fromisoformat(cfg.inference.initial_time)
    n_steps = cfg.inference.n_steps

    # Dataset
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

    # Models
    if "regression" in cfg.model.diffusion_conditions:
        net = Module.from_checkpoint(cfg.inference.regression_checkpoint)
        regression_model = net.to(device).eval()
    else:
        regression_model = None
    flowcast_model = _load_flowcast_student(cfg, dataset, invariant_tensor, device)

    # FlowCast sampler hyperparameters (Hydra config -> kwargs).
    fc_kwargs = dict(
        num_steps=int(cfg.inference.flowcast.num_steps),
        sigma_data=float(cfg.model.sigma_data),
        solver=str(cfg.inference.flowcast.solver),
    )
    print(f"[flowcast] sampler kwargs: {fc_kwargs}")

    # Output zarr -- reuse the same writers as inference.py. The "edm" group
    # holds the FlowCast-corrected prediction, the "noedm" group holds the
    # bare regression mean (M_t alone) for direct comparison.
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
                state_pred_fc = state_pred.clone()       # FlowCast-corrected
                state_pred_reg = state_pred.clone()      # regression-only

            assert state_pred_fc.shape == (1, len(state_channels)) + dataset.image_shape()
            assert state_pred_reg.shape == (1, len(state_channels)) + dataset.image_shape()

            # de-norm + expm1 for qpepre is handled inside denormalize_state
            write_inference_results_zarr(
                dataset.denormalize_state(state_pred_fc.cpu().numpy())[0],
                dataset.denormalize_state(state_pred_reg.cpu().numpy())[0],
                dataset.denormalize_state(data["state"][0].cpu().numpy()),
                edm_prediction_group,
                noedm_prediction_group,
                target_group,
                output_state_channels,
                vardict_state,
                i,
            )

            # Build condition + run frozen regression -> M_t in state_pred
            (condition, _, state_pred) = build_network_condition_and_target(
                background,
                [state_pred, state_pred],
                invariant_tensor,
                regression_net=regression_model,
                condition_list=cfg.model.diffusion_conditions,
                regression_condition_list=cfg.model.regression_conditions,
            )

            if state_pred is None:
                state_pred = torch.zeros_like(state_pred_fc)

            state_pred_reg = state_pred.clone()

            # FlowCast residual R_t (already in raw, un-standardised units).
            residual = flowcast_model_forward(
                flowcast_model,
                condition,
                state_pred.shape,
                **fc_kwargs,
            )

            state_pred[0, :] += residual[0].float()
            state_pred_fc = state_pred.clone()

            # Plot
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
