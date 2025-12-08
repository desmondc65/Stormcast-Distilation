# trainer.py

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

"""Main training loop with per-field CSV logging."""

import os
import time
import csv
from collections import defaultdict

import numpy as np
import xarray as xr
import datetime
import torch
import psutil
from physicsnemo.models import Module
from physicsnemo.distributed import DistributedManager
from physicsnemo.metrics.diffusion import EDMLoss
from physicsnemo.utils.diffusion import InfiniteSampler

from physicsnemo.launch.utils import save_checkpoint, load_checkpoint
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper
from .nn import (
    diffusion_model_forward,
    regression_loss_fn,
    get_preconditioned_architecture,
    build_network_condition_and_target,
)
from .plots import validation_plot
from datasets import dataset_classes
from datasets.dataset import worker_init
import matplotlib.pyplot as plt
import wandb
from .spectrum import ps1d_plots
from torch.nn.utils import clip_grad_norm_


logger = PythonLogger("train")


def _distributed_ready() -> bool:
    """Return True when torch.distributed default group is available."""
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def print_dataset_info(dataset, name="dataset"):
    logger0 = dataset.logger0 if hasattr(dataset, "logger0") else print
    logger0.info(f"--- {name} info ---")
    logger0.info(f"date_ranges: {dataset.date_ranges}")
    logger0.info(f"LowRes_zarrs: {getattr(dataset, 'LowRes_zarrs', None)}")
    logger0.info(f"HighRes_zarrs: {getattr(dataset, 'HighRes_zarrs', None)}")
    logger0.info(f"LowRes_channels: {getattr(dataset, 'LowRes_channels', None)}")
    logger0.info(f"HighRes_channels: {getattr(dataset, 'HighRes_channels', None)}")
    logger0.info(f"n_samples_total: {getattr(dataset, 'n_samples_total', None)}")
    logger0.info(f"valid_samples (first 5): {getattr(dataset, 'valid_samples', None)[:5]}")
    logger0.info(f"image_shape: {dataset.image_shape()}")
    logger0.info(f"background_channels: {dataset.background_channels()}")
    logger0.info(f"state_channels: {dataset.state_channels()}")
    logger0.info(f"invariants: {dataset.get_invariants()}")
    logger0.info(f"-------------------")


# ---------- CSV helpers ----------

def _append_rows_csv(path: str, rows: list, fieldnames: list):
    """Append rows to CSV, writing header if file does not exist."""
    dirn = os.path.dirname(path)
    if dirn:
        os.makedirs(dirn, exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def _sanitize_field(name: str) -> str:
    return "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in name)


def _log_train_loss_csv(rundir: str, step: int, loss_value: float):
    path = os.path.join(rundir, "train_loss.csv")
    _append_rows_csv(path, [{"step": step, "loss": float(loss_value)}], ["step", "loss"])


def _log_valid_loss_csv(rundir: str, step: int, loss_value: float):
    path = os.path.join(rundir, "valid_loss.csv")
    _append_rows_csv(path, [{"step": step, "loss": float(loss_value)}], ["step", "loss"])


def _log_rmse_field_csv(rundir: str, step: int, field: str, value: float):
    field_tag = _sanitize_field(field)
    path = os.path.join(rundir, f"rmse_{field_tag}.csv")
    _append_rows_csv(path, [{"step": step, "rmse": float(value)}], ["step", "rmse"])


def _log_mae_field_csv(rundir: str, step: int, field: str, value: float):
    field_tag = _sanitize_field(field)
    path = os.path.join(rundir, f"mae_{field_tag}.csv")
    _append_rows_csv(path, [{"step": step, "mae": float(value)}], ["step", "mae"])


def _log_ps1d_field_csv(rundir: str, step: int, field: str,
                        k: np.ndarray, pk_gen: np.ndarray, pk_tar: np.ndarray):
    """Write PS1D rows: one row per k for the given field and step."""
    field_tag = _sanitize_field(field)
    path = os.path.join(rundir, f"ps1d_{field_tag}.csv")
    ratio = np.divide(pk_gen, pk_tar, out=np.zeros_like(pk_gen), where=(pk_tar != 0))
    rows = [
        {"step": step, "k": float(ki), "Pk_gen": float(g), "Pk_tar": float(t), "ratio": float(r)}
        for ki, g, t, r in zip(k.tolist(), pk_gen.tolist(), pk_tar.tolist(), ratio.tolist())
    ]
    _append_rows_csv(path, rows, ["step", "k", "Pk_gen", "Pk_tar", "ratio"])


def _plot_loss_curves(rundir: str, experiment_name: str):
    """Plot training and validation loss curves from CSV files.
    
    Args:
        rundir: The main run directory containing loss CSV files.
        experiment_name: The name of the experiment for the plot title.
    """
    try:
        train_csv = os.path.join(rundir, "train_loss.csv")
        valid_csv = os.path.join(rundir, "valid_loss.csv")
        
        # Check if files exist
        if not os.path.exists(train_csv):
            logger.warn(f"Training loss CSV not found: {train_csv}")
            return
            
        # Read training loss
        train_steps = []
        train_losses = []
        with open(train_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                train_steps.append(int(row['step']))
                train_losses.append(float(row['loss']))
        
        # Read validation loss if exists
        valid_steps = []
        valid_losses = []
        if os.path.exists(valid_csv):
            with open(valid_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    valid_steps.append(int(row['step']))
                    valid_losses.append(float(row['loss']))
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if train_steps and train_losses:
            ax.plot(train_steps, train_losses, label='Training Loss', alpha=0.7, linewidth=1)
        
        if valid_steps and valid_losses:
            ax.plot(valid_steps, valid_losses, label='Validation Loss', 
                   marker='o', markersize=4, linewidth=2)
        
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'{experiment_name} - Training Progress', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Save the plot (will replace existing file)
        plot_path = os.path.join(rundir, "loss_curves.png")
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
    except Exception as e:
        logger.warn(f"Failed to plot loss curves: {e}")


def _save_validation_netcdf(
    rundir: str,
    step: int,
    sample_idx: int,
    field_name: str,
    generated_data: np.ndarray,
    truth_data: np.ndarray,
    coords: dict = None,
    var_attrs: dict = None,
):
    """Saves validation output (generated and truth) to a NetCDF file.

    Args:
        rundir: The main run directory.
        step: The current training step.
        sample_idx: The index of the sample in the batch (e.g., 'i').
        field_name: The name of the variable (e.g., 't2m').
        generated_data: The 2D numpy array from the model.
        truth_data: The 2D numpy array of the ground truth.
        coords: A dictionary of coordinates, e.g.,
                {'lat': lat_array, 'lon': lon_array}.
                If None, default integer (y, x) coordinates are used.
        var_attrs: A dictionary of variable attributes, e.g.,
                   {'units': 'K', 'long_name': '2-meter Temperature'}.
    """
    try:
        # 1. Create output directory
        nc_dir = os.path.join(rundir, "netcdf_outputs", field_name)
        os.makedirs(nc_dir, exist_ok=True)
        nc_filename = os.path.join(nc_dir, f"{step}_{sample_idx}_{field_name}.nc")

        # Ensure numpy arrays
        gen = np.asarray(generated_data)
        tar = np.asarray(truth_data)

        # 2. Set up coordinates
        if coords is None:
            # Create default integer coordinates if none provided
            H, W = gen.shape
            y_coords = np.arange(H)
            x_coords = np.arange(W)
            coord_dims = ("y", "x")
            coord_map = {"y": y_coords, "x": x_coords}
        else:
            # Use provided coordinates (e.g., {'lat': lat_array, 'lon': lon_array})
            coord_dims = tuple(coords.keys())
            coord_map = coords

        # 3. Create the xarray.Dataset
        ds = xr.Dataset(
            data_vars={
                "generated": (coord_dims, gen),
                "truth": (coord_dims, tar),
            },
            coords=coord_map,
            attrs={
                "title": f"Validation Output for {field_name}",
                "field": field_name,
                "step": int(step),
                "sample_index": int(sample_idx),
                "creation_date": datetime.datetime.now().isoformat(),
            },
        )

        # 4. Add variable-specific attributes (metadata)
        if var_attrs is None:
            var_attrs = {"units": "unknown", "long_name": f"Unknown {field_name}"}

        ds["generated"].attrs["long_name"] = f"Generated {var_attrs.get('long_name', field_name)}"
        ds["generated"].attrs["units"] = var_attrs.get("units", "unknown")
        ds["truth"].attrs["long_name"] = f"Truth {var_attrs.get('long_name', field_name)}"
        ds["truth"].attrs["units"] = var_attrs.get("units", "unknown")

        # 5. Save to file
        ds.to_netcdf(nc_filename)
        ds.close()

    except Exception as e:
        logger.warn(f"Failed to write NetCDF at step {step} for {field_name}: {e}")


# ---------- Main training ----------

def training_loop(cfg):

    # Initialize.
    start_time = time.time()
    dist = DistributedManager()
    device = dist.device
    logger0 = RankZeroLoggingWrapper(logger, dist)

    # Shorthand for config items
    batch_size = cfg.training.batch_size
    if cfg.training.batch_size_per_gpu == "auto":
        local_batch_size = batch_size // dist.world_size
    else:
        local_batch_size = cfg.training.batch_size_per_gpu
    assert batch_size % (local_batch_size * dist.world_size) == 0
    num_accumulation_rounds = batch_size // (local_batch_size * dist.world_size)

    log_to_wandb = cfg.training.log_to_wandb

    loss_type = cfg.training.loss
    if loss_type == "regression":
        net_name = "regression"
    elif loss_type == "edm":
        net_name = "diffusion"
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    condition_list = (
        cfg.model.regression_conditions
        if net_name == "regression"
        else cfg.model.diffusion_conditions
    )

    # Seed and Performance settings
    np.random.seed((cfg.training.seed * dist.world_size + dist.rank) % (1 << 31))
    torch.manual_seed(cfg.training.seed)
    torch.backends.cudnn.benchmark = cfg.training.cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    fp_optimizations = cfg.training.fp_optimizations
    enable_amp = fp_optimizations.startswith("amp")
    amp_dtype = torch.float16 if fp_optimizations == "amp-fp16" else torch.bfloat16

    total_train_steps = cfg.training.total_train_steps

    # Load dataset.
    logger0.info("Loading dataset...")
    dataset_cls = dataset_classes[cfg.dataset.name]
    logger0.info(f"Dataset class: {dataset_cls.__name__}")
    logger0.info(f"Dataset type: {cfg.dataset.name}")
    del cfg.dataset.name

    dataset_train = dataset_cls(cfg.dataset, train=True)
    dataset_valid = dataset_cls(cfg.dataset, train=False)
    print_dataset_info(dataset_train, "Train")

    background_channels = dataset_train.background_channels()
    state_channels = dataset_train.state_channels()

    sampler = InfiniteSampler(
        dataset=dataset_train,
        rank=dist.rank,
        num_replicas=dist.world_size,
        seed=cfg.training.seed,
    )
    valid_sampler = InfiniteSampler(
        dataset=dataset_valid,
        rank=dist.rank,
        num_replicas=dist.world_size,
        seed=cfg.training.seed,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=local_batch_size,
        num_workers=cfg.training.num_data_workers,
        sampler=sampler,
        worker_init_fn=worker_init,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )
    valid_data_loader = torch.utils.data.DataLoader(
        dataset=dataset_valid,
        batch_size=local_batch_size,
        num_workers=cfg.training.num_data_workers,
        sampler=valid_sampler,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )

    dataset_iterator = iter(data_loader)
    valid_dataset_iterator = iter(valid_data_loader)

    # load pretrained regression net if training diffusion
    if "regression" in condition_list:
        regression_net = Module.from_checkpoint(cfg.model.regression_weights)
        if cfg.training.compile_model:
            regression_net = torch.compile(regression_net)
        regression_net = regression_net.to(device)
    else:
        regression_net = None

    invariant_array = dataset_train.get_invariants()
    if invariant_array is not None:
        invariant_tensor = torch.from_numpy(invariant_array).to(device)
        invariant_tensor = invariant_tensor.unsqueeze(0)
        invariant_tensor = invariant_tensor.repeat(local_batch_size, 1, 1, 1)
    else:
        invariant_tensor = None

    # Construct network
    logger0.info("Constructing network...")
    num_condition_channels = {
        "state": len(state_channels),
        "background": len(background_channels),
        "regression": len(state_channels),
        "invariant": 0 if invariant_tensor is None else invariant_tensor.shape[1],
    }
    num_condition_channels = sum(num_condition_channels[c] for c in condition_list)

    logger0.info(f"model conditions {condition_list}")
    logger0.info(f"background_channels {background_channels}")
    logger0.info(f"state_channels {state_channels}")
    logger0.info(f"num_condition_channels {num_condition_channels}")

    net = get_preconditioned_architecture(
        name=net_name,
        img_resolution=dataset_train.image_shape(),
        target_channels=len(state_channels),
        conditional_channels=num_condition_channels,
        spatial_embedding=cfg.model.spatial_pos_embed,
        attn_resolutions=list(cfg.model.attn_resolutions),
    )

    net.train().requires_grad_(True).to(device)

    # Setup optimizer.
    logger0.info("Setting up optimizer...")
    if cfg.training.loss == "regression":
        loss_fn = regression_loss_fn
    elif cfg.training.loss == "edm":
        loss_fn = EDMLoss(P_mean=cfg.model.P_mean)
    else:
        raise ValueError(f"Unknown training.loss: {cfg.training.loss}")
    if cfg.training.compile_model:
        loss_fn = torch.compile(loss_fn)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.training.lr)
    augment_pipe = None
    ddp = torch.nn.parallel.DistributedDataParallel(
        net, device_ids=[device], broadcast_buffers=False
    )

    # Resume training from previous snapshot.
    ckpt_path = os.path.join(cfg.training.rundir, f"checkpoints_{net_name}")
    logger0.info(f'Trying to resume training state from "{ckpt_path}"...')
    total_steps = load_checkpoint(
        path=ckpt_path,
        models=net,
        optimizer=optimizer,
        epoch=None
        if cfg.training.resume_checkpoint == "latest"
        else cfg.training.resume_checkpoint,
    )

    if total_steps == 0:
        logger0.info(f"No resumable training state found.")
        init_weights = cfg.training.initial_weights
        if init_weights is None:
            logger0.info(f"Starting training from scratch...")
        else:
            logger0.info(f"Starting training from weights saved in {init_weights}...")
            if init_weights.endswith(".mdlus"):
                net.load(init_weights)
            elif init_weights.endswith(".pt"):
                model_dict = torch.load(init_weights, map_location=net.device)
                net.load_state(model_dict, strict=True)

    # Train.
    logger0.info(
        f"Training up to {total_train_steps} steps starting from step {total_steps}..."
    )
    wandb_logs = {}
    done = total_steps >= total_train_steps

    train_start = time.time()
    avg_train_loss = 0.0
    train_steps = 0
    valid_time = -1.0
    val_loss = -1.0
    validation_counter = 0  # Track number of validations performed

    rundir = cfg.training.rundir
    os.makedirs(rundir, exist_ok=True)
    while not done:
        optimizer.zero_grad(set_to_none=True)

        for _ in range(num_accumulation_rounds):
            batch = next(dataset_iterator)
            background = batch["background"].to(device=device, dtype=torch.float32)
            state = [s.to(device=device, dtype=torch.float32) for s in batch["state"]]

            with torch.autocast("cuda", dtype=amp_dtype, enabled=enable_amp):
                (condition, target, reg_out) = build_network_condition_and_target(
                    background,
                    state,
                    invariant_tensor,
                    regression_net=regression_net,
                    condition_list=condition_list,
                    regression_condition_list=cfg.model.regression_conditions,
                )
                loss = loss_fn(
                    net=ddp,
                    images=target,
                    condition=condition,
                    augment_pipe=augment_pipe,
                )

            if log_to_wandb:
                channelwise_loss = loss.mean(dim=(0, 2, 3))
                channelwise_loss_dict = {
                    f"ChLoss/{ch}": channelwise_loss[i].item()
                    for (i, ch) in enumerate(state_channels)
                }
                wandb_logs["channelwise_loss"] = channelwise_loss_dict

            loss_value = (loss.sum() / len(state_channels))
            loss_value.backward()

        if cfg.training.clip_grad_norm > 0:
            clip_grad_norm_(net.parameters(), cfg.training.clip_grad_norm)

        for g in optimizer.param_groups:
            g["lr"] = cfg.training.lr * min(
                total_steps / max(cfg.training.lr_rampup_steps, 1e-8), 1.0
            )
            if log_to_wandb:
                wandb_logs["lr"] = g["lr"]
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0.0, posinf=1e5, neginf=-1e5, out=param.grad)

        optimizer.step()

        if dist.world_size > 1 and _distributed_ready():
            torch.distributed.barrier()
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)

        avg_train_loss += loss.mean().detach().cpu().item()
        train_steps += 1

        if log_to_wandb:
            wandb_logs["loss"] = loss.mean().detach().cpu().item()

        total_steps += 1
        done = total_steps >= total_train_steps

        if dist.rank == 0 and total_steps % cfg.training.print_progress_freq == 0 and train_steps > 0:
            try:
                # Log the average training loss over the print_progress_freq interval
                avg_loss_to_log = avg_train_loss / train_steps
                _log_train_loss_csv(rundir, total_steps, float(avg_loss_to_log))
            except Exception as e:
                logger0.warn(f"Failed to write train_loss.csv at step {total_steps}: {e}")

        # Validation
        if total_steps % cfg.training.validation_freq == 0:
            validation_counter += 1  # Increment validation counter
            valid_start = time.time()
            logger0.info(f"[Validation] Starting validation at training step {total_steps}...")
            batch = next(valid_dataset_iterator)

            with torch.no_grad():
                background = batch["background"].to(device=device, dtype=torch.float32)
                state = [s.to(device=device, dtype=torch.float32) for s in batch["state"]]
                with torch.autocast("cuda", dtype=amp_dtype, enabled=enable_amp):
                    (condition, target, reg_out) = build_network_condition_and_target(
                        background,
                        state,
                        invariant_tensor,
                        regression_net=regression_net,
                        condition_list=condition_list,
                        regression_condition_list=cfg.model.regression_conditions,
                    )

                    loss_kwargs = {"return_model_outputs": True} if net_name == "regression" else {}
                    valid_loss = loss_fn(
                        net=net,
                        images=target,
                        condition=condition,
                        augment_pipe=augment_pipe,
                        **loss_kwargs,
                    )

                    if net_name == "diffusion":
                        output_images = diffusion_model_forward(
                            net,
                            condition,
                            state[1].shape,
                            sampler_args=dict(cfg.sampler.args),
                        )
                        if "regression" in condition_list:
                            output_images += reg_out
                        del reg_out
                    else:
                        (valid_loss, output_images) = valid_loss
                        if log_to_wandb:
                            channelwise_valid_loss = valid_loss.mean(dim=[0, 2, 3])
                            channelwise_valid_loss_dict = {
                                f"ChLoss_valid/{state_channels[i]}": channelwise_valid_loss[i].item()
                                for i in range(len(state_channels))
                            }
                            wandb_logs["channelwise_valid_loss"] = channelwise_valid_loss_dict

                if dist.world_size > 1 and _distributed_ready():
                    torch.distributed.barrier()
                    torch.distributed.all_reduce(valid_loss, op=torch.distributed.ReduceOp.AVG)

                val_loss = valid_loss.mean().detach().cpu().item()
                if log_to_wandb:
                    wandb_logs["valid_loss"] = val_loss

            if dist.rank == 0:
                # Accumulators
                rmse_acc = defaultdict(float)
                mae_acc = defaultdict(float)
                rmse_count = defaultdict(int)

                # PS1D accumulators per field
                ps_acc = {
                    f: {"Pk_gen_sum": None, "Pk_tar_sum": None, "k": None, "count": 0}
                    for f in cfg.training.validation_plot_variables
                }

                for i in range(output_images.shape[0]):
                    image = output_images[i].detach().cpu().numpy()
                    fields = cfg.training.validation_plot_variables

                    figs, spec_ratios, ps_numeric = ps1d_plots(
                        torch.as_tensor(output_images[i]).detach().cpu(),
                        torch.as_tensor(state[1][i]).detach().cpu(),
                        fields,
                        state_channels
                    )

                    k = ps_numeric["k"]           # (L,)
                    Pk_gen = ps_numeric["Pk_gen"] # (C,L)
                    Pk_tar = ps_numeric["Pk_tar"] # (C,L)

                    for f_ in fields:
                        cidx = state_channels.index(f_)
                        g_arr = Pk_gen[cidx]
                        t_arr = Pk_tar[cidx]
                        acc = ps_acc[f_]
                        if acc["Pk_gen_sum"] is None:
                            acc["Pk_gen_sum"] = g_arr.copy()
                            acc["Pk_tar_sum"] = t_arr.copy()
                            acc["k"] = k.copy()
                        else:
                            acc["Pk_gen_sum"] += g_arr
                            acc["Pk_tar_sum"] += t_arr
                        acc["count"] += 1

                    # RMSE/MAE per field for this sample
                    for f_ in fields:
                        f_index = state_channels.index(f_)
                        generated = image[f_index]
                        truth = state[1][i, f_index].detach().cpu().numpy()
                        diff = generated - truth
                        rmse_acc[f_] += float(np.sqrt(np.mean(diff ** 2)))
                        mae_acc[f_] += float(np.mean(np.abs(diff)))
                        rmse_count[f_] += 1

                    # Save plots
                    for f_ in fields:
                        f_index = state_channels.index(f_)
                        image_dir = os.path.join(cfg.training.rundir, "images", f_)
                        os.makedirs(image_dir, exist_ok=True)

                        generated = image[f_index]
                        truth = state[1][i, f_index].detach().cpu().numpy()

                        fig = validation_plot(
                            generated,
                            truth,
                            f_,
                            experiment_name=cfg.training.experiment_name,
                            step=total_steps,
                        )
                        fig.savefig(os.path.join(image_dir, f"{total_steps}_{i}_{f_}.png"))

                        specfig = "PS1D_" + f_
                        figs[specfig].savefig(
                            os.path.join(image_dir, f"{total_steps}_{i}_{f_}_spec.png")
                        )
                        # Optionally save NetCDF outputs for generated and truth fields
                        output_nc_enabled = getattr(cfg.training, "output_nc", False)
                        output_nc_freq = getattr(cfg.training, "output_nc_freq", 1)
                        
                        # Check if we should output NetCDF: enabled AND validation counter is a multiple of freq
                        if output_nc_enabled and (validation_counter % output_nc_freq == 0):
                            try:
                                logger0.info(
                                    f"Saving NetCDF for {f_} at step {total_steps}, sample {i}"
                                )
                                _save_validation_netcdf(
                                    rundir,
                                    total_steps,
                                    i,
                                    f_,
                                    generated,
                                    truth,
                                )
                            except Exception as e:
                                logger0.warn(
                                    f"Failed to write NetCDF for {f_} at step {total_steps}, sample {i}: {e}"
                                )
                        if log_to_wandb:
                            for figname, plot in figs.items():
                                wandb_logs[figname] = wandb.Image(plot)
                            wandb_logs.update({f"generated_{f_}": wandb.Image(fig)})

                    plt.close("all")

                # valid_loss.csv
                try:
                    _log_valid_loss_csv(rundir, total_steps, val_loss)
                except Exception as e:
                    logger0.warn(f"Failed to write valid_loss.csv at step {total_steps}: {e}")

                # rmse_<field>.csv & mae_<field>.csv
                try:
                    for f_ in cfg.training.validation_plot_variables:
                        count = max(rmse_count[f_], 1)
                        rmse_mean = rmse_acc[f_] / count
                        mae_mean = mae_acc[f_] / count
                        _log_rmse_field_csv(rundir, total_steps, f_, rmse_mean)
                        _log_mae_field_csv(rundir, total_steps, f_, mae_mean)
                except Exception as e:
                    logger0.warn(f"Failed to write rmse_/mae_ CSVs at step {total_steps}: {e}")

                # ps1d_<field>.csv (avg over batch for each field)
                try:
                    for f_ in cfg.training.validation_plot_variables:
                        acc = ps_acc[f_]
                        if acc["count"] == 0:
                            continue
                        k = acc["k"]
                        pk_gen_mean = acc["Pk_gen_sum"] / acc["count"]
                        pk_tar_mean = acc["Pk_tar_sum"] / acc["count"]
                        _log_ps1d_field_csv(rundir, total_steps, f_, k, pk_gen_mean, pk_tar_mean)
                except Exception as e:
                    logger0.warn(f"Failed to write ps1d_<field>.csv at step {total_steps}: {e}")

                if log_to_wandb:
                    wandb_logs.update(spec_ratios)
                    wandb.log(wandb_logs, step=total_steps)

                # Plot loss curves after validation (rank 0 only)
                try:
                    _plot_loss_curves(rundir, cfg.training.experiment_name)
                except Exception as e:
                    logger0.warn(f"Failed to plot loss curves at step {total_steps}: {e}")

            valid_time = time.time() - valid_start
            logger0.info(
                f"[Validation] Completed at step {total_steps}: loss={val_loss:.4f}, time={valid_time: .2f}s"
            )

        # Console stats
        current_time = time.time()
        if total_steps % cfg.training.print_progress_freq == 0:
            fields = []
            fields += [f"steps {total_steps:<5d}"]
            fields += [f"samples {total_steps*batch_size}"]
            fields += [f"tot_time {current_time - start_time: .2f}"]
            fields += [
                f"step_time {(current_time - train_start - max(valid_time, 0.0)) / max(train_steps, 1): .2f}"
            ]
            fields += [f"valid_time {max(valid_time, 0.0): .2f}"]
            fields += [f"cpumem {psutil.Process(os.getpid()).memory_info().rss / 2**30:<6.2f}"]
            fields += [f"gpumem {torch.cuda.max_memory_allocated(device) / 2**30:<6.2f}"]
            fields += [f"train_loss {avg_train_loss/max(train_steps,1):<6.3f}"]
            fields += [f"val_loss {val_loss:<6.3f}"]
            logger0.info(" ".join(fields))

            train_steps = 0
            train_start = time.time()
            avg_train_loss = 0.0
            torch.cuda.reset_peak_memory_stats()

        # Checkpointing
        if (
            (done or total_steps % cfg.training.checkpoint_freq == 0)
            and total_steps != 0
            and dist.rank == 0
        ):
            save_checkpoint(
                path=os.path.join(cfg.training.rundir, f"checkpoints_{net_name}"),
                models=net,
                optimizer=optimizer,
                epoch=total_steps,
            )

    if _distributed_ready():
        torch.distributed.barrier()
    logger0.info("\nExiting...")
