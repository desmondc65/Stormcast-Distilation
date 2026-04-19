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

"""FlowCast (Conditional Flow Matching) training loop.

Trains an I-CFM vector field v_theta on the residual manifold R_t = X_t - M_t
of the StormCast two-stage setup. The regression model M is kept frozen and
used purely as conditioning / residual anchor, so inference is
``M_t + Euler(v_theta)`` exactly as the existing EDM stack does for the
diffusion residual.
"""

import contextlib
import os
import time
from collections import defaultdict

import numpy as np
import torch
import psutil
import matplotlib.pyplot as plt
import wandb

from physicsnemo.models import Module
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.diffusion import InfiniteSampler
from physicsnemo.launch.utils import save_checkpoint, load_checkpoint
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper

from .nn import (
    get_preconditioned_architecture,
    build_network_condition_and_target,
    flowcast_model_forward,
)
from .flowcast_loss import FlowCastLoss
from .ema import ExponentialMovingAverage
from .plots import validation_plot
from .spectrum import ps1d_plots
from .trainer import (
    _log_train_loss_csv,
    _log_valid_loss_csv,
    _log_rmse_field_csv,
    _log_mae_field_csv,
    _log_ps1d_field_csv,
    _plot_loss_curves,
    _save_validation_netcdf,
    print_dataset_info,
)
from datasets import dataset_classes
from datasets.dataset import worker_init
from torch.nn.utils import clip_grad_norm_


logger = PythonLogger("train_flowcast")


def _distributed_ready() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def flowcast_training_loop(cfg):
    """Main training loop for Conditional Flow Matching on the StormCast residual."""

    # Initialize
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
    net_name = "flowcast"
    condition_list = cfg.model.diffusion_conditions

    # Seed and performance settings
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

    # Load dataset
    logger0.info("Loading dataset...")
    dataset_cls = dataset_classes[cfg.dataset.name]
    logger0.info(f"Dataset class: {dataset_cls.__name__}")
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

    # Load pretrained regression net if 'regression' conditioning is requested
    if "regression" in condition_list:
        regression_net = Module.from_checkpoint(cfg.model.regression_weights)
        if cfg.training.compile_model:
            regression_net = torch.compile(regression_net)
        regression_net = regression_net.to(device).eval().requires_grad_(False)
    else:
        regression_net = None

    invariant_array = dataset_train.get_invariants()
    if invariant_array is not None:
        invariant_tensor = torch.from_numpy(invariant_array).to(device)
        invariant_tensor = invariant_tensor.unsqueeze(0)
        invariant_tensor = invariant_tensor.repeat(local_batch_size, 1, 1, 1)
    else:
        invariant_tensor = None

    # Construct network dimensions
    logger0.info("Constructing networks...")
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

    arch_kwargs = dict(
        img_resolution=dataset_train.image_shape(),
        target_channels=len(state_channels),
        conditional_channels=num_condition_channels,
        spatial_embedding=cfg.model.spatial_pos_embed,
        attn_resolutions=list(cfg.model.attn_resolutions),
    )

    # Build student (FlowCastPrecond)
    student = get_preconditioned_architecture(name="flowcast", **arch_kwargs)
    student.train().requires_grad_(True).to(device)
    # Carry sigma_data / time_scale from the config down to the wrapper so the
    # sampler and loss stay consistent.
    student.sigma_data = float(cfg.model.sigma_data)
    student.time_scale = float(cfg.model.time_scale)

    # EMA target network (used for inference + validation sampling).
    ema_decay = float(cfg.training.ema_decay)
    ema = ExponentialMovingAverage(student, decay=ema_decay)
    ema_net = get_preconditioned_architecture(name="flowcast", **arch_kwargs)
    ema_net = ema_net.to(device).eval().requires_grad_(False)
    ema_net.sigma_data = float(cfg.model.sigma_data)
    ema_net.time_scale = float(cfg.model.time_scale)
    ema.apply_shadow(ema_net)

    # Optional channel weighting / spectral regularizer (same knobs as CD).
    channel_weights = getattr(cfg.training, "channel_weights", None)
    if channel_weights is not None:
        channel_weights = list(channel_weights)

    spectral_channel_names = getattr(cfg.training, "spectral_channels", None)
    if spectral_channel_names:
        spectral_channels = [state_channels.index(f) for f in spectral_channel_names]
    else:
        spectral_channels = None
    spectral_weight = float(getattr(cfg.training, "spectral_weight", 0.0))

    loss_fn = FlowCastLoss(
        sigma_data=cfg.model.sigma_data,
        sigma_path=cfg.training.sigma_path,
        t_eps=float(getattr(cfg.training, "t_eps", 1e-5)),
        channel_weights=channel_weights,
        spectral_channels=spectral_channels,
        spectral_weight=spectral_weight,
    )
    # AdamW to match the FlowCast paper (vs. Adam in CD/PD). Weight decay is
    # a small regularizer on the velocity network.
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=cfg.training.lr,
        betas=tuple(cfg.training.adam_betas),
        weight_decay=cfg.training.weight_decay,
    )
    ddp = torch.nn.parallel.DistributedDataParallel(
        student, device_ids=[device], broadcast_buffers=False
    )

    # Resume
    ckpt_path = os.path.join(cfg.training.rundir, f"checkpoints_{net_name}")
    logger0.info(f'Trying to resume training state from "{ckpt_path}"...')
    total_steps = load_checkpoint(
        path=ckpt_path,
        models=student,
        optimizer=optimizer,
        epoch=None
        if cfg.training.resume_checkpoint == "latest"
        else cfg.training.resume_checkpoint,
    )

    if total_steps == 0:
        logger0.info("No resumable training state found.")
        init_weights = cfg.training.initial_weights
        if init_weights is None or init_weights == "":
            logger0.info("Starting FlowCast training from random initialization...")
        else:
            logger0.info(f"Starting from weights saved in {init_weights}...")
            if init_weights.endswith(".mdlus"):
                student.load(init_weights)
            elif init_weights.endswith(".pt"):
                model_dict = torch.load(init_weights, map_location=student.device)
                student.load_state(model_dict, strict=True)
    else:
        logger0.info(f"Resumed from step {total_steps}.")

    # Restore EMA state if available
    ema_ckpt_path = os.path.join(cfg.training.rundir, "ema_state.pt")
    if os.path.exists(ema_ckpt_path):
        logger0.info(f"Restoring EMA state from {ema_ckpt_path}...")
        ema_state = torch.load(ema_ckpt_path, map_location=device)
        ema.load_state_dict(ema_state)
        ema.apply_shadow(ema_net)

    # Train
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
    validation_counter = 0

    rundir = cfg.training.rundir
    os.makedirs(rundir, exist_ok=True)

    while not done:
        optimizer.zero_grad(set_to_none=True)

        for accum_idx in range(num_accumulation_rounds):
            batch = next(dataset_iterator)
            background = batch["background"].to(device=device, dtype=torch.float32)
            state = [s.to(device=device, dtype=torch.float32) for s in batch["state"]]

            # Only sync gradients on the final micro-batch of the accumulation
            # window; earlier rounds run under no_sync() to avoid redundant
            # all-reduces.
            is_last_accum = accum_idx == num_accumulation_rounds - 1
            sync_ctx = (
                ddp.no_sync()
                if (dist.world_size > 1 and not is_last_accum)
                else contextlib.nullcontext()
            )
            with sync_ctx:
                with torch.autocast("cuda", dtype=amp_dtype, enabled=enable_amp):
                    (condition, target, reg_out) = build_network_condition_and_target(
                        background,
                        state,
                        invariant_tensor,
                        regression_net=regression_net,
                        condition_list=condition_list,
                        regression_condition_list=cfg.model.regression_conditions,
                    )
                    loss_dict = loss_fn(
                        student=ddp,
                        images=target,
                        condition=condition,
                    )

                loss_value = loss_dict["loss"]
                loss_value.backward()

        if cfg.training.clip_grad_norm > 0:
            clip_grad_norm_(student.parameters(), cfg.training.clip_grad_norm)

        # Cosine LR with linear warmup (FlowCast paper: 1% warmup, cosine to
        # min_lr_ratio). If resume_from > warmup, this just rolls into cosine.
        warmup = max(cfg.training.lr_warmup_steps, 1)
        min_lr = cfg.training.lr * cfg.training.min_lr_ratio
        if total_steps < warmup:
            lr_now = cfg.training.lr * (total_steps / warmup)
        else:
            progress = (total_steps - warmup) / max(total_train_steps - warmup, 1)
            progress = min(max(progress, 0.0), 1.0)
            lr_now = min_lr + 0.5 * (cfg.training.lr - min_lr) * (
                1.0 + np.cos(np.pi * progress)
            )
        for g in optimizer.param_groups:
            g["lr"] = lr_now
            if log_to_wandb:
                wandb_logs["lr"] = lr_now

        for param in student.parameters():
            if param.grad is not None:
                torch.nan_to_num(
                    param.grad, nan=0.0, posinf=1e5, neginf=-1e5, out=param.grad
                )

        optimizer.step()

        # Fixed-decay EMA (FlowCast paper: 0.999).
        ema.update(student, decay=ema_decay)
        ema.apply_shadow(ema_net)

        loss_scalar = loss_value.detach()
        pointwise_scalar = loss_dict["pointwise"]
        spectral_scalar = loss_dict["spectral"]

        if dist.world_size > 1 and _distributed_ready():
            torch.distributed.barrier()
            torch.distributed.all_reduce(loss_scalar, op=torch.distributed.ReduceOp.AVG)
            torch.distributed.all_reduce(
                pointwise_scalar, op=torch.distributed.ReduceOp.AVG
            )
            torch.distributed.all_reduce(
                spectral_scalar, op=torch.distributed.ReduceOp.AVG
            )

        avg_train_loss += loss_scalar.cpu().item()
        train_steps += 1

        if log_to_wandb:
            wandb_logs["loss"] = loss_scalar.cpu().item()
            wandb_logs["loss_pointwise"] = pointwise_scalar.cpu().item()
            wandb_logs["loss_spectral"] = spectral_scalar.cpu().item()

        total_steps += 1
        done = total_steps >= total_train_steps

        if (
            dist.rank == 0
            and total_steps % cfg.training.print_progress_freq == 0
            and train_steps > 0
        ):
            try:
                avg_loss_to_log = avg_train_loss / train_steps
                _log_train_loss_csv(rundir, total_steps, float(avg_loss_to_log))
            except Exception as e:
                logger0.warn(f"Failed to write train_loss.csv at step {total_steps}: {e}")

        # Validation (uses EMA weights for the sampler, per FlowCast paper)
        if total_steps % cfg.training.validation_freq == 0:
            validation_counter += 1
            valid_start = time.time()
            logger0.info(
                f"[Validation] Starting validation at training step {total_steps}..."
            )
            batch = next(valid_dataset_iterator)

            with torch.no_grad():
                background = batch["background"].to(device=device, dtype=torch.float32)
                state = [
                    s.to(device=device, dtype=torch.float32) for s in batch["state"]
                ]
                with torch.autocast("cuda", dtype=amp_dtype, enabled=enable_amp):
                    (condition, target, reg_out) = build_network_condition_and_target(
                        background,
                        state,
                        invariant_tensor,
                        regression_net=regression_net,
                        condition_list=condition_list,
                        regression_condition_list=cfg.model.regression_conditions,
                    )

                    # Euler ODE sampling from EMA weights.
                    valid_num_steps = int(cfg.training.valid_num_steps)
                    residual = flowcast_model_forward(
                        ema_net,
                        condition,
                        state[1].shape,
                        num_steps=valid_num_steps,
                        sigma_data=cfg.model.sigma_data,
                        solver=cfg.training.solver,
                    )
                    if "regression" in condition_list and reg_out is not None:
                        output_images = residual + reg_out
                    else:
                        output_images = residual

                    # Evaluate the CFM objective against the EMA weights so
                    # the validation loss tracks the same net used for the
                    # sampler and for the eventually-deployed checkpoint.
                    valid_loss_dict = loss_fn(
                        student=ema_net,
                        images=target,
                        condition=condition,
                    )
                    valid_loss_scalar = valid_loss_dict["loss"].detach()

                if dist.world_size > 1 and _distributed_ready():
                    torch.distributed.barrier()
                    torch.distributed.all_reduce(
                        valid_loss_scalar, op=torch.distributed.ReduceOp.AVG
                    )

                val_loss = valid_loss_scalar.cpu().item()
                if log_to_wandb:
                    wandb_logs["valid_loss"] = val_loss

            if dist.rank == 0:
                rmse_acc = defaultdict(float)
                mae_acc = defaultdict(float)
                rmse_count = defaultdict(int)

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
                        state_channels,
                    )

                    k = ps_numeric["k"]
                    Pk_gen = ps_numeric["Pk_gen"]
                    Pk_tar = ps_numeric["Pk_tar"]

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

                    for f_ in fields:
                        f_index = state_channels.index(f_)
                        generated = image[f_index]
                        truth = state[1][i, f_index].detach().cpu().numpy()
                        diff = generated - truth
                        rmse_acc[f_] += float(np.sqrt(np.mean(diff**2)))
                        mae_acc[f_] += float(np.mean(np.abs(diff)))
                        rmse_count[f_] += 1

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
                        fig.savefig(
                            os.path.join(image_dir, f"{total_steps}_{i}_{f_}.png")
                        )

                        specfig = "PS1D_" + f_
                        figs[specfig].savefig(
                            os.path.join(
                                image_dir, f"{total_steps}_{i}_{f_}_spec.png"
                            )
                        )

                        output_nc_enabled = getattr(cfg.training, "output_nc", False)
                        output_nc_freq = getattr(cfg.training, "output_nc_freq", 1)
                        if output_nc_enabled and (
                            validation_counter % output_nc_freq == 0
                        ):
                            try:
                                _save_validation_netcdf(
                                    rundir, total_steps, i, f_, generated, truth,
                                )
                            except Exception as e:
                                logger0.warn(
                                    f"Failed to write NetCDF for {f_} at step {total_steps}: {e}"
                                )

                        if log_to_wandb:
                            for figname, plot in figs.items():
                                wandb_logs[figname] = wandb.Image(plot)
                            wandb_logs.update({f"generated_{f_}": wandb.Image(fig)})

                    plt.close("all")

                try:
                    _log_valid_loss_csv(rundir, total_steps, val_loss)
                except Exception as e:
                    logger0.warn(
                        f"Failed to write valid_loss.csv at step {total_steps}: {e}"
                    )

                try:
                    for f_ in cfg.training.validation_plot_variables:
                        count = max(rmse_count[f_], 1)
                        rmse_mean = rmse_acc[f_] / count
                        mae_mean = mae_acc[f_] / count
                        _log_rmse_field_csv(rundir, total_steps, f_, rmse_mean)
                        _log_mae_field_csv(rundir, total_steps, f_, mae_mean)
                except Exception as e:
                    logger0.warn(
                        f"Failed to write rmse_/mae_ CSVs at step {total_steps}: {e}"
                    )

                try:
                    for f_ in cfg.training.validation_plot_variables:
                        acc = ps_acc[f_]
                        if acc["count"] == 0:
                            continue
                        k = acc["k"]
                        pk_gen_mean = acc["Pk_gen_sum"] / acc["count"]
                        pk_tar_mean = acc["Pk_tar_sum"] / acc["count"]
                        _log_ps1d_field_csv(
                            rundir, total_steps, f_, k, pk_gen_mean, pk_tar_mean
                        )
                except Exception as e:
                    logger0.warn(
                        f"Failed to write ps1d CSVs at step {total_steps}: {e}"
                    )

                if log_to_wandb:
                    wandb_logs.update(spec_ratios)
                    wandb.log(wandb_logs, step=total_steps)

                try:
                    _plot_loss_curves(rundir, cfg.training.experiment_name)
                except Exception as e:
                    logger0.warn(
                        f"Failed to plot loss curves at step {total_steps}: {e}"
                    )

            valid_time = time.time() - valid_start
            logger0.info(
                f"[Validation] Completed at step {total_steps}: loss={val_loss:.4f}, time={valid_time: .2f}s"
            )

        # Console stats
        current_time = time.time()
        if total_steps % cfg.training.print_progress_freq == 0:
            fields = []
            fields += [f"steps {total_steps:<5d}"]
            fields += [f"samples {total_steps * batch_size}"]
            fields += [f"tot_time {current_time - start_time: .2f}"]
            fields += [
                f"step_time {(current_time - train_start - max(valid_time, 0.0)) / max(train_steps, 1): .2f}"
            ]
            fields += [f"valid_time {max(valid_time, 0.0): .2f}"]
            fields += [
                f"cpumem {psutil.Process(os.getpid()).memory_info().rss / 2**30:<6.2f}"
            ]
            fields += [
                f"gpumem {torch.cuda.max_memory_allocated(device) / 2**30:<6.2f}"
            ]
            fields += [f"train_loss {avg_train_loss / max(train_steps, 1):<6.3f}"]
            fields += [f"val_loss {val_loss:<6.3f}"]
            fields += [f"lr {lr_now:.2e}"]
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
                models=student,
                optimizer=optimizer,
                epoch=total_steps,
            )
            torch.save(ema.state_dict(), ema_ckpt_path)
            logger0.info(f"Saved checkpoint and EMA state at step {total_steps}.")

    if _distributed_ready():
        torch.distributed.barrier()
    logger0.info("\nExiting...")
