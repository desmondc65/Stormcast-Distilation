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

"""Adversarial Diffusion Distillation (ADD) training loop.

Distils the 18-step StormCast EDM into a 1-4 step student using a combined
adversarial + score-distillation objective.  Only the EDM residual component
is distilled; the deterministic regression model is left untouched.

Three networks are maintained:
  - ADD-Student  (trainable, initialized from teacher weights)
  - DM-Teacher   (frozen copy of pretrained EDM)
  - Discriminator (trainable, custom multi-scale for 99-channel data)
"""

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
    add_model_forward,
)
from .ema import ExponentialMovingAverage
from .discriminator import MultiScaleDiscriminator
from .add_loss import ADDLoss
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


logger = PythonLogger("train_add")


def _distributed_ready() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _student_generate(student, condition, shape, sigma_max, num_steps,
                      sigma_min=0.002, rho=7.0):
    """Generate residual with the ADD-Student using 1-N denoising steps.

    For 1-step: sample z ~ N(0, sigma_max^2 I), denoise once.
    For N>1 steps: use a simple deterministic Euler schedule.

    Args:
        student: ADD-Student model (EDMPrecond).
        condition: Conditioning tensor [B, C_cond, H, W].
        shape: Output shape [B, C, H, W].
        sigma_max: Maximum noise level.
        num_steps: Number of denoising steps (1-4).
        sigma_min: Minimum noise level.
        rho: Karras schedule exponent.

    Returns:
        Generated residual [B, C, H, W].
    """
    device = condition.device
    dtype = condition.dtype
    B = shape[0]

    if num_steps == 1:
        z = torch.randn(*shape, device=device, dtype=dtype) * sigma_max
        sigma = torch.full([B], sigma_max, device=device, dtype=dtype)
        return student(z, sigma, condition=condition)

    # Multi-step Euler sampling with Karras schedule
    step_indices = torch.arange(num_steps, device=device, dtype=dtype)
    sigma_max_inv = sigma_max ** (1 / rho)
    sigma_min_inv = sigma_min ** (1 / rho)
    t_steps = (sigma_max_inv + step_indices / (num_steps - 1) * (sigma_min_inv - sigma_max_inv)) ** rho
    t_steps = torch.cat([t_steps, torch.zeros(1, device=device)])  # t_N = 0

    x = torch.randn(*shape, device=device, dtype=dtype) * t_steps[0]
    for i in range(num_steps):
        sigma = torch.full([B], t_steps[i].item(), device=device, dtype=dtype)
        denoised = student(x, sigma, condition=condition)
        d = (x - denoised) / t_steps[i]
        x = x + d * (t_steps[i + 1] - t_steps[i])

    return x


def add_training_loop(cfg):
    """Main training loop for Adversarial Diffusion Distillation."""

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
    net_name = "add"
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

    # Load pretrained regression net if needed
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

    # ── 1. Build ADD-Student (EDMPrecond, initialized from teacher) ──
    student = get_preconditioned_architecture(name="add", **arch_kwargs)
    student.train().requires_grad_(True).to(device)

    # ── 2. Load frozen DM-Teacher ──
    logger0.info(f"Loading frozen DM-Teacher from {cfg.model.teacher_weights}...")
    teacher = Module.from_checkpoint(cfg.model.teacher_weights)
    teacher = teacher.to(device).eval().requires_grad_(False)

    # Initialize student from teacher weights
    logger0.info("Initializing ADD-Student from teacher weights...")
    student.model.load_state_dict(teacher.model.state_dict())

    # ── 3. Build custom discriminator ──
    disc_cond_channels = num_condition_channels if cfg.training.condition_discriminator else 0
    discriminator = MultiScaleDiscriminator(
        in_channels=len(state_channels),
        cond_channels=disc_cond_channels,
        base_ch=cfg.training.disc_base_ch,
        num_scales=cfg.training.disc_num_scales,
    ).to(device)
    logger0.info(
        f"Discriminator: {sum(p.numel() for p in discriminator.parameters())/1e6:.1f}M params, "
        f"{cfg.training.disc_num_scales} scales, base_ch={cfg.training.disc_base_ch}"
    )

    # EMA for the student
    ema = ExponentialMovingAverage(student, decay=cfg.training.ema_decay)
    ema_net = get_preconditioned_architecture(name="add", **arch_kwargs)
    ema_net = ema_net.to(device).eval().requires_grad_(False)
    ema.apply_shadow(ema_net)

    # ── Loss ──
    loss_fn = ADDLoss(
        sigma_min=cfg.model.sigma_min,
        sigma_max=cfg.model.sigma_max,
        sigma_data=cfg.model.sigma_data,
        P_mean=cfg.model.P_mean,
        P_std=cfg.training.P_std,
        lambda_distill=cfg.training.lambda_distill,
        r1_gamma=cfg.training.r1_gamma,
        use_nfsd=cfg.training.use_nfsd,
    )

    # ── Optimizers ──
    logger0.info("Setting up optimizers...")
    optimizer_G = torch.optim.Adam(
        student.parameters(), lr=cfg.training.lr_G, betas=(0.0, 0.999)
    )
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), lr=cfg.training.lr_D, betas=(0.0, 0.999)
    )

    ddp_student = torch.nn.parallel.DistributedDataParallel(
        student, device_ids=[device], broadcast_buffers=False
    )
    ddp_disc = torch.nn.parallel.DistributedDataParallel(
        discriminator, device_ids=[device], broadcast_buffers=False
    )

    # Resume training
    ckpt_path = os.path.join(cfg.training.rundir, f"checkpoints_{net_name}")
    logger0.info(f'Trying to resume training state from "{ckpt_path}"...')
    total_steps = load_checkpoint(
        path=ckpt_path,
        models=student,
        optimizer=optimizer_G,
        epoch=None
        if cfg.training.resume_checkpoint == "latest"
        else cfg.training.resume_checkpoint,
    )

    # Try to resume discriminator
    disc_ckpt_path = os.path.join(cfg.training.rundir, "checkpoints_disc")
    disc_steps = load_checkpoint(
        path=disc_ckpt_path,
        models=discriminator,
        optimizer=optimizer_D,
        epoch=None
        if cfg.training.resume_checkpoint == "latest"
        else cfg.training.resume_checkpoint,
    )

    if total_steps == 0:
        logger0.info("No resumable training state found.")
        init_weights = cfg.training.initial_weights
        if init_weights is None or init_weights == "":
            logger0.info("Starting ADD distillation from teacher-initialized weights...")
        else:
            logger0.info(f"Starting from weights saved in {init_weights}...")
            if init_weights.endswith(".mdlus"):
                student.load(init_weights)
            elif init_weights.endswith(".pt"):
                model_dict = torch.load(init_weights, map_location=student.device)
                student.load_state(model_dict, strict=True)
    else:
        logger0.info(f"Resumed from step {total_steps}.")

    # Restore EMA state
    ema_ckpt_path = os.path.join(cfg.training.rundir, "ema_state.pt")
    if os.path.exists(ema_ckpt_path):
        logger0.info(f"Restoring EMA state from {ema_ckpt_path}...")
        ema_state = torch.load(ema_ckpt_path, map_location=device)
        ema.load_state_dict(ema_state)
        ema.apply_shadow(ema_net)

    # ── Train ──
    logger0.info(
        f"Training up to {total_train_steps} steps starting from step {total_steps}..."
    )
    wandb_logs = {}
    done = total_steps >= total_train_steps

    train_start = time.time()
    avg_loss_G = 0.0
    avg_loss_D = 0.0
    train_steps = 0
    valid_time = -1.0
    val_loss_G = -1.0
    validation_counter = 0

    rundir = cfg.training.rundir
    os.makedirs(rundir, exist_ok=True)

    num_student_steps = cfg.training.num_student_steps  # 1-4

    while not done:
        # ──────────────────────────────────────────────────
        # Step A: Generate student samples
        # ──────────────────────────────────────────────────
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

        # Student generates residual prediction (no grad needed — only used
        # by discriminator which detaches it; Step C re-generates with grads)
        with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype, enabled=enable_amp):
            student_output = _student_generate(
                ddp_student,
                condition,
                target.shape,
                sigma_max=cfg.model.sigma_max,
                num_steps=num_student_steps,
            )

        disc_condition = condition if cfg.training.condition_discriminator else None

        # ──────────────────────────────────────────────────
        # LR rampup (applied before both D and G steps for consistency)
        # ──────────────────────────────────────────────────
        lr_ramp = min(total_steps / max(cfg.training.lr_rampup_steps, 1e-8), 1.0)
        for g in optimizer_G.param_groups:
            g["lr"] = cfg.training.lr_G * lr_ramp
        for g in optimizer_D.param_groups:
            g["lr"] = cfg.training.lr_D * lr_ramp

        # ──────────────────────────────────────────────────
        # Step B: Update Discriminator
        # ──────────────────────────────────────────────────
        optimizer_D.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=enable_amp):
            loss_D, log_D = loss_fn.discriminator_loss(
                real=target,
                fake=student_output,
                discriminator=ddp_disc,
                disc_condition=disc_condition,
            )
        loss_D.backward()

        if cfg.training.clip_grad_norm > 0:
            clip_grad_norm_(discriminator.parameters(), cfg.training.clip_grad_norm)
        for param in discriminator.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0.0, posinf=1e5, neginf=-1e5, out=param.grad)
        optimizer_D.step()

        # ──────────────────────────────────────────────────
        # Step C: Update Student (Generator)
        # ──────────────────────────────────────────────────
        optimizer_G.zero_grad(set_to_none=True)

        # Re-generate with gradients for the student
        with torch.autocast("cuda", dtype=amp_dtype, enabled=enable_amp):
            student_output = _student_generate(
                ddp_student,
                condition,
                target.shape,
                sigma_max=cfg.model.sigma_max,
                num_steps=num_student_steps,
            )
            loss_G, log_G = loss_fn.generator_loss(
                student_output=student_output,
                teacher=teacher,
                discriminator=ddp_disc,
                condition=condition,
                disc_condition=disc_condition,
            )
        loss_G.backward()

        if cfg.training.clip_grad_norm > 0:
            clip_grad_norm_(student.parameters(), cfg.training.clip_grad_norm)

        for param in student.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0.0, posinf=1e5, neginf=-1e5, out=param.grad)
        optimizer_G.step()

        # Update EMA
        ema.update(student, decay=cfg.training.ema_decay)
        ema.apply_shadow(ema_net)

        if dist.world_size > 1 and _distributed_ready():
            torch.distributed.barrier()

        avg_loss_G += log_G["loss_G_total"]
        avg_loss_D += log_D["loss_D_total"]
        train_steps += 1

        if log_to_wandb:
            wandb_logs.update(log_G)
            wandb_logs.update(log_D)
            wandb_logs["lr_G"] = optimizer_G.param_groups[0]["lr"]
            wandb_logs["lr_D"] = optimizer_D.param_groups[0]["lr"]

        total_steps += 1
        done = total_steps >= total_train_steps

        if (
            dist.rank == 0
            and total_steps % cfg.training.print_progress_freq == 0
            and train_steps > 0
        ):
            try:
                avg_g = avg_loss_G / train_steps
                _log_train_loss_csv(rundir, total_steps, float(avg_g))
            except Exception as e:
                logger0.warn(f"Failed to write train_loss.csv at step {total_steps}: {e}")

        # ──────────────────────────────────────────────────
        # Validation
        # ──────────────────────────────────────────────────
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

                    # Generate with EMA student
                    output_images = add_model_forward(
                        ema_net,
                        condition,
                        state[1].shape,
                        sigma_max=cfg.model.sigma_max,
                        num_steps=num_student_steps,
                    )
                    if "regression" in condition_list and reg_out is not None:
                        output_images = output_images + reg_out

            if dist.rank == 0:
                # Accumulators
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

                # valid_loss.csv (log generator loss as the validation metric)
                try:
                    _log_valid_loss_csv(rundir, total_steps, avg_loss_G / max(train_steps, 1))
                except Exception as e:
                    logger0.warn(
                        f"Failed to write valid_loss.csv at step {total_steps}: {e}"
                    )

                # rmse/mae CSVs
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

                # ps1d CSVs
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
                f"[Validation] Completed at step {total_steps}: "
                f"G_loss={avg_loss_G / max(train_steps, 1):.4f}, "
                f"D_loss={avg_loss_D / max(train_steps, 1):.4f}, "
                f"time={valid_time:.2f}s"
            )

        # Console stats
        current_time = time.time()
        if total_steps % cfg.training.print_progress_freq == 0:
            fields = []
            fields += [f"steps {total_steps:<5d}"]
            fields += [f"samples {total_steps * batch_size}"]
            fields += [f"tot_time {current_time - start_time:.2f}"]
            fields += [
                f"step_time {(current_time - train_start - max(valid_time, 0.0)) / max(train_steps, 1):.2f}"
            ]
            fields += [f"valid_time {max(valid_time, 0.0):.2f}"]
            fields += [
                f"cpumem {psutil.Process(os.getpid()).memory_info().rss / 2**30:<6.2f}"
            ]
            fields += [
                f"gpumem {torch.cuda.max_memory_allocated(device) / 2**30:<6.2f}"
            ]
            fields += [f"loss_G {avg_loss_G / max(train_steps, 1):<6.3f}"]
            fields += [f"loss_D {avg_loss_D / max(train_steps, 1):<6.3f}"]
            fields += [f"lambda_distill {cfg.training.lambda_distill:.2f}"]
            logger0.info(" ".join(fields))

            train_steps = 0
            train_start = time.time()
            avg_loss_G = 0.0
            avg_loss_D = 0.0
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
                optimizer=optimizer_G,
                epoch=total_steps,
            )
            # Save discriminator
            save_checkpoint(
                path=os.path.join(cfg.training.rundir, "checkpoints_disc"),
                models=discriminator,
                optimizer=optimizer_D,
                epoch=total_steps,
            )
            # Save EMA state
            torch.save(ema.state_dict(), ema_ckpt_path)
            logger0.info(f"Saved checkpoint at step {total_steps}.")

    if _distributed_ready():
        torch.distributed.barrier()
    logger0.info("\nExiting...")
