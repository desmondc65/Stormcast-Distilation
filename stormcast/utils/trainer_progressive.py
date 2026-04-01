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

"""Progressive Distillation training loop (Salimans & Ho, 2022).

Iteratively halves the number of sampling steps required by a pre-trained
EDM diffusion model. Each phase trains a student to replicate the teacher's
two-step ODE integration in a single step, then promotes the student to
teacher for the next phase.
"""

import copy
import math
import os
import time
from collections import defaultdict

import numpy as np
import psutil
import torch
import matplotlib.pyplot as plt
import wandb

from physicsnemo.models import Module
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.diffusion import InfiniteSampler
from physicsnemo.launch.utils import save_checkpoint, load_checkpoint
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.experimental.metrics.diffusion import ProgressiveDistillationLoss

from .nn import (
    get_preconditioned_architecture,
    build_network_condition_and_target,
    diffusion_model_forward,
)
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


logger = PythonLogger("train_progressive")


def _distributed_ready() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def compute_num_phases(initial_steps: int, target_steps: int) -> int:
    """Compute the number of distillation phases needed.

    Each phase halves the step count. Returns ceil(log2(initial/target)).
    """
    if target_steps >= initial_steps:
        return 0
    return math.ceil(math.log2(initial_steps / target_steps))


def progressive_distillation_loop(cfg):
    """Main training loop for Progressive Distillation.

    Iterates through distillation phases, halving the sampling step count
    each time. Within each phase, trains a student EDMPrecond to match
    the teacher's two-step ODE output with a single step.
    """

    start_time = time.time()
    dist = DistributedManager()
    device = dist.device
    logger0 = RankZeroLoggingWrapper(logger, dist)

    # ── Hyperparameters ──────────────────────────────────────────────────
    batch_size = cfg.training.batch_size
    if cfg.training.batch_size_per_gpu == "auto":
        local_batch_size = batch_size // dist.world_size
    else:
        local_batch_size = cfg.training.batch_size_per_gpu
    assert batch_size % (local_batch_size * dist.world_size) == 0
    num_accumulation_rounds = batch_size // (local_batch_size * dist.world_size)

    log_to_wandb = cfg.training.log_to_wandb
    condition_list = cfg.model.diffusion_conditions

    initial_steps = cfg.training.initial_num_steps
    target_steps = cfg.training.target_num_steps
    steps_per_phase = cfg.training.steps_per_phase
    num_phases = compute_num_phases(initial_steps, target_steps)

    logger0.info(
        f"Progressive Distillation: {initial_steps} -> {target_steps} steps "
        f"over {num_phases} phases, {steps_per_phase} training steps each."
    )

    # ── Seed & performance ───────────────────────────────────────────────
    np.random.seed((cfg.training.seed * dist.world_size + dist.rank) % (1 << 31))
    torch.manual_seed(cfg.training.seed)
    torch.backends.cudnn.benchmark = cfg.training.cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    fp_optimizations = cfg.training.fp_optimizations
    enable_amp = fp_optimizations.startswith("amp")
    amp_dtype = torch.float16 if fp_optimizations == "amp-fp16" else torch.bfloat16

    # ── Dataset ──────────────────────────────────────────────────────────
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

    # ── Regression net (optional conditioning) ───────────────────────────
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

    # ── Network dimensions ───────────────────────────────────────────────
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

    # ── Loss function ────────────────────────────────────────────────────
    loss_fn = ProgressiveDistillationLoss(
        sigma_min=cfg.model.sigma_min,
        sigma_max=cfg.model.sigma_max,
        sigma_data=cfg.model.sigma_data,
        rho=cfg.training.rho,
        num_steps=initial_steps,
        loss_weighting=cfg.training.loss_weighting,
    )

    # ── Load initial teacher ─────────────────────────────────────────────
    logger0.info(f"Loading pre-trained teacher from {cfg.model.teacher_weights}...")
    teacher = Module.from_checkpoint(cfg.model.teacher_weights)
    teacher = teacher.to(device).eval().requires_grad_(False)

    rundir = cfg.training.rundir
    os.makedirs(rundir, exist_ok=True)

    # ── Determine resume state ───────────────────────────────────────────
    resume_phase, resume_step = _detect_resume_state(rundir, num_phases)
    if resume_phase > 0:
        logger0.info(
            f"Resuming from phase {resume_phase}, step {resume_step}. "
            f"Loading teacher from phase {resume_phase - 1} student checkpoint."
        )

    global_step = 0  # Cumulative step counter across all phases

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE LOOP
    # ══════════════════════════════════════════════════════════════════════
    for phase in range(num_phases):
        N_student = initial_steps // (2 ** (phase + 1))
        N_student = max(N_student, target_steps)
        loss_fn.set_num_steps(N_student)

        phase_dir = os.path.join(rundir, f"phase_{phase}")
        os.makedirs(phase_dir, exist_ok=True)

        logger0.info(
            f"\n{'='*60}\n"
            f"  Phase {phase}/{num_phases - 1}: "
            f"Teacher {initial_steps // (2 ** phase)} steps -> "
            f"Student {N_student} steps\n"
            f"{'='*60}"
        )

        # Skip fully completed phases on resume
        if phase < resume_phase:
            teacher = _load_phase_student_as_teacher(
                phase_dir, arch_kwargs, device, logger0
            )
            global_step += steps_per_phase
            continue

        # ── Load teacher for this phase (from previous student) ──────
        if phase > 0 and phase == resume_phase:
            prev_phase_dir = os.path.join(rundir, f"phase_{phase - 1}")
            teacher = _load_phase_student_as_teacher(
                prev_phase_dir, arch_kwargs, device, logger0
            )
        elif phase > 0:
            # teacher was already set at end of previous phase iteration
            pass

        # ── Build student, initialized from teacher ──────────────────
        student = get_preconditioned_architecture(name="diffusion", **arch_kwargs)
        student = student.to(device)
        student.load_state_dict(teacher.state_dict())
        student.train().requires_grad_(True)

        optimizer = torch.optim.Adam(
            student.parameters(), lr=cfg.training.lr
        )
        ddp = torch.nn.parallel.DistributedDataParallel(
            student, device_ids=[device], broadcast_buffers=False
        )

        # ── Resume within phase ──────────────────────────────────────
        ckpt_path = os.path.join(phase_dir, "checkpoints")
        phase_step = 0
        loaded = load_checkpoint(
            path=ckpt_path,
            models=student,
            optimizer=optimizer,
            epoch=None
            if cfg.training.resume_checkpoint == "latest"
            else cfg.training.resume_checkpoint,
        )
        if loaded > 0:
            phase_step = loaded
            logger0.info(f"  Resumed phase {phase} from step {phase_step}.")

        # ── Phase training loop ──────────────────────────────────────
        wandb_logs = {}
        avg_train_loss = 0.0
        train_steps_logged = 0
        train_start = time.time()
        valid_time = -1.0
        val_loss = -1.0
        validation_counter = 0

        while phase_step < steps_per_phase:
            optimizer.zero_grad(set_to_none=True)

            for _ in range(num_accumulation_rounds):
                batch = next(dataset_iterator)
                background = batch["background"].to(
                    device=device, dtype=torch.float32
                )
                state = [
                    s.to(device=device, dtype=torch.float32) for s in batch["state"]
                ]

                with torch.autocast("cuda", dtype=amp_dtype, enabled=enable_amp):
                    condition, target, reg_out = build_network_condition_and_target(
                        background,
                        state,
                        invariant_tensor,
                        regression_net=regression_net,
                        condition_list=condition_list,
                        regression_condition_list=cfg.model.regression_conditions,
                    )
                    loss = loss_fn(
                        student=ddp,
                        teacher=teacher,
                        images=target,
                        condition=condition,
                    )

                loss_value = loss.sum() / len(state_channels)
                loss_value.backward()

            # ── Gradient processing & optimizer step ──────────────────
            if cfg.training.clip_grad_norm > 0:
                clip_grad_norm_(student.parameters(), cfg.training.clip_grad_norm)

            lr_warmup = min(
                phase_step / max(cfg.training.lr_rampup_steps, 1e-8), 1.0
            )
            for g in optimizer.param_groups:
                g["lr"] = cfg.training.lr * lr_warmup
                if log_to_wandb:
                    wandb_logs["lr"] = g["lr"]

            for param in student.parameters():
                if param.grad is not None:
                    torch.nan_to_num(
                        param.grad,
                        nan=0.0,
                        posinf=1e5,
                        neginf=-1e5,
                        out=param.grad,
                    )

            optimizer.step()

            # ── Distributed loss sync ─────────────────────────────────
            if dist.world_size > 1 and _distributed_ready():
                torch.distributed.barrier()
                torch.distributed.all_reduce(
                    loss, op=torch.distributed.ReduceOp.AVG
                )

            # Log MSE against ground truth (not distillation loss)
            with torch.no_grad():
                gt_sampler_args = dict(
                    num_steps=N_student,
                    sigma_min=cfg.model.sigma_min,
                    sigma_max=cfg.model.sigma_max,
                    rho=cfg.training.rho,
                    solver="euler",
                )
                train_output = diffusion_model_forward(
                    student, condition, target.shape, gt_sampler_args
                )
                if "regression" in condition_list and reg_out is not None:
                    train_output = train_output + reg_out
                gt_mse = ((train_output - target) ** 2).mean().cpu().item()

            avg_train_loss += gt_mse
            train_steps_logged += 1
            phase_step += 1
            global_step += 1

            if log_to_wandb:
                wandb_logs["loss"] = gt_mse
                wandb_logs["phase"] = phase
                wandb_logs["N_student"] = N_student

            # ── CSV logging ───────────────────────────────────────────
            if (
                dist.rank == 0
                and phase_step % cfg.training.print_progress_freq == 0
                and train_steps_logged > 0
            ):
                try:
                    avg_loss_to_log = avg_train_loss / train_steps_logged
                    _log_train_loss_csv(phase_dir, global_step, float(avg_loss_to_log))
                except Exception as e:
                    logger0.warning(
                        f"Failed to write train_loss.csv at step {global_step}: {e}"
                    )

            # ── Validation ────────────────────────────────────────────
            if phase_step % cfg.training.validation_freq == 0:
                validation_counter += 1
                valid_start = time.time()
                logger0.info(
                    f"[Phase {phase} | Validation] Step {phase_step}/{steps_per_phase}..."
                )
                batch = next(valid_dataset_iterator)

                with torch.no_grad():
                    background = batch["background"].to(
                        device=device, dtype=torch.float32
                    )
                    state = [
                        s.to(device=device, dtype=torch.float32)
                        for s in batch["state"]
                    ]
                    with torch.autocast(
                        "cuda", dtype=amp_dtype, enabled=enable_amp
                    ):
                        condition, target, reg_out = build_network_condition_and_target(
                            background,
                            state,
                            invariant_tensor,
                            regression_net=regression_net,
                            condition_list=condition_list,
                            regression_condition_list=cfg.model.regression_conditions,
                        )

                        # Sample with student using N_student steps
                        sampler_args = dict(
                            num_steps=N_student,
                            sigma_min=cfg.model.sigma_min,
                            sigma_max=cfg.model.sigma_max,
                            rho=cfg.training.rho,
                            solver="euler",
                        )
                        output_images = diffusion_model_forward(
                            student, condition, state[1].shape, sampler_args
                        )
                        if "regression" in condition_list and reg_out is not None:
                            output_images = output_images + reg_out

                        # Validation loss: MSE against ground truth
                        valid_loss = ((output_images - target) ** 2).mean()

                    if dist.world_size > 1 and _distributed_ready():
                        torch.distributed.barrier()
                        torch.distributed.all_reduce(
                            valid_loss, op=torch.distributed.ReduceOp.AVG
                        )

                    val_loss = valid_loss.detach().cpu().item()
                    if log_to_wandb:
                        wandb_logs["valid_loss"] = val_loss

                # ── Validation metrics (rank 0 only) ──────────────────
                if dist.rank == 0:
                    _log_validation_metrics(
                        cfg,
                        phase_dir,
                        global_step,
                        validation_counter,
                        output_images,
                        state,
                        state_channels,
                        val_loss,
                        wandb_logs,
                        log_to_wandb,
                        logger0,
                    )

                valid_time = time.time() - valid_start
                logger0.info(
                    f"[Validation] loss={val_loss:.4f}, time={valid_time:.2f}s"
                )

            # ── Console stats ─────────────────────────────────────────
            current_time = time.time()
            if phase_step % cfg.training.print_progress_freq == 0:
                fields = [
                    f"phase {phase}",
                    f"step {phase_step}/{steps_per_phase}",
                    f"global {global_step}",
                    f"N_student {N_student}",
                    f"tot_time {current_time - start_time:.2f}",
                    f"step_time {(current_time - train_start - max(valid_time, 0.0)) / max(train_steps_logged, 1):.2f}",
                    f"cpumem {psutil.Process(os.getpid()).memory_info().rss / 2**30:<6.2f}",
                    f"gpumem {torch.cuda.max_memory_allocated(device) / 2**30:<6.2f}",
                    f"train_loss {avg_train_loss / max(train_steps_logged, 1):<6.3f}",
                    f"val_loss {val_loss:<6.3f}",
                ]
                logger0.info(" ".join(fields))

                train_steps_logged = 0
                train_start = time.time()
                avg_train_loss = 0.0
                torch.cuda.reset_peak_memory_stats()

            # ── Checkpointing ─────────────────────────────────────────
            done_phase = phase_step >= steps_per_phase
            if (
                (done_phase or phase_step % cfg.training.checkpoint_freq == 0)
                and phase_step != 0
                and dist.rank == 0
            ):
                save_checkpoint(
                    path=ckpt_path,
                    models=student,
                    optimizer=optimizer,
                    epoch=phase_step,
                )
                # Also save the bare student model for next-phase teacher loading
                student.save(os.path.join(phase_dir, "student_final.mdlus"))
                logger0.info(
                    f"Saved checkpoint at phase {phase}, step {phase_step}."
                )

            if log_to_wandb and dist.rank == 0:
                wandb.log(wandb_logs, step=global_step)
                wandb_logs = {}

        # ── Phase complete: promote student to teacher ────────────────
        logger0.info(f"Phase {phase} complete. Student ({N_student} steps) promoted to teacher.")
        teacher = copy.deepcopy(student)
        teacher.eval().requires_grad_(False)

        # Clean up phase-local objects
        del ddp, optimizer, student
        torch.cuda.empty_cache()

        # Stop if student reached target
        if N_student <= target_steps:
            logger0.info(
                f"Target step count {target_steps} reached. Distillation complete."
            )
            break

    if _distributed_ready():
        torch.distributed.barrier()
    logger0.info(f"\nProgressive Distillation finished in {time.time() - start_time:.1f}s.")


# ══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════════════════════


def _detect_resume_state(rundir: str, num_phases: int):
    """Detect which phase/step to resume from based on saved checkpoints.

    Returns (resume_phase, resume_step). If no checkpoints exist, returns (0, 0).
    """
    for phase in range(num_phases - 1, -1, -1):
        phase_dir = os.path.join(rundir, f"phase_{phase}")
        student_path = os.path.join(phase_dir, "student_final.mdlus")
        if os.path.exists(student_path):
            # This phase is fully complete; resume from the NEXT phase
            return phase + 1, 0
        ckpt_dir = os.path.join(phase_dir, "checkpoints")
        if os.path.isdir(ckpt_dir) and any(
            f.endswith(".pt") for f in os.listdir(ckpt_dir)
        ):
            # This phase has partial checkpoints; resume within it
            return phase, 0  # load_checkpoint will find the latest
    return 0, 0


def _load_phase_student_as_teacher(phase_dir, arch_kwargs, device, logger0):
    """Load a completed phase's student checkpoint as a frozen teacher."""
    student_path = os.path.join(phase_dir, "student_final.mdlus")
    logger0.info(f"Loading teacher from {student_path}...")
    teacher = Module.from_checkpoint(student_path)
    teacher = teacher.to(device).eval().requires_grad_(False)
    return teacher


def _log_validation_metrics(
    cfg,
    phase_dir,
    global_step,
    validation_counter,
    output_images,
    state,
    state_channels,
    val_loss,
    wandb_logs,
    log_to_wandb,
    logger0,
):
    """Compute and log per-field RMSE, MAE, PS1D, and images (rank 0 only)."""
    fields = cfg.training.validation_plot_variables

    rmse_acc = defaultdict(float)
    mae_acc = defaultdict(float)
    rmse_count = defaultdict(int)
    ps_acc = {
        f: {"Pk_gen_sum": None, "Pk_tar_sum": None, "k": None, "count": 0}
        for f in fields
    }

    for i in range(output_images.shape[0]):
        image = output_images[i].detach().cpu().numpy()

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
            image_dir = os.path.join(phase_dir, "images", f_)
            os.makedirs(image_dir, exist_ok=True)

            generated = image[f_index]
            truth = state[1][i, f_index].detach().cpu().numpy()

            fig = validation_plot(
                generated,
                truth,
                f_,
                experiment_name=cfg.training.experiment_name,
                step=global_step,
            )
            fig.savefig(os.path.join(image_dir, f"{global_step}_{i}_{f_}.png"))

            specfig = "PS1D_" + f_
            figs[specfig].savefig(
                os.path.join(image_dir, f"{global_step}_{i}_{f_}_spec.png")
            )

            output_nc_enabled = getattr(cfg.training, "output_nc", False)
            output_nc_freq = getattr(cfg.training, "output_nc_freq", 1)
            if output_nc_enabled and (validation_counter % output_nc_freq == 0):
                try:
                    _save_validation_netcdf(
                        phase_dir, global_step, i, f_, generated, truth
                    )
                except Exception as e:
                    logger0.warning(
                        f"Failed to write NetCDF for {f_} at step {global_step}: {e}"
                    )

            if log_to_wandb:
                for figname, plot in figs.items():
                    wandb_logs[figname] = wandb.Image(plot)
                wandb_logs.update({f"generated_{f_}": wandb.Image(fig)})

        plt.close("all")

    # CSV logs
    try:
        _log_valid_loss_csv(phase_dir, global_step, val_loss)
    except Exception as e:
        logger0.warning(f"Failed to write valid_loss.csv at step {global_step}: {e}")

    try:
        for f_ in fields:
            count = max(rmse_count[f_], 1)
            rmse_mean = rmse_acc[f_] / count
            mae_mean = mae_acc[f_] / count
            _log_rmse_field_csv(phase_dir, global_step, f_, rmse_mean)
            _log_mae_field_csv(phase_dir, global_step, f_, mae_mean)
    except Exception as e:
        logger0.warning(f"Failed to write rmse/mae CSVs at step {global_step}: {e}")

    try:
        for f_ in fields:
            acc = ps_acc[f_]
            if acc["count"] == 0:
                continue
            k = acc["k"]
            pk_gen_mean = acc["Pk_gen_sum"] / acc["count"]
            pk_tar_mean = acc["Pk_tar_sum"] / acc["count"]
            _log_ps1d_field_csv(phase_dir, global_step, f_, k, pk_gen_mean, pk_tar_mean)
    except Exception as e:
        logger0.warning(f"Failed to write ps1d CSVs at step {global_step}: {e}")

    if log_to_wandb:
        wandb_logs.update(spec_ratios)

    try:
        _plot_loss_curves(phase_dir, cfg.training.experiment_name)
    except Exception as e:
        logger0.warning(f"Failed to plot loss curves at step {global_step}: {e}")
