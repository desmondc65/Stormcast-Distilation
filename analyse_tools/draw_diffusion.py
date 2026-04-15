"""Visualise the EDM teacher's 18-step Heun sampling trajectory on one
validation sample and plot the u10 channel as a horizontal film strip.

Mirrors the teacher inference path used by ``analyse_progressive.py``
(regression mean M_t + EDM residual, Karras ρ-schedule, Heun solver), but
instruments the sampling loop so we keep the **denoised estimate**
``x̂_0 = net(x_cur, σ)`` after every one of the ``num_steps`` iterations.
Plotting ``x_next`` (the noisy sample mid-trajectory) just shows noise at
high σ — the denoised estimate is what actually progresses from blurry to
sharp across the strip.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
STORMCAST_ROOT = REPO_ROOT / "stormcast"
sys.path.insert(0, str(STORMCAST_ROOT))

from physicsnemo.distributed import DistributedManager  # noqa: E402
from physicsnemo.models import Module  # noqa: E402
from physicsnemo.models.diffusion import EDMPrecond  # noqa: E402

from datasets import dataset_classes  # noqa: E402
from utils.nn import build_network_condition_and_target  # noqa: E402


# Local defaults mirror analyse_progressive.py / CLAUDE.md §1
DEFAULT_DATA_ROOT = REPO_ROOT / "exp_3_train_2_5_yrs_val_1yr_tp1" / "zarr_exp3_L_24_H_24_train_2_5_years_full"
DEFAULT_REGRESSION = REPO_ROOT / "exp_3_train_2_5_yrs_val_1yr_tp1" / "exp_3_reg_L_24_H_4_train_2_5_years" / "0" / "checkpoints_regression" / "StormCastUNet.0.14000.mdlus"
DEFAULT_TEACHER = REPO_ROOT / "exp_3_train_2_5_yrs_val_1yr_tp1" / "exp_3_dif_L_24_H_4_train_2_5_years" / "0" / "checkpoints_diffusion" / "EDMPrecond.0.70000.mdlus"

DIFFUSION_CONDITIONS = ["state", "regression", "invariant"]
REGRESSION_CONDITIONS = ["state", "background", "invariant"]


def make_dataset_cfg(data_location: Path, valid_dates: tuple[str, str]):
    return OmegaConf.create({
        "location": str(data_location),
        "HighRes_img_size": [224, 128],
        "dt": 1,
        "exp_train_zarrs": ["stormcast_test_train"],
        "train_dates": ["2019/08/01", "2020/08/31"],
        "exp_valid_zarrs": ["stormcast_test_valid"],
        "valid_dates": list(valid_dates),
        "invariants": ["lsm", "orog"],
        "input_channels": "all",
        "diffusion_channels": ["t2m", "u10", "v10", "qpepre"],
        "kept_LowRes_channels": "all",
        "kept_HighRes_channels": "all",
    })


@torch.no_grad()
def heun_sample_with_trace(
    net: torch.nn.Module,
    latents: torch.Tensor,
    condition: torch.Tensor,
    num_steps: int = 18,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    dtype: torch.dtype = torch.float64,
) -> tuple[list[torch.Tensor], list[float]]:
    """Heun Karras sampler that returns every intermediate ``x_next``.

    Matches ``physicsnemo.utils.diffusion.deterministic_sampler`` with
    ``discretization='edm'``, ``schedule='linear'``, ``scaling='none'``,
    ``solver='heun'``, ``S_churn=0``. Conditioning is passed via the
    EDMPrecond ``condition=`` kwarg.
    """
    assert isinstance(net, EDMPrecond), "teacher is expected to be EDMPrecond"

    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    step_indices = torch.arange(num_steps, dtype=dtype, device=latents.device)
    sigma_steps = (
        sigma_max ** (1 / rho)
        + step_indices / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = net.round_sigma(sigma_steps)
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0

    x_next = latents.to(dtype) * t_steps[0]
    trace: list[torch.Tensor] = []
    sigmas: list[float] = []

    for i in range(num_steps):
        t_cur, t_next = t_steps[i], t_steps[i + 1]
        x_cur = x_next

        denoised = net(x_cur, t_cur, condition=condition).to(dtype)
        trace.append(denoised.detach().clone())
        sigmas.append(float(t_cur))

        d_cur = (x_cur - denoised) / t_cur
        x_next = x_cur + (t_next - t_cur) * d_cur

        if i < num_steps - 1:
            denoised2 = net(x_next, t_next, condition=condition).to(dtype)
            d_prime = (x_next - denoised2) / t_next
            x_next = x_cur + (t_next - t_cur) * 0.5 * (d_cur + d_prime)

    return trace, sigmas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-location", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--regression-checkpoint", type=Path, default=DEFAULT_REGRESSION)
    ap.add_argument("--teacher-checkpoint", type=Path, default=DEFAULT_TEACHER)
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / "analyse_tools" / "teacher_diffusion_u10.png")
    ap.add_argument("--sample-index", type=int, default=0)
    ap.add_argument("--num-steps", type=int, default=18)
    ap.add_argument("--sigma-min", type=float, default=0.002)
    ap.add_argument("--sigma-max", type=float, default=80.0)
    ap.add_argument("--rho", type=float, default=7.0)
    ap.add_argument("--valid-dates", nargs=2,
                    default=["2022/01/01", "2022/12/31"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--channel", type=str, default="u10")
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    DistributedManager.initialize()
    device = DistributedManager().device
    torch.cuda.empty_cache()

    dataset_cfg = make_dataset_cfg(args.data_location, tuple(args.valid_dates))
    dataset_cls = dataset_classes["data_loader_rwrf_era5_stable.Dataset"]
    dataset = dataset_cls(dataset_cfg, train=False)
    channels = dataset.state_channels()
    if args.channel not in channels:
        raise SystemExit(f"channel {args.channel!r} not in dataset channels {channels}")
    ch_idx = channels.index(args.channel)
    print(f"[dataset] {len(dataset)} samples, channels={channels}, plotting {args.channel} (idx {ch_idx})")

    invariant_array = dataset.get_invariants()
    invariant_tensor = (
        torch.from_numpy(invariant_array).to(device=device, dtype=torch.float32).unsqueeze(0)
        if invariant_array is not None else None
    )

    batch = dataset[args.sample_index]
    state_in_t = batch["state"][0].to(device=device, dtype=torch.float32).unsqueeze(0)
    state_tar_t = batch["state"][1].to(device=device, dtype=torch.float32).unsqueeze(0)
    background_t = batch["background"].to(device=device, dtype=torch.float32).unsqueeze(0)

    print(f"[regression] {args.regression_checkpoint}")
    regression_model = Module.from_checkpoint(str(args.regression_checkpoint)).to(device).eval()
    print(f"[teacher]    {args.teacher_checkpoint}")
    teacher = Module.from_checkpoint(str(args.teacher_checkpoint)).to(device).eval()

    with torch.no_grad():
        condition, _, regression_output = build_network_condition_and_target(
            background_t, [state_in_t, state_in_t], invariant_tensor,
            regression_net=regression_model,
            condition_list=DIFFUSION_CONDITIONS,
            regression_condition_list=REGRESSION_CONDITIONS,
        )
        if regression_output is None:
            regression_output = torch.zeros_like(state_in_t)

        torch.manual_seed(args.seed)
        latents = torch.randn(*state_in_t.shape, device=device, dtype=condition.dtype)

        trace, sigmas = heun_sample_with_trace(
            teacher, latents, condition,
            num_steps=args.num_steps,
            sigma_min=args.sigma_min, sigma_max=args.sigma_max, rho=args.rho,
        )

    reg_np = regression_output.cpu().numpy()[0]  # (C, H, W)
    frames = []
    for x in trace:
        resid = x.to(torch.float32).cpu().numpy()[0]
        full = reg_np + resid
        full_denorm = dataset.denormalize_state(full.copy())
        frames.append(full_denorm[ch_idx])

    truth = dataset.denormalize_state(state_tar_t.cpu().numpy()[0].copy())[ch_idx]

    # Shared vmin/vmax across the strip so the denoising progression is visible.
    stacked = np.stack(frames + [truth])
    vmin, vmax = float(stacked.min()), float(stacked.max())

    n = len(frames)
    fig, axes = plt.subplots(1, n, figsize=(1.5 * n + 1.0, 2.8), squeeze=False,
                             constrained_layout=True)
    for i, (ax, f, sig) in enumerate(zip(axes[0], frames, sigmas)):
        im = ax.imshow(f, origin="lower", cmap="viridis",
                       vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"step {i+1}\nσ={sig:.2g}", fontsize=8)
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.015, pad=0.01, shrink=0.9)
    fig.suptitle(f"EDM teacher Heun sampling trajectory — {args.channel} "
                 f"(sample idx {args.sample_index}, N={n} steps)", fontsize=11)
    fig.savefig(args.output, dpi=140)
    plt.close(fig)
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
