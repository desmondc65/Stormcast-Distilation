#!/usr/bin/env python3
"""Per-step inference visualizations for StormCast (EDM) and FlowCast (CFM).

Picks N (default 3) evenly-spaced initial times from the 2022 validation
year, runs both samplers, and writes ONE bare PNG per (channel, step) so
the sampling trajectory can be watched frame-by-frame, separately for each
variable:

    StormCast (EDM, default 18 Heun steps)  -> 18 PNGs/timestamp/channel
    FlowCast  (CFM, default 10 Euler steps) -> 10 PNGs/timestamp/channel
    + 1 truth.png and 1 regression.png per channel per timestamp

Each PNG is a bare ``imshow`` with ``origin='lower'``, default viridis,
``aspect='auto'`` -- no title, no axes, no colorbar, no frame. ``vmin/vmax``
is locked to the truth field for the same (timestamp, channel) so the same
color mapping is used across every step, making convergence into the
correct value range visible (early diffusion steps saturate -- that's the
sigma_max noise floor, by design).

This script mirrors the EDM Heun loop in
``physicsnemo.utils.diffusion.deterministic_sampler`` and the FlowCast Euler
loop in ``stormcast.utils.nn.flowcast_model_forward`` -- it is NOT a wrapper
around them, because those return only the final state. The math is the
same: linear-schedule EDM with the Karras rho-spaced sigma schedule, no
S_churn / no patching / no class labels (matching the StormCast-on-RWRF
inference path).

Two views are written for every sampler step:

    forecast view  -- M_t + R_t denormalized to physical units (the actual
                      one-hour-ahead state prediction)
    residual view  -- forecast - regression_phys, i.e. the raw residual R_t
                      in physical units (with the qpepre log1p nonlinearity
                      handled correctly because each term is denormalized
                      independently)

The forecast view uses vmin/vmax taken globally from the truth fields; the
residual view uses vmin/vmax taken globally from (truth - regression). Both
scales stay fixed across every step and every timestamp.

Output:
    <output-dir>/time_NN_<timestamp>/
        truth/<channel>.png
        regression/<channel>.png
        truth_residual/<channel>.png
        stormcast/<channel>/step_KK.png            (forecast, KK = 01..N_diff)
        stormcast_residual/<channel>/step_KK.png   (raw residual)
        flowcast/<channel>/step_KK.png             (forecast, KK = 01..N_flow)
        flowcast_residual/<channel>/step_KK.png    (raw residual)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

# --- repo path bootstrap -----------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
STORMCAST_ROOT = REPO_ROOT / "stormcast"
sys.path.insert(0, str(STORMCAST_ROOT))

from physicsnemo.distributed import DistributedManager  # noqa: E402
from physicsnemo.models import Module  # noqa: E402

from datasets import dataset_classes  # noqa: E402
from utils.nn import build_network_condition_and_target  # noqa: E402


PLOT_CHANNELS = ["t2m", "u10", "v10", "qpepre"]

DIFFUSION_CONDITIONS = ["state", "regression", "invariant"]
REGRESSION_CONDITIONS = ["state", "background", "invariant"]

CLEANED_CHANNEL_ORDER = ["u10", "v10", "t2m", "qpepre"]


def make_dataset_cfg(data_loc, valid_dates, hr_size, qpepre_log1p, kept_HR):
    return OmegaConf.create(
        {
            "location": str(data_loc),
            "HighRes_img_size": list(hr_size),
            "dt": 1,
            "exp_train_zarrs": ["stormcast_test_train"],
            "train_dates": ["2019/08/01", "2021/12/31"],
            "exp_valid_zarrs": ["stormcast_test_valid"],
            "valid_dates": list(valid_dates),
            "invariants": ["lsm", "orog"],
            "input_channels": "all",
            "diffusion_channels": list(kept_HR),
            "kept_LowRes_channels": "all",
            "kept_HighRes_channels": list(kept_HR),
            "qpepre_log1p": qpepre_log1p,
        }
    )


def build_dataset(data_loc, valid_dates, hr_size, qpepre_log1p, kept_HR, device):
    cfg = make_dataset_cfg(data_loc, valid_dates, hr_size, qpepre_log1p, kept_HR)
    dataset_cls = dataset_classes["data_loader_rwrf_era5_stable.Dataset"]
    dataset = dataset_cls(cfg, train=False)
    inv = dataset.get_invariants()
    invariant_tensor = (
        torch.from_numpy(inv).to(device=device, dtype=torch.float32).unsqueeze(0)
        if inv is not None
        else None
    )
    return dataset, invariant_tensor


# --- Samplers that record intermediates --------------------------------------
@torch.no_grad()
def edm_sampler_with_intermediates(
    net, latents, condition, num_steps=18, sigma_min=0.002, sigma_max=80.0,
    rho=7.0, solver="heun",
):
    """EDM linear-schedule sampler (matches deterministic_sampler with default
    EDM discretization, S_churn=0, no patching, no class labels) but yields the
    sample state x_next AFTER each of the ``num_steps`` outer iterations.

    Returns a list of ``num_steps`` tensors, each with the same shape as
    ``latents``. The k-th entry is the sample at the end of step k+1 (so
    entry 0 is the result of the first Euler/Heun step and the last entry is
    the final clean sample equivalent to what ``deterministic_sampler``
    returns).
    """
    device = latents.device
    dtype = torch.float64

    step_indices = torch.arange(num_steps, device=device, dtype=dtype)
    sigma_steps = (
        sigma_max ** (1 / rho)
        + step_indices / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    sigma_steps = net.round_sigma(sigma_steps)
    t_steps = torch.cat([sigma_steps, torch.zeros_like(sigma_steps[:1])])

    x_next = latents.to(dtype) * t_steps[0]
    intermediates = []

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        # No S_churn -> x_hat = x_cur, t_hat = t_cur.
        denoised = net(
            x_cur.to(latents.dtype), t_cur, condition=condition,
        ).to(dtype)
        d_cur = (x_cur - denoised) / t_cur
        h = t_next - t_cur

        if solver == "euler" or i == num_steps - 1:
            x_next = x_cur + h * d_cur
        else:
            x_prime = x_cur + h * d_cur
            denoised_prime = net(
                x_prime.to(latents.dtype), t_next, condition=condition,
            ).to(dtype)
            d_prime = (x_prime - denoised_prime) / t_next
            x_next = x_cur + h * 0.5 * (d_cur + d_prime)

        intermediates.append(
            (i, float(t_cur), x_next.clone().to(latents.dtype))
        )
    return intermediates


@torch.no_grad()
def flowcast_sampler_with_intermediates(
    net, shape, condition, num_steps=10, sigma_data=0.5,
    t_start=0.0, t_end=1.0, solver="euler",
):
    """FlowCast Euler ODE sampler (matches flowcast_model_forward) but yields
    the un-standardized residual ``z * sigma_data`` after each step."""
    device = condition.device
    dtype = condition.dtype

    z = torch.randn(*shape, device=device, dtype=dtype)
    dt = (t_end - t_start) / num_steps
    intermediates = []

    for i in range(num_steps):
        t_i = t_start + i * dt
        t_vec = torch.full([shape[0]], t_i, device=device, dtype=dtype)
        if solver == "midpoint":
            v_half = net(z, t_vec, condition=condition)
            t_half = t_vec + 0.5 * dt
            z_half = z + v_half * (0.5 * dt)
            v = net(z_half, t_half, condition=condition)
        else:  # euler
            v = net(z, t_vec, condition=condition)
        z = z + v * dt
        intermediates.append(
            (i, float(t_i + dt), (z * sigma_data).clone())
        )
    return intermediates


# --- Plotting ---------------------------------------------------------------
def _global_vlims_from_truths(truth_phys_list, channels):
    """Per-channel (vmin, vmax) taken across the truth fields at every
    timestamp -- one fixed scale per channel for the whole run, so the same
    color mapping is used for every step PNG everywhere."""
    out = {}
    for j, ch in enumerate(channels):
        stacks = [t[j].ravel() for t in truth_phys_list]
        arr = np.concatenate(stacks)
        out[ch] = (float(np.min(arr)), float(np.max(arr)))
    return out


def plot_bare(out_path, field_phys, vlim):
    """Render a single (H, W) field as a bare PNG: ``origin='lower'``,
    default viridis colormap, ``vmin/vmax`` pinned to ``vlim``. The output
    PNG has exactly ``field_phys.shape`` pixels (one grid cell per pixel),
    so domain length / width and aspect ratio are preserved verbatim across
    every saved frame -- no axes / colorbar / frame / padding."""
    vmin, vmax = vlim
    plt.imsave(
        out_path, field_phys, vmin=vmin, vmax=vmax,
        cmap="viridis", origin="lower",
    )


def plot_fields_per_channel(out_dir, fields_phys, channels, vlims, suffix=""):
    """Write one bare PNG per channel into ``out_dir``.

    ``suffix`` (e.g. ``"_step_05"``) is appended to the channel name in the
    output filename; pass an empty string for truth / regression where each
    channel only has one PNG.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for j, ch in enumerate(channels):
        out = out_dir / f"{ch}{suffix}.png"
        plot_bare(out, fields_phys[j], vlims[ch])


def residual_to_state_phys(residual, M_t, dataset, plot_idx):
    """Given a residual tensor in the standardized state space (shape (1,C,H,W)),
    add the regression mean, denormalize to physical units, and return the
    plot_idx-reordered (C_plot, H, W) numpy array.
    """
    x_t = M_t + residual
    arr = dataset.denormalize_state(x_t.cpu().numpy()[0].copy())
    return arr[plot_idx]


# --- Main -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--regression", type=Path, required=True,
                    help="Cleaned StormCastUNet regression checkpoint.")
    ap.add_argument("--diffusion", type=Path, required=True,
                    help="EDMPrecond diffusion checkpoint (cleaned-grid).")
    ap.add_argument("--flowcast", type=Path, required=True,
                    help="FlowCastPrecond checkpoint (cleaned-grid).")
    ap.add_argument("--valid-dates", nargs=2, default=["2022/01/01", "2022/12/31"])
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--n-times", type=int, default=3,
                    help="Number of evenly-spaced initial times (default 3).")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--diffusion-num-steps", type=int, default=18)
    ap.add_argument("--sigma-min", type=float, default=0.002)
    ap.add_argument("--sigma-max", type=float, default=80.0)
    ap.add_argument("--rho", type=float, default=7.0)
    ap.add_argument("--diffusion-solver", choices=("heun", "euler"), default="heun")

    ap.add_argument("--flowcast-num-steps", type=int, default=10)
    ap.add_argument("--flowcast-solver", choices=("euler", "midpoint"), default="euler")
    ap.add_argument("--sigma-data", type=float, default=0.5)

    args = ap.parse_args()

    DistributedManager.initialize()
    device = DistributedManager().device
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"[device] {device}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ data
    print(f"[data] building dataset @ {args.data}")
    dataset, invariant = build_dataset(
        args.data, tuple(args.valid_dates), (192, 96),
        qpepre_log1p=True, kept_HR=CLEANED_CHANNEL_ORDER,
        device=device,
    )
    state_channels = list(dataset.state_channels())
    print(f"[data] pairs={len(dataset)} channels={state_channels}")

    n = len(dataset)
    if n < args.n_times + 1:
        raise SystemExit(f"not enough samples ({n}) for {args.n_times} times")
    t0_indices = (
        np.linspace(0, n - 2, args.n_times).astype(int).tolist()
    )
    print(f"[seqs] n_times={len(t0_indices)} first={t0_indices[0]} last={t0_indices[-1]}")

    plot_idx = [state_channels.index(c) for c in PLOT_CHANNELS]

    # --------------------------------------------------------------- models
    print(f"[load] regression {args.regression}")
    regression = Module.from_checkpoint(str(args.regression)).to(device).eval()
    print(f"[load] diffusion  {args.diffusion}")
    edm = Module.from_checkpoint(str(args.diffusion)).to(device).eval()
    print(f"[load] flowcast   {args.flowcast}")
    flow = Module.from_checkpoint(str(args.flowcast)).to(device).eval()

    # ------------------------------------------------ pre-pass: global vlims
    # Walk every timestamp once to load truth fields AND run the regression
    # mean, then compute one fixed (vmin, vmax) per channel for (a) the
    # forecast view (anchored to the truth's range) and (b) the residual
    # view (anchored to (truth - regression)'s range). The same vlims are
    # then used for every step PNG and every timestamp.
    print("[vlims] pre-pass: loading truth + running regression for color scales")
    cached = []  # one entry per timestamp; see fields below
    truths_phys, truth_residuals_phys = [], []
    for t0 in t0_indices:
        data = dataset[t0]
        background = data["background"].to(device=device, dtype=torch.float32).unsqueeze(0)
        state_in = data["state"][0].to(device=device, dtype=torch.float32).unsqueeze(0)
        state_tar = data["state"][1].to(device=device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            condition, _, M_t = build_network_condition_and_target(
                background, [state_in, state_in], invariant,
                regression_net=regression,
                condition_list=DIFFUSION_CONDITIONS,
                regression_condition_list=REGRESSION_CONDITIONS,
            )
            if M_t is None:
                M_t = torch.zeros_like(state_in)

        truth_phys = dataset.denormalize_state(
            state_tar.cpu().numpy()[0].copy()
        )[plot_idx]
        regression_phys = dataset.denormalize_state(
            M_t.cpu().numpy()[0].copy()
        )[plot_idx]
        # Physical-unit truth residual = denorm(X) - denorm(M_t). For qpepre
        # this captures the log1p nonlinearity correctly because each term is
        # denormalized independently.
        truth_residual_phys = truth_phys - regression_phys

        cached.append(dict(
            t0=t0,
            condition=condition,
            state_in=state_in,
            M_t=M_t,
            truth_phys=truth_phys,
            regression_phys=regression_phys,
            truth_residual_phys=truth_residual_phys,
        ))
        truths_phys.append(truth_phys)
        truth_residuals_phys.append(truth_residual_phys)

    vlims = _global_vlims_from_truths(truths_phys, PLOT_CHANNELS)
    vlims_res = _global_vlims_from_truths(truth_residuals_phys, PLOT_CHANNELS)
    for ch in PLOT_CHANNELS:
        vmin, vmax = vlims[ch]
        rmin, rmax = vlims_res[ch]
        print(f"[vlims]   {ch}: forecast=[{vmin:.4g}, {vmax:.4g}]  "
              f"residual=[{rmin:.4g}, {rmax:.4g}]")

    # --------------------------------------------------------------- loop
    t_start_all = time.perf_counter()
    for i, entry in enumerate(cached):
        t_s = time.perf_counter()
        t0 = entry["t0"]
        condition = entry["condition"]
        state_in = entry["state_in"]
        M_t = entry["M_t"]
        truth_phys = entry["truth_phys"]
        regression_phys = entry["regression_phys"]
        truth_residual_phys = entry["truth_residual_phys"]

        try:
            ts = dataset.valid_samples[t0 + 1]
            ts_str = ts.strftime("%Y-%m-%d_%H")
            ts_label = ts.strftime("%Y-%m-%d %H:00 UTC")
        except (AttributeError, IndexError):
            ts_str = f"idx{t0}"
            ts_label = f"t0_idx={t0}"
        tdir = args.output_dir / f"time_{i + 1:02d}_{ts_str}"
        tdir.mkdir(parents=True, exist_ok=True)

        # Reference panels: truth + regression in the forecast view, and
        # (truth - regression) in the residual view.
        plot_fields_per_channel(tdir / "truth", truth_phys, PLOT_CHANNELS, vlims)
        plot_fields_per_channel(
            tdir / "regression", regression_phys, PLOT_CHANNELS, vlims,
        )
        plot_fields_per_channel(
            tdir / "truth_residual", truth_residual_phys, PLOT_CHANNELS, vlims_res,
        )

        # Pre-make per-channel subdirs for both samplers, both views.
        for ch in PLOT_CHANNELS:
            (tdir / "stormcast" / ch).mkdir(parents=True, exist_ok=True)
            (tdir / "stormcast_residual" / ch).mkdir(parents=True, exist_ok=True)
            (tdir / "flowcast" / ch).mkdir(parents=True, exist_ok=True)
            (tdir / "flowcast_residual" / ch).mkdir(parents=True, exist_ok=True)

        def _write_step(method_dir, residual_dir, phys, k):
            """Write the forecast view (M_t + R_t denormalized) and the
            residual view (forecast - regression_phys) for one sampler step."""
            for j, ch in enumerate(PLOT_CHANNELS):
                plot_bare(
                    method_dir / ch / f"step_{k + 1:02d}.png",
                    phys[j], vlims[ch],
                )
                plot_bare(
                    residual_dir / ch / f"step_{k + 1:02d}.png",
                    phys[j] - regression_phys[j], vlims_res[ch],
                )

        # --- StormCast (EDM) ---
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        latents = torch.randn(*state_in.shape, device=device, dtype=torch.float32)
        edm_steps = edm_sampler_with_intermediates(
            edm, latents, condition,
            num_steps=args.diffusion_num_steps,
            sigma_min=args.sigma_min, sigma_max=args.sigma_max,
            rho=args.rho, solver=args.diffusion_solver,
        )
        for k, (_, _t_cur, x_next) in enumerate(edm_steps):
            phys = residual_to_state_phys(x_next, M_t, dataset, plot_idx)
            _write_step(
                tdir / "stormcast", tdir / "stormcast_residual", phys, k,
            )

        # --- FlowCast (CFM) ---
        torch.manual_seed(args.seed + 1)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + 1)
        flow_steps = flowcast_sampler_with_intermediates(
            flow, state_in.shape, condition,
            num_steps=args.flowcast_num_steps,
            sigma_data=args.sigma_data,
            solver=args.flowcast_solver,
        )
        for k, (_, _t_now, residual) in enumerate(flow_steps):
            phys = residual_to_state_phys(residual, M_t, dataset, plot_idx)
            _write_step(
                tdir / "flowcast", tdir / "flowcast_residual", phys, k,
            )

        dt = time.perf_counter() - t_s
        total = time.perf_counter() - t_start_all
        # 3 reference + 2 views * (diff_steps + flow_steps) panels per channel
        n_per_t = len(PLOT_CHANNELS) * (
            3 + 2 * (args.diffusion_num_steps + args.flowcast_num_steps)
        )
        print(
            f"[time {i + 1}/{len(t0_indices)}] idx={t0:<5}  {ts_label}  "
            f"wrote={n_per_t}  dt={dt:5.1f}s  total={total / 60:5.1f}m"
        )

    meta = args.output_dir / "metadata.txt"
    with open(meta, "w") as f:
        f.write(f"data: {args.data}\n")
        f.write(f"valid_dates: {args.valid_dates}\n")
        f.write(f"n_times: {args.n_times}\n")
        f.write(f"t0_indices: {t0_indices}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"channels (plot order): {PLOT_CHANNELS}\n")
        f.write(f"diffusion: num_steps={args.diffusion_num_steps} solver={args.diffusion_solver} "
                f"sigma_min={args.sigma_min} sigma_max={args.sigma_max} rho={args.rho}\n")
        f.write(f"flowcast:  num_steps={args.flowcast_num_steps} solver={args.flowcast_solver} "
                f"sigma_data={args.sigma_data}\n")
        f.write(f"regression: {args.regression}\n")
        f.write(f"diffusion:  {args.diffusion}\n")
        f.write(f"flowcast:   {args.flowcast}\n")
    print(f"[done] outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
