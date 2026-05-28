#!/usr/bin/env python3
"""12-hour autoregressive rollout figure: one PNG per variable.

Row order and labels match the column convention of
``run_single_time_exp.py`` -- legacy StormCast, RWRF truth, cleaned
StormCast, CFM -- with the actual grid shape substituted into each
header instead of a generic "original/new domain" tag:

    +--------+--------+--------+--------+
    | StormCast (224x128)               |   row 0: EDM teacher, legacy
    +--------+--------+--------+--------+
    | RWRF (192x96)                     |   row 1: ground truth, cleaned
    +--------+--------+--------+--------+
    | StormCast (192x96)                |   row 2: EDM teacher, cleaned
    +--------+--------+--------+--------+
    | CFM (192x96)                      |   row 3: FlowCast student
    +--------+--------+--------+--------+
       +1h     +3h     +6h     +12h

All three model rows use checkpoints at the matched ~2 M training-sample
budget. Rows are displayed at their native grids -- the original-domain
row is 224x128 while the rest are 192x96 cleaned. They share the same
figure cell size with ``aspect='auto'`` so the original-domain row is
visibly stretched; that is intentional since the two domains differ.

The script does:
    1. build the original-domain (224x128, raw qpepre, channel order
       [t2m, u10, v10, qpepre]) and new-domain (192x96, log1p, channel
       order [u10, v10, t2m, qpepre]) validation datasets and align t0 by
       hourly index;
    2. load the two regressions and three residual heads;
    3. run a deterministic single-member rollout to ``+12h`` for all three
       methods;
    4. for each of ``t2m, u10, v10, qpepre`` write one figure to
       ``--output-dir/rollout_12h_<channel>.png``.

Defaults pin the 2 M-sample checkpoints (EDMPrecond.0.70000 on the
original domain, EDMPrecond.0.31000 on the new domain,
FlowCastPrecond.0.20000 for FlowCast). Typical invocation::

    python experiment_scripts/plot_rollout_12h.py --t0-idx 4128

Pass ``--t0-idx`` to choose the initial validation-set sample (defaults to
mid-summer 2022). Pass ``--hours-to-plot 1 3 6 12`` to tweak the columns.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
STORMCAST_ROOT = REPO_ROOT / "stormcast"
sys.path.insert(0, str(STORMCAST_ROOT))

from physicsnemo.distributed import DistributedManager  # noqa: E402
from physicsnemo.models import Module  # noqa: E402

from datasets import dataset_classes  # noqa: E402
from utils.nn import (  # noqa: E402
    build_network_condition_and_target,
    diffusion_model_forward,
    flowcast_model_forward,
)


# ---- Defaults pinned to ~2 M-sample checkpoints ----------------------------
# Original-domain (224x128, raw qpepre) -- matches run_main_experiment.sh.
DEFAULT_ORIG_DATA = (
    REPO_ROOT / "exp_3_train_2_5_yrs_val_1yr_tp1"
    / "zarr_exp3_L_24_H_24_train_2_5_years_full"
)
DEFAULT_ORIG_REG = (
    REPO_ROOT / "exp_3_train_2_5_yrs_val_1yr_tp1"
    / "exp_3_reg_L_24_H_4_train_2_5_years/0"
    / "checkpoints_regression/StormCastUNet.0.7500.mdlus"
)
DEFAULT_ORIG_EDM = (
    REPO_ROOT / "exp_3_train_2_5_yrs_val_1yr_tp1"
    / "exp_3_dif_L_24_H_4_train_2_5_years/0"
    / "checkpoints_diffusion/EDMPrecond.0.70000.mdlus"
)

# New-domain (cleaned 192x96, log1p qpepre) -- both EDM and FlowCast checkpoints
# below sit at ~2 M training samples.
DEFAULT_NEW_DATA = (
    REPO_ROOT / "exp_3_train_2_5_yrs_val_1yr_tp1"
    / "zarr_exp3_L_24_H_24_train_2_5_years_full_cleaned_4_27_2026"
)
DEFAULT_NEW_REG = (
    REPO_ROOT / "runs/regression_zettabyte_v1_cleaned_4_27_2026"
    / "regression_zettabyte_cleaned_4_27_2026/run_0"
    / "checkpoints_regression/StormCastUNet.0.8000.mdlus"
)
DEFAULT_NEW_EDM = (
    REPO_ROOT / "runs/diffusion_zettabyte_v1_cleaned_4_27_2026"
    / "diffusion_zettabyte_cleaned_4_27_2026/run_0"
    / "checkpoints_diffusion/EDMPrecond.0.31000.mdlus"
)
DEFAULT_NEW_FLOW = (
    REPO_ROOT / "runs/flowcast_zettabyte_v1_cleaned_4_27_2026"
    / "flowcast_zettabyte_cleaned_4_27_2026/run_0"
    / "checkpoints_flowcast/FlowCastPrecond.0.20000.mdlus"
)

ORIG_CHANNELS = ["t2m", "u10", "v10", "qpepre"]
NEW_CHANNELS = ["u10", "v10", "t2m", "qpepre"]

DIFFUSION_CONDITIONS = ["state", "regression", "invariant"]
REGRESSION_CONDITIONS = ["state", "background", "invariant"]

CHANNEL_UNITS = {"t2m": "K", "u10": "m/s", "v10": "m/s", "qpepre": "mm/h"}


# ---- Dataset / inference plumbing -------------------------------------------
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


def build_dataset(cfg, device):
    dataset_cls = dataset_classes["data_loader_rwrf_era5_stable.Dataset"]
    ds = dataset_cls(cfg, train=False)
    inv = ds.get_invariants()
    invariant_tensor = (
        torch.from_numpy(inv).to(device=device, dtype=torch.float32).unsqueeze(0)
        if inv is not None
        else None
    )
    return ds, invariant_tensor


def rollout(*, model, method, regression, invariant, dataset, t0_idx, n_steps,
            sampler_kwargs, device):
    """Single deterministic autoregressive rollout, returns (pred, truth) in
    physical units, shape (T, C, H, W)."""
    state_pred = None
    preds, truths = [], []
    for i in range(n_steps):
        data = dataset[t0_idx + i]
        background = (
            data["background"].to(device=device, dtype=torch.float32).unsqueeze(0)
        )
        if state_pred is None:
            state_pred = (
                data["state"][0].to(device=device, dtype=torch.float32).unsqueeze(0)
            )
        with torch.no_grad():
            condition, _, M_t = build_network_condition_and_target(
                background,
                [state_pred, state_pred],
                invariant,
                regression_net=regression,
                condition_list=DIFFUSION_CONDITIONS,
                regression_condition_list=REGRESSION_CONDITIONS,
            )
            if M_t is None:
                M_t = torch.zeros_like(state_pred)
            if method == "diffusion":
                residual = diffusion_model_forward(
                    model, condition, state_pred.shape, sampler_args=sampler_kwargs
                )
            elif method == "flowcast":
                residual = flowcast_model_forward(
                    model, condition, state_pred.shape, **sampler_kwargs
                )
            else:
                raise ValueError(f"unknown method {method!r}")
        x_t = M_t + residual
        preds.append(dataset.denormalize_state(x_t.cpu().numpy()[0].copy()))
        truths.append(dataset.denormalize_state(data["state"][1].cpu().numpy().copy()))
        state_pred = x_t
    return np.stack(preds), np.stack(truths)


# ---- Plotting --------------------------------------------------------------
def _channel_scale(channel, stack):
    """Return (vmin, vmax, cmap) for the given channel across a stacked field."""
    if channel == "qpepre":
        vmax = float(np.quantile(stack, 0.995) + 1e-3)
        return 0.0, vmax, "Blues"
    fm = float(np.mean(stack))
    fabs = float(np.quantile(np.abs(stack - fm), 0.99) + 1e-9)
    return fm - fabs, fm + fabs, "RdBu_r"


def plot_channel_grid(
    *,
    channel: str,
    truth_new,            # (T, H, W) in physical units, new-domain grid
    orig_pred,            # (T, H_o, W_o) -- can be None if --skip-original
    new_edm_pred,         # (T, H, W) -- can be None if --skip-new-edm
    flow_pred,            # (T, H, W) -- can be None
    hours,                # list of ints, e.g. [1,3,6,12]
    out_path: Path,
    t0_label: str,
):
    # Row sequence matches the column order in run_single_time_exp.py:
    # legacy StormCast, RWRF truth, cleaned StormCast, CFM. Headers carry
    # the actual grid shape (e.g. "StormCast (224x128)") instead of a
    # generic "original/new domain" tag.
    cln_shape = f"{truth_new.shape[-2]}x{truth_new.shape[-1]}"
    rows_data = []
    if orig_pred is not None:
        leg_shape = f"{orig_pred.shape[-2]}x{orig_pred.shape[-1]}"
        rows_data.append((f"StormCast ({leg_shape})", orig_pred))
    rows_data.append((f"RWRF ({cln_shape})", truth_new))
    if new_edm_pred is not None:
        rows_data.append((f"StormCast ({cln_shape})", new_edm_pred))
    if flow_pred is not None:
        rows_data.append((f"CFM ({cln_shape})", flow_pred))

    n_rows = len(rows_data)
    n_cols = len(hours)

    # Shared colour scale across every panel in this figure.
    sample_panels = []
    for _, arr in rows_data:
        for h in hours:
            sample_panels.append(arr[h - 1])
    flat_stack = np.concatenate([p.reshape(-1) for p in sample_panels])
    vmin, vmax, cmap = _channel_scale(channel, flat_stack)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.8 * n_cols + 1.5, 2.4 * n_rows + 0.6),
        squeeze=False,
    )

    im_last = None
    for r, (label, arr) in enumerate(rows_data):
        for c, h in enumerate(hours):
            ax = axes[r, c]
            im_last = ax.imshow(
                arr[h - 1],
                origin="lower",
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                aspect="auto",
                interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"+{h}h", fontsize=11)
            if c == 0:
                ax.set_ylabel(label, fontsize=10)
            for s in ax.spines.values():
                s.set_linewidth(0.6)

    unit = CHANNEL_UNITS.get(channel, "")
    cbar = fig.colorbar(
        im_last, ax=axes, fraction=0.025, pad=0.015, shrink=0.92,
    )
    cbar.set_label(f"{channel} [{unit}]", fontsize=10)

    fig.suptitle(
        f"12 h autoregressive rollout — {channel}  ({t0_label})",
        fontsize=12, y=0.995,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


# ---- Main ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--original-data", type=Path, default=DEFAULT_ORIG_DATA)
    ap.add_argument("--original-regression", type=Path, default=DEFAULT_ORIG_REG)
    ap.add_argument("--original-diffusion", type=Path, default=DEFAULT_ORIG_EDM)
    ap.add_argument("--new-data", type=Path, default=DEFAULT_NEW_DATA)
    ap.add_argument("--new-regression", type=Path, default=DEFAULT_NEW_REG)
    ap.add_argument("--new-diffusion", type=Path, default=DEFAULT_NEW_EDM)
    ap.add_argument("--new-flowcast", type=Path, default=DEFAULT_NEW_FLOW)

    ap.add_argument("--valid-dates", nargs=2, default=["2022/01/01", "2022/12/31"])
    ap.add_argument("--original-hr-size", nargs=2, type=int, default=[224, 128])
    ap.add_argument("--new-hr-size", nargs=2, type=int, default=[192, 96])

    ap.add_argument(
        "--t0-idx", type=int, default=4128,
        help="Validation-set hourly index for the rollout start. Default 4128 "
             "= 2022-06-23 00:00 (8760 hourly samples in 2022).",
    )
    ap.add_argument("--n-steps", type=int, default=12,
                    help="Autoregressive horizon in hours.")
    ap.add_argument("--hours-to-plot", type=int, nargs="+", default=[1, 3, 6, 12],
                    help="Lead-time columns to render (in hours after t0).")
    ap.add_argument("--channels-to-plot", nargs="+",
                    default=["t2m", "u10", "v10", "qpepre"],
                    help="Variables to render -- one PNG per variable.")

    ap.add_argument("--diffusion-num-steps", type=int, default=18)
    ap.add_argument("--diffusion-solver", choices=("heun", "euler"), default="heun")
    ap.add_argument("--sigma-min", type=float, default=0.002)
    ap.add_argument("--sigma-max", type=float, default=80.0)
    ap.add_argument("--rho", type=float, default=7.0)
    ap.add_argument("--flowcast-num-steps", type=int, default=10)
    ap.add_argument("--flowcast-solver", choices=("euler", "midpoint"), default="euler")
    ap.add_argument("--sigma-data", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "experiment_scripts/results/rollout_12h",
    )
    ap.add_argument("--skip-original", action="store_true")
    ap.add_argument("--skip-new-edm", action="store_true")
    ap.add_argument("--skip-flowcast", action="store_true")

    args = ap.parse_args()

    DistributedManager.initialize()
    device = DistributedManager().device
    if device.type == "cuda":
        torch.cuda.empty_cache()

    max_hour = max(args.hours_to_plot)
    if max_hour > args.n_steps:
        raise SystemExit(
            f"--hours-to-plot has +{max_hour}h but --n-steps={args.n_steps}h"
        )

    print(f"[device] {device}")
    print(f"[t0] valid index {args.t0_idx} (rollout to +{args.n_steps}h)")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # ---- New-domain pipeline: dataset, regression, EDM, FlowCast ----------
    new_cfg = make_dataset_cfg(
        args.new_data, args.valid_dates, args.new_hr_size,
        qpepre_log1p=True, kept_HR=NEW_CHANNELS,
    )
    new_ds, new_inv = build_dataset(new_cfg, device)
    new_ch = list(new_ds.state_channels())
    print(f"[data] new-domain: samples={len(new_ds)}  channels={new_ch}")
    if args.t0_idx + args.n_steps > len(new_ds):
        raise SystemExit(
            f"--t0-idx {args.t0_idx} + n_steps {args.n_steps} > new-domain len "
            f"{len(new_ds)}"
        )

    print(f"[load] new-domain regression {args.new_regression}")
    new_reg = (
        Module.from_checkpoint(str(args.new_regression)).to(device).eval()
    )

    new_truth = None
    new_edm_pred = None
    if not args.skip_new_edm:
        print(f"[load] new-domain diffusion {args.new_diffusion}")
        new_edm = (
            Module.from_checkpoint(str(args.new_diffusion)).to(device).eval()
        )
        diff_kwargs = dict(
            num_steps=args.diffusion_num_steps,
            sigma_min=args.sigma_min, sigma_max=args.sigma_max,
            rho=args.rho, solver=args.diffusion_solver,
        )
        print(f"[run] new-domain EDM rollout ({args.n_steps}h, NFE/step="
              f"{2 * args.diffusion_num_steps if args.diffusion_solver == 'heun' else args.diffusion_num_steps})")
        t0 = time.perf_counter()
        new_edm_pred, new_truth = rollout(
            model=new_edm, method="diffusion",
            regression=new_reg, invariant=new_inv,
            dataset=new_ds, t0_idx=args.t0_idx, n_steps=args.n_steps,
            sampler_kwargs=diff_kwargs, device=device,
        )
        print(f"[run]   done in {time.perf_counter() - t0:.1f}s")
        del new_edm
        if device.type == "cuda":
            torch.cuda.empty_cache()

    flow_pred = None
    if not args.skip_flowcast:
        print(f"[load] flowcast {args.new_flowcast}")
        flow_model = (
            Module.from_checkpoint(str(args.new_flowcast)).to(device).eval()
        )
        flow_kwargs = dict(
            num_steps=args.flowcast_num_steps,
            sigma_data=args.sigma_data, solver=args.flowcast_solver,
        )
        print(f"[run] FlowCast rollout (NFE={args.flowcast_num_steps})")
        t0 = time.perf_counter()
        flow_pred, flow_truth = rollout(
            model=flow_model, method="flowcast",
            regression=new_reg, invariant=new_inv,
            dataset=new_ds, t0_idx=args.t0_idx, n_steps=args.n_steps,
            sampler_kwargs=flow_kwargs, device=device,
        )
        print(f"[run]   done in {time.perf_counter() - t0:.1f}s")
        if new_truth is None:
            new_truth = flow_truth
        del flow_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    del new_reg
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---- Original-domain pipeline -----------------------------------------
    orig_pred = None
    orig_ch = None
    if not args.skip_original:
        orig_cfg = make_dataset_cfg(
            args.original_data, args.valid_dates, args.original_hr_size,
            qpepre_log1p=False, kept_HR=ORIG_CHANNELS,
        )
        orig_ds, orig_inv = build_dataset(orig_cfg, device)
        orig_ch = list(orig_ds.state_channels())
        print(f"[data] original-domain: samples={len(orig_ds)}  channels={orig_ch}")
        if args.t0_idx + args.n_steps > len(orig_ds):
            raise SystemExit(
                f"--t0-idx {args.t0_idx} + n_steps {args.n_steps} > original-domain "
                f"len {len(orig_ds)}"
            )

        print(f"[load] original-domain regression {args.original_regression}")
        orig_reg = (
            Module.from_checkpoint(str(args.original_regression)).to(device).eval()
        )
        print(f"[load] original-domain diffusion {args.original_diffusion}")
        orig_edm = (
            Module.from_checkpoint(str(args.original_diffusion)).to(device).eval()
        )
        diff_kwargs = dict(
            num_steps=args.diffusion_num_steps,
            sigma_min=args.sigma_min, sigma_max=args.sigma_max,
            rho=args.rho, solver=args.diffusion_solver,
        )
        print(f"[run] original-domain EDM rollout ({args.n_steps}h)")
        t0 = time.perf_counter()
        orig_pred, _ = rollout(
            model=orig_edm, method="diffusion",
            regression=orig_reg, invariant=orig_inv,
            dataset=orig_ds, t0_idx=args.t0_idx, n_steps=args.n_steps,
            sampler_kwargs=diff_kwargs, device=device,
        )
        print(f"[run]   done in {time.perf_counter() - t0:.1f}s")
        del orig_edm, orig_reg
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if new_truth is None:
        raise SystemExit(
            "No rollout produced a truth field. At least one of "
            "--skip-new-edm / --skip-flowcast must be False."
        )

    # Translate a t0 index to a human-readable label (each step = 1h after
    # the first valid date).
    y0, m0, d0 = (int(x) for x in args.valid_dates[0].split("/"))
    import datetime as _dt
    t0_label = (
        _dt.datetime(y0, m0, d0) + _dt.timedelta(hours=args.t0_idx)
    ).strftime("t0 = %Y-%m-%d %H:00 UTC")

    # ---- Render one PNG per channel ----------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for ch in args.channels_to_plot:
        truth_arr = new_truth[:, new_ch.index(ch)]
        edm_arr = (
            new_edm_pred[:, new_ch.index(ch)] if new_edm_pred is not None else None
        )
        flow_arr = (
            flow_pred[:, new_ch.index(ch)] if flow_pred is not None else None
        )
        orig_arr = (
            orig_pred[:, orig_ch.index(ch)] if orig_pred is not None else None
        )

        plot_channel_grid(
            channel=ch,
            truth_new=truth_arr,
            orig_pred=orig_arr,
            new_edm_pred=edm_arr,
            flow_pred=flow_arr,
            hours=args.hours_to_plot,
            out_path=args.output_dir / f"rollout_12h_{ch}.png",
            t0_label=t0_label,
        )

    print(f"[done] outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
