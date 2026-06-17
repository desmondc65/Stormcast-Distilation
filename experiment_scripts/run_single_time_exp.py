#!/usr/bin/env python3
"""Single-time qualitative comparison across the three thesis models.

For N evenly-spaced initial times across the 2022 validation year, runs each
of

    1. orig-StormCast       (legacy EDM teacher on 224x128, raw mm/h qpepre)
    2. cleaned-StormCast    (cleaned EDM teacher on 192x96, log1p qpepre)
    3. FlowCast             (cleaned CFM student on 192x96, log1p qpepre)

forecasting --lead-time hours ahead, and renders ONE PNG per initial time with

    2x2 macro-grid of variables, each macro-cell a 1x4 strip of models:
        macro-col 1 (top -> bottom):  t2m,  qpepre
        macro-col 2 (top -> bottom):  v10,  u10
    each strip cols = StormCast (legacy res) | RWRF (cleaned res)
                    | StormCast (cleaned res) | CFM (cleaned res)

Column resolutions are read off the data and substituted into the headers
(e.g. ``StormCast (224x128)`` vs ``StormCast (192x96)``). The 'RWRF' truth
column is the cleaned-dataset target, since that's the grid on which the
cleaned-StormCast and CFM columns live; the legacy column is at its native
larger field of view but stretched vertically by ~14 % so the four columns
share a common cell shape (the alternative is uneven inter-column gaps).
A colorbar is drawn on the right of every row, with shared vmin/vmax across
the four panels of that row.

Colour conventions match the trainer-time validation_plot exactly: default
``viridis`` colormap for every channel, ``vmin/vmax`` taken as the true
min/max across the four row panels (no quantile clipping). All channels are
shown in physical units after ``Dataset.denormalize_state`` (qpepre in raw
mm/h, with the log1p inverted).

Output: ``<output-dir>/time_{NN}_lead_{L}h.png`` + ``metadata.txt``.
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
from utils.nn import (  # noqa: E402
    build_network_condition_and_target,
    diffusion_model_forward,
    flowcast_model_forward,
    meanflow_model_forward,
)


# Variables to render, in plot order (rows top -> bottom).
PLOT_CHANNELS = ["t2m", "u10", "v10", "qpepre"]

# Physical units after Dataset.denormalize_state. Matches CHANNEL_UNITS in
# compare_diffusion_vs_flowcast.py.
CHANNEL_UNITS = {"t2m": "K", "u10": "m/s", "v10": "m/s", "qpepre": "mm/h"}
CHANNEL_LABELS = {ch: f"{ch} [{u}]" for ch, u in CHANNEL_UNITS.items()}
# Match stormcast/utils/plots.py:validation_plot — it never passes a ``cmap``,
# so every channel inherits the matplotlib default (``viridis``). Override per
# channel here ONLY if you want to diverge from the trainer's look.
CHANNEL_CMAPS: dict[str, str] = {}

# Conditioning bundles must match config/model/{diffusion,flowcast}.yaml.
DIFFUSION_CONDITIONS = ["state", "regression", "invariant"]
REGRESSION_CONDITIONS = ["state", "background", "invariant"]

# Training channel orders for the two dataset layouts.
LEGACY_CHANNEL_ORDER = ["t2m", "u10", "v10", "qpepre"]
CLEANED_CHANNEL_ORDER = ["u10", "v10", "t2m", "qpepre"]


# --- Dataset / model wiring --------------------------------------------------
def make_dataset_cfg(data_loc, valid_dates, hr_size, qpepre_log1p, kept_channels):
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
            "diffusion_channels": list(kept_channels),
            "kept_LowRes_channels": "all",
            "kept_HighRes_channels": list(kept_channels),
            "qpepre_log1p": qpepre_log1p,
        }
    )


def build_dataset(data_loc, valid_dates, hr_size, qpepre_log1p, kept_channels, device):
    cfg = make_dataset_cfg(data_loc, valid_dates, hr_size, qpepre_log1p, kept_channels)
    dataset_cls = dataset_classes["data_loader_rwrf_era5_stable.Dataset"]
    dataset = dataset_cls(cfg, train=False)
    inv = dataset.get_invariants()
    invariant_tensor = (
        torch.from_numpy(inv).to(device=device, dtype=torch.float32).unsqueeze(0)
        if inv is not None
        else None
    )
    return dataset, invariant_tensor


def rollout_one(
    *,
    model,
    method,
    regression,
    invariant,
    dataset,
    t0_idx,
    n_steps,
    sampler_kwargs,
    device,
):
    """One autoregressive rollout. Returns ``(preds, truths)`` as
    ``(T, C, H, W)`` numpy arrays in PHYSICAL units."""
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
            (condition, _, M_t) = build_network_condition_and_target(
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
            elif method == "meanflow":
                residual = meanflow_model_forward(
                    model, condition, state_pred.shape, **sampler_kwargs
                )
            else:
                raise ValueError(f"unknown method {method!r}")
        x_t = M_t + residual
        preds.append(dataset.denormalize_state(x_t.cpu().numpy()[0].copy()))
        truths.append(dataset.denormalize_state(data["state"][1].cpu().numpy().copy()))
        state_pred = x_t
    return np.stack(preds), np.stack(truths)


def run_leg(
    *,
    label,
    model,
    method,
    regression,
    invariant,
    dataset,
    t0_indices,
    lead_time,
    sampler_kwargs,
    device,
    seed,
):
    """Roll out every initial time and return the lead-time field per time.

    Returns ``(preds_last, truths_last)`` each as ``(N, C, H, W)`` with
    ``N == len(t0_indices)``. Only the lead-time slice is kept (intermediate
    hours are discarded -- we only render the +lead-time hour).
    """
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    preds, truths = [], []
    t_leg = time.perf_counter()
    print(f"[run] {label} starting: {len(t0_indices)} times x lead={lead_time}h",
          flush=True)
    for i, t0 in enumerate(t0_indices):
        t_s = time.perf_counter()
        p_seq, t_seq = rollout_one(
            model=model, method=method, regression=regression,
            invariant=invariant, dataset=dataset, t0_idx=t0,
            n_steps=lead_time, sampler_kwargs=sampler_kwargs, device=device,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        preds.append(p_seq[-1])
        truths.append(t_seq[-1])
        elapsed = time.perf_counter() - t_s
        leg_el = time.perf_counter() - t_leg
        rate = leg_el / (i + 1)
        eta = rate * (len(t0_indices) - (i + 1))
        print(
            f"[run] {label}  t={i + 1}/{len(t0_indices)} t0={t0:<5} "
            f"rollout={elapsed:5.1f}s  avg={rate:4.1f}s  "
            f"elapsed={leg_el / 60:5.1f}m  eta={eta / 60:5.1f}m",
            flush=True,
        )
    total_min = (time.perf_counter() - t_leg) / 60
    print(f"[run] {label} finished in {total_min:.1f}m", flush=True)
    return np.stack(preds), np.stack(truths)


# --- Plotting ---------------------------------------------------------------
def _row_color_limits(arrs):
    """True min/max across the row's panels, matching the trainer's
    ``validation_plot`` (it does ``vmin=min(...), vmax=max(...)`` over the
    panels rather than any quantile clip)."""
    stack = np.concatenate([a.ravel() for a in arrs])
    return float(np.min(stack)), float(np.max(stack))


CELL_VARS = [["t2m", "v10"], ["qpepre", "u10"]]


def plot_one_time(
    out_path,
    lead_h,
    truth,            # (4, H_c, W_c)
    legacy_pred,      # (4, H_l, W_l)
    cleaned_pred,     # (4, H_c, W_c)
    flow_pred,        # (4, H_c, W_c)
    time_label,
    meanflow_pred=None,   # (4, H_c, W_c) or None (opt-in extra column)
):
    """2x2 macro grid of variables; each macro cell is a 1x4 strip
    (orig StormCast + RWRF truth + cleaned StormCast + FlowCast/CFM) with
    its own colorbar on the right. Layout matches CELL_VARS. When
    ``meanflow_pred`` is given, a 5th ``MeanFlow`` column is appended to each
    strip; otherwise the figure is identical to the 4-column default."""
    has_mf = meanflow_pred is not None
    leg_shape = f"{legacy_pred.shape[-2]}x{legacy_pred.shape[-1]}"
    cln_shape = f"{truth.shape[-2]}x{truth.shape[-1]}"
    col_titles = [
        f"StormCast ({leg_shape})",
        f"RWRF ({cln_shape})",
        f"StormCast ({cln_shape})",
        f"CFM ({cln_shape})",
    ]
    if has_mf:
        col_titles.append(f"MeanFlow ({cln_shape})")
    ncols = len(col_titles)
    ch_to_idx = {ch: i for i, ch in enumerate(PLOT_CHANNELS)}

    fig = plt.figure(figsize=(5 * ncols, 10), constrained_layout=True)
    subfigs = fig.subfigures(2, 2, wspace=0.04, hspace=0.06)

    for mr in range(2):
        for mc in range(2):
            ch = CELL_VARS[mr][mc]
            r = ch_to_idx[ch]
            arrs = [legacy_pred[r], truth[r], cleaned_pred[r], flow_pred[r]]
            if has_mf:
                arrs.append(meanflow_pred[r])
            vmin, vmax = _row_color_limits(arrs)
            cmap = CHANNEL_CMAPS.get(ch)
            sf = subfigs[mr, mc]
            axes = sf.subplots(1, ncols, squeeze=False)[0]
            im = None
            for c, arr in enumerate(arrs):
                im = axes[c].imshow(
                    arr, origin="lower", vmin=vmin, vmax=vmax,
                    cmap=cmap, aspect="auto",
                )
                axes[c].set_xticks([])
                axes[c].set_yticks([])
                if mr == 0:
                    axes[c].set_title(col_titles[c], fontsize=12)
            axes[0].set_ylabel(CHANNEL_LABELS[ch], fontsize=13)
            cbar = sf.colorbar(
                im, ax=axes.tolist(), location="right",
                fraction=0.025, pad=0.01, shrink=0.92,
                label=CHANNEL_UNITS[ch],
            )
            cbar.ax.tick_params(labelsize=10)
            cbar.ax.yaxis.label.set_size(12)

    fig.suptitle(f"{time_label}  |  +{lead_h}h forecast", fontsize=15)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# --- Main -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--legacy-data", type=Path, required=True)
    ap.add_argument("--legacy-reg", type=Path, required=True)
    ap.add_argument("--legacy-edm", type=Path, required=True)
    ap.add_argument("--cleaned-data", type=Path, required=True)
    ap.add_argument("--cleaned-reg", type=Path, required=True)
    ap.add_argument("--cleaned-edm", type=Path, required=True)
    ap.add_argument("--cleaned-flow", type=Path, required=True)
    # MeanFlow leg is opt-in: only added when a checkpoint path is supplied.
    ap.add_argument("--cleaned-meanflow", type=Path, default=None,
                    help="Optional MeanFlow student .mdlus; adds a 5th MeanFlow "
                         "column when provided. Omit to keep the 4-column figure.")

    ap.add_argument("--valid-dates", nargs=2, default=["2022/01/01", "2022/12/31"])
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--n-times", type=int, default=10,
                    help="Number of evenly-spaced initial times (one PNG each).")
    ap.add_argument("--extra-t0", type=int, nargs="*", default=None,
                    help="Explicit init-time indices APPENDED after the evenly-"
                         "spaced --n-times set (e.g. a hand-picked heavy-rain "
                         "case). Each is numbered time_{N+k} in output order, so "
                         "this extends an existing set without disturbing it. "
                         "The plotted field is the +lead-time hour, so to feature "
                         "a rainy target hour H pass --extra-t0 (H - lead).")
    ap.add_argument("--lead-time", type=int, default=1,
                    help="Forecast horizon (hours). Only the +lead-time field is plotted.")
    ap.add_argument("--seed", type=int, default=0)

    # Diffusion sampler (matches compare_diffusion_vs_flowcast.py).
    ap.add_argument("--diffusion-num-steps", type=int, default=18)
    ap.add_argument("--sigma-min", type=float, default=0.002)
    ap.add_argument("--sigma-max", type=float, default=80.0)
    ap.add_argument("--rho", type=float, default=7.0)
    ap.add_argument("--diffusion-solver", choices=("heun", "euler"), default="heun")

    # FlowCast sampler.
    ap.add_argument("--flowcast-num-steps", type=int, default=10)
    ap.add_argument("--flowcast-solver", choices=("euler", "midpoint"), default="euler")
    ap.add_argument("--sigma-data", type=float, default=0.5)

    # MeanFlow sampler (average-velocity; shares --sigma-data with FlowCast).
    ap.add_argument("--meanflow-num-steps", type=int, default=2)

    args = ap.parse_args()

    DistributedManager.initialize()
    device = DistributedManager().device
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"[device] {device}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ data
    print("[data] building legacy dataset")
    legacy_ds, legacy_inv = build_dataset(
        args.legacy_data, tuple(args.valid_dates), (224, 128),
        qpepre_log1p=False, kept_channels=LEGACY_CHANNEL_ORDER, device=device,
    )
    legacy_channels = list(legacy_ds.state_channels())
    print(f"[data] legacy  pairs={len(legacy_ds)} channels={legacy_channels}")

    print("[data] building cleaned dataset")
    cleaned_ds, cleaned_inv = build_dataset(
        args.cleaned_data, tuple(args.valid_dates), (192, 96),
        qpepre_log1p=True, kept_channels=CLEANED_CHANNEL_ORDER, device=device,
    )
    cleaned_channels = list(cleaned_ds.state_channels())
    print(f"[data] cleaned pairs={len(cleaned_ds)} channels={cleaned_channels}")

    n_usable = min(len(legacy_ds), len(cleaned_ds))
    if n_usable < args.n_times + args.lead_time:
        raise SystemExit(
            f"not enough samples ({n_usable}) for {args.n_times} times x lead {args.lead_time}h"
        )
    t0_indices = (
        np.linspace(0, n_usable - args.lead_time - 1, args.n_times)
        .astype(int)
        .tolist()
    )
    if args.extra_t0:
        hi = n_usable - args.lead_time - 1
        extra = [int(t) for t in args.extra_t0 if 0 <= int(t) <= hi]
        dropped = [int(t) for t in args.extra_t0 if not (0 <= int(t) <= hi)]
        if dropped:
            print(f"[seqs] WARN out-of-range --extra-t0 dropped: {dropped} (valid 0..{hi})")
        t0_indices += extra
        print(f"[seqs] appended {len(extra)} hand-picked t0(s): {extra}")
    print(
        f"[seqs] n_times={len(t0_indices)} lead={args.lead_time}h "
        f"first={t0_indices[0]} last={t0_indices[-1]}"
    )

    # ------------------------------------------------------------------ leg A
    print(f"[load] legacy regression {args.legacy_reg}")
    legacy_reg = Module.from_checkpoint(str(args.legacy_reg)).to(device).eval()
    print(f"[load] legacy diffusion  {args.legacy_edm}")
    legacy_edm = Module.from_checkpoint(str(args.legacy_edm)).to(device).eval()
    diffusion_kwargs = dict(
        num_steps=args.diffusion_num_steps,
        sigma_min=args.sigma_min, sigma_max=args.sigma_max,
        rho=args.rho, solver=args.diffusion_solver,
    )
    legacy_preds, legacy_truths = run_leg(
        label="orig_stormcast", model=legacy_edm, method="diffusion",
        regression=legacy_reg, invariant=legacy_inv, dataset=legacy_ds,
        t0_indices=t0_indices, lead_time=args.lead_time,
        sampler_kwargs=diffusion_kwargs, device=device, seed=args.seed,
    )
    del legacy_edm, legacy_reg
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ leg B
    print(f"[load] cleaned regression {args.cleaned_reg}")
    cleaned_reg = Module.from_checkpoint(str(args.cleaned_reg)).to(device).eval()
    print(f"[load] cleaned diffusion  {args.cleaned_edm}")
    cleaned_edm = Module.from_checkpoint(str(args.cleaned_edm)).to(device).eval()
    cleaned_preds, cleaned_truths = run_leg(
        label="cleaned_stormcast", model=cleaned_edm, method="diffusion",
        regression=cleaned_reg, invariant=cleaned_inv, dataset=cleaned_ds,
        t0_indices=t0_indices, lead_time=args.lead_time,
        sampler_kwargs=diffusion_kwargs, device=device, seed=args.seed,
    )
    del cleaned_edm
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ leg C
    print(f"[load] cleaned flowcast   {args.cleaned_flow}")
    flow_model = Module.from_checkpoint(str(args.cleaned_flow)).to(device).eval()
    flowcast_kwargs = dict(
        num_steps=args.flowcast_num_steps,
        sigma_data=args.sigma_data,
        solver=args.flowcast_solver,
    )
    flow_preds, _ = run_leg(
        label="flowcast", model=flow_model, method="flowcast",
        regression=cleaned_reg, invariant=cleaned_inv, dataset=cleaned_ds,
        t0_indices=t0_indices, lead_time=args.lead_time,
        sampler_kwargs=flowcast_kwargs, device=device, seed=args.seed + 1,
    )
    del flow_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ leg D
    # MeanFlow (opt-in): mirrors the FlowCast leg exactly but swaps the
    # multi-step Euler ODE for the few-step average-velocity sampler. Reuses
    # the cleaned regression, dataset and invariants. Disabled (None preds)
    # unless --cleaned-meanflow is supplied -- the figure then stays 4-column.
    meanflow_preds = None
    if args.cleaned_meanflow is not None:
        print(f"[load] cleaned meanflow   {args.cleaned_meanflow}")
        meanflow_model = (
            Module.from_checkpoint(str(args.cleaned_meanflow)).to(device).eval()
        )
        meanflow_kwargs = dict(
            num_steps=args.meanflow_num_steps,
            sigma_data=args.sigma_data,
        )
        meanflow_preds, _ = run_leg(
            label="meanflow", model=meanflow_model, method="meanflow",
            regression=cleaned_reg, invariant=cleaned_inv, dataset=cleaned_ds,
            t0_indices=t0_indices, lead_time=args.lead_time,
            sampler_kwargs=meanflow_kwargs, device=device, seed=args.seed + 2,
        )
        del meanflow_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        print("[skip] meanflow leg disabled (pass --cleaned-meanflow to enable)")

    del cleaned_reg
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ plot
    leg_idx = [legacy_channels.index(c) for c in PLOT_CHANNELS]
    cln_idx = [cleaned_channels.index(c) for c in PLOT_CHANNELS]
    print(f"[plot] rendering {len(t0_indices)} figures into {args.output_dir}")
    for i, t0 in enumerate(t0_indices):
        leg_p = legacy_preds[i][leg_idx]      # (4, 224, 128)
        cln_p = cleaned_preds[i][cln_idx]     # (4, 192, 96)
        flw_p = flow_preds[i][cln_idx]        # (4, 192, 96)
        truth = cleaned_truths[i][cln_idx]    # (4, 192, 96)
        mf_p = meanflow_preds[i][cln_idx] if meanflow_preds is not None else None
        try:
            ts = cleaned_ds.valid_samples[t0 + args.lead_time]
            time_label = ts.strftime("%Y-%m-%d %H:00 UTC")
        except (AttributeError, IndexError):
            time_label = f"t0_idx={t0}"
        out_path = args.output_dir / f"time_{i + 1:02d}_lead_{args.lead_time}h.png"
        plot_one_time(
            out_path, args.lead_time, truth, leg_p, cln_p, flw_p, time_label,
            meanflow_pred=mf_p,
        )
        print(f"[plot] wrote {out_path}")

    meta = args.output_dir / "metadata.txt"
    with open(meta, "w") as f:
        f.write(f"valid_dates: {args.valid_dates}\n")
        f.write(f"lead_time: {args.lead_time} h\n")
        f.write(f"n_times: {args.n_times}\n")
        if args.extra_t0:
            f.write(f"extra_t0: {[int(t) for t in args.extra_t0]}\n")
        f.write(f"total_panels: {len(t0_indices)}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"t0_indices: {t0_indices}\n")
        f.write(f"diffusion: num_steps={args.diffusion_num_steps} solver={args.diffusion_solver}\n")
        f.write(f"flowcast:  num_steps={args.flowcast_num_steps} solver={args.flowcast_solver}\n")
        if args.cleaned_meanflow is not None:
            f.write(f"meanflow:  num_steps={args.meanflow_num_steps}\n")
        f.write(f"legacy_data:  {args.legacy_data}\n")
        f.write(f"legacy_reg:   {args.legacy_reg}\n")
        f.write(f"legacy_edm:   {args.legacy_edm}\n")
        f.write(f"cleaned_data: {args.cleaned_data}\n")
        f.write(f"cleaned_reg:  {args.cleaned_reg}\n")
        f.write(f"cleaned_edm:  {args.cleaned_edm}\n")
        f.write(f"cleaned_flow: {args.cleaned_flow}\n")
        if args.cleaned_meanflow is not None:
            f.write(f"cleaned_meanflow: {args.cleaned_meanflow}\n")
    print(f"[done] {args.output_dir}")


if __name__ == "__main__":
    main()
