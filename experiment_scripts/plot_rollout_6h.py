#!/usr/bin/env python3
"""6-hour HOURLY autoregressive rollout figure: one PNG per variable.

Sibling of ``plot_rollout_12h.py``. Same rows, same plumbing -- the only
difference is the horizon: this script rolls out to ``+6h`` and renders
EVERY hourly lead time as its own column (``+1h .. +6h``) instead of the
coarse ``+1h, +3h, +6h, +12h`` spacing of the 12-hour figure.

Row order and labels match the column convention of
``run_single_time_exp.py`` -- RWRF truth, legacy StormCast, cleaned
StormCast, CFM, MeanFlow as bold left-hand row headers -- with the actual
grid shape substituted into each header instead of a generic
"original/new domain" tag, and the lead times across the top:

                +1h     +2h     +3h     +4h     +5h     +6h
              +-----+ +-----+ +-----+ +-----+ +-----+ +-----+
    RWRF      |     | |     | |     | |     | |     | |     |   ground truth
    (192x96)  +-----+ +-----+ +-----+ +-----+ +-----+ +-----+
    StormCast |     | |     | |     | |     | |     | |     |   EDM, legacy
    (224x128) +-----+ +-----+ +-----+ +-----+ +-----+ +-----+
    StormCast |     | |     | |     | |     | |     | |     |   EDM, cleaned
    (192x96)  +-----+ +-----+ +-----+ +-----+ +-----+ +-----+
    CFM       |     | |     | |     | |     | |     | |     |   FlowCast
    (192x96)  +-----+ +-----+ +-----+ +-----+ +-----+ +-----+
    MeanFlow  |     | |     | |     | |     | |     | |     |   avg-velocity
    (192x96)  +-----+ +-----+ +-----+ +-----+ +-----+ +-----+

All model rows use checkpoints at the matched ~2 M training-sample
budget. Every panel is drawn at its NATIVE data aspect ratio
(``aspect='equal'``; the figure cell size is derived from the field's H:W),
so the 192x96 cleaned fields and the 224x128 legacy fields are never
stretched -- the legacy row simply letterboxes slightly inside its cell.

The script does:
    1. build the original-domain (224x128, raw qpepre, channel order
       [t2m, u10, v10, qpepre]) and new-domain (192x96, log1p, channel
       order [u10, v10, t2m, qpepre]) validation datasets and align t0 by
       hourly index;
    2. load the two regressions and three residual heads;
    3. run a deterministic single-member rollout to ``+6h`` for all three
       methods;
    4. for each of ``t2m, u10, v10, qpepre`` write one figure to
       ``--output-dir/rollout_6h_<channel>.png``.

Defaults pin the 2 M-sample checkpoints (EDMPrecond.0.70000 on the
original domain, EDMPrecond.0.31000 on the new domain,
FlowCastPrecond.0.20000 for FlowCast). Typical invocation::

    python experiment_scripts/plot_rollout_6h.py --t0-idx 4128

Pass ``--t0-idx`` to choose the initial validation-set sample (defaults to
mid-summer 2022). Pass ``--hours-to-plot 1 2 3 4 5 6`` to tweak the columns.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import thesis_style as ts  # noqa: E402  viridis-consistent palette (see thesis_style.py)

from physicsnemo.distributed import DistributedManager  # noqa: E402
from physicsnemo.models import Module  # noqa: E402

from datasets import dataset_classes  # noqa: E402
from utils.nn import (  # noqa: E402
    build_network_condition_and_target,
    diffusion_model_forward,
    flowcast_model_forward,
    meanflow_model_forward,
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
# MeanFlow shares the FlowCast SongUNet backbone and conditioning bundle, so it
# loads and rolls out exactly like the FlowCast leg; only the sampler helper
# (meanflow_model_forward, no solver kwarg) differs. Same ~2 M-sample budget.
DEFAULT_NEW_MEANFLOW = (
    REPO_ROOT / "runs/meanflow_zettabyte_v1_cleaned_4_27_2026"
    / "meanflow_zettabyte_cleaned_4_27_2026/run_0"
    / "checkpoints_meanflow/MeanFlowPrecond.0.20000.mdlus"
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


# ---- Plotting --------------------------------------------------------------
def _channel_scale(channel, stack):
    """Return (vmin, vmax, cmap) for the given channel across a stacked field.

    Thesis colour convention (thesis_style / plot_weight_comparison_grid):
    every field panel uses the viridis ramp. qpepre clips the extreme tail so
    convective structure stays legible; the other channels share the global
    min/max across all panels (clim_for style — viridis is sequential, so no
    mean-centring).
    """
    if channel == "qpepre":
        vmax = float(np.quantile(stack, 0.995) + 1e-3)
        return 0.0, vmax, ts.FIELD_CMAP
    return float(stack.min()), float(stack.max()), ts.FIELD_CMAP


def plot_channel_grid(
    *,
    channel: str,
    truth_new,            # (T, H, W) in physical units, new-domain grid
    orig_pred,            # (T, H_o, W_o) -- can be None if --skip-original
    new_edm_pred,         # (T, H, W) -- can be None if --skip-new-edm
    flow_pred,            # (T, H, W) -- can be None
    mf_pred,              # (T, H, W) -- can be None if --skip-meanflow
    hours,                # list of ints, e.g. [1,3,6,12]
    out_path: Path,
    t0_label: str,
):
    # Row order: RWRF truth on TOP as the reference, then the model rows
    # (legacy StormCast, cleaned StormCast, CFM, MeanFlow). Model labels are
    # the bold left-hand row headers (carrying the actual grid shape, e.g.
    # "StormCast (224x128)"); the lead times run across the TOP as the column
    # titles "+1h .. +6h".
    cln_shape = f"{truth_new.shape[-2]}x{truth_new.shape[-1]}"
    rows_data = [(f"RWRF ({cln_shape})", truth_new)]
    if orig_pred is not None:
        leg_shape = f"{orig_pred.shape[-2]}x{orig_pred.shape[-1]}"
        rows_data.append((f"StormCast ({leg_shape})", orig_pred))
    if new_edm_pred is not None:
        rows_data.append((f"StormCast ({cln_shape})", new_edm_pred))
    if flow_pred is not None:
        rows_data.append((f"CFM ({cln_shape})", flow_pred))
    if mf_pred is not None:
        rows_data.append((f"MeanFlow ({cln_shape})", mf_pred))

    n_rows = len(rows_data)
    n_cols = len(hours)

    # Shared colour scale across every panel in this figure.
    sample_panels = []
    for _, arr in rows_data:
        for h in hours:
            sample_panels.append(arr[h - 1])
    flat_stack = np.concatenate([p.reshape(-1) for p in sample_panels])
    vmin, vmax, cmap = _channel_scale(channel, flat_stack)

    # Size each cell to the field's NATIVE H:W so panels are never stretched
    # (the user-facing convention: keep the pictures' ratio as they are).
    ph, pw = truth_new.shape[-2], truth_new.shape[-1]
    panel_h = 3.0
    panel_w = panel_h * (pw / ph)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(panel_w * n_cols + 1.6, panel_h * n_rows + 0.6),
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
                aspect="equal",
                interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"+{h}h", fontsize=11)
            if c == 0:
                ax.set_ylabel(label, fontsize=10, fontweight="bold")
            for s in ax.spines.values():
                s.set_linewidth(0.6)

    unit = CHANNEL_UNITS.get(channel, "")
    cbar = fig.colorbar(
        im_last, ax=axes, fraction=0.025, pad=0.015, shrink=0.92,
    )
    cbar.set_label(f"{channel} [{unit}]", fontsize=10)

    fig.suptitle(
        f"6 h hourly autoregressive rollout — {channel}  ({t0_label})",
        fontsize=12, y=0.995,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def render_case(sub: Path, t0_label: str, truth, new_ch, *, orig_pred, orig_ch,
                edm_pred, flow_pred, mf_pred, hours, channels_to_plot):
    """Render all requested channel grids for one case (used by both the
    inference path and --replot)."""
    for ch in channels_to_plot:
        plot_channel_grid(
            channel=ch,
            truth_new=truth[:, new_ch.index(ch)],
            orig_pred=(orig_pred[:, orig_ch.index(ch)]
                       if orig_pred is not None else None),
            new_edm_pred=(edm_pred[:, new_ch.index(ch)]
                          if edm_pred is not None else None),
            flow_pred=(flow_pred[:, new_ch.index(ch)]
                       if flow_pred is not None else None),
            mf_pred=(mf_pred[:, new_ch.index(ch)]
                     if mf_pred is not None else None),
            hours=hours,
            out_path=sub / f"rollout_6h_{ch}.png",
            t0_label=t0_label,
        )


def write_candidates_md(out_dir: Path, candidates, n_steps: int):
    """Rain-activity ranking table used to pick a thesis case study."""
    if not candidates:
        return
    md = out_dir / "candidates.md"
    with md.open("w") as f:
        f.write("# Rollout candidates — truth qpepre activity over the "
                f"+1..+{n_steps}h window\n\n")
        f.write("Sorted by wet-pixel fraction at >=1 mm/h (most active first).\n\n")
        f.write("| t0 idx | t0 (UTC) | max rain (mm/h) | wet >=0.1 (%) | "
                "wet >=1.0 (%) | figures |\n")
        f.write("|---:|---|---:|---:|---:|---|\n")
        for t0_idx, dt0, mx, w01, w10, name in sorted(
                candidates, key=lambda c: -c[4]):
            f.write(f"| {t0_idx} | {dt0:%Y-%m-%d %H:00} | {mx:.1f} | "
                    f"{w01:.1f} | {w10:.2f} | `{name}/` |\n")
    print(f"[done] candidate summary -> {md}")


def qpepre_stats(truth, new_ch, t0_idx, dt0, sub_name):
    """(t0, datetime, max, wet>=0.1%, wet>=1.0%, dirname) tuple, or None."""
    if "qpepre" not in new_ch:
        return None
    qp = truth[:, new_ch.index("qpepre")]  # (T, H, W) mm/h
    return (t0_idx, dt0, float(qp.max()),
            100.0 * float((qp >= 0.1).mean()),
            100.0 * float((qp >= 1.0).mean()), sub_name)


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
    ap.add_argument("--new-meanflow", type=Path, default=DEFAULT_NEW_MEANFLOW)

    ap.add_argument("--valid-dates", nargs=2, default=["2022/01/01", "2022/12/31"])
    ap.add_argument("--original-hr-size", nargs=2, type=int, default=[224, 128])
    ap.add_argument("--new-hr-size", nargs=2, type=int, default=[192, 96])

    ap.add_argument(
        "--t0-idx", type=int, nargs="+", default=[4128],
        help="Validation-set hourly indices for the rollout starts (one figure "
             "set per index). Default 4128 = 2022-06-23 00:00.",
    )
    ap.add_argument(
        "--auto-t0", type=int, default=None, metavar="N",
        help="Ignore --t0-idx and instead pick N evenly-spaced initial times "
             "across the validation year (leaving room for --n-steps). Use for "
             "candidate generation, e.g. --auto-t0 20.",
    )
    ap.add_argument("--n-steps", type=int, default=6,
                    help="Autoregressive horizon in hours.")
    ap.add_argument("--hours-to-plot", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6],
                    help="Lead-time columns to render (in hours after t0). "
                         "Default is every hour out to +6h.")
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
    ap.add_argument("--meanflow-num-steps", type=int, default=2,
                    help="MeanFlow average-velocity segments (NFE). Default 2; "
                         "set 1 for one-NFE sampling. No solver kwarg.")
    ap.add_argument("--sigma-data", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--ensemble", type=int, default=1,
        help="Members per (model, t0): each member is an independently-seeded "
             "autoregressive rollout (seed, seed+1, ...) and the plotted field "
             "is the member mean — matching the ensemble-mean convention of the "
             "scoreboard and qualitative panels. Default 1 = deterministic.",
    )
    ap.add_argument(
        "--replot", action="store_true",
        help="Skip ALL inference: re-render figures from the cached "
             "t0_*/fields.npz files under --output-dir (written by previous "
             "runs). Use after style-only changes; --t0-idx is ignored.",
    )

    ap.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "experiment_scripts/results/rollout_6h",
    )
    ap.add_argument("--skip-original", action="store_true")
    ap.add_argument("--skip-new-edm", action="store_true")
    ap.add_argument("--skip-flowcast", action="store_true")
    ap.add_argument("--skip-meanflow", action="store_true")

    args = ap.parse_args()

    # ---- Replot mode: no GPU, no datasets, no models -----------------------
    if args.replot:
        import datetime as _dt
        npzs = sorted(args.output_dir.glob("t0_*/fields.npz"))
        if not npzs:
            raise SystemExit(f"--replot: no t0_*/fields.npz under {args.output_dir}")
        print(f"[replot] {len(npzs)} cached case(s) under {args.output_dir}")
        candidates = []
        for p in npzs:
            z = np.load(p)
            truth = z["truth"]
            new_ch = [str(c) for c in z["new_ch"]]
            t0_idx = int(z["t0_idx"])
            t0_label = str(z["t0_label"])
            dt0 = _dt.datetime.fromisoformat(str(z["t0_iso"]))
            render_case(
                p.parent, t0_label, truth, new_ch,
                orig_pred=z["orig"] if "orig" in z else None,
                orig_ch=[str(c) for c in z["orig_ch"]] if "orig_ch" in z else None,
                edm_pred=z["edm"] if "edm" in z else None,
                flow_pred=z["flow"] if "flow" in z else None,
                mf_pred=z["meanflow"] if "meanflow" in z else None,
                hours=args.hours_to_plot,
                channels_to_plot=args.channels_to_plot,
            )
            st = qpepre_stats(truth, new_ch, t0_idx, dt0, p.parent.name)
            if st is not None:
                candidates.append(st)
        write_candidates_md(args.output_dir, candidates, args.n_steps)
        print(f"[done] outputs in {args.output_dir}")
        return

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

    # ---- Datasets (build both up-front so t0 bounds account for each) -----
    new_cfg = make_dataset_cfg(
        args.new_data, args.valid_dates, args.new_hr_size,
        qpepre_log1p=True, kept_HR=NEW_CHANNELS,
    )
    new_ds, new_inv = build_dataset(new_cfg, device)
    new_ch = list(new_ds.state_channels())
    print(f"[data] new-domain: samples={len(new_ds)}  channels={new_ch}")

    orig_ds = orig_inv = orig_ch = None
    if not args.skip_original:
        orig_cfg = make_dataset_cfg(
            args.original_data, args.valid_dates, args.original_hr_size,
            qpepre_log1p=False, kept_HR=ORIG_CHANNELS,
        )
        orig_ds, orig_inv = build_dataset(orig_cfg, device)
        orig_ch = list(orig_ds.state_channels())
        print(f"[data] original-domain: samples={len(orig_ds)}  channels={orig_ch}")

    # ---- Resolve the initial-time list -------------------------------------
    t0_limit = len(new_ds) - args.n_steps - 1
    if orig_ds is not None:
        t0_limit = min(t0_limit, len(orig_ds) - args.n_steps - 1)
    if args.auto_t0:
        t0_list = sorted(
            dict.fromkeys(np.linspace(0, t0_limit, args.auto_t0).astype(int).tolist())
        )
    else:
        t0_list = list(args.t0_idx)
        bad = [t for t in t0_list if t < 0 or t > t0_limit]
        if bad:
            raise SystemExit(f"--t0-idx {bad} out of range [0, {t0_limit}]")

    import datetime as _dt
    y0, m0, d0 = (int(x) for x in args.valid_dates[0].split("/"))
    base_dt = _dt.datetime(y0, m0, d0)

    def t0_datetime(t0_idx: int) -> _dt.datetime:
        return base_dt + _dt.timedelta(hours=int(t0_idx))

    print(f"[t0] {len(t0_list)} initial time(s), rollout to +{args.n_steps}h: "
          + ", ".join(f"{t}({t0_datetime(t):%m-%d %H}Z)" for t in t0_list))

    def _seed_all(member: int = 0, seed_offset: int = 0):
        """Reseed before each rollout so every (model, t0, member) draw is
        reproducible; member e uses seed + seed_offset + e. ``seed_offset``
        decorrelates the per-leg RNG (e.g. MeanFlow uses +2)."""
        torch.manual_seed(args.seed + seed_offset + member)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + seed_offset + member)

    def _ens_rollout(seed_offset: int = 0, **kw):
        """Ensemble-mean rollout: average of --ensemble independently-seeded
        member rollouts (each autoregressive on its own predictions). Returns
        (pred_mean, truth) like rollout()."""
        acc = None
        truth = None
        for e in range(args.ensemble):
            _seed_all(e, seed_offset=seed_offset)
            pred, truth = rollout(**kw)
            acc = pred if acc is None else acc + pred
        return acc / args.ensemble, truth

    print(f"[load] new-domain regression {args.new_regression}")
    new_reg = (
        Module.from_checkpoint(str(args.new_regression)).to(device).eval()
    )

    # Per-t0 prediction stores: {t0_idx: (T, C, H, W) physical units}.
    new_truths: dict[int, np.ndarray] = {}
    new_edm_preds: dict[int, np.ndarray] = {}
    flow_preds: dict[int, np.ndarray] = {}
    meanflow_preds: dict[int, np.ndarray] = {}
    orig_preds: dict[int, np.ndarray] = {}

    diff_kwargs = dict(
        num_steps=args.diffusion_num_steps,
        sigma_min=args.sigma_min, sigma_max=args.sigma_max,
        rho=args.rho, solver=args.diffusion_solver,
    )

    if not args.skip_new_edm:
        print(f"[load] new-domain diffusion {args.new_diffusion}")
        new_edm = (
            Module.from_checkpoint(str(args.new_diffusion)).to(device).eval()
        )
        nfe_step = (2 * args.diffusion_num_steps if args.diffusion_solver == "heun"
                    else args.diffusion_num_steps)
        print(f"[run] new-domain EDM rollouts ({args.n_steps}h, NFE/step={nfe_step})")
        for i, t0_idx in enumerate(t0_list):
            tw = time.perf_counter()
            pred, truth = _ens_rollout(
                model=new_edm, method="diffusion",
                regression=new_reg, invariant=new_inv,
                dataset=new_ds, t0_idx=t0_idx, n_steps=args.n_steps,
                sampler_kwargs=diff_kwargs, device=device,
            )
            new_edm_preds[t0_idx] = pred
            new_truths.setdefault(t0_idx, truth)
            print(f"[run]   EDM {i + 1}/{len(t0_list)} t0={t0_idx} "
                  f"({t0_datetime(t0_idx):%m-%d %H}Z) in {time.perf_counter() - tw:.1f}s")
        del new_edm
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not args.skip_flowcast:
        print(f"[load] flowcast {args.new_flowcast}")
        flow_model = (
            Module.from_checkpoint(str(args.new_flowcast)).to(device).eval()
        )
        flow_kwargs = dict(
            num_steps=args.flowcast_num_steps,
            sigma_data=args.sigma_data, solver=args.flowcast_solver,
        )
        print(f"[run] FlowCast rollouts (NFE={args.flowcast_num_steps})")
        for i, t0_idx in enumerate(t0_list):
            tw = time.perf_counter()
            pred, truth = _ens_rollout(
                model=flow_model, method="flowcast",
                regression=new_reg, invariant=new_inv,
                dataset=new_ds, t0_idx=t0_idx, n_steps=args.n_steps,
                sampler_kwargs=flow_kwargs, device=device,
            )
            flow_preds[t0_idx] = pred
            new_truths.setdefault(t0_idx, truth)
            print(f"[run]   CFM {i + 1}/{len(t0_list)} t0={t0_idx} "
                  f"({t0_datetime(t0_idx):%m-%d %H}Z) in {time.perf_counter() - tw:.1f}s")
        del flow_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not args.skip_meanflow:
        print(f"[load] meanflow {args.new_meanflow}")
        mf_model = (
            Module.from_checkpoint(str(args.new_meanflow)).to(device).eval()
        )
        # MeanFlow sampler: no solver kwarg (average-velocity segments).
        meanflow_kwargs = dict(
            num_steps=args.meanflow_num_steps,
            sigma_data=args.sigma_data,
        )
        print(f"[run] MeanFlow rollouts (NFE={args.meanflow_num_steps})")
        for i, t0_idx in enumerate(t0_list):
            tw = time.perf_counter()
            pred, truth = _ens_rollout(
                model=mf_model, method="meanflow",
                regression=new_reg, invariant=new_inv,
                dataset=new_ds, t0_idx=t0_idx, n_steps=args.n_steps,
                sampler_kwargs=meanflow_kwargs, device=device,
                seed_offset=2,
            )
            meanflow_preds[t0_idx] = pred
            new_truths.setdefault(t0_idx, truth)
            print(f"[run]   MeanFlow {i + 1}/{len(t0_list)} t0={t0_idx} "
                  f"({t0_datetime(t0_idx):%m-%d %H}Z) in {time.perf_counter() - tw:.1f}s")
        del mf_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    del new_reg
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---- Original-domain pipeline (dataset already built up-front) ---------
    if not args.skip_original:
        print(f"[load] original-domain regression {args.original_regression}")
        orig_reg = (
            Module.from_checkpoint(str(args.original_regression)).to(device).eval()
        )
        print(f"[load] original-domain diffusion {args.original_diffusion}")
        orig_edm = (
            Module.from_checkpoint(str(args.original_diffusion)).to(device).eval()
        )
        print(f"[run] original-domain EDM rollouts ({args.n_steps}h)")
        for i, t0_idx in enumerate(t0_list):
            tw = time.perf_counter()
            pred, _ = _ens_rollout(
                model=orig_edm, method="diffusion",
                regression=orig_reg, invariant=orig_inv,
                dataset=orig_ds, t0_idx=t0_idx, n_steps=args.n_steps,
                sampler_kwargs=diff_kwargs, device=device,
            )
            orig_preds[t0_idx] = pred
            print(f"[run]   legacy {i + 1}/{len(t0_list)} t0={t0_idx} "
                  f"({t0_datetime(t0_idx):%m-%d %H}Z) in {time.perf_counter() - tw:.1f}s")
        del orig_edm, orig_reg
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not new_truths:
        raise SystemExit(
            "No rollout produced a truth field. At least one of "
            "--skip-new-edm / --skip-flowcast / --skip-meanflow must be False."
        )

    # ---- Cache fields + render one PNG per (t0, channel) -------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = []  # (t0_idx, datetime, max_rain, wet0p1, wet1p0, subdir)
    for t0_idx in t0_list:
        dt0 = t0_datetime(t0_idx)
        sub = args.output_dir / f"t0_{t0_idx:04d}_{dt0:%Y%m%d_%H}"
        sub.mkdir(parents=True, exist_ok=True)
        t0_label = dt0.strftime("t0 = %Y-%m-%d %H:00 UTC")
        if args.ensemble > 1:
            t0_label += f", {args.ensemble}-member mean"
        truth = new_truths[t0_idx]

        # Persist the (expensive) fields so --replot can re-render any future
        # style change with zero inference.
        cache = dict(
            truth=truth, new_ch=np.array(new_ch),
            t0_idx=np.int64(t0_idx), t0_label=np.str_(t0_label),
            t0_iso=np.str_(dt0.isoformat()),
        )
        if t0_idx in new_edm_preds:
            cache["edm"] = new_edm_preds[t0_idx]
        if t0_idx in flow_preds:
            cache["flow"] = flow_preds[t0_idx]
        if t0_idx in meanflow_preds:
            cache["meanflow"] = meanflow_preds[t0_idx]
        if t0_idx in orig_preds:
            cache["orig"] = orig_preds[t0_idx]
            cache["orig_ch"] = np.array(orig_ch)
        np.savez_compressed(sub / "fields.npz", **cache)

        render_case(
            sub, t0_label, truth, new_ch,
            orig_pred=orig_preds.get(t0_idx),
            orig_ch=orig_ch,
            edm_pred=new_edm_preds.get(t0_idx),
            flow_pred=flow_preds.get(t0_idx),
            mf_pred=meanflow_preds.get(t0_idx),
            hours=args.hours_to_plot,
            channels_to_plot=args.channels_to_plot,
        )
        st = qpepre_stats(truth, new_ch, t0_idx, dt0, sub.name)
        if st is not None:
            candidates.append(st)

    write_candidates_md(args.output_dir, candidates, args.n_steps)
    print(f"[done] outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
