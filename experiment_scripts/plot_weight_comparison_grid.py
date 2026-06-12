#!/usr/bin/env python3
"""Single-step (+1h) comparison grid: rows = variable, columns = weight type.

For each chosen date this renders one figure laid out as::

    rows    = [t2m, u10, v10, qpepre]                      (physical units)
    columns = truth | legacy EDM | cleaned EDM | cleaned FlowCast

The four columns are:

  1. **truth**         — the RWRF target ``M_{t+1}`` read from the *cleaned*
                          192x96 / log1p dataset.
  2. **legacy_edm**    — EDM teacher trained on the *uncropped* 224x128 raw-mm/h
                          dataset (channel order ``[t2m, u10, v10, qpepre]``).
                          Regression : ``StormCastUNet.0.7500.mdlus``
                          Diffusion  : ``EDMPrecond.0.70000.mdlus``
  3. **cleaned_edm**   — EDM teacher trained on the cleaned 192x96 / log1p
                          dataset (channel order ``[u10, v10, t2m, qpepre]``).
                          Regression : ``StormCastUNet.0.8000.mdlus``
                          Diffusion  : ``EDMPrecond.0.20000.mdlus``
  4. **cleaned_flow**  — FlowCast (CFM) student, same cleaned dataset/regression.
                          FlowCast   : ``FlowCastPrecond.0.20000.mdlus``

Each prediction is a single generative step on top of the frozen regression
mean (``M_{t+1} = mu_{t+1} + r_{t+1}``) — NOT an autoregressive rollout — so the
panels show the model's one-step skill at the chosen valid time. Everything is
plotted in post-``denormalize_state`` physical units (K, m/s, m/s, mm/h), so the
qpepre column is comparable across the raw-mm/h and log1p legs.

Caveat baked into the figure: column 2 lives on the legacy 224x128 grid while
the truth / cleaned columns are 192x96 — the *fields of view* differ (~36 % crop),
so compare patterns and magnitudes, not pixel-for-pixel positions.

This mirrors the model wiring in ``compare_diffusion_vs_flowcast.py`` /
``run_main_experiment.sh`` (same conditioning lists, channel orders, samplers and
``Module.from_checkpoint`` loading) so results line up with the §6.6 scoreboard.

Examples
--------
    # 4 evenly-spaced dates across the 2022 validation year (auto-picked)
    python plot_weight_comparison_grid.py

    # explicit dates (initial time t; the plot is the +1h forecast)
    python plot_weight_comparison_grid.py --dates 2022-06-12T00 2022-08-07T12 \
        2022-09-23T06 2022-12-01T18
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

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

# --- Static config -----------------------------------------------------------
# Row order (the variables to plot). Each dataset stores these in a different
# channel order; we look the index up per-dataset via state_channels().
ROW_VARS = ["t2m", "u10", "v10", "qpepre"]

CHANNEL_UNITS = {"t2m": "K", "u10": "m/s", "v10": "m/s", "qpepre": "mm/h"}
CHANNEL_LABELS = {ch: f"{ch}\n[{u}]" for ch, u in CHANNEL_UNITS.items()}

# Match the validation snapshots saved during training
# (stormcast/utils/plots.py:validation_plot): default "viridis" colormap, no
# fixed clim — each row auto-scales to the shared global min/max across its
# panels, origin="lower", true (equal) aspect.
FIELD_CMAP = "viridis"


def clim_for(arrs):
    """Shared (vmin, vmax) = global min/max across the row's panels — exactly
    what validation_plot() does across its generated/truth pair."""
    stack = np.concatenate([a.ravel() for a in arrs])
    return float(np.min(stack)), float(np.max(stack))

# Conditioning bundles — must match config/model/{diffusion,flowcast}.yaml and
# compare_diffusion_vs_flowcast.py.
DIFFUSION_CONDITIONS = ["state", "regression", "invariant"]
REGRESSION_CONDITIONS = ["state", "background", "invariant"]

# Default checkpoints / datasets (all under the repo root). Overridable on the CLI.
LEGACY_DATA = REPO_ROOT / "exp_3_train_2_5_yrs_val_1yr_tp1" / "zarr_exp3_L_24_H_24_train_2_5_years_full"
LEGACY_REG = (
    REPO_ROOT
    / "exp_3_train_2_5_yrs_val_1yr_tp1/exp_3_reg_L_24_H_4_train_2_5_years/0"
    / "checkpoints_regression/StormCastUNet.0.7500.mdlus"
)
LEGACY_EDM = (
    REPO_ROOT
    / "exp_3_train_2_5_yrs_val_1yr_tp1/exp_3_dif_L_24_H_4_train_2_5_years/0"
    / "checkpoints_diffusion/EDMPrecond.0.70000.mdlus"
)

CLEANED_DATA = REPO_ROOT / "exp_3_train_2_5_yrs_val_1yr_tp1" / "zarr_exp3_L_24_H_24_train_2_5_years_full_cleaned_4_27_2026"
CLEANED_REG = (
    REPO_ROOT
    / "runs/regression_zettabyte_v1_cleaned_4_27_2026/regression_zettabyte_cleaned_4_27_2026/run_0"
    / "checkpoints_regression/StormCastUNet.0.8000.mdlus"
)
CLEANED_EDM = (
    REPO_ROOT
    / "runs/diffusion_zettabyte_v1_cleaned_4_27_2026/diffusion_zettabyte_cleaned_4_27_2026/run_0"
    / "checkpoints_diffusion/EDMPrecond.0.20000.mdlus"
)
CLEANED_FLOW = (
    REPO_ROOT
    / "runs/flowcast_zettabyte_v1_cleaned_4_27_2026/flowcast_zettabyte_cleaned_4_27_2026/run_0"
    / "checkpoints_flowcast/FlowCastPrecond.0.20000.mdlus"
)
CLEANED_MEANFLOW = (
    REPO_ROOT
    / "runs/meanflow_zettabyte_v1_cleaned_4_27_2026/meanflow_zettabyte_cleaned_4_27_2026/run_0"
    / "checkpoints_meanflow/MeanFlowPrecond.0.20000.mdlus"
)

# Column display labels (top-row titles).
COL_TITLES = [
    "truth\n(cleaned 192×96)",
    "legacy EDM\n(224×128, raw)",
    "cleaned EDM\n(192×96, log1p)",
    "cleaned FlowCast\n(192×96, log1p)",
]
# Appended as a 5th column only when the MeanFlow checkpoint is present.
MEANFLOW_TITLE = "cleaned MeanFlow\n(192×96, log1p)"


# --- Dataset wiring (mirrors compare_diffusion_vs_flowcast.make_dataset_cfg) --
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


def date_index_map(dataset) -> dict[datetime, int]:
    """Map each input timestamp (datetime) -> global sample index."""
    return {ts: i for i, ts in enumerate(dataset.valid_samples)}


# --- Inference (single +1h step on top of the frozen regression mean) --------
def get_truth(dataset, idx) -> np.ndarray:
    """Return the de-normalized target M_{t+1} for sample idx, shape (C, H, W)."""
    data = dataset[idx]
    return dataset.denormalize_state(data["state"][1].cpu().numpy().copy())


def predict_single(*, method, gen_model, regression, invariant, dataset, idx, sampler_kwargs, device) -> np.ndarray:
    """One generative step at sample idx. Returns M_{t+1} in physical units (C, H, W)."""
    data = dataset[idx]
    background = data["background"].to(device=device, dtype=torch.float32).unsqueeze(0)
    state_t = data["state"][0].to(device=device, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        (condition, _, mu) = build_network_condition_and_target(
            background,
            [state_t, state_t],
            invariant,
            regression_net=regression,
            condition_list=DIFFUSION_CONDITIONS,
            regression_condition_list=REGRESSION_CONDITIONS,
        )
        if mu is None:
            mu = torch.zeros_like(state_t)
        if method == "diffusion":
            residual = diffusion_model_forward(
                gen_model, condition, state_t.shape, sampler_args=sampler_kwargs
            )
        elif method == "flowcast":
            residual = flowcast_model_forward(
                gen_model, condition, state_t.shape, **sampler_kwargs
            )
        elif method == "meanflow":
            residual = meanflow_model_forward(
                gen_model, condition, state_t.shape, **sampler_kwargs
            )
        else:
            raise ValueError(f"unknown method {method!r}")
        x_t1 = mu + residual
    return dataset.denormalize_state(x_t1.cpu().numpy()[0].copy())


# --- Plotting ----------------------------------------------------------------
def plot_grid(out_path: Path, valid_time: datetime, columns: list[dict]):
    """columns: list of dicts {title, array (C,H,W), channels: [name,...]}."""
    nrows, ncols = len(ROW_VARS), len(columns)
    # Size each column to the (portrait) data aspect so the equal-aspect panels
    # sit close together instead of floating in wide, half-empty cells.
    ph, pw = columns[0]["array"].shape[-2], columns[0]["array"].shape[-1]
    panel_h = 3.0
    panel_w = panel_h * (pw / ph)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * panel_w + 2.2, nrows * panel_h + 0.8),
        constrained_layout=True, squeeze=False,
    )
    try:  # tighten inter-column spacing (matplotlib >= 3.6 layout engine)
        fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.03, wspace=0.0, hspace=0.04)
    except Exception:
        pass

    for r, var in enumerate(ROW_VARS):
        # Pull this variable from every column (each in its own channel order).
        arrs = [col["array"][col["channels"].index(var)] for col in columns]
        vmin, vmax = clim_for(arrs)

        im = None
        for c, (col, arr) in enumerate(zip(columns, arrs)):
            ax = axes[r, c]
            # aspect="auto" fills each (portrait-proportioned) cell so columns
            # pack evenly and tightly; the figure size carries the portrait look.
            im = ax.imshow(arr, origin="lower", vmin=vmin, vmax=vmax, cmap=FIELD_CMAP, aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(col["title"], fontsize=10)
            if c == 0:
                ax.set_ylabel(CHANNEL_LABELS[var], fontsize=10, rotation=0, ha="right", va="center", labelpad=22)
        # One shared colorbar per row (geometry follows plots.py: fraction/pad).
        fig.colorbar(im, ax=axes[r, :].tolist(), fraction=0.046, pad=0.04,
                     label=CHANNEL_UNITS[var])

    fig.suptitle(
        f"Valid {valid_time:%Y-%m-%d %H:00} UTC   (+1h one-step forecast)\n"
        "rows = variable, columns = weight type — col 2 on legacy 224×128 grid, others 192×96",
        fontsize=12,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


# --- Main --------------------------------------------------------------------
def parse_date(s: str) -> datetime:
    """Accept 'YYYY-MM-DD', 'YYYY-MM-DDTHH', or full ISO. Returns naive UTC datetime."""
    s = s.strip()
    if len(s) == 13 and "T" in s:  # 'YYYY-MM-DDTHH'
        s = s + ":00:00"
    return datetime.fromisoformat(s)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--legacy-data", type=Path, default=LEGACY_DATA)
    ap.add_argument("--legacy-regression", type=Path, default=LEGACY_REG)
    ap.add_argument("--legacy-diffusion", type=Path, default=LEGACY_EDM)
    ap.add_argument("--cleaned-data", type=Path, default=CLEANED_DATA)
    ap.add_argument("--cleaned-regression", type=Path, default=CLEANED_REG)
    ap.add_argument("--cleaned-diffusion", type=Path, default=CLEANED_EDM)
    ap.add_argument("--cleaned-flowcast", type=Path, default=CLEANED_FLOW)
    ap.add_argument("--cleaned-meanflow", type=Path, default=CLEANED_MEANFLOW,
                    help="MeanFlow checkpoint for the 5th column. Skipped gracefully "
                         "if the path does not exist.")
    ap.add_argument("--valid-dates", nargs=2, default=["2022/01/01", "2022/12/31"],
                    help="Validation-year window for BOTH datasets.")

    ap.add_argument(
        "--dates", nargs="+", default=None,
        help="Initial times t (the plot is the +1h forecast). Formats: "
             "'YYYY-MM-DD', 'YYYY-MM-DDTHH', or full ISO. Default: 4 evenly-spaced "
             "dates from the common validation set.",
    )
    ap.add_argument("--n-dates", type=int, default=20,
                    help="How many evenly-spaced dates to auto-pick when --dates is omitted.")

    ap.add_argument("--output-dir", type=Path,
                    default=REPO_ROOT / "experiment_scripts" / "results" / "weight_comparison_grid")

    # Sampler hyperparameters (match the canonical compare harness / configs).
    ap.add_argument("--diffusion-num-steps", type=int, default=18)
    ap.add_argument("--sigma-min", type=float, default=0.002)
    ap.add_argument("--sigma-max", type=float, default=80.0)
    ap.add_argument("--rho", type=float, default=7.0)
    ap.add_argument("--diffusion-solver", choices=("heun", "euler"), default="heun")
    ap.add_argument("--flowcast-num-steps", type=int, default=10)
    ap.add_argument("--flowcast-solver", choices=("euler", "midpoint"), default="euler")
    ap.add_argument("--meanflow-num-steps", type=int, default=2,
                    help="MeanFlow segments (NFE). 2 by default; 1 for one-NFE.")
    ap.add_argument("--sigma-data", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    DistributedManager.initialize()
    device = DistributedManager().device
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"[device] {device}")

    # --- datasets (legacy 224x128 raw, cleaned 192x96 log1p) ----------------
    print(f"[data] legacy  {args.legacy_data}")
    legacy_ds, legacy_inv = build_dataset(
        args.legacy_data, tuple(args.valid_dates), (224, 128),
        qpepre_log1p=False, kept_channels=["t2m", "u10", "v10", "qpepre"], device=device,
    )
    print(f"[data] cleaned {args.cleaned_data}")
    cleaned_ds, cleaned_inv = build_dataset(
        args.cleaned_data, tuple(args.valid_dates), (192, 96),
        qpepre_log1p=True, kept_channels=["u10", "v10", "t2m", "qpepre"], device=device,
    )
    legacy_channels = list(legacy_ds.state_channels())
    cleaned_channels = list(cleaned_ds.state_channels())
    print(f"[data] legacy channels={legacy_channels}  cleaned channels={cleaned_channels}")
    print(f"[data] legacy pairs={len(legacy_ds)}  cleaned pairs={len(cleaned_ds)}")

    legacy_map = date_index_map(legacy_ds)
    cleaned_map = date_index_map(cleaned_ds)
    common = sorted(set(legacy_map) & set(cleaned_map))
    if not common:
        raise SystemExit("No overlapping valid timestamps between the two datasets.")

    # --- choose dates -------------------------------------------------------
    if args.dates:
        wanted = [parse_date(d) for d in args.dates]
        chosen = []
        for dt in wanted:
            if dt not in cleaned_map:
                print(f"[warn] {dt} not in cleaned validation set — skipping")
                continue
            if dt not in legacy_map:
                print(f"[warn] {dt} not in legacy validation set — skipping")
                continue
            chosen.append(dt)
        if not chosen:
            raise SystemExit("None of the requested --dates are present in both datasets.")
    else:
        pick = np.linspace(0, len(common) - 1, args.n_dates).astype(int)
        chosen = [common[i] for i in dict.fromkeys(pick.tolist())]
    print(f"[dates] {len(chosen)} date(s): " + ", ".join(f"{d:%Y-%m-%d %H:00}" for d in chosen))

    # --- load models (all via Module.from_checkpoint, matching the harness) -
    # NB: for FlowCast we load the step-20000 .mdlus directly (NOT ema_state.pt):
    # the run's ema_state.pt is the EMA shadow at the *final* step, not step 20000,
    # so it would not match this checkpoint. The compare harness loads the same way.
    print("[load] regression (legacy / cleaned)")
    legacy_reg = Module.from_checkpoint(str(args.legacy_regression)).to(device).eval()
    cleaned_reg = Module.from_checkpoint(str(args.cleaned_regression)).to(device).eval()
    print("[load] legacy EDM / cleaned EDM / cleaned FlowCast")
    legacy_edm = Module.from_checkpoint(str(args.legacy_diffusion)).to(device).eval()
    cleaned_edm = Module.from_checkpoint(str(args.cleaned_diffusion)).to(device).eval()
    cleaned_flow = Module.from_checkpoint(str(args.cleaned_flowcast)).to(device).eval()

    # Opt-in 5th column: cleaned MeanFlow. Skip gracefully if the checkpoint is
    # missing so the 4-column grid still renders without it.
    cleaned_meanflow = None
    if args.cleaned_meanflow is not None and Path(args.cleaned_meanflow).exists():
        print(f"[load] cleaned MeanFlow {args.cleaned_meanflow}")
        cleaned_meanflow = Module.from_checkpoint(str(args.cleaned_meanflow)).to(device).eval()
    else:
        print(f"[skip] cleaned MeanFlow checkpoint not found ({args.cleaned_meanflow}) "
              "— rendering the 4-column grid without it")

    edm_kwargs = dict(
        num_steps=args.diffusion_num_steps, sigma_min=args.sigma_min,
        sigma_max=args.sigma_max, rho=args.rho, solver=args.diffusion_solver,
    )
    fc_kwargs = dict(
        num_steps=args.flowcast_num_steps, sigma_data=args.sigma_data,
        solver=args.flowcast_solver,
    )
    # MeanFlow sampler takes NO solver kwarg (see utils.nn.meanflow_model_forward).
    mf_kwargs = dict(num_steps=args.meanflow_num_steps, sigma_data=args.sigma_data)

    # --- per-date figure ----------------------------------------------------
    for dt in chosen:
        idx_clean = cleaned_map[dt]
        idx_leg = legacy_map[dt]
        valid_time = dt + timedelta(hours=1)
        print(f"[run] {dt:%Y-%m-%d %H:00}  cleaned_idx={idx_clean} legacy_idx={idx_leg}")

        # Seed before each generative call so the noise draw is reproducible.
        def _seed():
            torch.manual_seed(args.seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(args.seed)

        truth = get_truth(cleaned_ds, idx_clean)
        _seed()
        col2 = predict_single(
            method="diffusion", gen_model=legacy_edm, regression=legacy_reg,
            invariant=legacy_inv, dataset=legacy_ds, idx=idx_leg,
            sampler_kwargs=edm_kwargs, device=device,
        )
        _seed()
        col3 = predict_single(
            method="diffusion", gen_model=cleaned_edm, regression=cleaned_reg,
            invariant=cleaned_inv, dataset=cleaned_ds, idx=idx_clean,
            sampler_kwargs=edm_kwargs, device=device,
        )
        _seed()
        col4 = predict_single(
            method="flowcast", gen_model=cleaned_flow, regression=cleaned_reg,
            invariant=cleaned_inv, dataset=cleaned_ds, idx=idx_clean,
            sampler_kwargs=fc_kwargs, device=device,
        )

        columns = [
            {"title": COL_TITLES[0], "array": truth, "channels": cleaned_channels},
            {"title": COL_TITLES[1], "array": col2, "channels": legacy_channels},
            {"title": COL_TITLES[2], "array": col3, "channels": cleaned_channels},
            {"title": COL_TITLES[3], "array": col4, "channels": cleaned_channels},
        ]

        # 5th column: cleaned MeanFlow (only when its checkpoint loaded). Reuses
        # the cleaned regression / dataset / invariant exactly like cleaned_flow.
        if cleaned_meanflow is not None:
            _seed()
            col5 = predict_single(
                method="meanflow", gen_model=cleaned_meanflow, regression=cleaned_reg,
                invariant=cleaned_inv, dataset=cleaned_ds, idx=idx_clean,
                sampler_kwargs=mf_kwargs, device=device,
            )
            columns.append(
                {"title": MEANFLOW_TITLE, "array": col5, "channels": cleaned_channels}
            )
        out_path = args.output_dir / f"weight_grid_{valid_time:%Y%m%d_%H}.png"
        plot_grid(out_path, valid_time, columns)

    print(f"[done] {args.output_dir}")


if __name__ == "__main__":
    main()
