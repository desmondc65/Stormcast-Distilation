"""Compare progressive-distillation phases against the EDM teacher on the
local validation split.

For each sample drawn from the validation zarr we run:
    - frozen regression model  →  mean M_t
    - EDM teacher (full N_T steps)    →  teacher prediction (M_t + ε_teacher)
    - each phase_k student (N_k steps) →  student_k prediction
and compute, per channel (t2m, u10, v10, qpepre):
    - RMSE / MAE vs. ground-truth target X_t
    - RMSE vs. teacher (how faithfully the student matches the teacher)
    - radially averaged 1-D power spectrum

The dataset and checkpoints default to the local absolute paths (see
``CLAUDE.md`` §1); the shipped Hydra configs point at NCDR-cluster paths that
don't exist on this workstation. Override via CLI flags.

References: ``stormcast/inference_ncdr.py`` (single-step inference loop),
``stormcast/utils/inference_util.py`` (CSV / plot helpers), and
``stormcast/utils/trainer_progressive.py`` (phase → num_steps schedule).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
STORMCAST_ROOT = REPO_ROOT / "stormcast"
sys.path.insert(0, str(STORMCAST_ROOT))

from physicsnemo.distributed import DistributedManager  # noqa: E402
from physicsnemo.models import Module  # noqa: E402

from datasets import dataset_classes  # noqa: E402
from utils.nn import (  # noqa: E402
    build_network_condition_and_target,
    diffusion_model_forward,
    progressive_distilled_forward,
)
from utils.spectrum import ps1d_plots  # noqa: E402


# --- Local defaults (from CLAUDE.md §1) -----------------------------------
DEFAULT_DATA_ROOT = REPO_ROOT / "exp_3_train_2_5_yrs_val_1yr_tp1" / "zarr_exp3_L_24_H_24_train_2_5_years_full"
DEFAULT_REGRESSION = REPO_ROOT / "exp_3_train_2_5_yrs_val_1yr_tp1" / "exp_3_reg_L_24_H_4_train_2_5_years" / "0" / "checkpoints_regression" / "StormCastUNet.0.14000.mdlus"
DEFAULT_TEACHER = REPO_ROOT / "exp_3_train_2_5_yrs_val_1yr_tp1" / "exp_3_dif_L_24_H_4_train_2_5_years" / "0" / "checkpoints_diffusion" / "EDMPrecond.0.70000.mdlus"
DEFAULT_PHASES = REPO_ROOT / "zettabyte_results" / "progressive" / "progressive_ncdr" / "run_0"


# ---------------------------------------------------------------------------
# Phase schedule
# ---------------------------------------------------------------------------
def phase_num_steps(initial: int, target: int, phase: int) -> int:
    """Mirror ``trainer_progressive.py``: N_k = max(initial // 2^(k+1), target)."""
    return max(initial // (2 ** (phase + 1)), target)


def discover_phases(phases_dir: Path) -> list[int]:
    out = []
    for d in sorted(phases_dir.glob("phase_*")):
        if (d / "student_final.mdlus").exists():
            try:
                out.append(int(d.name.split("_")[-1]))
            except ValueError:
                pass
    return sorted(out)


# ---------------------------------------------------------------------------
# Dataset config (in-code OmegaConf so we don't depend on Hydra paths that
# reference the NCDR cluster)
# ---------------------------------------------------------------------------
def make_dataset_cfg(data_location: Path, valid_dates: tuple[str, str]) -> "OmegaConf":
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


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
DIFFUSION_CONDITIONS = ["state", "regression", "invariant"]
REGRESSION_CONDITIONS = ["state", "background", "invariant"]


def run_model(
    model: Module,
    background: torch.Tensor,
    state_in: torch.Tensor,
    invariant_tensor: Optional[torch.Tensor],
    regression_model: Optional[Module],
    mode: str,
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
) -> torch.Tensor:
    """Full regression + diffusion/progressive-sampler single-step forward."""
    with torch.no_grad():
        condition, _, regression_output = build_network_condition_and_target(
            background,
            [state_in, state_in],
            invariant_tensor,
            regression_net=regression_model,
            condition_list=DIFFUSION_CONDITIONS,
            regression_condition_list=REGRESSION_CONDITIONS,
        )
        if regression_output is None:
            regression_output = torch.zeros_like(state_in)

        if mode == "teacher":
            sampler_args = dict(
                num_steps=num_steps, sigma_min=sigma_min, sigma_max=sigma_max,
                rho=rho, solver="heun",
            )
            diffusion_output = diffusion_model_forward(
                model, condition, state_in.shape, sampler_args=sampler_args
            )
        else:
            diffusion_output = progressive_distilled_forward(
                model, condition, state_in.shape,
                num_steps=num_steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho,
            )
    return regression_output + diffusion_output


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def per_channel_rmse(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((pred - ref) ** 2, axis=(-2, -1)))


def per_channel_mae(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(pred - ref), axis=(-2, -1))


def mean_radial_logpsd(fields: np.ndarray, channels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Sample-averaged radial power spectrum per channel.

    Args:
        fields: (S, C, H, W) denormalised predictions or truth.
    Returns:
        (k, Pk_mean) with k:(K,) and Pk_mean:(C, K).
    """
    S = fields.shape[0]
    acc = None
    k = None
    for s in range(S):
        t = torch.as_tensor(fields[s])
        _, _, num = ps1d_plots(t, t, fields=channels, diffusion_channels=channels)
        plt.close("all")
        if acc is None:
            acc = np.zeros_like(num["Pk_gen"])
            k = num["k"]
        acc += num["Pk_gen"]
    return k, acc / S


def logpsd_l1_vs_truth(pred: np.ndarray, truth: np.ndarray,
                       channels: list[str]) -> np.ndarray:
    """Per-channel L1 distance between log10 radial power spectra — the
    weather analogue of FID: it measures how well the model reproduces the
    distribution of spatial scales (i.e. "spectral realism") rather than
    per-pixel error.
    """
    _, pk_pred = mean_radial_logpsd(pred, channels)
    _, pk_true = mean_radial_logpsd(truth, channels)
    eps = 1e-12
    return np.mean(np.abs(np.log10(pk_pred + eps) - np.log10(pk_true + eps)), axis=-1)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_metric_bars(metrics: dict[str, np.ndarray], channels: list[str],
                     title: str, out_path: Path):
    if not metrics:
        return
    labels = list(metrics.keys())
    width = 0.8 / max(len(labels), 1)
    x = np.arange(len(channels))
    fig, ax = plt.subplots(figsize=(1.6 * len(channels) + 2, 4.5))
    for i, lab in enumerate(labels):
        ax.bar(x + i * width, metrics[lab], width, label=lab)
    ax.set_xticks(x + width * (len(labels) - 1) / 2)
    ax.set_xticklabels(channels)
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_field_panels(preds: dict[str, np.ndarray], truth: np.ndarray,
                      channels: list[str], out_path: Path,
                      title: str = "Per-phase predictions"):
    panels = {"truth": truth, **preds}
    labels = list(panels.keys())
    n_rows = len(channels)
    n_cols = len(labels)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.2 * n_cols + 0.8, 2.8 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )
    # Per-channel colormap — qpepre is sparse, nonneg, heavy-tailed; default
    # viridis washes it out. Use a white→blue ramp anchored at 0.
    cmaps = {"qpepre": "Blues"}
    for r, ch in enumerate(channels):
        ch_name = str(ch)
        stacked = np.stack([p[r] for p in panels.values()])
        if ch_name == "qpepre":
            vmin, vmax = 0.0, float(np.quantile(stacked, 0.995))
        else:
            vmin = float(stacked.min())
            vmax = float(stacked.max())
        cmap = cmaps.get(ch_name, "viridis")
        for c, lab in enumerate(labels):
            ax = axes[r, c]
            im = ax.imshow(panels[lab][r], origin="lower",
                           vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(lab, fontsize=10)
            if c == 0:
                ax.set_ylabel(ch_name, fontsize=10)
        # Steal a thin slice from the full row so every column stays aligned.
        fig.colorbar(im, ax=list(axes[r, :]), fraction=0.03, pad=0.02,
                     shrink=0.95)
    fig.suptitle(title, fontsize=12)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_spectra(spectra: dict[str, dict], channels: list[str], out_path: Path):
    n = len(channels)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.0), squeeze=False)
    cmap = plt.get_cmap("tab10")
    for ci, ch in enumerate(channels):
        ax = axes[0, ci]
        for idx, (lab, s) in enumerate(spectra.items()):
            ax.loglog(s["k"], s["Pk"][ci], color=cmap(idx % 10),
                      label=lab, linewidth=1.1,
                      linestyle="--" if lab == "truth" else "-")
        ax.set_title(ch)
        ax.set_xlabel("k")
        if ci == 0:
            ax.set_ylabel("P(k)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Radially averaged power spectra (averaged over samples)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_training_curves(phases_dir: Path, phases: list[int], out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.get_cmap("tab10")
    plotted = False
    for i, p in enumerate(phases):
        csv_path = phases_dir / f"phase_{p}" / "valid_loss.csv"
        if not csv_path.exists():
            continue
        steps, losses = [], []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                steps.append(int(row["step"]))
                losses.append(float(row["loss"]))
        if steps:
            ax.plot(steps, losses, color=cmap(i % 10), label=f"phase {p}", linewidth=1.2)
            plotted = True
    if not plotted:
        plt.close(fig); return
    ax.set_xlabel("global step")
    ax.set_ylabel("validation MSE")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax.set_title("Progressive distillation — validation loss per phase")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_per_channel_rmse_curves(phases_dir: Path, phases: list[int],
                                 channels: list[str], out_path: Path):
    n = len(channels)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.8), squeeze=False)
    cmap = plt.get_cmap("tab10")
    any_plotted = False
    for ci, ch in enumerate(channels):
        ax = axes[0, ci]
        for i, p in enumerate(phases):
            csv_path = phases_dir / f"phase_{p}" / f"rmse_{ch}.csv"
            if not csv_path.exists():
                continue
            steps, vals = [], []
            with open(csv_path) as f:
                for row in csv.DictReader(f):
                    steps.append(int(row["step"]))
                    vals.append(float(row["rmse"]))
            if steps:
                ax.plot(steps, vals, color=cmap(i % 10), label=f"phase {p}", linewidth=1.0)
                any_plotted = True
        ax.set_title(ch)
        ax.set_xlabel("step")
        if ci == 0:
            ax.set_ylabel("RMSE (normalised)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    if not any_plotted:
        plt.close(fig); return
    fig.suptitle("Per-channel validation RMSE over training", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# NFE sweep ablation (Fig. 4 analogue from Salimans & Ho 2022)
# ---------------------------------------------------------------------------
def _teacher_preds_at_nfe(
    teacher, sample_inputs, invariant_tensor, regression_model,
    dataset, num_steps, sigma_min, sigma_max, rho, stochastic, seed,
):
    sampler_args = dict(num_steps=num_steps, sigma_min=sigma_min,
                        sigma_max=sigma_max, rho=rho, solver="heun")
    if stochastic:
        sampler_args.update(S_churn=40.0, S_min=0.0,
                            S_max=float("inf"), S_noise=1.0)
    preds = []
    for sidx, (bg, st_in, _) in enumerate(sample_inputs):
        torch.manual_seed(seed + sidx)
        with torch.no_grad():
            condition, _, reg_out = build_network_condition_and_target(
                bg, [st_in, st_in], invariant_tensor,
                regression_net=regression_model,
                condition_list=DIFFUSION_CONDITIONS,
                regression_condition_list=REGRESSION_CONDITIONS,
            )
            if reg_out is None:
                reg_out = torch.zeros_like(st_in)
            diff_out = diffusion_model_forward(
                teacher, condition, st_in.shape, sampler_args=sampler_args
            )
            pred = reg_out + diff_out
        preds.append(dataset.denormalize_state(pred.cpu().numpy()[0].copy()))
    return np.stack(preds, axis=0)


def run_ablation(
    teacher_ckpt, phases, phases_dir, sample_inputs, truth_arr,
    invariant_tensor, regression_model, dataset, channels,
    nfe_list, initial_num_steps, target_num_steps,
    sigma_min, sigma_max, rho, device, seed, output_dir,
):
    rows = []

    print(f"[ablation] teacher sweep on NFEs={nfe_list}")
    teacher = Module.from_checkpoint(str(teacher_ckpt)).to(device).eval()
    for mode_name, stochastic in [("deterministic", False), ("stochastic", True)]:
        for N in nfe_list:
            if N < 2:
                continue
            use_cuda = device.type == "cuda"
            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            preds = _teacher_preds_at_nfe(
                teacher, sample_inputs, invariant_tensor, regression_model,
                dataset, N, sigma_min, sigma_max, rho, stochastic, seed,
            )
            if use_cuda:
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / len(sample_inputs) * 1000.0
            rmse = per_channel_rmse(preds, truth_arr).mean(axis=0)
            logpsd = logpsd_l1_vs_truth(preds, truth_arr, channels)
            rows.append({"method": mode_name, "nfe": N,
                         "rmse": rmse, "logpsd_l1": logpsd, "ms": ms})
            print(f"[ablation] {mode_name:>13s} N={N:>3d}  "
                  f"RMSE={rmse.mean():.3f}  logPSD_L1={logpsd.mean():.3f}  "
                  f"({ms:.1f} ms)")
    del teacher
    torch.cuda.empty_cache()

    print("[ablation] distilled students at native N")
    for p in phases:
        N = phase_num_steps(initial_num_steps, target_num_steps, p)
        ckpt = phases_dir / f"phase_{p}" / "student_final.mdlus"
        student = Module.from_checkpoint(str(ckpt)).to(device).eval()
        use_cuda = device.type == "cuda"
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        preds = []
        for sidx, (bg, st_in, _) in enumerate(sample_inputs):
            torch.manual_seed(seed + sidx)
            pred = run_model(student, bg, st_in, invariant_tensor, regression_model,
                             mode="student", num_steps=N,
                             sigma_min=sigma_min, sigma_max=sigma_max, rho=rho)
            preds.append(dataset.denormalize_state(pred.cpu().numpy()[0].copy()))
        if use_cuda:
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / len(sample_inputs) * 1000.0
        preds = np.stack(preds, axis=0)
        rmse = per_channel_rmse(preds, truth_arr).mean(axis=0)
        logpsd = logpsd_l1_vs_truth(preds, truth_arr, channels)
        rows.append({"method": "distilled", "nfe": N,
                     "rmse": rmse, "logpsd_l1": logpsd, "ms": ms})
        print(f"[ablation] {'distilled':>13s} N={N:>3d}  "
              f"RMSE={rmse.mean():.3f}  logPSD_L1={logpsd.mean():.3f}  "
              f"({ms:.1f} ms)")
        del student
        torch.cuda.empty_cache()

    # CSV
    csv_path = output_dir / "ablation.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = (["method", "nfe", "ms_per_sample"]
                  + [f"rmse_{c}" for c in channels]
                  + ["rmse_mean"]
                  + [f"logpsd_l1_{c}" for c in channels]
                  + ["logpsd_l1_mean"])
        w.writerow(header)
        for r in rows:
            w.writerow(
                [r["method"], r["nfe"], f"{r['ms']:.3f}"]
                + [f"{x:.6f}" for x in r["rmse"]]
                + [f"{float(r['rmse'].mean()):.6f}"]
                + [f"{x:.6f}" for x in r["logpsd_l1"]]
                + [f"{float(r['logpsd_l1'].mean()):.6f}"]
            )
    print(f"[ablation] {csv_path}")

    # Plot: RMSE vs NFE + logPSD-L1 vs NFE, three lines.
    by_method: dict[str, list] = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)
    colors = {"deterministic": "tab:blue", "stochastic": "tab:green",
              "distilled": "tab:red"}
    markers = {"deterministic": "o", "stochastic": "s", "distilled": "D"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for key, ax, ylabel in [
        ("rmse", axes[0], "RMSE vs truth (mean over channels) ↓ lower is better"),
        ("logpsd_l1", axes[1], "log-PSD L1 vs truth (FID analogue) ↓ lower is better"),
    ]:
        for m, recs in by_method.items():
            recs = sorted(recs, key=lambda r: r["nfe"])
            xs = [r["nfe"] for r in recs]
            ys = [float(r[key].mean()) for r in recs]
            ax.plot(xs, ys, marker=markers[m], color=colors[m],
                    label=m.capitalize(), linewidth=1.3)
        ax.set_xlabel("sampling steps (NFE)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Distilled vs Teacher-at-reduced-steps (Salimans & Ho 2022 Fig. 4 analogue)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / "ablation_nfe_sweep.png", dpi=140)
    plt.close(fig)

    # Per-channel log-PSD L1 panel (qpepre often behaves differently)
    fig, axes = plt.subplots(1, len(channels), figsize=(3.6 * len(channels), 4.0),
                             squeeze=False)
    for ci, ch in enumerate(channels):
        ax = axes[0, ci]
        for m, recs in by_method.items():
            recs = sorted(recs, key=lambda r: r["nfe"])
            xs = [r["nfe"] for r in recs]
            ys = [float(r["logpsd_l1"][ci]) for r in recs]
            ax.plot(xs, ys, marker=markers[m], color=colors[m],
                    label=m.capitalize(), linewidth=1.2)
        ax.set_xlabel("NFE")
        ax.set_title(ch)
        if ci == 0:
            ax.set_ylabel("log-PSD L1 ↓ lower is better")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Per-channel log-PSD L1 vs NFE")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_dir / "ablation_logpsd_per_channel.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# NetCDF writer
# ---------------------------------------------------------------------------
def save_prediction_nc(out_path: Path, prediction: np.ndarray, channels: list[str],
                       latitude: np.ndarray, longitude: np.ndarray, label: str):
    H, W = prediction.shape[1], prediction.shape[2]
    ds = xr.Dataset(
        coords={
            "y": np.arange(H), "x": np.arange(W),
            "latitude": (["y", "x"], latitude),
            "longitude": (["y", "x"], longitude),
        },
        attrs={"description": f"Progressive-distillation comparison — {label}"},
    )
    for i, ch in enumerate(channels):
        ds[ch] = (["y", "x"], prediction[i])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path, format="NETCDF4")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases-dir", type=Path, default=DEFAULT_PHASES)
    ap.add_argument("--data-location", type=Path, default=DEFAULT_DATA_ROOT,
                    help="Root of the local zarr dataset (has LowRes/, HighRes/, invariants/).")
    ap.add_argument("--regression-checkpoint", type=Path, default=DEFAULT_REGRESSION)
    ap.add_argument("--teacher-checkpoint", type=Path, default=DEFAULT_TEACHER)
    ap.add_argument("--output-dir", type=Path,
                    default=REPO_ROOT / "analyse_tools" / "progressive_comparison_v1")
    ap.add_argument("--initial-num-steps", type=int, default=18,
                    help="Teacher step count (matches "
                         "stormcast/zettabyte_scripts/train_progressive.sh: "
                         "initial_num_steps=18 → phase schedule 9→4→2→1). "
                         "Each phase_k student uses max(initial // 2^(k+1), target).")
    ap.add_argument("--target-num-steps", type=int, default=1)
    ap.add_argument("--teacher-steps", type=int, default=None,
                    help="Sampling steps for the teacher reference "
                         "(default: --initial-num-steps).")
    ap.add_argument("--sigma-min", type=float, default=0.002)
    ap.add_argument("--sigma-max", type=float, default=80.0)
    ap.add_argument("--rho", type=float, default=7.0)
    ap.add_argument("--n-samples", type=int, default=16,
                    help="Number of validation samples to evaluate (averaged).")
    ap.add_argument("--sample-indices", type=int, nargs="*", default=None,
                    help="Explicit sample indices; overrides --n-samples.")
    ap.add_argument("--valid-dates", nargs=2,
                    default=["2022/01/01", "2022/12/31"],
                    help="Validation date range (YYYY/MM/DD YYYY/MM/DD). The "
                         "local zarr covers 2022 full year.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Fixed latent-noise seed so every model sees the same z.")
    ap.add_argument("--skip-teacher", action="store_true")
    ap.add_argument("--ablation", action="store_true", default=True,
                    help="Sweep teacher NFE (deterministic + stochastic) and compare "
                         "against distilled students — reproduces the Fig. 4 "
                         "ablation from Salimans & Ho 2022 in weather-metric space.")
    ap.add_argument("--ablation-nfes", type=int, nargs="+",
                    default=[2, 4, 8, 18],
                    help="Step counts to sweep for the teacher in --ablation mode.")
    args = ap.parse_args()

    if args.teacher_steps is None:
        args.teacher_steps = args.initial_num_steps

    args.output_dir.mkdir(parents=True, exist_ok=True)

    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device
    torch.cuda.empty_cache()

    # --- dataset ---
    dataset_cfg = make_dataset_cfg(args.data_location, tuple(args.valid_dates))
    print(f"[dataset] {args.data_location} (valid {args.valid_dates})")
    dataset_cls = dataset_classes["data_loader_rwrf_era5_stable.Dataset"]
    dataset = dataset_cls(dataset_cfg, train=False)
    channels = dataset.state_channels()
    print(f"[dataset] {len(dataset)} valid samples, channels={channels}")

    latitude = np.asarray(dataset.HighRes_lat.values)
    longitude = np.asarray(dataset.HighRes_lon.values)

    invariant_array = dataset.get_invariants()
    invariant_tensor = (
        torch.from_numpy(invariant_array).to(device=device, dtype=torch.float32).unsqueeze(0)
        if invariant_array is not None else None
    )

    if args.sample_indices:
        sample_indices = list(args.sample_indices)
    else:
        rng = np.random.default_rng(args.seed)
        sample_indices = rng.choice(len(dataset), size=min(args.n_samples, len(dataset)),
                                    replace=False).tolist()
    print(f"[samples] indices = {sample_indices}")

    # --- models ---
    print(f"[regression] {args.regression_checkpoint}")
    regression_model = Module.from_checkpoint(str(args.regression_checkpoint)).to(device).eval()

    phases = discover_phases(args.phases_dir)
    if not phases:
        raise SystemExit(f"No phase_*/student_final.mdlus under {args.phases_dir}")
    print(f"[phases] found {phases}")

    # Accumulators: dict[label] -> list[(C, H, W)] across samples
    preds_all: dict[str, list[np.ndarray]] = defaultdict(list)
    truth_all: list[np.ndarray] = []

    # --- pre-load each model once, run over all samples ---
    model_plan = []
    if not args.skip_teacher:
        model_plan.append(("teacher", args.teacher_checkpoint, "teacher", args.teacher_steps))
    for p in phases:
        label = f"phase_{p}_N{phase_num_steps(args.initial_num_steps, args.target_num_steps, p)}"
        ckpt = args.phases_dir / f"phase_{p}" / "student_final.mdlus"
        n_k = phase_num_steps(args.initial_num_steps, args.target_num_steps, p)
        model_plan.append((label, ckpt, "student", n_k))

    # Cache denormalised inputs/targets once
    sample_inputs = []
    for idx in sample_indices:
        batch = dataset[idx]
        state_in_t = batch["state"][0].to(device=device, dtype=torch.float32).unsqueeze(0)
        state_tar_t = batch["state"][1].to(device=device, dtype=torch.float32).unsqueeze(0)
        background_t = batch["background"].to(device=device, dtype=torch.float32).unsqueeze(0)
        sample_inputs.append((background_t, state_in_t, state_tar_t))
        truth_all.append(dataset.denormalize_state(state_tar_t.cpu().numpy()[0].copy()))

    timings: dict[str, list[float]] = {}
    use_cuda = device.type == "cuda"
    for label, ckpt, mode, n_steps in model_plan:
        print(f"[{label}] loading {ckpt} (mode={mode}, N={n_steps})")
        model = Module.from_checkpoint(str(ckpt)).to(device).eval()

        # Warmup: first CUDA launch includes kernel compile/cache costs.
        bg0, st0, _ = sample_inputs[0]
        torch.manual_seed(args.seed)
        _ = run_model(model, bg0, st0, invariant_tensor, regression_model,
                      mode=mode, num_steps=n_steps,
                      sigma_min=args.sigma_min, sigma_max=args.sigma_max, rho=args.rho)
        if use_cuda:
            torch.cuda.synchronize()

        per_sample_ms: list[float] = []
        for sidx, (bg, st_in, _) in enumerate(sample_inputs):
            torch.manual_seed(args.seed + sidx)
            if use_cuda:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                pred = run_model(model, bg, st_in, invariant_tensor, regression_model,
                                 mode=mode, num_steps=n_steps,
                                 sigma_min=args.sigma_min, sigma_max=args.sigma_max, rho=args.rho)
                end.record()
                torch.cuda.synchronize()
                dt_ms = start.elapsed_time(end)
            else:
                t0 = time.perf_counter()
                pred = run_model(model, bg, st_in, invariant_tensor, regression_model,
                                 mode=mode, num_steps=n_steps,
                                 sigma_min=args.sigma_min, sigma_max=args.sigma_max, rho=args.rho)
                dt_ms = (time.perf_counter() - t0) * 1000.0
            per_sample_ms.append(dt_ms)
            preds_all[label].append(dataset.denormalize_state(pred.cpu().numpy()[0].copy()))
        timings[label] = per_sample_ms
        mean_ms = float(np.mean(per_sample_ms))
        std_ms = float(np.std(per_sample_ms))
        print(f"[{label}] N={n_steps} steps  →  {mean_ms:.2f} ± {std_ms:.2f} ms/sample "
              f"(min {min(per_sample_ms):.2f}, max {max(per_sample_ms):.2f})")
        del model
        torch.cuda.empty_cache()

    # --- metrics: vs. truth and vs. teacher ---
    def _stack(lst): return np.stack(lst, axis=0)   # (S, C, H, W)
    truth_arr = _stack(truth_all)
    preds_arr = {lab: _stack(v) for lab, v in preds_all.items()}

    rmse_vs_truth = {lab: per_channel_rmse(arr, truth_arr).mean(axis=0)
                     for lab, arr in preds_arr.items()}
    mae_vs_truth = {lab: per_channel_mae(arr, truth_arr).mean(axis=0)
                    for lab, arr in preds_arr.items()}

    rmse_vs_teacher = {}
    if "teacher" in preds_arr:
        teacher_arr = preds_arr["teacher"]
        rmse_vs_teacher = {lab: per_channel_rmse(arr, teacher_arr).mean(axis=0)
                           for lab, arr in preds_arr.items() if lab != "teacher"}

    # FID-analogue: radial log-PSD L1 distance vs truth, per channel.
    print("[fidelity] computing log-PSD L1 distance vs truth…")
    logpsd_l1 = {lab: logpsd_l1_vs_truth(arr, truth_arr, channels)
                 for lab, arr in preds_arr.items()}

    # CSV summary
    csv_path = args.output_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "metric", *channels])
        for lab, v in rmse_vs_truth.items():
            w.writerow([lab, "rmse_vs_truth", *[f"{x:.6f}" for x in v]])
        for lab, v in mae_vs_truth.items():
            w.writerow([lab, "mae_vs_truth", *[f"{x:.6f}" for x in v]])
        for lab, v in rmse_vs_teacher.items():
            w.writerow([lab, "rmse_vs_teacher", *[f"{x:.6f}" for x in v]])
        for lab, v in logpsd_l1.items():
            w.writerow([lab, "logpsd_l1_vs_truth", *[f"{x:.6f}" for x in v]])
    print(f"[metrics] {csv_path}")

    plot_metric_bars(logpsd_l1, channels,
                     "Per-channel log-PSD L1 vs. truth (FID analogue) ↓ lower is better",
                     args.output_dir / "logpsd_l1_vs_truth.png")

    # Timing CSV + bar plot
    n_steps_map = {lab: n for lab, _, _, n in model_plan}
    timing_csv = args.output_dir / "timing.csv"
    with open(timing_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "num_steps", "n_samples",
                    "mean_ms", "std_ms", "min_ms", "max_ms", "median_ms"])
        for lab, ts in timings.items():
            arr = np.asarray(ts)
            w.writerow([lab, n_steps_map[lab], len(arr),
                        f"{arr.mean():.4f}", f"{arr.std():.4f}",
                        f"{arr.min():.4f}", f"{arr.max():.4f}",
                        f"{np.median(arr):.4f}"])
    print(f"[timing]  {timing_csv}")

    if timings:
        labels = list(timings.keys())
        means = [float(np.mean(timings[l])) for l in labels]
        stds = [float(np.std(timings[l])) for l in labels]
        fig, ax = plt.subplots(figsize=(1.4 * len(labels) + 2, 4.5))
        xs = np.arange(len(labels))
        ax.bar(xs, means, yerr=stds, capsize=4,
               color=[plt.get_cmap("tab10")(i % 10) for i in range(len(labels))])
        for x, lab, m in zip(xs, labels, means):
            ax.text(x, m, f"{m:.1f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("inference time (ms / sample) ↓ lower is better")
        ax.set_title(f"Inference latency ({'GPU' if use_cuda else 'CPU'}, "
                     f"post-warmup, batch=1)")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(args.output_dir / "inference_time.png", dpi=150)
        plt.close(fig)

    # Bar plots
    plot_metric_bars(rmse_vs_truth, channels,
                     "Per-channel RMSE vs. ground truth ↓ lower is better",
                     args.output_dir / "rmse_vs_truth.png")
    plot_metric_bars(mae_vs_truth, channels,
                     "Per-channel MAE vs. ground truth ↓ lower is better",
                     args.output_dir / "mae_vs_truth.png")
    if rmse_vs_teacher:
        plot_metric_bars(rmse_vs_teacher, channels,
                         "Per-channel RMSE vs. teacher ↓ lower is better",
                         args.output_dir / "rmse_vs_teacher.png")

    # Field panels for the first sample
    fields_dir = args.output_dir / "fields"
    fields_dir.mkdir(parents=True, exist_ok=True)
    for sidx, idx in enumerate(sample_indices):
        sample_preds = {lab: arr[sidx] for lab, arr in preds_arr.items()}
        plot_field_panels(
            sample_preds, truth_arr[sidx], channels,
            fields_dir / f"fields_sample{sidx:03d}_idx{idx}.png",
            title=f"Per-phase predictions (sample {sidx}, dataset idx {idx})",
        )
    print(f"[fields]  {len(sample_indices)} panels → {fields_dir}")

    # Save NetCDF for every (sample, model)
    for sidx, idx in enumerate(sample_indices):
        save_prediction_nc(args.output_dir / "nc" / f"sample{sidx}_idx{idx}_truth.nc",
                           truth_arr[sidx], channels, latitude, longitude, "truth")
        for lab, arr in preds_arr.items():
            save_prediction_nc(args.output_dir / "nc" / f"sample{sidx}_idx{idx}_{lab}.nc",
                               arr[sidx], channels, latitude, longitude, lab)

    # Power spectra (averaged across samples)
    print("[spectra] averaging across samples…")
    spectra: dict[str, dict] = {}
    # Truth spectrum
    t0 = torch.as_tensor(truth_arr[0])
    _, _, num0 = ps1d_plots(t0, t0, fields=channels, diffusion_channels=channels)
    plt.close("all")
    k = num0["k"]
    acc = {"truth": np.zeros_like(num0["Pk_gen"])}
    for lab in preds_arr:
        acc[lab] = np.zeros_like(num0["Pk_gen"])
    for s in range(len(sample_indices)):
        t = torch.as_tensor(truth_arr[s])
        _, _, num_t = ps1d_plots(t, t, fields=channels, diffusion_channels=channels)
        acc["truth"] += num_t["Pk_gen"]
        for lab, arr in preds_arr.items():
            p = torch.as_tensor(arr[s])
            _, _, num_p = ps1d_plots(p, p, fields=channels, diffusion_channels=channels)
            acc[lab] += num_p["Pk_gen"]
        plt.close("all")
    for lab in acc:
        acc[lab] /= len(sample_indices)
        spectra[lab] = {"k": k, "Pk": acc[lab]}

    plot_spectra(spectra, channels, args.output_dir / "spectra.png")

    # Training-curve artefacts (from per-phase CSVs)
    plot_training_curves(args.phases_dir, phases, args.output_dir / "valid_loss_per_phase.png")
    plot_per_channel_rmse_curves(args.phases_dir, phases, channels,
                                 args.output_dir / "rmse_per_channel_per_phase.png")

    if args.ablation:
        print("\n[ablation] NFE sweep — this will run the teacher at multiple step counts…")
        run_ablation(
            teacher_ckpt=args.teacher_checkpoint,
            phases=phases,
            phases_dir=args.phases_dir,
            sample_inputs=sample_inputs,
            truth_arr=truth_arr,
            invariant_tensor=invariant_tensor,
            regression_model=regression_model,
            dataset=dataset,
            channels=channels,
            nfe_list=args.ablation_nfes,
            initial_num_steps=args.initial_num_steps,
            target_num_steps=args.target_num_steps,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            rho=args.rho,
            device=device,
            seed=args.seed,
            output_dir=args.output_dir,
        )

    print(f"\n✓ analysis complete → {args.output_dir}")


if __name__ == "__main__":
    main()
