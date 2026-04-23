"""Cross-method evaluation of the three distilled students vs the EDM teacher.

Implements the headline experiments listed in
``stormcast/experiments/metrics_exp_ablations.md``:

  * **E1**  Full-year (subsampled) single-step deterministic + precip benchmark.
  * **E5**  NFE-quality Pareto sweep per method.
  * **E6**  Computational benchmark (ms/sample, NFE).
  * **Visual quality panels** (Fig. 1 / Fig. 4 analogue) — per-channel
    side-by-side heatmaps for ``t2m``, ``u10``, ``v10``, ``qpepre`` of
    {ground truth, teacher@18, PD-1, CD-1, CFM-10}.

The implementation mirrors ``analyse_tools/analyse_progressive.py`` but
generalises across all four methods. Default checkpoint paths are the ones
listed in the user's request:

    - Teacher    : exp_3_dif_L_24_H_4_train_2_5_years/0/checkpoints_diffusion/EDMPrecond.0.70000.mdlus
    - Progressive: zettabyte_results/progressive/.../phase_2/checkpoints/EDMPrecond.0.32500.mdlus
    - Consistency: zettabyte_results/consistency/.../checkpoints_consistency/ConsistencyPrecond.0.37500.mdlus
    - FlowCast   : zettabyte_results/flowcast/.../checkpoints_flowcast/FlowCastPrecond.0.37500.mdlus

Note on consistency: the .mdlus file holds the *online* student, not the EMA
shadow (``ema_state.pt``). CLAUDE.md §7 warns that EMA gives better samples;
this script honours the explicit user choice.

References:
    stormcast/utils/nn.py  — sampler entry-points
    stormcast/utils/spectrum.py — power-spectrum helper
    analyse_tools/analyse_progressive.py — single-method analogue
"""

from __future__ import annotations

import argparse
import csv
import json
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

# --- repo path bootstrap --------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
STORMCAST_ROOT = REPO_ROOT / "stormcast"
sys.path.insert(0, str(STORMCAST_ROOT))

from physicsnemo.distributed import DistributedManager  # noqa: E402
from physicsnemo.models import Module  # noqa: E402

from datasets import dataset_classes  # noqa: E402
from utils.nn import (  # noqa: E402
    build_network_condition_and_target,
    consistency_model_forward,
    diffusion_model_forward,
    flowcast_model_forward,
    progressive_distilled_forward,
)
from utils.spectrum import ps1d_plots  # noqa: E402


# --- Defaults from CLAUDE.md §1 + user-requested checkpoints --------------
DEFAULT_DATA_ROOT = REPO_ROOT / "exp_3_train_2_5_yrs_val_1yr_tp1" / "zarr_exp3_L_24_H_24_train_2_5_years_full"
DEFAULT_REGRESSION = REPO_ROOT / "exp_3_train_2_5_yrs_val_1yr_tp1" / "exp_3_reg_L_24_H_4_train_2_5_years" / "0" / "checkpoints_regression" / "StormCastUNet.0.14000.mdlus"
DEFAULT_TEACHER = REPO_ROOT / "exp_3_train_2_5_yrs_val_1yr_tp1" / "exp_3_dif_L_24_H_4_train_2_5_years" / "0" / "checkpoints_diffusion" / "EDMPrecond.0.70000.mdlus"
DEFAULT_PD = REPO_ROOT / "zettabyte_results" / "progressive" / "progressive_ncdr" / "run_0" / "phase_2" / "checkpoints" / "EDMPrecond.0.32500.mdlus"
DEFAULT_CD = REPO_ROOT / "zettabyte_results" / "consistency" / "consistency" / "consistency_ncdr" / "run_0" / "checkpoints_consistency" / "ConsistencyPrecond.0.37500.mdlus"
DEFAULT_FC = REPO_ROOT / "zettabyte_results" / "flowcast" / "flowcast" / "flowcast_ncdr" / "run_0" / "checkpoints_flowcast" / "FlowCastPrecond.0.37500.mdlus"

DIFFUSION_CONDITIONS = ["state", "regression", "invariant"]
REGRESSION_CONDITIONS = ["state", "background", "invariant"]

# Categorical thresholds for qpepre in mm/h (metrics_exp_ablations.md §1.2).
PRECIP_THRESHOLDS = [0.1, 1.0, 2.5, 5.0, 10.0, 20.0, 50.0]
# FSS pooling windows in pixels (≈ 3km grid → 9, 21, 45, 93 km).
FSS_WINDOWS = [3, 7, 15, 31]


# ---------------------------------------------------------------------------
# Dataset config (in-code OmegaConf — avoids the NCDR cluster paths in
# stormcast/config/dataset/era5_rwrf_qpepre.yaml)
# ---------------------------------------------------------------------------
def make_dataset_cfg(data_location: Path,
                     valid_dates: tuple[str, str]) -> "OmegaConf":
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
# Generic forward pass: regression mean + (method-specific) residual sampler
# ---------------------------------------------------------------------------
def run_forward(
    model: Module,
    method: str,
    background: torch.Tensor,
    state_in: torch.Tensor,
    invariant_tensor: Optional[torch.Tensor],
    regression_model: Optional[Module],
    *,
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    sigma_data: float,
    flow_solver: str = "euler",
) -> torch.Tensor:
    """Method-aware single-step forward. Returns the de-normalised prediction in
    *standardised* space (caller calls ``dataset.denormalize_state``).
    """
    with torch.no_grad():
        condition, _, regression_output = build_network_condition_and_target(
            background,
            [state_in, state_in],  # target unused at inference
            invariant_tensor,
            regression_net=regression_model,
            condition_list=DIFFUSION_CONDITIONS,
            regression_condition_list=REGRESSION_CONDITIONS,
        )
        if regression_output is None:
            regression_output = torch.zeros_like(state_in)

        if method == "teacher":
            sampler_args = dict(num_steps=num_steps, sigma_min=sigma_min,
                                sigma_max=sigma_max, rho=rho, solver="heun")
            residual = diffusion_model_forward(
                model, condition, state_in.shape, sampler_args=sampler_args
            )
        elif method == "progressive":
            residual = progressive_distilled_forward(
                model, condition, state_in.shape,
                num_steps=num_steps, sigma_min=sigma_min,
                sigma_max=sigma_max, rho=rho,
            )
        elif method == "consistency":
            residual = consistency_model_forward(
                model, condition, state_in.shape,
                sigma_max=sigma_max, sigma_min=sigma_min, rho=rho,
                num_steps=num_steps,
            )
        elif method == "flowcast":
            residual = flowcast_model_forward(
                model, condition, state_in.shape,
                num_steps=num_steps, sigma_data=sigma_data,
                solver=flow_solver,
            )
        else:
            raise ValueError(f"Unknown method {method!r}")

    return regression_output + residual


# ---------------------------------------------------------------------------
# Section 1.1 — deterministic pixel-space metrics
# ---------------------------------------------------------------------------
def per_channel_rmse(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((pred - ref) ** 2, axis=(-2, -1)))


def per_channel_mae(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(pred - ref), axis=(-2, -1))


def per_channel_bias(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.mean(pred - ref, axis=(-2, -1))


def per_channel_r2(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """R² with sample-mean climatology baseline. (S, C, H, W) -> (C,)"""
    ss_res = np.sum((pred - ref) ** 2, axis=(0, -2, -1))
    mean_ref = np.mean(ref, axis=(0, -2, -1), keepdims=True)
    ss_tot = np.sum((ref - mean_ref) ** 2, axis=(0, -2, -1))
    eps = 1e-12
    return 1.0 - ss_res / (ss_tot + eps)


def per_channel_acc(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Anomaly correlation against the per-pixel sample-mean climatology of the
    held-out batch. Approximates StormCast §4.1.
    """
    clim = np.mean(ref, axis=0, keepdims=True)
    a = pred - clim
    b = ref - clim
    num = np.sum(a * b, axis=(0, -2, -1))
    den = np.sqrt(np.sum(a * a, axis=(0, -2, -1))
                  * np.sum(b * b, axis=(0, -2, -1)))
    eps = 1e-12
    return num / (den + eps)


# ---------------------------------------------------------------------------
# Section 1.2 — precipitation categorical + spatial skill
# ---------------------------------------------------------------------------
def _contingency(pred_pos: np.ndarray, ref_pos: np.ndarray) -> dict:
    tp = float(np.sum(pred_pos & ref_pos))
    fp = float(np.sum(pred_pos & ~ref_pos))
    fn = float(np.sum(~pred_pos & ref_pos))
    tn = float(np.sum(~pred_pos & ~ref_pos))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def categorical_scores(pred: np.ndarray, ref: np.ndarray,
                       thresholds: list[float]) -> dict:
    """Per-threshold CSI / FAR / POD / HSS / frequency bias for a single 2D
    field flattened across all samples. Inputs in physical units (mm/h)."""
    out: dict[float, dict[str, float]] = {}
    for tau in thresholds:
        cont = _contingency(pred >= tau, ref >= tau)
        tp, fp, fn, tn = cont["tp"], cont["fp"], cont["fn"], cont["tn"]
        eps = 1e-12
        csi = tp / (tp + fp + fn + eps)
        far = fp / (tp + fp + eps) if (tp + fp) > 0 else float("nan")
        pod = tp / (tp + fn + eps) if (tp + fn) > 0 else float("nan")
        # Heidke skill score
        n = tp + fp + fn + tn
        e = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n + eps)
        hss = (tp + tn - e) / (n - e + eps) if (n - e) > 0 else float("nan")
        bias = (tp + fp) / (tp + fn + eps) if (tp + fn) > 0 else float("nan")
        out[tau] = {"csi": csi, "far": far, "pod": pod,
                    "hss": hss, "bias_freq": bias,
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn}
    return out


def _box_pool(field: np.ndarray, w: int) -> np.ndarray:
    """Mean-pool a (S, H, W) field with a square window (no padding)."""
    if w <= 1:
        return field.copy()
    # cumulative-sum trick for exact mean — handles only valid (interior) boxes
    pad = w // 2
    arr = np.pad(field, ((0, 0), (pad, pad), (pad, pad)), mode="edge")
    cs = np.cumsum(np.cumsum(arr, axis=1), axis=2)
    cs = np.pad(cs, ((0, 0), (1, 0), (1, 0)), mode="constant")
    H, W = field.shape[1], field.shape[2]
    out = (
        cs[:, w:H + w, w:W + w]
        - cs[:, :H, w:W + w]
        - cs[:, w:H + w, :W]
        + cs[:, :H, :W]
    ) / (w * w)
    return out


def fss(pred: np.ndarray, ref: np.ndarray, tau: float, w: int) -> float:
    """Fractions Skill Score on one (S, H, W) field for threshold tau and box
    half-width w (Roberts & Lean 2008)."""
    p_bin = (pred >= tau).astype(np.float32)
    r_bin = (ref >= tau).astype(np.float32)
    pf = _box_pool(p_bin, w)
    rf = _box_pool(r_bin, w)
    num = np.mean((pf - rf) ** 2)
    den = np.mean(pf * pf) + np.mean(rf * rf) + 1e-12
    return float(1.0 - num / den)


def precip_distributional(pred: np.ndarray, ref: np.ndarray) -> dict:
    """Wet-fraction, tail quantiles, Wasserstein-1 on log1p intensities."""
    eps = 1e-6
    wet_pred = float(np.mean(pred >= 0.1))
    wet_ref = float(np.mean(ref >= 0.1))
    quantiles = [0.90, 0.95, 0.99, 0.999]
    qp = np.quantile(pred, quantiles)
    qr = np.quantile(ref, quantiles)
    # 1D Wasserstein on positive samples (log1p) — sub-sample if huge.
    p_pos = np.log1p(np.maximum(pred[pred > 0.1], eps))
    r_pos = np.log1p(np.maximum(ref[ref > 0.1], eps))
    n = min(len(p_pos), len(r_pos), 200_000)
    if n < 32:
        w1 = float("nan")
    else:
        rng = np.random.default_rng(0)
        p_s = np.sort(rng.choice(p_pos, size=n, replace=False))
        r_s = np.sort(rng.choice(r_pos, size=n, replace=False))
        w1 = float(np.mean(np.abs(p_s - r_s)))
    return {
        "wet_frac_pred": wet_pred,
        "wet_frac_ref": wet_ref,
        "wet_frac_ratio": wet_pred / (wet_ref + 1e-12),
        **{f"q{int(q * 1000) / 10}_pred": float(v) for q, v in zip(quantiles, qp)},
        **{f"q{int(q * 1000) / 10}_ref": float(v) for q, v in zip(quantiles, qr)},
        "wasserstein1_log1p": w1,
    }


# ---------------------------------------------------------------------------
# Section 1.3 — spectral / texture
# ---------------------------------------------------------------------------
def mean_radial_psd(fields: np.ndarray, channels: list[str]
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Sample-averaged radial power spectrum per channel.
    fields: (S, C, H, W) -> (k, Pk) with Pk:(C, K)."""
    S = fields.shape[0]
    acc = None
    k = None
    for s in range(S):
        t = torch.as_tensor(fields[s])
        _, _, num = ps1d_plots(t, t, fields=channels,
                               diffusion_channels=channels)
        plt.close("all")
        if acc is None:
            acc = np.zeros_like(num["Pk_gen"])
            k = num["k"]
        acc += num["Pk_gen"]
    return k, acc / S


def logpsd_l1(pred: np.ndarray, ref: np.ndarray,
              channels: list[str]) -> np.ndarray:
    """Per-channel L1 distance between log10 radial power spectra — the
    weather analogue of an FID score (FlowCast App. D)."""
    _, pk_pred = mean_radial_psd(pred, channels)
    _, pk_ref = mean_radial_psd(ref, channels)
    eps = 1e-12
    return np.mean(np.abs(np.log10(pk_pred + eps)
                          - np.log10(pk_ref + eps)), axis=-1)


def gradient_magnitude_rmse(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Per-channel RMSE of the gradient magnitude — sharpness sentinel.
    Inputs (S, C, H, W) -> (C,)."""
    def grad_mag(x):
        gy, gx = np.gradient(x, axis=(-2, -1))
        return np.sqrt(gy * gy + gx * gx)
    gp = grad_mag(pred)
    gr = grad_mag(ref)
    return np.sqrt(np.mean((gp - gr) ** 2, axis=(0, -2, -1)))


def effective_resolution(pk_pred: np.ndarray, pk_ref: np.ndarray,
                         k: np.ndarray, db_drop: float = 3.0) -> np.ndarray:
    """Smallest wavelength (1/k_cutoff) where 10·log10(pk_pred/pk_ref) drops
    ≥ db_drop. Returns one value per channel; NaN if never drops below."""
    out = np.full(pk_pred.shape[0], np.nan, dtype=np.float64)
    eps = 1e-12
    diff_db = 10.0 * (np.log10(pk_pred + eps) - np.log10(pk_ref + eps))
    for c in range(pk_pred.shape[0]):
        below = diff_db[c] <= -db_drop
        if np.any(below):
            ki = np.argmax(below)  # first crossing
            kk = max(k[ki], 1e-6)
            out[c] = 1.0 / kk
    return out


# ---------------------------------------------------------------------------
# Visual-quality panels — Fig. 1 / Fig. 4 of the doc
# ---------------------------------------------------------------------------
def plot_visual_panel(samples: dict[str, np.ndarray], truth: np.ndarray,
                      channels: list[str], out_path: Path, title: str):
    """Render a (channels × methods+1) heatmap grid for one validation sample.

    samples: dict[label] -> (C, H, W) prediction in physical units.
    truth:   (C, H, W) physical-unit ground truth.
    """
    panels = {"truth": truth, **samples}
    labels = list(panels.keys())
    cmaps = {"qpepre": "Blues", "t2m": "RdBu_r", "u10": "RdBu_r", "v10": "RdBu_r"}
    n_rows, n_cols = len(channels), len(labels)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.2 * n_cols + 1.0, 2.7 * n_rows),
        squeeze=False, constrained_layout=True,
    )
    for r, ch in enumerate(channels):
        stacked = np.stack([p[r] for p in panels.values()])
        if ch == "qpepre":
            vmin, vmax = 0.0, float(np.quantile(stacked, 0.995) + 1e-3)
        else:
            vabs = float(np.quantile(np.abs(stacked - np.mean(stacked)), 0.99))
            cm = float(np.mean(stacked))
            vmin, vmax = cm - vabs, cm + vabs
        cmap = cmaps.get(ch, "viridis")
        im = None
        for c, lab in enumerate(labels):
            ax = axes[r, c]
            im = ax.imshow(panels[lab][r], origin="lower", vmin=vmin,
                           vmax=vmax, cmap=cmap, aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(lab, fontsize=10)
            if c == 0:
                ax.set_ylabel(ch, fontsize=10)
        fig.colorbar(im, ax=list(axes[r, :]), fraction=0.03, pad=0.02,
                     shrink=0.95)
    fig.suptitle(title, fontsize=12)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_method_bar(metrics: dict[str, np.ndarray], channels: list[str],
                    title: str, out_path: Path):
    if not metrics:
        return
    labels = list(metrics.keys())
    width = 0.8 / max(len(labels), 1)
    x = np.arange(len(channels))
    fig, ax = plt.subplots(figsize=(1.6 * len(channels) + 2, 4.5))
    cmap = plt.get_cmap("tab10")
    for i, lab in enumerate(labels):
        ax.bar(x + i * width, metrics[lab], width, label=lab,
               color=cmap(i % 10))
    ax.set_xticks(x + width * (len(labels) - 1) / 2)
    ax.set_xticklabels(channels)
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_spectra(spectra: dict[str, dict], channels: list[str], out_path: Path):
    n = len(channels)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.0), squeeze=False)
    cmap = plt.get_cmap("tab10")
    for ci, ch in enumerate(channels):
        ax = axes[0, ci]
        for idx, (lab, s) in enumerate(spectra.items()):
            ax.loglog(s["k"], s["Pk"][ci], color=cmap(idx % 10),
                      label=lab, linewidth=1.2,
                      linestyle="--" if lab == "truth" else "-")
        ax.set_title(ch)
        ax.set_xlabel("k")
        if ci == 0:
            ax.set_ylabel("P(k)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Radially averaged power spectra (per channel, sample-mean)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_pareto(rows: list[dict], channels: list[str], out_path: Path):
    """Plot E5 Pareto: NFE vs {RMSE_mean, log-PSD-L1_mean}, one trace per
    method (4 colours). Open marker for the teacher.
    """
    by_method: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)
    colors = {"teacher": "tab:gray", "progressive": "tab:red",
              "consistency": "tab:blue", "flowcast": "tab:green"}
    markers = {"teacher": "o", "progressive": "D",
               "consistency": "s", "flowcast": "^"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for key, ax, ylabel in [
        ("rmse_mean", axes[0], "RMSE vs truth (channel mean) ↓"),
        ("logpsd_l1_mean", axes[1], "log-PSD L1 vs truth (channel mean) ↓"),
    ]:
        for m, recs in by_method.items():
            recs = sorted(recs, key=lambda r: r["nfe"])
            xs = [r["nfe"] for r in recs]
            ys = [float(r[key]) for r in recs]
            ax.plot(xs, ys, marker=markers.get(m, "o"),
                    color=colors.get(m, "k"), label=m,
                    linewidth=1.4, markersize=7,
                    markerfacecolor="white" if m == "teacher" else colors.get(m, "k"))
        ax.set_xscale("log")
        ax.set_xlabel("NFE (log scale)")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.suptitle("NFE-quality Pareto (E5 / metrics_exp_ablations.md §3.5)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Inference driver
# ---------------------------------------------------------------------------
def run_method_on_samples(
    *, model: Module, method: str, num_steps: int,
    sample_inputs: list, invariant_tensor, regression_model,
    sigma_min: float, sigma_max: float, rho: float,
    sigma_data: float, flow_solver: str, dataset, seed: int,
    use_cuda: bool,
) -> tuple[np.ndarray, list[float]]:
    """Returns (preds[S, C, H, W] in physical units, per-sample ms list)."""
    bg0, st0, _ = sample_inputs[0]
    torch.manual_seed(seed)
    _ = run_forward(model, method, bg0, st0, invariant_tensor,
                    regression_model, num_steps=num_steps,
                    sigma_min=sigma_min, sigma_max=sigma_max, rho=rho,
                    sigma_data=sigma_data, flow_solver=flow_solver)
    if use_cuda:
        torch.cuda.synchronize()

    preds, per_ms = [], []
    for sidx, (bg, st_in, _) in enumerate(sample_inputs):
        torch.manual_seed(seed + sidx)
        if use_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            pred = run_forward(model, method, bg, st_in, invariant_tensor,
                               regression_model, num_steps=num_steps,
                               sigma_min=sigma_min, sigma_max=sigma_max,
                               rho=rho, sigma_data=sigma_data,
                               flow_solver=flow_solver)
            end.record()
            torch.cuda.synchronize()
            dt_ms = start.elapsed_time(end)
        else:
            t0 = time.perf_counter()
            pred = run_forward(model, method, bg, st_in, invariant_tensor,
                               regression_model, num_steps=num_steps,
                               sigma_min=sigma_min, sigma_max=sigma_max,
                               rho=rho, sigma_data=sigma_data,
                               flow_solver=flow_solver)
            dt_ms = (time.perf_counter() - t0) * 1000.0
        per_ms.append(dt_ms)
        preds.append(dataset.denormalize_state(pred.cpu().numpy()[0].copy()))
    return np.stack(preds, axis=0), per_ms


# ---------------------------------------------------------------------------
# NetCDF writer (kept compatible with analyse_progressive)
# ---------------------------------------------------------------------------
def save_prediction_nc(out_path: Path, prediction: np.ndarray,
                       channels: list[str], latitude, longitude, label: str):
    H, W = prediction.shape[1], prediction.shape[2]
    ds = xr.Dataset(
        coords={"y": np.arange(H), "x": np.arange(W),
                "latitude": (["y", "x"], latitude),
                "longitude": (["y", "x"], longitude)},
        attrs={"description": f"Cross-method comparison — {label}"},
    )
    for i, ch in enumerate(channels):
        ds[ch] = (["y", "x"], prediction[i])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path, format="NETCDF4")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-location", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--regression-checkpoint", type=Path, default=DEFAULT_REGRESSION)
    ap.add_argument("--teacher-checkpoint", type=Path, default=DEFAULT_TEACHER)
    ap.add_argument("--progressive-checkpoint", type=Path, default=DEFAULT_PD)
    ap.add_argument("--consistency-checkpoint", type=Path, default=DEFAULT_CD)
    ap.add_argument("--flowcast-checkpoint", type=Path, default=DEFAULT_FC)
    ap.add_argument("--output-dir", type=Path,
                    default=REPO_ROOT / "stormcast" / "experiments"
                    / "results_cross_method")
    ap.add_argument("--n-samples", type=int, default=64,
                    help="Number of validation samples for E1 + visual + spectra.")
    ap.add_argument("--sample-stride", type=int, default=144,
                    help="Stride (hours) for E1 subsampling. Default 6×24=144 → "
                         "~one sample per 6 days. Use 6 for the full §3.1 protocol.")
    ap.add_argument("--n-vis-panels", type=int, default=6,
                    help="How many per-sample heatmap panels to write.")
    ap.add_argument("--valid-dates", nargs=2,
                    default=["2022/01/01", "2022/12/31"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma-min", type=float, default=0.002)
    ap.add_argument("--sigma-max", type=float, default=80.0)
    ap.add_argument("--rho", type=float, default=7.0)
    ap.add_argument("--sigma-data", type=float, default=0.5)
    ap.add_argument("--teacher-nfe", type=int, default=18,
                    help="Reference teacher NFE for E1 panels and spectra.")
    ap.add_argument("--pd-nfe", type=int, default=2,
                    help="PD inference NFE. With initial=18, target=1 the phase "
                         "schedule is 18→9→4→2→1; phase_2 → 2-step student "
                         "(see trainer_progressive.py and CLAUDE.md §3.1).")
    ap.add_argument("--cd-nfe", type=int, default=1)
    ap.add_argument("--cfm-nfe", type=int, default=10)
    ap.add_argument("--pareto-teacher-nfes", type=int, nargs="+",
                    default=[2, 4, 8, 18])
    ap.add_argument("--pareto-cd-nfes", type=int, nargs="+",
                    default=[1, 2, 4, 8])
    ap.add_argument("--pareto-cfm-nfes", type=int, nargs="+",
                    default=[1, 2, 4, 10, 25])
    ap.add_argument("--pareto-pd-nfes", type=int, nargs="+",
                    default=[1, 2, 4],
                    help="PD phase_2 student was distilled to 2 steps; the "
                         "1- and 4-step entries report 'naive step-down/up' "
                         "behaviour for the same weights.")
    ap.add_argument("--skip-pareto", action="store_true",
                    help="Skip the E5 NFE sweep (saves a lot of compute).")
    ap.add_argument("--skip-nc", action="store_true",
                    help="Skip per-sample NetCDF dumps.")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics").mkdir(exist_ok=True)
    (args.output_dir / "figures").mkdir(exist_ok=True)
    (args.output_dir / "panels").mkdir(exist_ok=True)
    if not args.skip_nc:
        (args.output_dir / "nc").mkdir(exist_ok=True)

    # ---- runtime / device ----
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device
    use_cuda = device.type == "cuda"
    torch.cuda.empty_cache() if use_cuda else None

    # ---- dataset ----
    cfg = make_dataset_cfg(args.data_location, tuple(args.valid_dates))
    print(f"[dataset] {args.data_location} valid={args.valid_dates}")
    dataset_cls = dataset_classes["data_loader_rwrf_era5_stable.Dataset"]
    dataset = dataset_cls(cfg, train=False)
    channels: list[str] = list(dataset.state_channels())
    print(f"[dataset] N={len(dataset)} channels={channels}")

    latitude = np.asarray(dataset.HighRes_lat.values)
    longitude = np.asarray(dataset.HighRes_lon.values)

    inv = dataset.get_invariants()
    invariant_tensor = (
        torch.from_numpy(inv).to(device=device, dtype=torch.float32).unsqueeze(0)
        if inv is not None else None
    )

    # ---- sample selection: E1 protocol (§3.1) — every Nth hour ----
    stride = max(int(args.sample_stride), 1)
    grid = list(range(0, len(dataset), stride))
    if len(grid) == 0:
        raise SystemExit("Empty sample grid — check --sample-stride / dataset size.")
    if args.n_samples >= len(grid):
        sample_indices = grid
    else:
        # Pick evenly spaced subset of the grid for tractability.
        idx = np.linspace(0, len(grid) - 1, args.n_samples).astype(int)
        sample_indices = [grid[i] for i in idx]
    print(f"[samples] {len(sample_indices)} indices "
          f"(stride={stride}h, first={sample_indices[0]}, last={sample_indices[-1]})")

    # ---- regression (frozen) ----
    print(f"[regression] {args.regression_checkpoint}")
    regression_model = Module.from_checkpoint(
        str(args.regression_checkpoint)).to(device).eval()

    # ---- pre-fetch sample tensors + truth ----
    sample_inputs: list[tuple] = []
    truth_list: list[np.ndarray] = []
    for idx in sample_indices:
        batch = dataset[idx]
        st_in = batch["state"][0].to(device=device, dtype=torch.float32).unsqueeze(0)
        st_tar = batch["state"][1].to(device=device, dtype=torch.float32).unsqueeze(0)
        bg = batch["background"].to(device=device, dtype=torch.float32).unsqueeze(0)
        sample_inputs.append((bg, st_in, st_tar))
        truth_list.append(dataset.denormalize_state(st_tar.cpu().numpy()[0].copy()))
    truth_arr = np.stack(truth_list, axis=0)

    # ---- E1: load each method once, run on all samples ----
    method_plan = [
        ("teacher",     args.teacher_checkpoint,     "teacher",     args.teacher_nfe),
        ("progressive", args.progressive_checkpoint, "progressive", args.pd_nfe),
        ("consistency", args.consistency_checkpoint, "consistency", args.cd_nfe),
        ("flowcast",    args.flowcast_checkpoint,    "flowcast",    args.cfm_nfe),
    ]
    label_for: dict[str, str] = {}
    preds: dict[str, np.ndarray] = {}
    timings: dict[str, list[float]] = {}
    nfe_for: dict[str, int] = {}

    for tag, ckpt, method, nfe in method_plan:
        label = f"{tag}_N{nfe}"
        label_for[tag] = label
        nfe_for[label] = nfe
        print(f"[{label}] loading {ckpt}")
        model = Module.from_checkpoint(str(ckpt)).to(device).eval()

        arr, ms = run_method_on_samples(
            model=model, method=method, num_steps=nfe,
            sample_inputs=sample_inputs, invariant_tensor=invariant_tensor,
            regression_model=regression_model,
            sigma_min=args.sigma_min, sigma_max=args.sigma_max, rho=args.rho,
            sigma_data=args.sigma_data, flow_solver="euler", dataset=dataset,
            seed=args.seed, use_cuda=use_cuda,
        )
        preds[label] = arr
        timings[label] = ms
        m, sd = float(np.mean(ms)), float(np.std(ms))
        print(f"[{label}] N={nfe}  →  {m:.2f} ± {sd:.2f} ms/sample  "
              f"(min {min(ms):.2f}, max {max(ms):.2f})")
        del model
        if use_cuda:
            torch.cuda.empty_cache()

    # ---- §1.1 deterministic metrics ----
    print("[E1] deterministic metrics")
    rmse = {lab: per_channel_rmse(arr, truth_arr).mean(axis=0)
            for lab, arr in preds.items()}
    mae = {lab: per_channel_mae(arr, truth_arr).mean(axis=0)
           for lab, arr in preds.items()}
    bias = {lab: per_channel_bias(arr, truth_arr).mean(axis=0)
            for lab, arr in preds.items()}
    r2 = {lab: per_channel_r2(arr, truth_arr) for lab, arr in preds.items()}
    acc = {lab: per_channel_acc(arr, truth_arr) for lab, arr in preds.items()}
    grad_rmse = {lab: gradient_magnitude_rmse(arr, truth_arr)
                 for lab, arr in preds.items()}

    # vs teacher (matching is the distillation objective)
    teacher_label = label_for["teacher"]
    rmse_vs_teacher = {lab: per_channel_rmse(arr, preds[teacher_label]).mean(axis=0)
                       for lab, arr in preds.items() if lab != teacher_label}

    # ---- §1.3 spectra + log-PSD-L1 vs truth ----
    print("[E1] log-PSD L1 vs truth (per channel)")
    psd_l1 = {lab: logpsd_l1(arr, truth_arr, channels)
              for lab, arr in preds.items()}

    print("[E1] effective resolution (per channel)")
    k_truth, pk_truth = mean_radial_psd(truth_arr, channels)
    eff_res: dict[str, np.ndarray] = {}
    spectra: dict[str, dict] = {"truth": {"k": k_truth, "Pk": pk_truth}}
    for lab, arr in preds.items():
        k_p, pk_p = mean_radial_psd(arr, channels)
        spectra[lab] = {"k": k_p, "Pk": pk_p}
        eff_res[lab] = effective_resolution(pk_p, pk_truth, k_p)

    # ---- §1.2 precip categorical / spatial / distributional ----
    print("[E1] precipitation skill (qpepre)")
    qpepre_idx = channels.index("qpepre")
    qp_truth = truth_arr[:, qpepre_idx, :, :]
    cat: dict[str, dict] = {}
    fss_metric: dict[str, dict] = {}
    distr: dict[str, dict] = {}
    for lab, arr in preds.items():
        qp = arr[:, qpepre_idx, :, :]
        cat[lab] = categorical_scores(qp, qp_truth, PRECIP_THRESHOLDS)
        fss_metric[lab] = {
            (tau, w): fss(qp, qp_truth, tau, w)
            for tau in [1.0, 5.0, 10.0]
            for w in FSS_WINDOWS
        }
        distr[lab] = precip_distributional(qp, qp_truth)

    # ---- writeout: deterministic + spectral CSV ----
    det_csv = args.output_dir / "metrics" / "deterministic_e1.csv"
    with open(det_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "metric", *channels])
        for lab in preds:
            w.writerow([lab, "rmse_vs_truth", *[f"{v:.6f}" for v in rmse[lab]]])
            w.writerow([lab, "mae_vs_truth", *[f"{v:.6f}" for v in mae[lab]]])
            w.writerow([lab, "bias_vs_truth", *[f"{v:.6f}" for v in bias[lab]]])
            w.writerow([lab, "r2_vs_truth", *[f"{v:.6f}" for v in r2[lab]]])
            w.writerow([lab, "acc_vs_truth", *[f"{v:.6f}" for v in acc[lab]]])
            w.writerow([lab, "grad_mag_rmse",
                        *[f"{v:.6f}" for v in grad_rmse[lab]]])
            w.writerow([lab, "logpsd_l1_vs_truth",
                        *[f"{v:.6f}" for v in psd_l1[lab]]])
            w.writerow([lab, "effective_resolution_pixels",
                        *[f"{v:.6f}" for v in eff_res[lab]]])
        for lab, vec in rmse_vs_teacher.items():
            w.writerow([lab, "rmse_vs_teacher", *[f"{v:.6f}" for v in vec]])
    print(f"[csv] {det_csv}")

    # ---- writeout: precip CSV ----
    pre_csv = args.output_dir / "metrics" / "precipitation_e1.csv"
    with open(pre_csv, "w", newline="") as f:
        w = csv.writer(f)
        head = (["method", "threshold_mm_h", "csi", "far", "pod", "hss",
                 "frequency_bias", "tp", "fp", "fn", "tn"])
        w.writerow(head)
        for lab in preds:
            for tau, sc in cat[lab].items():
                w.writerow([lab, tau,
                            f"{sc['csi']:.6f}", f"{sc['far']:.6f}",
                            f"{sc['pod']:.6f}", f"{sc['hss']:.6f}",
                            f"{sc['bias_freq']:.6f}",
                            f"{sc['tp']:.0f}", f"{sc['fp']:.0f}",
                            f"{sc['fn']:.0f}", f"{sc['tn']:.0f}"])
    print(f"[csv] {pre_csv}")

    fss_csv = args.output_dir / "metrics" / "fss_e1.csv"
    with open(fss_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "threshold_mm_h", "window_px", "fss"])
        for lab in preds:
            for (tau, ww), v in fss_metric[lab].items():
                w.writerow([lab, tau, ww, f"{v:.6f}"])
    print(f"[csv] {fss_csv}")

    distr_csv = args.output_dir / "metrics" / "precip_distributional_e1.csv"
    with open(distr_csv, "w", newline="") as f:
        keys = list(next(iter(distr.values())).keys())
        w = csv.writer(f)
        w.writerow(["method", *keys])
        for lab in preds:
            w.writerow([lab, *[f"{distr[lab][k]:.6f}" for k in keys]])
    print(f"[csv] {distr_csv}")

    # ---- writeout: timing / E6 ----
    tim_csv = args.output_dir / "metrics" / "timing_e6.csv"
    with open(tim_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "nfe", "n_samples",
                    "mean_ms", "std_ms", "min_ms", "max_ms", "median_ms",
                    "speedup_vs_teacher"])
        teacher_mean = float(np.mean(timings[teacher_label]))
        for lab, ms in timings.items():
            arr = np.asarray(ms)
            w.writerow([lab, nfe_for[lab], len(arr),
                        f"{arr.mean():.4f}", f"{arr.std():.4f}",
                        f"{arr.min():.4f}", f"{arr.max():.4f}",
                        f"{np.median(arr):.4f}",
                        f"{teacher_mean / arr.mean():.4f}"])
    print(f"[csv] {tim_csv}")

    # ---- bar charts (figures) ----
    figs = args.output_dir / "figures"
    plot_method_bar(rmse, channels, "RMSE vs truth ↓", figs / "e1_rmse.png")
    plot_method_bar(mae, channels, "MAE vs truth ↓", figs / "e1_mae.png")
    plot_method_bar({k: np.abs(v) for k, v in bias.items()}, channels,
                    "|Bias| vs truth ↓", figs / "e1_abs_bias.png")
    plot_method_bar(acc, channels, "ACC vs truth ↑", figs / "e1_acc.png")
    plot_method_bar(psd_l1, channels, "log-PSD-L1 vs truth ↓ (FID analogue)",
                    figs / "e1_logpsd_l1.png")
    plot_method_bar(grad_rmse, channels,
                    "Gradient-magnitude RMSE ↓ (sharpness)",
                    figs / "e1_grad_rmse.png")
    if rmse_vs_teacher:
        plot_method_bar(rmse_vs_teacher, channels,
                        "RMSE vs teacher (distillation faithfulness) ↓",
                        figs / "e1_rmse_vs_teacher.png")

    # ---- spectra figure (Fig. 1 analogue) ----
    plot_spectra(spectra, channels, figs / "e1_rapsd.png")

    # ---- precip CSI / FAR / HSS curves vs threshold ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), squeeze=False)
    cmap = plt.get_cmap("tab10")
    for j, key in enumerate(["csi", "far", "hss"]):
        ax = axes[0, j]
        for i, lab in enumerate(preds):
            ys = [cat[lab][tau][key] for tau in PRECIP_THRESHOLDS]
            ax.plot(PRECIP_THRESHOLDS, ys, marker="o", color=cmap(i % 10),
                    label=lab, linewidth=1.4)
        ax.set_xscale("log")
        ax.set_xlabel("threshold (mm/h)")
        ax.set_ylabel(key.upper())
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Precipitation categorical skill vs threshold (E1 / §1.2)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(figs / "e1_precip_curves.png", dpi=140)
    plt.close(fig)

    # ---- visual quality panels (Fig. 4 analogue) ----
    n_panels = min(args.n_vis_panels, len(sample_indices))
    pick = np.linspace(0, len(sample_indices) - 1, n_panels).astype(int)
    for sidx in pick:
        sample = {lab: arr[sidx] for lab, arr in preds.items()}
        plot_visual_panel(
            sample, truth_arr[sidx], channels,
            args.output_dir / "panels" /
            f"panel_sample{sidx:03d}_idx{sample_indices[sidx]}.png",
            title=f"Per-method comparison — dataset idx {sample_indices[sidx]}",
        )
    print(f"[panels] {n_panels} → {args.output_dir / 'panels'}")

    # ---- NetCDF dumps (per-sample) ----
    if not args.skip_nc:
        for sidx in pick:
            idx = sample_indices[sidx]
            save_prediction_nc(args.output_dir / "nc" /
                               f"sample{sidx:03d}_idx{idx}_truth.nc",
                               truth_arr[sidx], channels, latitude, longitude,
                               "truth")
            for lab, arr in preds.items():
                save_prediction_nc(args.output_dir / "nc" /
                                   f"sample{sidx:03d}_idx{idx}_{lab}.nc",
                                   arr[sidx], channels, latitude, longitude,
                                   lab)

    # ---- E5: NFE Pareto ----
    if not args.skip_pareto:
        print("\n[E5] NFE-quality Pareto sweep")
        rows: list[dict] = []
        sweep_plan = [
            ("teacher",     args.teacher_checkpoint,     "teacher",
             args.pareto_teacher_nfes),
            ("progressive", args.progressive_checkpoint, "progressive",
             args.pareto_pd_nfes),
            ("consistency", args.consistency_checkpoint, "consistency",
             args.pareto_cd_nfes),
            ("flowcast",    args.flowcast_checkpoint,    "flowcast",
             args.pareto_cfm_nfes),
        ]
        for tag, ckpt, method, nfes in sweep_plan:
            print(f"[E5] loading {tag} for sweep")
            model = Module.from_checkpoint(str(ckpt)).to(device).eval()
            for N in nfes:
                if method == "teacher" and N < 2:
                    continue  # Heun sampler requires N>=2
                arr, ms = run_method_on_samples(
                    model=model, method=method, num_steps=N,
                    sample_inputs=sample_inputs,
                    invariant_tensor=invariant_tensor,
                    regression_model=regression_model,
                    sigma_min=args.sigma_min, sigma_max=args.sigma_max,
                    rho=args.rho, sigma_data=args.sigma_data,
                    flow_solver="euler", dataset=dataset, seed=args.seed,
                    use_cuda=use_cuda,
                )
                rmse_vec = per_channel_rmse(arr, truth_arr).mean(axis=0)
                psd_vec = logpsd_l1(arr, truth_arr, channels)
                qp = arr[:, qpepre_idx, :, :]
                csi10 = categorical_scores(qp, qp_truth, [10.0])[10.0]["csi"]
                row = {
                    "method": tag, "nfe": N,
                    "ms_mean": float(np.mean(ms)),
                    "rmse_mean": float(rmse_vec.mean()),
                    "logpsd_l1_mean": float(psd_vec.mean()),
                    "csi_qpepre_10mm": csi10,
                    **{f"rmse_{c}": float(rmse_vec[ci])
                       for ci, c in enumerate(channels)},
                    **{f"logpsd_{c}": float(psd_vec[ci])
                       for ci, c in enumerate(channels)},
                }
                rows.append(row)
                print(f"[E5] {tag:>11s} N={N:>3d}  RMSE={row['rmse_mean']:.3f}  "
                      f"logPSD_L1={row['logpsd_l1_mean']:.3f}  "
                      f"CSI@10={csi10:.3f}  ({row['ms_mean']:.1f} ms)")
            del model
            if use_cuda:
                torch.cuda.empty_cache()

        # CSV + plot
        e5_csv = args.output_dir / "metrics" / "pareto_e5.csv"
        with open(e5_csv, "w", newline="") as f:
            keys = list(rows[0].keys())
            w = csv.writer(f)
            w.writerow(keys)
            for r in rows:
                w.writerow([r[k] for k in keys])
        print(f"[csv] {e5_csv}")
        plot_pareto(rows, channels, figs / "e5_nfe_pareto.png")

    # ---- summary JSON for downstream tooling ----
    summary = {
        "channels": channels,
        "n_samples": len(sample_indices),
        "sample_indices": list(sample_indices),
        "valid_dates": list(args.valid_dates),
        "checkpoints": {
            "regression": str(args.regression_checkpoint),
            "teacher": str(args.teacher_checkpoint),
            "progressive": str(args.progressive_checkpoint),
            "consistency": str(args.consistency_checkpoint),
            "flowcast": str(args.flowcast_checkpoint),
        },
        "e1_native_nfe": {label_for[t]: nfe_for[label_for[t]]
                          for t in label_for},
        "e1_rmse_vs_truth": {lab: rmse[lab].tolist() for lab in preds},
        "e1_logpsd_l1": {lab: psd_l1[lab].tolist() for lab in preds},
        "e1_acc": {lab: acc[lab].tolist() for lab in preds},
        "e1_csi_qpepre": {
            lab: {str(t): cat[lab][t]["csi"] for t in PRECIP_THRESHOLDS}
            for lab in preds
        },
        "e6_timing_ms_mean": {lab: float(np.mean(ms))
                              for lab, ms in timings.items()},
    }
    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[summary] {args.output_dir / 'summary.json'}")

    print(f"\n✓ all outputs → {args.output_dir}")


if __name__ == "__main__":
    main()
