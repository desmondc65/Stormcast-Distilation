"""Shared evaluation helpers for ablation analysis scripts.

Single-step (1h) aggregated metrics + autoregressive case studies for the
Stormcast/FlowCast/EDM students. The two CLI entry points
(`log1p_ablation.py`, `qpw_ablation.py`) compose these helpers.

Local dataset paths live under the project's ``exp_3_train_2_5_yrs_val_1yr_tp1``
directory. Models were trained on 192x96 crops; we read that from the run's
hydra overrides where possible, and otherwise fall back to (192, 96).
"""

from __future__ import annotations

import os
import sys
import csv
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterable, Sequence

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

# --- Make stormcast/ importable. -----------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STORMCAST_DIR = os.path.join(PROJECT_ROOT, "stormcast")
if STORMCAST_DIR not in sys.path:
    sys.path.insert(0, STORMCAST_DIR)

from physicsnemo.distributed import DistributedManager  # noqa: E402
from physicsnemo.models import Module  # noqa: E402

from datasets import dataset_classes  # noqa: E402
from utils.nn import (  # noqa: E402
    build_network_condition_and_target,
    diffusion_model_forward,
    flowcast_model_forward,
    get_preconditioned_architecture,
)
from utils.spectrum import ps1d_plots  # noqa: E402


# --- Local dataset locations. --------------------------------------------------
DATA_LOG1P = os.path.join(
    PROJECT_ROOT,
    "exp_3_train_2_5_yrs_val_1yr_tp1",
    "zarr_exp3_L_24_H_24_train_2_5_years_full_cleaned_4_27_2026",
)
DATA_RAW = os.path.join(
    PROJECT_ROOT,
    "exp_3_train_2_5_yrs_val_1yr_tp1",
    "zarr_exp3_L_24_H_24_train_2_5_years_full_cleaned_4_27_2026_raw",
)

# CSI thresholds (mm/h) for qpepre.
QPEPRE_THRESHOLDS = (0.1, 1.0, 5.0, 10.0, 20.0)

# Default case-study initial times (UTC). Picked across the 2022 valid year
# to give one summer convective day, one frontal/typhoon-season day and one
# winter NE-monsoon day.
DEFAULT_CASE_STUDIES = (
    "2022-07-15T00:00:00",
    "2022-09-12T00:00:00",
    "2022-12-08T00:00:00",
)


# ------------------------------------------------------------------------------
# Dataset / model loaders
# ------------------------------------------------------------------------------
def make_dataset_cfg(
    location: str,
    *,
    qpepre_log1p: bool,
    img_size=(192, 96),
    valid_dates=("2022/01/01", "2022/12/31"),
):
    """Build the Hydra-style cfg snippet that ``Dataset(params, train=False)`` expects."""
    return OmegaConf.create(
        {
            "name": "data_loader_rwrf_era5_stable.Dataset",
            "location": location,
            "HighRes_img_size": list(img_size),
            "dt": 1,
            "exp_train_zarrs": ["stormcast_test_train"],
            "train_dates": ["2019/08/01", "2021/12/31"],
            "exp_valid_zarrs": ["stormcast_test_valid"],
            "valid_dates": list(valid_dates),
            "invariants": ["lsm", "orog"],
            "input_channels": "all",
            "diffusion_channels": ["t2m", "u10", "v10", "qpepre"],
            "kept_LowRes_channels": "all",
            "kept_HighRes_channels": ["u10", "v10", "t2m", "qpepre"],
            "qpepre_log1p": bool(qpepre_log1p),
        }
    )


def open_validation_dataset(location: str, qpepre_log1p: bool, img_size=(192, 96)):
    DistributedManager.initialize()
    cfg = make_dataset_cfg(location, qpepre_log1p=qpepre_log1p, img_size=img_size)
    cls = dataset_classes[cfg.name]
    return cls(cfg, train=False)


def load_regression(checkpoint_path: str, device) -> Module:
    net = Module.from_checkpoint(checkpoint_path).to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def load_diffusion(checkpoint_path: str, device) -> Module:
    net = Module.from_checkpoint(checkpoint_path).to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def load_flowcast(
    *,
    img_resolution: tuple[int, int],
    target_channels: int,
    conditional_channels: int,
    spatial_pos_embed: bool,
    attn_resolutions: Sequence[int],
    sigma_data: float,
    time_scale: float,
    ema_path: str,
    device,
) -> Module:
    """Construct a fresh FlowCastPrecond and stamp in the EMA shadow weights."""
    net = get_preconditioned_architecture(
        name="flowcast",
        img_resolution=tuple(img_resolution),
        target_channels=int(target_channels),
        conditional_channels=int(conditional_channels),
        spatial_embedding=bool(spatial_pos_embed),
        attn_resolutions=list(attn_resolutions),
    )
    net = net.to(device).eval().requires_grad_(False)
    net.sigma_data = float(sigma_data)
    net.time_scale = float(time_scale)

    ema_state = torch.load(ema_path, map_location=device)
    if "shadow_params" in ema_state and "param_names" in ema_state:
        names = ema_state["param_names"]
        shadows = ema_state["shadow_params"]
        target_state = dict(net.state_dict())
        for i, name in enumerate(names):
            if name in target_state:
                target_state[name] = shadows[i].to(target_state[name].dtype)
        net.load_state_dict(target_state, strict=False)
    else:
        net.load_state_dict(ema_state, strict=False)
    return net


# ------------------------------------------------------------------------------
# Sampler wrappers
# ------------------------------------------------------------------------------
@dataclass
class StudentSpec:
    """Generic spec for whichever student is being evaluated."""

    name: str
    kind: str  # 'edm' or 'flowcast' or 'regression'
    model: Module | None  # None for kind=='regression'
    sampler_kwargs: dict = field(default_factory=dict)


def sample_residual(spec: StudentSpec, condition: torch.Tensor, target_shape):
    """Run one student forward to produce the residual R_t (raw units)."""
    if spec.kind == "regression":
        return torch.zeros(target_shape, device=condition.device, dtype=condition.dtype)
    if spec.kind == "edm":
        return diffusion_model_forward(
            spec.model,
            condition,
            target_shape,
            sampler_args=spec.sampler_kwargs,
        )
    if spec.kind == "flowcast":
        return flowcast_model_forward(
            spec.model,
            condition,
            target_shape,
            **spec.sampler_kwargs,
        )
    raise ValueError(f"Unknown student kind: {spec.kind!r}")


# ------------------------------------------------------------------------------
# Metric accumulators
# ------------------------------------------------------------------------------
@dataclass
class MetricAccumulator:
    state_channels: tuple[str, ...]
    qpepre_idx: int | None
    sum_sq: np.ndarray  # (C,)
    sum_abs: np.ndarray  # (C,)
    n_pix: int = 0

    # qpepre CSI tables: hits / false-alarms / misses per threshold
    csi_hits: dict[float, int] = field(default_factory=dict)
    csi_fa: dict[float, int] = field(default_factory=dict)
    csi_miss: dict[float, int] = field(default_factory=dict)
    csi_correct_neg: dict[float, int] = field(default_factory=dict)

    # Wet-pixel stats and tail quantiles for qpepre
    wet_frac_pred_sum: float = 0.0
    wet_frac_true_sum: float = 0.0
    n_samples: int = 0
    qpepre_pred_p99: list = field(default_factory=list)
    qpepre_true_p99: list = field(default_factory=list)

    # PS1D accumulators (per channel)
    ps_k: np.ndarray | None = None
    ps_pred: np.ndarray | None = None  # (C, L)
    ps_true: np.ndarray | None = None  # (C, L)
    ps_count: int = 0

    @classmethod
    def new(cls, state_channels):
        chans = tuple(state_channels)
        try:
            qpepre_idx = chans.index("qpepre")
        except ValueError:
            qpepre_idx = None
        C = len(chans)
        acc = cls(
            state_channels=chans,
            qpepre_idx=qpepre_idx,
            sum_sq=np.zeros(C),
            sum_abs=np.zeros(C),
        )
        for thr in QPEPRE_THRESHOLDS:
            acc.csi_hits[thr] = 0
            acc.csi_fa[thr] = 0
            acc.csi_miss[thr] = 0
            acc.csi_correct_neg[thr] = 0
        return acc

    def update(self, pred_real: np.ndarray, true_real: np.ndarray):
        """Update with one (C, H, W) sample in real units (mm/h, K, m/s)."""
        assert pred_real.shape == true_real.shape
        diff = pred_real - true_real
        self.sum_sq += (diff ** 2).reshape(diff.shape[0], -1).sum(axis=1)
        self.sum_abs += np.abs(diff).reshape(diff.shape[0], -1).sum(axis=1)
        self.n_pix += diff[0].size
        self.n_samples += 1

        if self.qpepre_idx is not None:
            qp_pred = pred_real[self.qpepre_idx]
            qp_true = true_real[self.qpepre_idx]
            self.wet_frac_pred_sum += float((qp_pred > 0.1).mean())
            self.wet_frac_true_sum += float((qp_true > 0.1).mean())
            self.qpepre_pred_p99.append(float(np.quantile(qp_pred, 0.99)))
            self.qpepre_true_p99.append(float(np.quantile(qp_true, 0.99)))
            for thr in QPEPRE_THRESHOLDS:
                p = qp_pred >= thr
                t = qp_true >= thr
                self.csi_hits[thr] += int(np.logical_and(p, t).sum())
                self.csi_fa[thr] += int(np.logical_and(p, ~t).sum())
                self.csi_miss[thr] += int(np.logical_and(~p, t).sum())
                self.csi_correct_neg[thr] += int(np.logical_and(~p, ~t).sum())

    def update_spectrum(self, pred_real: np.ndarray, true_real: np.ndarray):
        """Update PSD accumulator. pred/true are (C,H,W) in real units."""
        from utils.spectrum import ps1d_plots as _ps1d  # local import

        figs, _, numeric = _ps1d(
            torch.as_tensor(pred_real), torch.as_tensor(true_real),
            list(self.state_channels), list(self.state_channels),
        )
        for fig in figs.values():
            plt.close(fig)
        if self.ps_k is None:
            self.ps_k = numeric["k"].copy()
            self.ps_pred = numeric["Pk_gen"].copy()
            self.ps_true = numeric["Pk_tar"].copy()
        else:
            self.ps_pred = self.ps_pred + numeric["Pk_gen"]
            self.ps_true = self.ps_true + numeric["Pk_tar"]
        self.ps_count += 1

    # ------------------ summarisation ------------------
    def summary(self) -> dict:
        n = max(self.n_pix, 1)
        rmse = np.sqrt(self.sum_sq / n)
        mae = self.sum_abs / n
        out = {
            "n_samples": self.n_samples,
            "rmse": {c: float(rmse[i]) for i, c in enumerate(self.state_channels)},
            "mae": {c: float(mae[i]) for i, c in enumerate(self.state_channels)},
        }
        if self.qpepre_idx is not None and self.n_samples > 0:
            csi = {}
            pod = {}
            far = {}
            bias = {}
            for thr in QPEPRE_THRESHOLDS:
                h = self.csi_hits[thr]
                f = self.csi_fa[thr]
                m = self.csi_miss[thr]
                csi[thr] = h / max(h + f + m, 1)
                pod[thr] = h / max(h + m, 1)
                far[thr] = f / max(h + f, 1)
                bias[thr] = (h + f) / max(h + m, 1)
            out["qpepre_csi"] = csi
            out["qpepre_pod"] = pod
            out["qpepre_far"] = far
            out["qpepre_bias"] = bias
            out["qpepre_wet_frac_pred"] = self.wet_frac_pred_sum / self.n_samples
            out["qpepre_wet_frac_true"] = self.wet_frac_true_sum / self.n_samples
            out["qpepre_p99_pred_mean"] = float(np.mean(self.qpepre_pred_p99))
            out["qpepre_p99_true_mean"] = float(np.mean(self.qpepre_true_p99))
        if self.ps_count > 0:
            out["ps_k"] = self.ps_k
            out["ps_pred"] = self.ps_pred / self.ps_count
            out["ps_true"] = self.ps_true / self.ps_count
        return out


# ------------------------------------------------------------------------------
# Single-step evaluation loop
# ------------------------------------------------------------------------------
def evaluate_single_step(
    *,
    spec: StudentSpec,
    regression_model: Module | None,
    dataset,
    device,
    diffusion_conditions=("state", "regression", "invariant"),
    regression_conditions=("state", "background", "invariant"),
    max_samples: int | None = None,
    spectrum_stride: int = 24,
    progress_label: str = "",
) -> dict:
    """Aggregated 1-step (1h) metrics over the validation set.

    For sample i we feed real X_{t-1} (input) and predict X_t. Comparison is
    done in real (de-normalised) units. Spectra are accumulated every
    ``spectrum_stride`` samples to keep the cost reasonable.
    """
    state_channels = dataset.state_channels()
    invariant_array = dataset.get_invariants()
    invariant_tensor = (
        torch.from_numpy(invariant_array).to(device).repeat(1, 1, 1, 1)
        if invariant_array is not None
        else None
    )

    n_total = len(dataset)
    if max_samples is not None:
        n_total = min(n_total, int(max_samples))

    acc = MetricAccumulator.new(state_channels)

    log_every = max(n_total // 20, 1)
    for i in range(n_total):
        data = dataset[i]
        background = data["background"].to(device=device, dtype=torch.float32).unsqueeze(0)
        x_prev = data["state"][0].to(device=device, dtype=torch.float32).unsqueeze(0)

        condition, _, reg_out = build_network_condition_and_target(
            background,
            [x_prev, x_prev],
            invariant_tensor,
            regression_net=regression_model,
            condition_list=list(diffusion_conditions),
            regression_condition_list=list(regression_conditions),
        )
        residual = sample_residual(spec, condition, x_prev.shape)
        if reg_out is None:
            x_pred = residual
        else:
            x_pred = reg_out + residual

        pred_real = dataset.denormalize_state(x_pred[0].detach().cpu().numpy().copy())
        true_real = dataset.denormalize_state(data["state"][1].detach().cpu().numpy().copy())
        acc.update(pred_real, true_real)
        if (i % spectrum_stride) == 0:
            acc.update_spectrum(pred_real, true_real)

        if (i % log_every) == 0:
            print(f"  [{progress_label}] step {i}/{n_total}", flush=True)

    return acc.summary()


# ------------------------------------------------------------------------------
# Autoregressive rollout case study
# ------------------------------------------------------------------------------
def autoregressive_rollout(
    *,
    spec: StudentSpec,
    regression_model: Module | None,
    dataset,
    initial_time: str,
    n_steps: int,
    device,
    diffusion_conditions=("state", "regression", "invariant"),
    regression_conditions=("state", "background", "invariant"),
) -> dict:
    """Roll out from ``initial_time`` for ``n_steps`` hours.

    Returns dict with arrays of (n_steps, C, H, W) for prediction and target
    (in real units), plus per-step per-channel RMSE/MAE.
    """
    state_channels = dataset.state_channels()
    invariant_array = dataset.get_invariants()
    invariant_tensor = (
        torch.from_numpy(invariant_array).to(device).repeat(1, 1, 1, 1)
        if invariant_array is not None
        else None
    )

    init_dt = datetime.fromisoformat(initial_time)
    if init_dt not in dataset.valid_samples:
        # find the next available timestamp
        for ts in dataset.valid_samples:
            if ts >= init_dt:
                init_dt = ts
                break
    start_idx = dataset.valid_samples.index(init_dt)
    if start_idx + n_steps > len(dataset):
        n_steps = len(dataset) - start_idx
    print(
        f"  [rollout] init={init_dt} (idx={start_idx}) n_steps={n_steps}",
        flush=True,
    )

    preds_real = []
    truths_real = []
    rmse_per_step = []
    mae_per_step = []

    state_pred = None
    for step in range(n_steps):
        data = dataset[start_idx + step]
        background = data["background"].to(device=device, dtype=torch.float32).unsqueeze(0)
        if step == 0:
            state_pred = data["state"][0].to(device=device, dtype=torch.float32).unsqueeze(0)

        condition, _, reg_out = build_network_condition_and_target(
            background,
            [state_pred, state_pred],
            invariant_tensor,
            regression_net=regression_model,
            condition_list=list(diffusion_conditions),
            regression_condition_list=list(regression_conditions),
        )
        residual = sample_residual(spec, condition, state_pred.shape)
        if reg_out is None:
            new_state = residual
        else:
            new_state = reg_out + residual

        pred_real = dataset.denormalize_state(new_state[0].detach().cpu().numpy().copy())
        true_real = dataset.denormalize_state(data["state"][1].detach().cpu().numpy().copy())
        diff = pred_real - true_real
        rmse_per_step.append(np.sqrt((diff ** 2).reshape(diff.shape[0], -1).mean(axis=1)))
        mae_per_step.append(np.abs(diff).reshape(diff.shape[0], -1).mean(axis=1))
        preds_real.append(pred_real)
        truths_real.append(true_real)
        state_pred = new_state

    return {
        "initial_time": init_dt.isoformat(),
        "state_channels": list(state_channels),
        "preds": np.stack(preds_real, axis=0),
        "truths": np.stack(truths_real, axis=0),
        "rmse_per_step": np.stack(rmse_per_step, axis=0),  # (T, C)
        "mae_per_step": np.stack(mae_per_step, axis=0),
    }


# ------------------------------------------------------------------------------
# Output helpers
# ------------------------------------------------------------------------------
def write_summary_csv(path: str, rows: list[dict], fieldnames: Sequence[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_psd_comparison(out_path: str, results: dict[str, dict], channel: str):
    """One PSD figure per channel, comparing all runs."""
    fig, (ax0, ax1) = plt.subplots(
        2, 1, gridspec_kw={"height_ratios": [2, 1], "hspace": 0}, figsize=(7, 5)
    )
    target_drawn = False
    for name, summary in results.items():
        if "ps_k" not in summary:
            continue
        chans = list(summary["state_channels"]) if "state_channels" in summary else None
        # state_channels are added separately by the orchestrator.
        chans = chans or ("u10", "v10", "t2m", "qpepre")
        try:
            cidx = chans.index(channel)
        except ValueError:
            continue
        k = summary["ps_k"]
        if not target_drawn:
            ax0.plot(k, summary["ps_true"][cidx], "k-", lw=1.5, label=f"truth ({channel})")
            target_drawn = True
        line, = ax0.plot(k, summary["ps_pred"][cidx], lw=1.0, label=name)
        ratio = np.divide(
            summary["ps_pred"][cidx], summary["ps_true"][cidx],
            out=np.zeros_like(summary["ps_pred"][cidx]),
            where=summary["ps_true"][cidx] != 0,
        )
        ax1.plot(k, ratio, lw=1.0, color=line.get_color())
    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.set_ylabel("PSD")
    ax0.legend(loc="best", fontsize=8)
    ax0.tick_params(axis="x", labelbottom=False)
    ax1.plot([k[0], k[-1]], [1.0, 1.0], "k--", lw=0.8)
    ax1.set_xscale("log")
    ax1.set_xlabel("Wavenumber")
    ax1.set_ylabel("ratio")
    ax1.set_ylim((0, 2))
    ax0.set_title(f"PS1D — {channel}")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_rollout_rmse(out_path: str, rollouts: dict[str, dict], channel: str):
    """Per-channel RMSE vs lead time across runs (averaged over case studies)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    drawn_any = False
    for name, cases in rollouts.items():
        if not cases:
            continue
        chans = cases[0]["state_channels"]
        try:
            cidx = chans.index(channel)
        except ValueError:
            continue
        # average across cases
        per_step = np.stack([c["rmse_per_step"][:, cidx] for c in cases], axis=0)
        mean = per_step.mean(axis=0)
        ax.plot(np.arange(1, mean.shape[0] + 1), mean, marker="o", label=name)
        drawn_any = True
    if not drawn_any:
        plt.close(fig)
        return
    ax.set_xlabel("Lead time (h)")
    ax.set_ylabel(f"RMSE ({channel})")
    ax.set_title(f"Rollout RMSE — {channel}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_rollout_qpepre_panel(out_path: str, rollouts: dict[str, list], case_idx: int = 0):
    """Side-by-side qpepre snapshots at +1/+6/+12h for each run vs truth."""
    if not rollouts:
        return
    runs = list(rollouts.keys())
    # use the first run to find chans / horizons
    first = rollouts[runs[0]]
    if not first or len(first) <= case_idx:
        return
    sample = first[case_idx]
    chans = sample["state_channels"]
    if "qpepre" not in chans:
        return
    qidx = chans.index("qpepre")
    T = sample["preds"].shape[0]
    horizons = sorted(set([min(0, T - 1), min(5, T - 1), min(11, T - 1)]))
    horizons = sorted(set(horizons))

    n_cols = 1 + len(runs)  # truth + each run
    n_rows = len(horizons)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2.6 * n_cols, 2.6 * n_rows), squeeze=False
    )
    vmax = max(
        sample["truths"][horizons[-1], qidx].max(),
        max(rollouts[r][case_idx]["preds"][horizons[-1], qidx].max() for r in runs),
        0.1,
    )
    for ri, h in enumerate(horizons):
        truth = rollouts[runs[0]][case_idx]["truths"][h, qidx]
        ax = axes[ri, 0]
        im = ax.imshow(truth, origin="lower", cmap="Blues", vmin=0, vmax=vmax)
        ax.set_ylabel(f"+{h+1}h")
        if ri == 0:
            ax.set_title("truth")
        ax.set_xticks([]); ax.set_yticks([])
        for ci, run in enumerate(runs):
            pred = rollouts[run][case_idx]["preds"][h, qidx]
            ax = axes[ri, 1 + ci]
            ax.imshow(pred, origin="lower", cmap="Blues", vmin=0, vmax=vmax)
            if ri == 0:
                ax.set_title(run, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(
        f"qpepre rollout — init {sample['initial_time']}", fontsize=10
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def write_markdown_summary(path: str, title: str, results: dict[str, dict]):
    """Write a human-readable summary across runs."""
    state_channels = ("u10", "v10", "t2m", "qpepre")
    lines = [f"# {title}", ""]
    lines.append("## Per-channel single-step (1h) metrics over 2022")
    lines.append("")
    header = ["run", "n_samples"]
    for c in state_channels:
        header += [f"rmse_{c}", f"mae_{c}"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for name, summ in results.items():
        row = [name, str(summ.get("n_samples", "-"))]
        for c in state_channels:
            row.append(f"{summ['rmse'].get(c, float('nan')):.4f}")
            row.append(f"{summ['mae'].get(c, float('nan')):.4f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## qpepre CSI (1h, full year)")
    lines.append("")
    csi_hdr = ["run"] + [f"CSI@{thr}" for thr in QPEPRE_THRESHOLDS]
    lines.append("| " + " | ".join(csi_hdr) + " |")
    lines.append("|" + "|".join(["---"] * len(csi_hdr)) + "|")
    for name, summ in results.items():
        if "qpepre_csi" not in summ:
            continue
        row = [name] + [f"{summ['qpepre_csi'][thr]:.3f}" for thr in QPEPRE_THRESHOLDS]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## qpepre wet-fraction & p99")
    lines.append("")
    lines.append("| run | wet_frac_pred | wet_frac_true | p99_pred (mm/h) | p99_true (mm/h) |")
    lines.append("|---|---|---|---|---|")
    for name, summ in results.items():
        if "qpepre_wet_frac_pred" not in summ:
            continue
        lines.append(
            f"| {name} | {summ['qpepre_wet_frac_pred']:.4f} | {summ['qpepre_wet_frac_true']:.4f} | "
            f"{summ['qpepre_p99_pred_mean']:.3f} | {summ['qpepre_p99_true_mean']:.3f} |"
        )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
