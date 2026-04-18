"""Shared helpers for the progressive-distillation experiment scripts.

Purpose
-------
These scripts produce the quantitative and qualitative results consumed by
Chapter 4 (Experiments and Results) of the thesis. They mirror the evaluation
protocol defined in ``NTU-Thesis-LaTeX-Template/contents/chapter04.tex`` —
deterministic and precipitation-specific skill, spectral diagnostics,
autoregressive rollout, qualitative case studies, and failure modes — for the
progressive-distillation students trained under
``zettabyte_results/progressive/progressive_ncdr/run_0/phase_{0,1,2}``.

The structure follows ``analyse_tools/analyse_progressive.py``: an in-code
``OmegaConf`` dataset config avoids the stale NCDR-cluster paths shipped in
Hydra and points at the local 2022 validation zarr.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
STORMCAST_ROOT = REPO_ROOT / "stormcast"
if str(STORMCAST_ROOT) not in sys.path:
    sys.path.insert(0, str(STORMCAST_ROOT))

from physicsnemo.distributed import DistributedManager  # noqa: E402
from physicsnemo.models import Module  # noqa: E402

from datasets import dataset_classes  # noqa: E402
from utils.nn import (  # noqa: E402
    build_network_condition_and_target,
    diffusion_model_forward,
    progressive_distilled_forward,
)

# ---------------------------------------------------------------------------
# Local defaults (CLAUDE.md §1, analyse_progressive.py)
# ---------------------------------------------------------------------------
DEFAULT_DATA_ROOT = REPO_ROOT / "exp_3_train_2_5_yrs_val_1yr_tp1" / "zarr_exp3_L_24_H_24_train_2_5_years_full"
DEFAULT_REGRESSION = REPO_ROOT / "exp_3_train_2_5_yrs_val_1yr_tp1" / "exp_3_reg_L_24_H_4_train_2_5_years" / "0" / "checkpoints_regression" / "StormCastUNet.0.14000.mdlus"
DEFAULT_TEACHER = REPO_ROOT / "exp_3_train_2_5_yrs_val_1yr_tp1" / "exp_3_dif_L_24_H_4_train_2_5_years" / "0" / "checkpoints_diffusion" / "EDMPrecond.0.70000.mdlus"
DEFAULT_PHASES = REPO_ROOT / "zettabyte_results" / "progressive" / "progressive_ncdr" / "run_0"
DEFAULT_RESULTS = Path(__file__).resolve().parent / "results"

DIFFUSION_CONDITIONS = ["state", "regression", "invariant"]
REGRESSION_CONDITIONS = ["state", "background", "invariant"]

# Teacher EDM hyperparameters — must match training exactly (CLAUDE.md §7).
SIGMA_MIN = 0.002
SIGMA_MAX = 80.0
RHO = 7.0
TEACHER_STEPS = 18                  # matches train_progressive.sh initial_num_steps
TARGET_STEPS = 1                    # final PD target

# Channel ordering in the 4-channel RWRF target zarr.
CHANNELS = ["t2m", "u10", "v10", "qpepre"]
CHANNEL_UNITS = {"t2m": "K", "u10": "m s$^{-1}$", "v10": "m s$^{-1}$", "qpepre": "mm h$^{-1}$"}
CHANNEL_CMAP = {"t2m": "RdBu_r", "u10": "RdBu_r", "v10": "RdBu_r", "qpepre": "Blues"}

# Precipitation thresholds (Chapter 4, sec. exp-metrics).
PRECIP_THRESHOLDS = (0.1, 1.0, 5.0, 10.0, 20.0)
FSS_NEIGHBOURHOODS = (5, 11, 21)


# ---------------------------------------------------------------------------
# Matplotlib style — elegant and formal, thesis-ready
# ---------------------------------------------------------------------------
def set_thesis_style() -> None:
    """Apply a uniform, serif, publication-quality matplotlib style."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "STIXGeneral", "Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "regular",
        "axes.labelsize": 10,
        "axes.labelweight": "regular",
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#222222",
        "axes.grid": True,
        "grid.linewidth": 0.4,
        "grid.color": "#B8B8B8",
        "grid.alpha": 0.55,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "legend.borderaxespad": 0.4,
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "lines.linewidth": 1.3,
        "lines.markersize": 4.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# A restrained, ordered palette used throughout the scripts. The first entry
# is the teacher baseline; the next three are the PD phases from coarsest to
# finest step count.
PALETTE = {
    "teacher":  "#111111",
    "phase_0":  "#0B5394",   # deep blue
    "phase_1":  "#6FA8DC",   # light blue
    "phase_2":  "#CC0000",   # red highlight for the one-step student
    "truth":    "#2E7D32",   # sober green for the reference field
}
MARKERS = {"teacher": "s", "phase_0": "o", "phase_1": "^", "phase_2": "D"}


def label_colour(label: str) -> str:
    for key, col in PALETTE.items():
        if label.startswith(key):
            return col
    # Deterministic fallback based on label hash.
    cmap = plt.get_cmap("Greys")
    return cmap(0.5)


def label_marker(label: str) -> str:
    for key, mk in MARKERS.items():
        if label.startswith(key):
            return mk
    return "o"


def save_figure(fig, out_dir: Path, stem: str) -> None:
    """Persist a figure as PDF (for LaTeX) + PNG (for quick review)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------
def make_dataset_cfg(data_location: Path,
                     valid_dates: Tuple[str, str] = ("2022/01/01", "2022/12/31")) -> OmegaConf:
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
        "diffusion_channels": CHANNELS,
        "kept_LowRes_channels": "all",
        "kept_HighRes_channels": "all",
    })


def load_dataset(data_location: Path = DEFAULT_DATA_ROOT,
                 valid_dates: Tuple[str, str] = ("2022/01/01", "2022/12/31")):
    cfg = make_dataset_cfg(data_location, valid_dates)
    dataset_cls = dataset_classes["data_loader_rwrf_era5_stable.Dataset"]
    dataset = dataset_cls(cfg, train=False)
    return dataset


# ---------------------------------------------------------------------------
# Model plan
# ---------------------------------------------------------------------------
@dataclass
class ModelEntry:
    label: str
    ckpt: Path
    mode: str          # "teacher" or "student"
    num_steps: int     # sampling NFE


def phase_num_steps(initial: int, target: int, phase: int) -> int:
    return max(initial // (2 ** (phase + 1)), target)


def discover_phases(phases_dir: Path) -> list[int]:
    phases = []
    for d in sorted(phases_dir.glob("phase_*")):
        if (d / "student_final.mdlus").exists():
            try:
                phases.append(int(d.name.split("_")[-1]))
            except ValueError:
                pass
    return sorted(phases)


def build_model_plan(phases_dir: Path, *,
                     teacher_ckpt: Path = DEFAULT_TEACHER,
                     initial_steps: int = TEACHER_STEPS,
                     target_steps: int = TARGET_STEPS,
                     teacher_steps: Optional[int] = None,
                     skip_teacher: bool = False) -> list[ModelEntry]:
    plan: list[ModelEntry] = []
    if not skip_teacher:
        plan.append(ModelEntry(
            label="teacher",
            ckpt=teacher_ckpt,
            mode="teacher",
            num_steps=teacher_steps or initial_steps,
        ))
    for p in discover_phases(phases_dir):
        n_k = phase_num_steps(initial_steps, target_steps, p)
        plan.append(ModelEntry(
            label=f"phase_{p}_N{n_k}",
            ckpt=phases_dir / f"phase_{p}" / "student_final.mdlus",
            mode="student",
            num_steps=n_k,
        ))
    return plan


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------
def run_model(model: Module,
              background: torch.Tensor,
              state_in: torch.Tensor,
              invariant_tensor: Optional[torch.Tensor],
              regression_model: Optional[Module],
              *,
              mode: str,
              num_steps: int,
              sigma_min: float = SIGMA_MIN,
              sigma_max: float = SIGMA_MAX,
              rho: float = RHO) -> torch.Tensor:
    """Regression + diffusion/progressive-sampler single-step forward."""
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
                num_steps=num_steps,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                rho=rho,
                solver="heun",
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


def load_model(ckpt: Path, device) -> Module:
    return Module.from_checkpoint(str(ckpt)).to(device).eval()


def init_device():
    DistributedManager.initialize()
    dist = DistributedManager()
    torch.cuda.empty_cache()
    return dist.device


def prepare_sample_inputs(dataset, indices: list[int], device):
    """Move a list of validation samples onto ``device`` and return
    ``(background, state_in, state_target_denorm, timestamp)`` tuples plus the
    ground-truth array stacked on CPU."""
    sample_inputs = []
    truths = []
    timestamps = []
    for idx in indices:
        batch = dataset[idx]
        st_in = batch["state"][0].to(device=device, dtype=torch.float32).unsqueeze(0)
        st_tar = batch["state"][1].to(device=device, dtype=torch.float32).unsqueeze(0)
        bg = batch["background"].to(device=device, dtype=torch.float32).unsqueeze(0)
        sample_inputs.append((bg, st_in, st_tar))
        truths.append(dataset.denormalize_state(st_tar.cpu().numpy()[0].copy()))
        timestamps.append(dataset.valid_samples[idx])
    return sample_inputs, np.stack(truths, axis=0), timestamps


def load_invariants(dataset, device):
    arr = dataset.get_invariants()
    if arr is None:
        return None
    return torch.from_numpy(arr).to(device=device, dtype=torch.float32).unsqueeze(0)


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------
def per_channel_rmse(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((pred - ref) ** 2, axis=(-2, -1)))


def per_channel_mae(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(pred - ref), axis=(-2, -1))


def contingency(pred: np.ndarray, truth: np.ndarray, tau: float) -> dict:
    """Sum hits / misses / false alarms over all samples and pixels."""
    fc = pred >= tau
    ob = truth >= tau
    hits = int(np.sum(fc & ob))
    misses = int(np.sum(~fc & ob))
    false_alarms = int(np.sum(fc & ~ob))
    correct_negs = int(np.sum(~fc & ~ob))
    return {"H": hits, "M": misses, "F": false_alarms, "CN": correct_negs}


def csi_from_contingency(c: dict) -> float:
    denom = c["H"] + c["M"] + c["F"]
    return float(c["H"] / denom) if denom else float("nan")


def fb_from_contingency(c: dict) -> float:
    denom = c["H"] + c["M"]
    return float((c["H"] + c["F"]) / denom) if denom else float("nan")


def fss_batch(pred: np.ndarray, truth: np.ndarray, tau: float, n: int) -> float:
    """Fractions Skill Score averaged over samples.

    Implements Roberts & Lean (2008): convolve the binary exceedance fields
    with an (n×n) box filter and compute the standard numerator / denominator.
    ``pred``/``truth`` are stacks of ``qpepre`` fields, shape (S, H, W).
    """
    from scipy.signal import fftconvolve
    S = pred.shape[0]
    kernel = np.ones((n, n), dtype=np.float64) / (n * n)
    num = 0.0
    den = 0.0
    for s in range(S):
        pf = (pred[s] >= tau).astype(np.float64)
        po = (truth[s] >= tau).astype(np.float64)
        pf_smoothed = fftconvolve(pf, kernel, mode="same")
        po_smoothed = fftconvolve(po, kernel, mode="same")
        num += float(np.sum((pf_smoothed - po_smoothed) ** 2))
        den += float(np.sum(pf_smoothed ** 2) + np.sum(po_smoothed ** 2))
    if den == 0.0:
        return float("nan")
    return float(1.0 - num / den)


def radial_log_psd(field: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-channel radially averaged power spectrum.

    Args:
        field: (C, H, W) single-sample prediction.
    Returns:
        (k, P(k)) with k of length min(H, W) // 2 and P(k) of shape (C, len(k)).
    """
    C, H, W = field.shape
    kmax = min(H, W) // 2
    fy = np.fft.fftfreq(H) * H
    fx = np.fft.fftfreq(W) * W
    kk = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    kr = np.round(kk).astype(int)
    mask = kr <= kmax
    out = np.zeros((C, kmax + 1), dtype=np.float64)
    for c in range(C):
        F = np.fft.fft2(field[c])
        P = (F * np.conj(F)).real
        sums = np.bincount(kr[mask], weights=P[mask], minlength=kmax + 1)
        counts = np.bincount(kr[mask], minlength=kmax + 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[c] = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    k = np.arange(kmax + 1)
    return k, out


def mean_radial_log_psd(stack: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sample-averaged radial PSD over a (S, C, H, W) stack."""
    S = stack.shape[0]
    acc = None
    k = None
    for s in range(S):
        k, pk = radial_log_psd(stack[s])
        if acc is None:
            acc = np.zeros_like(pk)
        acc += pk
    return k, acc / S


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------
def choose_sample_indices(dataset, n_samples: int, seed: int = 0,
                          stride: Optional[int] = None) -> list[int]:
    """Return indices drawn from the validation split.

    If ``stride`` is given, returns every ``stride``-th sample instead of a
    random draw — this gives smoother temporal coverage for rollout and
    failure-mode diagnostics at a controlled budget.
    """
    N = len(dataset)
    if stride is not None and stride > 0:
        return list(range(0, N, stride))[:n_samples]
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(N, size=min(n_samples, N), replace=False).tolist())


# ---------------------------------------------------------------------------
# Banner for script logs
# ---------------------------------------------------------------------------
def banner(msg: str) -> None:
    bar = "=" * max(20, len(msg) + 4)
    print(f"\n{bar}\n  {msg}\n{bar}")


def timeit(label: str):
    class _Ctx:
        def __enter__(self_inner):
            self_inner.t0 = time.perf_counter()
            return self_inner
        def __exit__(self_inner, *a):
            print(f"[{label}] {time.perf_counter() - self_inner.t0:.1f} s")
    return _Ctx()
