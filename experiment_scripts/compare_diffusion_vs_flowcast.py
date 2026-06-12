#!/usr/bin/env python3
"""Side-by-side comparison of the EDM diffusion teacher and the FlowCast student.

Implements ``experiment_scripts/analysis_plan.md``: report
    CRPS ↓  CSI-M ↑  CSI-P16 ↑  FSS-P16-M ↑  HSS-M ↑  FAR-M ↓  Time/Seq.(s)
plus per-channel RMSE for ``[u10, v10, t2m, qpepre]`` and prediction PNG
panels per channel.

Both models share the same frozen StormCast regression mean ``M_t`` and only
differ in how the residual ``R_t`` is sampled.  The script:

  1. builds the cleaned RWRF/ERA5 validation dataset (192x96, qpepre = log1p mm/h);
  2. picks ``--n-sequences`` evenly-spaced initial times;
  3. autoregressively rolls each sequence ``--n-steps`` hours forward;
  4. repeats with ``--ensemble`` random seeds so kernel CRPS is well-defined;
  5. aggregates all metrics into ``scoreboard.{md,csv}`` and writes per-channel
     truth-vs-diffusion-vs-flowcast PNG panels.

The defaults mirror the canonical zettabyte training scripts
(``stormcast/zettabyte_scripts/train_{diffusion,flowcast,regression}.sh``)
so checkpoints, channel order, and qpepre encoding stay aligned.

Run with ``--help`` for the full option list.
"""

from __future__ import annotations

import argparse
import csv
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
from physicsnemo.metrics.general.crps import kcrps  # noqa: E402
from physicsnemo.models import Module  # noqa: E402

from datasets import dataset_classes  # noqa: E402
from utils.nn import (  # noqa: E402
    build_network_condition_and_target,
    diffusion_model_forward,
    flowcast_model_forward,
    meanflow_model_forward,
)


# --- Defaults ---------------------------------------------------------------
DEFAULT_DATA = (
    REPO_ROOT
    / "exp_3_train_2_5_yrs_val_1yr_tp1"
    / "zarr_exp3_L_24_H_24_train_2_5_years_full_cleaned_4_27_2026"
)
DEFAULT_REGRESSION = (
    REPO_ROOT
    / "runs/regression_zettabyte_v1_cleaned_4_27_2026"
    / "regression_zettabyte_cleaned_4_27_2026/run_0"
    / "checkpoints_regression/StormCastUNet.0.8000.mdlus"
)
DEFAULT_DIFFUSION = (
    REPO_ROOT
    / "runs/diffusion_zettabyte_v1_cleaned_4_27_2026"
    / "diffusion_zettabyte_cleaned_4_27_2026/run_0"
    / "checkpoints_diffusion/EDMPrecond.0.30000.mdlus"
)
DEFAULT_FLOWCAST = (
    REPO_ROOT
    / "runs/flowcast_zettabyte_v1_cleaned_4_27_2026"
    / "flowcast_zettabyte_cleaned_4_27_2026/run_0"
    / "checkpoints_flowcast/FlowCastPrecond.0.25000.mdlus"
)
# MeanFlow shares the FlowCast SongUNet backbone and conditioning bundle, so it
# is loaded and rolled out exactly like the FlowCast leg; only the sampler
# (few-step average velocity) differs. Loaded from the online-student .mdlus
# (same convention as DEFAULT_FLOWCAST above) so MeanFlow-vs-FlowCast stays an
# A/B on the objective alone.
DEFAULT_MEANFLOW = (
    REPO_ROOT
    / "runs/meanflow_zettabyte_v1_cleaned_4_27_2026"
    / "meanflow_zettabyte_cleaned_4_27_2026/run_0"
    / "checkpoints_meanflow/MeanFlowPrecond.0.20000.mdlus"
)

# Canonical training channel order (kept_HighRes_channels in the train scripts).
KEPT_HIGHRES = ["u10", "v10", "t2m", "qpepre"]

# Physical units after Dataset.denormalize_state — see
# stormcast/datasets/data_loader_rwrf_era5_stable.py:362. log1p is undone for
# qpepre when qpepre_log1p=True so this table holds for both encodings.
CHANNEL_UNITS = {
    "t2m": "K",
    "u10": "m/s",
    "v10": "m/s",
    "qpepre": "mm/h",
}
# Pretty labels (with units) for plot titles / colorbars.
CHANNEL_LABELS = {ch: f"{ch} [{u}]" for ch, u in CHANNEL_UNITS.items()}

# Conditioning bundles must match config/model/{diffusion,flowcast}.yaml.
DIFFUSION_CONDITIONS = ["state", "regression", "invariant"]
REGRESSION_CONDITIONS = ["state", "background", "invariant"]

# qpepre categorical-skill thresholds (mm/h).
PRECIP_THRESHOLDS = [0.1, 1.0, 5.0, 10.0, 16.0, 32.0]
P16_THRESHOLD = 16.0

# FSS pooling half-widths (cells). Cleaned grid spacing ~2 km, so 3/7/15 cells
# corresponds to roughly 12/30/62 km neighbourhood diameters.
FSS_WINDOWS_PIX = [3, 7, 15]


# --- Dataset wiring ---------------------------------------------------------
def make_dataset_cfg(
    data_loc: Path,
    valid_dates: tuple[str, str],
    hr_size: tuple[int, int],
    qpepre_log1p: bool,
    kept_channels: list[str],
) -> OmegaConf:
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


def build_dataset(args, device):
    cfg = make_dataset_cfg(
        args.data_location,
        tuple(args.valid_dates),
        tuple(args.hr_size),
        qpepre_log1p=args.qpepre_log1p,
        kept_channels=args.kept_channels,
    )
    dataset_cls = dataset_classes["data_loader_rwrf_era5_stable.Dataset"]
    dataset = dataset_cls(cfg, train=False)
    inv = dataset.get_invariants()
    invariant_tensor = (
        torch.from_numpy(inv).to(device=device, dtype=torch.float32).unsqueeze(0)
        if inv is not None
        else None
    )
    return dataset, invariant_tensor


# --- Metric primitives -------------------------------------------------------
def per_channel_rmse(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """RMSE over (H, W) for each leading slot. (..., C, H, W) -> (..., C)."""
    return np.sqrt(np.mean((pred - ref) ** 2, axis=(-2, -1)))


def _contingency(pred_pos: np.ndarray, ref_pos: np.ndarray):
    tp = float(np.sum(pred_pos & ref_pos))
    fp = float(np.sum(pred_pos & ~ref_pos))
    fn = float(np.sum(~pred_pos & ref_pos))
    tn = float(np.sum(~pred_pos & ~ref_pos))
    return tp, fp, fn, tn


def categorical_scores(pred: np.ndarray, ref: np.ndarray, thresholds):
    """Per-threshold CSI / FAR / POD / HSS over a stacked field."""
    out = {}
    for tau in thresholds:
        tp, fp, fn, tn = _contingency(pred >= tau, ref >= tau)
        eps = 1e-12
        n = tp + fp + fn + tn
        csi = tp / (tp + fp + fn + eps)
        far = fp / (tp + fp + eps) if (tp + fp) > 0 else float("nan")
        pod = tp / (tp + fn + eps) if (tp + fn) > 0 else float("nan")
        e = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n + eps)
        hss = (tp + tn - e) / (n - e + eps) if (n - e) > 0 else float("nan")
        out[tau] = {"csi": csi, "far": far, "pod": pod, "hss": hss}
    return out


def _box_pool(field: np.ndarray, w: int) -> np.ndarray:
    """Mean-pool the last 2 dims with a (2w+1) square box (edge padded).

    Implementation uses a summed-area table so cost is O(N) regardless of w.
    """
    if w <= 0:
        return field.copy()
    win = 2 * w + 1
    pad = w
    arr = np.pad(field, [(0, 0)] * (field.ndim - 2) + [(pad, pad), (pad, pad)], mode="edge")
    cs = np.cumsum(np.cumsum(arr, axis=-1), axis=-2)
    cs = np.pad(cs, [(0, 0)] * (cs.ndim - 2) + [(1, 0), (1, 0)], mode="constant")
    H, W = field.shape[-2], field.shape[-1]
    out = (
        cs[..., win : H + win, win : W + win]
        - cs[..., :H, win : W + win]
        - cs[..., win : H + win, :W]
        + cs[..., :H, :W]
    ) / (win * win)
    return out


def fss(pred: np.ndarray, ref: np.ndarray, tau: float, w: int) -> float:
    """Fractions Skill Score (Roberts & Lean 2008) at threshold tau, half-width w."""
    p_bin = (pred >= tau).astype(np.float32)
    r_bin = (ref >= tau).astype(np.float32)
    pf = _box_pool(p_bin, w)
    rf = _box_pool(r_bin, w)
    num = np.mean((pf - rf) ** 2)
    den = np.mean(pf * pf) + np.mean(rf * rf) + 1e-12
    return float(1.0 - num / den)


def crps_field(preds_ens: np.ndarray, truth: np.ndarray) -> float:
    """Mean kernel CRPS over the field. preds_ens (K, ...), truth (...)."""
    pred_t = torch.from_numpy(np.ascontiguousarray(preds_ens))
    obs_t = torch.from_numpy(np.ascontiguousarray(truth))
    val = kcrps(pred_t, obs_t, dim=0, biased=False)
    return float(val.mean().item())


# --- Inference / autoregressive rollout -------------------------------------
def rollout(
    *,
    model,
    method: str,
    regression,
    invariant,
    dataset,
    t0_idx: int,
    n_steps: int,
    sampler_kwargs: dict,
    device,
):
    """One autoregressive rollout. Returns (pred_seq, truth_seq) in PHYSICAL units.

    pred_seq[k] is the k-th predicted hour (X_{t0+1+k}), truth_seq[k] is the
    matching observation. The predicted state is fed back as ``state[0]`` for
    the next step exactly as in stormcast/inference{,_flowcast}.py.
    """
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


def run_method_ensemble(
    *,
    model,
    method,
    regression,
    invariant,
    dataset,
    t0_indices,
    n_steps,
    n_ensemble,
    sampler_kwargs,
    device,
    seed,
):
    """Build ``(K, S, T, C, H, W)`` ensemble + ``(S, T, C, H, W)`` truth + per-seq times.

    Emits one ``[run]`` progress line per finished (member, sequence) so the
    wrapper isn't silent for the full K*S*T*NFE work block — the first
    sequence absorbs torch.compile autotune (5-15 min on cold cache) and
    will be much slower than the rest, which is useful to see live.
    """
    n_seq = len(t0_indices)
    total = n_ensemble * n_seq
    pred_ens = []
    truth_seq = None
    times: list[float] = []
    t_method = time.perf_counter()
    print(
        f"[run] {method} ensemble starting: K={n_ensemble} x S={n_seq} x T={n_steps} "
        f"= {total} sequence-rollouts",
        flush=True,
    )
    for k in range(n_ensemble):
        torch.manual_seed(seed + 1000 * k)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed + 1000 * k)
        member_preds = []
        for s, t0 in enumerate(t0_indices):
            t_w = time.perf_counter()
            p_seq, t_seq = rollout(
                model=model,
                method=method,
                regression=regression,
                invariant=invariant,
                dataset=dataset,
                t0_idx=t0,
                n_steps=n_steps,
                sampler_kwargs=sampler_kwargs,
                device=device,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t_w
            if k == 0:
                times.append(elapsed)
                if truth_seq is None:
                    truth_seq = np.empty((n_seq,) + t_seq.shape, dtype=t_seq.dtype)
                truth_seq[s] = t_seq
            member_preds.append(p_seq)
            done = k * n_seq + s + 1
            method_elapsed = time.perf_counter() - t_method
            # Crude ETA: pretend the average rate from now on matches the
            # average rate so far. The first sequence is inflated by
            # torch.compile so early ETAs are pessimistic; they tighten up
            # by the second member.
            rate = method_elapsed / done
            eta_s = rate * (total - done)
            print(
                f"[run] {method}  k={k + 1}/{n_ensemble} seq={s + 1}/{n_seq} "
                f"({done}/{total})  t0={t0:<5}  rollout={elapsed:5.1f}s  "
                f"avg={rate:4.1f}s/seq  elapsed={method_elapsed/60:5.1f}m  "
                f"eta={eta_s/60:5.1f}m",
                flush=True,
            )
        pred_ens.append(np.stack(member_preds))
    pred_ens = np.stack(pred_ens)
    total_min = (time.perf_counter() - t_method) / 60
    print(f"[run] {method} ensemble finished in {total_min:.1f}m", flush=True)
    return pred_ens, truth_seq, np.asarray(times)


# --- Aggregation -------------------------------------------------------------
def aggregate(preds_ens: np.ndarray, truth: np.ndarray, channels: list[str]) -> dict:
    """Compute every scoreboard metric for a single method.

    preds_ens: (K, S, T, C, H, W) physical units.
    truth:     (S, T, C, H, W) physical units.
    """
    K, S, T, C, H, W = preds_ens.shape
    qp_idx = channels.index("qpepre")

    pred_mean = preds_ens.mean(axis=0)  # (S, T, C, H, W)

    # Per-channel RMSE of the ensemble mean against truth.
    rmse_all = per_channel_rmse(
        pred_mean.reshape(S * T, C, H, W),
        truth.reshape(S * T, C, H, W),
    )
    rmse_per_channel = rmse_all.mean(axis=0)

    # Per-lead-time per-channel RMSE: averaged over sequences and pixels for
    # each (T, C). Shape (T, C). Lead time index k corresponds to +(k+1) h.
    rmse_per_step = np.sqrt(((pred_mean - truth) ** 2).mean(axis=(0, -2, -1)))

    # qpepre categorical / FSS on the ensemble mean.
    qp_pred = pred_mean[..., qp_idx, :, :].reshape(-1, H, W)
    qp_truth = truth[..., qp_idx, :, :].reshape(-1, H, W)
    cat = categorical_scores(qp_pred, qp_truth, PRECIP_THRESHOLDS)
    csi_per = np.array([cat[t]["csi"] for t in PRECIP_THRESHOLDS])
    far_per = np.array([cat[t]["far"] for t in PRECIP_THRESHOLDS])
    hss_per = np.array([cat[t]["hss"] for t in PRECIP_THRESHOLDS])

    fss_p16 = np.array([fss(qp_pred, qp_truth, P16_THRESHOLD, w) for w in FSS_WINDOWS_PIX])

    # Per-channel kernel CRPS over (S*T, H, W).
    crps_per_channel = np.zeros(C, dtype=np.float64)
    for c in range(C):
        crps_per_channel[c] = crps_field(
            preds_ens[..., c, :, :].reshape(K, S * T, H, W),
            truth[..., c, :, :].reshape(S * T, H, W),
        )

    return {
        "rmse_per_channel": rmse_per_channel,
        "rmse_per_step": rmse_per_step,  # (T, C)
        "crps_per_channel": crps_per_channel,
        "crps_mean": float(np.mean(crps_per_channel)),
        "csi_per": csi_per,
        "csi_mean": float(np.nanmean(csi_per)),
        "csi_p16": float(cat[P16_THRESHOLD]["csi"]),
        "fss_p16_per": fss_p16,
        "fss_p16_mean": float(np.nanmean(fss_p16)),
        "hss_per": hss_per,
        "hss_mean": float(np.nanmean(hss_per)),
        "far_per": far_per,
        "far_mean": float(np.nanmean(far_per)),
    }


# --- Outputs -----------------------------------------------------------------
def write_scoreboard(
    metrics_by_method: dict,
    channels: list[str],
    time_per_seq_mean: dict,
    out_path_md: Path,
):
    headers = [
        "method",
        "Time/Seq.(s)",
        "CRPS↓",
        "CSI-M↑",
        "CSI-P16↑",
        "FSS-P16-M↑",
        "HSS-M↑",
        "FAR-M↓",
    ]
    headers += [f"RMSE_{ch}↓" for ch in channels]
    rows = []
    for m, met in metrics_by_method.items():
        row = [
            m,
            f"{time_per_seq_mean[m]:.3f}",
            f"{met['crps_mean']:.4f}",
            f"{met['csi_mean']:.4f}",
            f"{met['csi_p16']:.4f}",
            f"{met['fss_p16_mean']:.4f}",
            f"{met['hss_mean']:.4f}",
            f"{met['far_mean']:.4f}",
        ]
        row += [f"{v:.4f}" for v in met["rmse_per_channel"]]
        rows.append(row)

    out_path_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path_md.with_suffix(".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    md = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        md.append("| " + " | ".join(r) + " |")
    out_path_md.write_text("\n".join(md) + "\n")
    print("\n".join(md))


def plot_panels(
    out_dir: Path,
    channels: list[str],
    *,
    truth_seq: np.ndarray,
    diffusion_pred_mean: np.ndarray,
    flowcast_pred_mean: np.ndarray | None,
    meanflow_pred_mean: np.ndarray | None = None,
    seq_idx_to_plot,
    steps_to_plot,
):
    """Per (sequence, step, channel) panel showing predictions and error fields.

    Output layout (per channel ``ch``)::

        out_dir/
            {ch}/seq{NN}_step{KK}.png   2-row figure:
                                        row 1 = truth | diffusion | [flowcast] (physical units)
                                        row 2 = -      | diffusion - truth | [flowcast - truth]
            diff/{ch}/seq{NN}_step{KK}.png   1-row figure of the error fields only.

    Both layouts share the same data; the ``diff/`` folder is convenient when
    you want to scrape a single failure mode across many sequences without
    re-rendering the field row. All fields are in their post-``denormalize_state``
    physical units (see ``CHANNEL_UNITS``).

    When ``flowcast_pred_mean`` is None the flowcast column is omitted (used by
    the legacy old-stormcast comparison where there is no FlowCast checkpoint).
    ``meanflow_pred_mean`` works the same way: when provided, an extra
    ``meanflow (mean)`` field column and ``meanflow − truth`` diff column are
    appended after the FlowCast ones; when None (the default) the layout is
    unchanged, so existing diffusion/flowcast callers are unaffected.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    diff_root = out_dir / "diff"
    diff_root.mkdir(parents=True, exist_ok=True)
    # Per-channel subdirs so downstream "give me all qpepre panels" is one ls.
    for ch in channels:
        (out_dir / ch).mkdir(parents=True, exist_ok=True)
        (diff_root / ch).mkdir(parents=True, exist_ok=True)

    # Thesis colour convention (matches plot_weight_comparison_grid.py /
    # validation_plot): all 2-D fields use the viridis ramp; signed (pred-truth)
    # fields keep a diverging map. Centralised in thesis_style so old and new
    # figures stay on one palette.
    try:
        from thesis_style import FIELD_CMAP, DIFF_CMAP
    except Exception:  # keep the harness importable even without thesis_style
        FIELD_CMAP, DIFF_CMAP = "viridis", "RdBu_r"
    field_cmaps = {ch: FIELD_CMAP for ch in ("qpepre", "t2m", "u10", "v10")}
    diff_cmap = DIFF_CMAP  # symmetric around zero for every channel
    has_flow = flowcast_pred_mean is not None
    has_mf = meanflow_pred_mean is not None

    field_titles = (
        ["truth", "diffusion (mean)"]
        + (["flowcast (mean)"] if has_flow else [])
        + (["meanflow (mean)"] if has_mf else [])
    )
    diff_titles = (
        ["diffusion − truth"]
        + (["flowcast − truth"] if has_flow else [])
        + (["meanflow − truth"] if has_mf else [])
    )
    ncols = len(field_titles)

    for s in seq_idx_to_plot:
        for step in steps_to_plot:
            truth = truth_seq[s, step]
            diff = diffusion_pred_mean[s, step]
            flow = flowcast_pred_mean[s, step] if has_flow else None
            mf = meanflow_pred_mean[s, step] if has_mf else None
            for c, ch in enumerate(channels):
                unit = CHANNEL_UNITS.get(ch, "")
                # ---- field row colour scale (shared across truth + predictions) ----
                field_arrs = (
                    [truth[c], diff[c]]
                    + ([flow[c]] if has_flow else [])
                    + ([mf[c]] if has_mf else [])
                )
                field_stack = np.stack(field_arrs)
                if ch == "qpepre":
                    # precip: clip the extreme tail so convective structure stays
                    # legible under the sequential viridis ramp.
                    f_vmin = 0.0
                    f_vmax = float(np.quantile(field_stack, 0.995) + 1e-3)
                else:
                    # viridis is sequential: share one global (vmin, vmax) across
                    # truth + predictions (clim_for style), not a mean-centred one.
                    f_vmin = float(field_stack.min())
                    f_vmax = float(field_stack.max())
                f_cmap = field_cmaps.get(ch, "viridis")

                # ---- diff row colour scale (shared across all prediction diffs) ----
                diff_arrs = (
                    [diff[c] - truth[c]]
                    + ([flow[c] - truth[c]] if has_flow else [])
                    + ([mf[c] - truth[c]] if has_mf else [])
                )
                diff_stack = np.stack(diff_arrs)
                d_abs = float(np.quantile(np.abs(diff_stack), 0.99) + 1e-9)
                d_vmin, d_vmax = -d_abs, d_abs

                # ---- 2-row combined panel (truth + preds; diffs) ----
                fig, axes = plt.subplots(
                    2, ncols, figsize=(11 * ncols / 3, 7.5), constrained_layout=True,
                    squeeze=False,
                )
                # Row 0: truth + predictions on the physical-unit colour scale.
                for ax, arr, title in zip(axes[0], field_arrs, field_titles):
                    im_f = ax.imshow(
                        arr, origin="lower", vmin=f_vmin, vmax=f_vmax,
                        cmap=f_cmap, aspect="auto",
                    )
                    ax.set_title(title)
                    ax.set_xticks([]); ax.set_yticks([])
                fig.colorbar(
                    im_f, ax=axes[0].tolist(),
                    fraction=0.03, pad=0.02, shrink=0.95, label=unit,
                )
                # Row 1: leave first cell blank to align with truth; then diffs.
                axes[1, 0].axis("off")
                for ax, arr, title in zip(axes[1, 1:], diff_arrs, diff_titles):
                    im_d = ax.imshow(
                        arr, origin="lower", vmin=d_vmin, vmax=d_vmax,
                        cmap=diff_cmap, aspect="auto",
                    )
                    ax.set_title(title)
                    ax.set_xticks([]); ax.set_yticks([])
                fig.colorbar(
                    im_d, ax=axes[1].tolist(),
                    fraction=0.03, pad=0.02, shrink=0.95, label=f"Δ {unit}".strip(),
                )

                fig.suptitle(
                    f"{CHANNEL_LABELS.get(ch, ch)}  |  seq={s}  step +{step + 1}h",
                    fontsize=11,
                )
                fname = f"seq{s:02d}_step{step:02d}.png"
                fig.savefig(out_dir / ch / fname, dpi=130)
                plt.close(fig)

                # ---- companion diff-only panel (single row), for scraping later ----
                fig2, axes2 = plt.subplots(
                    1, max(1, len(diff_arrs)),
                    figsize=(5 * max(1, len(diff_arrs)), 4),
                    constrained_layout=True, squeeze=False,
                )
                for ax, arr, title in zip(axes2[0], diff_arrs, diff_titles):
                    im_d = ax.imshow(
                        arr, origin="lower", vmin=d_vmin, vmax=d_vmax,
                        cmap=diff_cmap, aspect="auto",
                    )
                    ax.set_title(title)
                    ax.set_xticks([]); ax.set_yticks([])
                fig2.colorbar(
                    im_d, ax=axes2[0].tolist(),
                    fraction=0.03, pad=0.02, shrink=0.95, label=f"Δ {unit}".strip(),
                )
                fig2.suptitle(
                    f"{CHANNEL_LABELS.get(ch, ch)}  prediction − truth  |  seq={s}  step +{step + 1}h",
                    fontsize=11,
                )
                fig2.savefig(diff_root / ch / fname, dpi=130)
                plt.close(fig2)


# --- Main -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-location", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--valid-dates", nargs=2, default=["2022/01/01", "2022/12/31"])
    ap.add_argument("--hr-size", nargs=2, type=int, default=[192, 96])
    ap.add_argument(
        "--qpepre-log1p",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set to match the dataset's qpepre encoding (default cleaned dataset is True; "
             "the legacy zarr_exp3_L_24_H_24_train_2_5_years_full dataset is False).",
    )
    ap.add_argument(
        "--kept-channels",
        nargs=4,
        default=KEPT_HIGHRES,
        help=(
            "HighRes channel order to feed the network. Cleaned/zettabyte runs use "
            "[u10, v10, t2m, qpepre]; the legacy old-dataset checkpoints "
            "(StormCastUNet.0.7500.mdlus / EDMPrecond.0.70000.mdlus) were trained "
            "with the natural zarr order [t2m, u10, v10, qpepre] — pass that here "
            "when scoring them."
        ),
    )

    ap.add_argument("--regression-checkpoint", type=Path, default=DEFAULT_REGRESSION)
    ap.add_argument("--diffusion-checkpoint", type=Path, default=DEFAULT_DIFFUSION)
    ap.add_argument("--flowcast-checkpoint", type=Path, default=DEFAULT_FLOWCAST)
    ap.add_argument(
        "--meanflow-checkpoint",
        type=Path,
        default=None,
        help=(
            "MeanFlow student .mdlus to add as an extra leg. The MeanFlow leg is "
            "OPT-IN: it runs only when this is given, so existing diffusion/flowcast "
            "comparisons are unaffected. Loaded the same way as the FlowCast .mdlus "
            f"(online student). Canonical cleaned checkpoint: {DEFAULT_MEANFLOW}."
        ),
    )
    ap.add_argument(
        "--skip-flowcast",
        action="store_true",
        help="Skip the FlowCast leg. Use this for the legacy/old-dataset comparison "
             "where no matching FlowCast checkpoint exists.",
    )
    ap.add_argument(
        "--skip-diffusion",
        action="store_true",
        help="Skip the EDM diffusion leg. Useful for FlowCast-only sweeps "
             "(e.g. qpw ablation) where one EDM reference is computed separately.",
    )

    ap.add_argument("--output-dir", type=Path,
                    default=REPO_ROOT / "experiment_scripts" / "results"
                    / "diffusion_vs_flowcast")
    ap.add_argument("--n-sequences", type=int, default=24,
                    help="Number of evenly-spaced initial times.")
    ap.add_argument("--n-steps", type=int, default=6,
                    help="Autoregressive forecast horizon (hours per sequence).")
    ap.add_argument("--ensemble", type=int, default=4,
                    help="Ensemble members per sequence — required for kernel CRPS.")
    ap.add_argument("--seed", type=int, default=0)

    # Diffusion sampler hyperparameters (match config/sampler/edm_deterministic.yaml
    # + train_diffusion.sh overrides).
    ap.add_argument("--diffusion-num-steps", type=int, default=18)
    ap.add_argument("--sigma-min", type=float, default=0.002)
    ap.add_argument("--sigma-max", type=float, default=80.0)
    ap.add_argument("--rho", type=float, default=7.0)
    ap.add_argument("--diffusion-solver", choices=("heun", "euler"), default="heun")

    # FlowCast sampler hyperparameters (match config/inference/flowcast.yaml).
    # Pass multiple integers to score the same FlowCast checkpoint at several
    # NFE settings in one go; each becomes its own row named flowcast_nfe<N>.
    # A single value keeps the legacy row name "flowcast" for back-compat.
    ap.add_argument("--flowcast-num-steps", type=int, nargs="+", default=[10])
    ap.add_argument("--flowcast-solver", choices=("euler", "midpoint"), default="euler")
    ap.add_argument("--sigma-data", type=float, default=0.5)

    # MeanFlow sampler hyperparameters (match config/inference/meanflow.yaml).
    # Few-step average-velocity sampler — num_steps is the NFE directly (1 =
    # one-step, 2 = the config default). Pass multiple integers to score the
    # same checkpoint at several NFE in one go; each becomes its own row named
    # meanflow_nfe<N>. A single value keeps the row name "meanflow". sigma_data
    # is shared with the FlowCast leg above.
    ap.add_argument("--meanflow-num-steps", type=int, nargs="+", default=[2])

    # Plotting
    ap.add_argument("--n-panels-seq", type=int, default=3,
                    help="Number of sequences to render in the panels directory.")
    ap.add_argument("--panel-steps", type=int, nargs="+", default=None,
                    help="Forecast steps to render. Default: first and last.")

    args = ap.parse_args()

    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"[device] {device}")
    print(f"[data] {args.data_location} valid={args.valid_dates} hr={args.hr_size}")

    dataset, invariant_tensor = build_dataset(args, device)
    channels = list(dataset.state_channels())
    if channels != list(args.kept_channels):
        print(
            f"[warn] dataset returned channels {channels}, expected {list(args.kept_channels)}; "
            f"continuing with the loader-provided order."
        )
    print(f"[data] usable pairs={len(dataset)} channels={channels}")

    if len(dataset) < args.n_sequences * args.n_steps + 1:
        raise SystemExit(
            f"Not enough validation samples ({len(dataset)}) for "
            f"{args.n_sequences} sequences of length {args.n_steps}."
        )

    # Evenly-spaced initial times across the validation set, leaving room for n_steps.
    t0_indices = (
        np.linspace(0, len(dataset) - args.n_steps - 1, args.n_sequences)
        .astype(int)
        .tolist()
    )
    print(
        f"[seqs] {len(t0_indices)} sequences x {args.n_steps} steps "
        f"(first={t0_indices[0]}, last={t0_indices[-1]})"
    )

    print(f"[load] regression {args.regression_checkpoint}")
    regression = (
        Module.from_checkpoint(str(args.regression_checkpoint)).to(device).eval()
    )

    diff_ens = None
    truth_seq = None
    diff_times = None
    if not args.skip_diffusion:
        # ---- diffusion ----
        diffusion_kwargs = dict(
            num_steps=args.diffusion_num_steps,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            rho=args.rho,
            solver=args.diffusion_solver,
        )
        print(f"[load] diffusion {args.diffusion_checkpoint}")
        diff_model = (
            Module.from_checkpoint(str(args.diffusion_checkpoint)).to(device).eval()
        )
        print(
            f"[run] diffusion ensemble (K={args.ensemble}, NFE/step≈"
            f"{2 * args.diffusion_num_steps if args.diffusion_solver == 'heun' else args.diffusion_num_steps})"
        )
        diff_ens, truth_seq, diff_times = run_method_ensemble(
            model=diff_model,
            method="diffusion",
            regression=regression,
            invariant=invariant_tensor,
            dataset=dataset,
            t0_indices=t0_indices,
            n_steps=args.n_steps,
            n_ensemble=args.ensemble,
            sampler_kwargs=diffusion_kwargs,
            device=device,
            seed=args.seed,
        )
        del diff_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        print("[skip] diffusion leg disabled via --skip-diffusion")

    # Each FlowCast NFE produces (name, ens, times). The model is loaded once
    # and reused; only sampler_kwargs changes per pass.
    flow_results: list[tuple[str, np.ndarray, np.ndarray]] = []
    if not args.skip_flowcast:
        nfe_list = list(args.flowcast_num_steps)
        single = len(nfe_list) == 1
        print(f"[load] flowcast {args.flowcast_checkpoint}")
        flow_model = (
            Module.from_checkpoint(str(args.flowcast_checkpoint)).to(device).eval()
        )
        for i, nfe in enumerate(nfe_list):
            method_name = "flowcast" if single else f"flowcast_nfe{nfe}"
            flowcast_kwargs = dict(
                num_steps=nfe,
                sigma_data=args.sigma_data,
                solver=args.flowcast_solver,
            )
            print(
                f"[run] {method_name} ensemble (K={args.ensemble}, "
                f"NFE={nfe if args.flowcast_solver == 'euler' else 2 * nfe})"
            )
            ens, _truth, times = run_method_ensemble(
                model=flow_model,
                method="flowcast",
                regression=regression,
                invariant=invariant_tensor,
                dataset=dataset,
                t0_indices=t0_indices,
                n_steps=args.n_steps,
                n_ensemble=args.ensemble,
                sampler_kwargs=flowcast_kwargs,
                device=device,
                seed=args.seed + 1 + i,
            )
            if truth_seq is None:
                truth_seq = _truth
            flow_results.append((method_name, ens, times))
        del flow_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        print("[skip] flowcast leg disabled via --skip-flowcast")

    # MeanFlow leg (opt-in): mirrors the FlowCast leg exactly but swaps the
    # multi-step Euler ODE for the few-step average-velocity sampler. One row
    # per requested NFE, named meanflow / meanflow_nfe<N>.
    mf_results: list[tuple[str, np.ndarray, np.ndarray]] = []
    if args.meanflow_checkpoint is not None:
        nfe_list = list(args.meanflow_num_steps)
        single = len(nfe_list) == 1
        print(f"[load] meanflow {args.meanflow_checkpoint}")
        mf_model = (
            Module.from_checkpoint(str(args.meanflow_checkpoint)).to(device).eval()
        )
        for i, nfe in enumerate(nfe_list):
            method_name = "meanflow" if single else f"meanflow_nfe{nfe}"
            meanflow_kwargs = dict(
                num_steps=nfe,
                sigma_data=args.sigma_data,
            )
            print(f"[run] {method_name} ensemble (K={args.ensemble}, NFE={nfe})")
            ens, _truth, times = run_method_ensemble(
                model=mf_model,
                method="meanflow",
                regression=regression,
                invariant=invariant_tensor,
                dataset=dataset,
                t0_indices=t0_indices,
                n_steps=args.n_steps,
                n_ensemble=args.ensemble,
                sampler_kwargs=meanflow_kwargs,
                device=device,
                seed=args.seed + 101 + i,
            )
            if truth_seq is None:
                truth_seq = _truth
            mf_results.append((method_name, ens, times))
        del mf_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        print("[skip] meanflow leg disabled (pass --meanflow-checkpoint to enable)")

    if truth_seq is None:
        raise SystemExit(
            "No leg produced a rollout — nothing to score. Enable at least one of "
            "the diffusion / flowcast / meanflow legs (the diffusion and flowcast "
            "legs run by default unless --skip-*; meanflow runs only with "
            "--meanflow-checkpoint)."
        )

    print("[score] aggregating...")
    metrics = {}
    times_mean = {}
    if diff_ens is not None:
        metrics["diffusion"] = aggregate(diff_ens, truth_seq, channels)
        times_mean["diffusion"] = float(diff_times.mean())
    for name, ens, times in flow_results:
        metrics[name] = aggregate(ens, truth_seq, channels)
        times_mean[name] = float(times.mean())
    for name, ens, times in mf_results:
        metrics[name] = aggregate(ens, truth_seq, channels)
        times_mean[name] = float(times.mean())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_scoreboard(metrics, channels, times_mean,
                     args.output_dir / "scoreboard.md")

    diff_mean = diff_ens.mean(axis=0) if diff_ens is not None else None
    # Panels show one FlowCast / MeanFlow trace each; pick the first NFE to keep
    # the figure readable. The full per-NFE breakdown lives in scoreboard.csv.
    flow_mean = flow_results[0][1].mean(axis=0) if flow_results else None
    mf_mean = mf_results[0][1].mean(axis=0) if mf_results else None

    panel_steps = (
        args.panel_steps
        if args.panel_steps is not None
        else sorted({0, args.n_steps - 1})
    )
    if diff_mean is None:
        # plot_panels assumes a diffusion column; skip rather than refactor.
        # Sweeps that bypass diffusion typically don't need per-sequence panels.
        print("[panels] skipped (no diffusion run; pass a diffusion checkpoint to render panels)")
    else:
        plot_panels(
            args.output_dir / "panels",
            channels,
            truth_seq=truth_seq,
            diffusion_pred_mean=diff_mean,
            flowcast_pred_mean=flow_mean,
            meanflow_pred_mean=mf_mean,
            seq_idx_to_plot=range(min(len(t0_indices), args.n_panels_seq)),
            steps_to_plot=panel_steps,
        )

    # Per-threshold detail dump (for plotting later).
    with open(args.output_dir / "per_threshold.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "metric", *PRECIP_THRESHOLDS])
        for m, met in metrics.items():
            w.writerow([m, "csi", *[f"{v:.6f}" for v in met["csi_per"]]])
            w.writerow([m, "far", *[f"{v:.6f}" for v in met["far_per"]]])
            w.writerow([m, "hss", *[f"{v:.6f}" for v in met["hss_per"]]])

    with open(args.output_dir / "fss_p16.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "metric", *FSS_WINDOWS_PIX])
        for m, met in metrics.items():
            w.writerow([m, f"fss_p{int(P16_THRESHOLD)}",
                        *[f"{v:.6f}" for v in met["fss_p16_per"]]])

    with open(args.output_dir / "rmse_per_channel.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", *channels])
        for m, met in metrics.items():
            w.writerow([m, *[f"{v:.6f}" for v in met["rmse_per_channel"]]])

    # Per-lead-time RMSE: one row per (method, +Nh) so it can drive the
    # RMSE-vs-lead-time line plots. Lead-time +1h is the first prediction
    # produced by the rollout.
    with open(args.output_dir / "rmse_per_step.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "step_h", *channels])
        for m, met in metrics.items():
            for k, row in enumerate(met["rmse_per_step"]):
                w.writerow([m, k + 1, *[f"{v:.6f}" for v in row]])

    with open(args.output_dir / "crps_per_channel.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", *channels])
        for m, met in metrics.items():
            w.writerow([m, *[f"{v:.6f}" for v in met["crps_per_channel"]]])

    print(f"[done] {args.output_dir}")


if __name__ == "__main__":
    main()
