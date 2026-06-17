#!/usr/bin/env python
"""qpepre channel-weight (qpw) ablation for FlowCast.

Sweeps the FlowCast student trained with channel_weights=[1,1,1,QPW] for
QPW in {1.0, 1.2, 1.4, 1.6, 1.8, 2.2, 2.4} and compares them at step 20000.

All qpw runs share:
    * dataset = cleaned (log1p) zarr
    * regression = StormCastUNet.0.8000.mdlus from the cleaned-log1p run
    * arch = same FlowCastPrecond as the canonical flowcast_zettabyte run

The point of the ablation is to find the qpw that maximises CSI on heavy
qpepre while keeping non-precip channels (t2m/u10/v10) RMSE flat.

Outputs (under ``experiment_scripts/results/qpw_ablation/``):
    - ``metrics.csv``                    per-qpw 1h metrics
    - ``csi_qpepre.csv``                 CSI/POD/FAR/BIAS/p99/wet-frac
    - ``rollout_rmse.csv``               per-step RMSE on case studies
    - ``qpw_sweep_<channel>.png``        metric-vs-qpw curves
    - ``psd_<channel>.png``              PSD overlay
    - ``rollout_rmse_<channel>.png``     RMSE-vs-leadtime
    - ``case_<n>_qpepre_panel.png``      qpepre rollout heatmaps
    - ``summary.md``

Usage:
    cd experiment_scripts
    python qpw_ablation.py
    python qpw_ablation.py --max-samples 200      # smoke-test
    python qpw_ablation.py --no-rollouts          # 1-step only
    python qpw_ablation.py --qpw 1.0 1.4 2.2      # subset of qpw values
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from glob import glob

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from _eval_utils import (  # noqa: E402
    DATA_LOG1P,
    DEFAULT_CASE_STUDIES,
    QPEPRE_THRESHOLDS,
    StudentSpec,
    autoregressive_rollout,
    evaluate_single_step,
    load_flowcast,
    load_regression,
    open_validation_dataset,
    plot_psd_comparison,
    plot_rollout_qpepre_panel,
    plot_rollout_rmse,
    write_markdown_summary,
)

PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
RUNS_ROOT = os.path.join(PROJECT_ROOT, "runs")
ABL_ROOT = os.path.join(RUNS_ROOT, "flowcast_qpw_ablation")

REG_LOG1P = os.path.join(
    RUNS_ROOT,
    "regression_zettabyte_v1_cleaned_4_27_2026/regression_zettabyte_cleaned_4_27_2026/run_0/checkpoints_regression/StormCastUNet.0.8000.mdlus",
)

FLOWCAST_SAMPLER_KWARGS = dict(num_steps=10, sigma_data=0.5, solver="euler")
FLOWCAST_ARCH = dict(
    img_resolution=(192, 96),
    target_channels=4,
    conditional_channels=10,  # 4 state + 4 regression + 2 invariant
    spatial_pos_embed=True,
    attn_resolutions=[],
    sigma_data=0.5,
    time_scale=1000.0,
)

# Default qpw sweep — all subdirs we ship with.
DEFAULT_QPW = (1.0, 1.2, 1.4, 1.6, 1.8, 2.2, 2.4)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out-dir", default=os.path.join(THIS_DIR, "results", "qpw_ablation"),
    )
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--spectrum-stride", type=int, default=24)
    p.add_argument("--rollout-steps", type=int, default=12)
    p.add_argument("--cases", nargs="*", default=list(DEFAULT_CASE_STUDIES))
    p.add_argument("--no-rollouts", action="store_true")
    p.add_argument(
        "--qpw",
        nargs="*",
        type=float,
        default=list(DEFAULT_QPW),
        help="Subset of qpw weights to evaluate.",
    )
    p.add_argument(
        "--checkpoint-step",
        type=int,
        default=20000,
        help="Training step of the FlowCastPrecond.*.mdlus to load.",
    )
    p.add_argument("--device", default=None)
    return p.parse_args()


def find_run_dir(qpw: float) -> str:
    """Map qpw value to the inner ``run_0`` directory.

    Files on disk look like:
        runs/flowcast_qpw_ablation/qpw1.0/flowcast_qpw1.0_cleaned_4_27_2026/run_0/...
    """
    label = f"qpw{qpw:g}"
    parent = os.path.join(ABL_ROOT, label)
    if not os.path.isdir(parent):
        raise FileNotFoundError(parent)
    matches = sorted(glob(os.path.join(parent, "*", "run_0")))
    if not matches:
        raise FileNotFoundError(f"No run_0 inside {parent}")
    return matches[0]


def make_qpw_specs(qpw_values, step, device):
    for qpw in qpw_values:
        run_dir = find_run_dir(qpw)
        ema_path = os.path.join(run_dir, "ema_state.pt")
        ckpt = os.path.join(
            run_dir, "checkpoints_flowcast", f"FlowCastPrecond.0.{step}.mdlus"
        )
        if not os.path.exists(ema_path):
            raise FileNotFoundError(ema_path)
        if not os.path.exists(ckpt):
            print(f"  [warn] {ckpt} missing — using EMA only.", flush=True)
        net = load_flowcast(
            ema_path=ema_path,
            **FLOWCAST_ARCH,
            device=device,
        )
        label = f"qpw{qpw:g}"
        yield label, qpw, StudentSpec(
            label, "flowcast", net, dict(FLOWCAST_SAMPLER_KWARGS)
        )


def plot_qpw_sweep(out_path: str, qpw_metric_pairs: list[tuple[float, float]], ylabel: str, title: str):
    """1-D scatter+line of metric vs qpw."""
    if not qpw_metric_pairs:
        return
    qpw_metric_pairs = sorted(qpw_metric_pairs, key=lambda x: x[0])
    xs = [q for q, _ in qpw_metric_pairs]
    ys = [m for _, m in qpw_metric_pairs]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel("qpw (qpepre channel weight)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[qpw_ablation] device={device}, qpw={args.qpw}, step={args.checkpoint_step}")

    dataset = open_validation_dataset(DATA_LOG1P, qpepre_log1p=True, img_size=(192, 96))
    regression_model = load_regression(REG_LOG1P, device)

    results: dict[str, dict] = {}
    rollouts: dict[str, list[dict]] = {}
    qpw_lookup: dict[str, float] = {}

    for label, qpw, spec in make_qpw_specs(args.qpw, args.checkpoint_step, device):
        print(f"\n=== {label} ===")
        qpw_lookup[label] = qpw
        summary = evaluate_single_step(
            spec=spec,
            regression_model=regression_model,
            dataset=dataset,
            device=device,
            max_samples=args.max_samples,
            spectrum_stride=args.spectrum_stride,
            progress_label=label,
        )
        summary["state_channels"] = list(dataset.state_channels())
        results[label] = summary

        case_results = []
        if not args.no_rollouts:
            for ts in args.cases:
                roll = autoregressive_rollout(
                    spec=spec,
                    regression_model=regression_model,
                    dataset=dataset,
                    initial_time=ts,
                    n_steps=args.rollout_steps,
                    device=device,
                )
                case_results.append(roll)
        rollouts[label] = case_results

        del spec.model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ------------------------------------------------------------------ outputs
    state_channels = ("u10", "v10", "t2m", "qpepre")

    # metrics.csv -- include a qpw column for quick plotting in pandas.
    rows = []
    for name, s in results.items():
        row = {"run": name, "qpw": qpw_lookup[name], "n_samples": s["n_samples"]}
        for c in state_channels:
            row[f"rmse_{c}"] = s["rmse"].get(c, float("nan"))
            row[f"mae_{c}"] = s["mae"].get(c, float("nan"))
        rows.append(row)
    fieldnames = ["run", "qpw", "n_samples"] + [
        f"{m}_{c}" for c in state_channels for m in ("rmse", "mae")
    ]
    with open(os.path.join(args.out_dir, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # csi_qpepre.csv
    csi_path = os.path.join(args.out_dir, "csi_qpepre.csv")
    with open(csi_path, "w", newline="") as f:
        hdr = ["run", "qpw"]
        for thr in QPEPRE_THRESHOLDS:
            hdr += [f"csi@{thr}", f"pod@{thr}", f"far@{thr}", f"bias@{thr}"]
        hdr += ["wet_frac_pred", "wet_frac_true", "p99_pred", "p99_true"]
        w = csv.writer(f)
        w.writerow(hdr)
        for name, s in results.items():
            if "qpepre_csi" not in s:
                continue
            row = [name, qpw_lookup[name]]
            for thr in QPEPRE_THRESHOLDS:
                row += [
                    f"{s['qpepre_csi'][thr]:.4f}",
                    f"{s['qpepre_pod'][thr]:.4f}",
                    f"{s['qpepre_far'][thr]:.4f}",
                    f"{s['qpepre_bias'][thr]:.4f}",
                ]
            row += [
                f"{s['qpepre_wet_frac_pred']:.4f}",
                f"{s['qpepre_wet_frac_true']:.4f}",
                f"{s['qpepre_p99_pred_mean']:.3f}",
                f"{s['qpepre_p99_true_mean']:.3f}",
            ]
            w.writerow(row)

    # rollout CSV
    if any(rollouts.values()):
        roll_path = os.path.join(args.out_dir, "rollout_rmse.csv")
        with open(roll_path, "w", newline="") as f:
            hdr = ["run", "qpw", "case_init", "lead_h"] + [f"rmse_{c}" for c in state_channels]
            w = csv.writer(f)
            w.writerow(hdr)
            for name, cases in rollouts.items():
                for case in cases:
                    chans = case["state_channels"]
                    rmse = case["rmse_per_step"]
                    for h in range(rmse.shape[0]):
                        row = [name, qpw_lookup[name], case["initial_time"], h + 1]
                        for c in state_channels:
                            cidx = chans.index(c)
                            row.append(f"{rmse[h, cidx]:.4f}")
                        w.writerow(row)

    # qpw-sweep curves: one plot per channel for RMSE; plus CSI@1mm and CSI@10mm.
    for c in state_channels:
        pairs = [(qpw_lookup[name], s["rmse"].get(c)) for name, s in results.items()]
        plot_qpw_sweep(
            os.path.join(args.out_dir, f"qpw_sweep_rmse_{c}.png"),
            pairs,
            f"RMSE ({c})",
            f"qpw sweep — RMSE_{c}",
        )

    for thr in (1.0, 10.0):
        pairs = [
            (qpw_lookup[name], s["qpepre_csi"][thr])
            for name, s in results.items()
            if "qpepre_csi" in s
        ]
        plot_qpw_sweep(
            os.path.join(args.out_dir, f"qpw_sweep_csi_{thr}mm.png"),
            pairs,
            f"CSI @ {thr} mm/h",
            f"qpw sweep — CSI@{thr}",
        )

    # PSD comparison + rollout plots (same primitives as log1p_ablation).
    for c in state_channels:
        plot_psd_comparison(
            os.path.join(args.out_dir, f"psd_{c}.png"), results, c
        )

    if any(rollouts.values()):
        for c in state_channels:
            plot_rollout_rmse(
                os.path.join(args.out_dir, f"rollout_rmse_{c}.png"),
                rollouts,
                c,
            )
        for ci, _ts in enumerate(args.cases):
            plot_rollout_qpepre_panel(
                os.path.join(args.out_dir, f"case_{ci}_qpepre_panel.png"),
                rollouts,
                case_idx=ci,
            )

    write_markdown_summary(
        os.path.join(args.out_dir, "summary.md"),
        f"qpw FlowCast ablation — step {args.checkpoint_step}",
        results,
    )

    cleaned = {}
    for name, s in results.items():
        d = dict(s)
        d["qpw"] = qpw_lookup[name]
        for k in ("ps_k", "ps_pred", "ps_true"):
            if k in d:
                d[k] = d[k].tolist()
        if "qpepre_csi" in d:
            for k2 in ("qpepre_csi", "qpepre_pod", "qpepre_far", "qpepre_bias"):
                d[k2] = {str(thr): float(v) for thr, v in d[k2].items()}
        cleaned[name] = d
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(cleaned, f, indent=2)

    print(f"\n[qpw_ablation] wrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
