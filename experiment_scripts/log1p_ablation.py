#!/usr/bin/env python
"""log1p-on-qpepre ablation.

Compares the four runs that share the same architecture but differ only in
whether qpepre was log1p-compressed at preprocessing time:

    * EDM diffusion              (log1p vs NO_log1p) at step 20000
    * FlowCast (flow-matching)   (log1p vs NO_log1p) at step 20000

Each run is paired with the regression checkpoint used during *its* training:
    log1p     -> StormCastUNet.0.8000.mdlus
    NO_log1p  -> StormCastUNet.0.10000.mdlus

The dataset path is also tied to the run: the cleaned (log1p) zarr for the
log1p runs and the *_raw zarr for the NO_log1p runs. ``qpepre_log1p`` is set
correspondingly so ``denormalize_state`` round-trips qpepre to mm/h
consistently across both data variants — this is what makes the comparison
apples-to-apples.

Outputs (under ``experiment_scripts/results/log1p_ablation/``):
    - ``metrics.csv``                    per-run aggregated 1h metrics
    - ``csi_qpepre.csv``                 CSI at five thresholds
    - ``rollout_rmse.csv``               per-step RMSE on case studies
    - ``psd_<channel>.png``              spectral comparison
    - ``rollout_rmse_<channel>.png``     RMSE-vs-leadtime plots
    - ``case_<n>_qpepre_panel.png``      qpepre rollout heatmaps
    - ``summary.md``                     markdown summary table

Usage:
    cd experiment_scripts
    python log1p_ablation.py                          # full run
    python log1p_ablation.py --max-samples 200        # quick smoke-test
    python log1p_ablation.py --no-rollouts            # 1-step only
    python log1p_ablation.py --include-regression     # add regression-only baseline
"""

from __future__ import annotations

import argparse
import os
import sys
import csv
import json

import numpy as np
import torch

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from _eval_utils import (  # noqa: E402
    DATA_LOG1P,
    DATA_RAW,
    DEFAULT_CASE_STUDIES,
    QPEPRE_THRESHOLDS,
    StudentSpec,
    autoregressive_rollout,
    evaluate_single_step,
    load_diffusion,
    load_flowcast,
    load_regression,
    open_validation_dataset,
    plot_psd_comparison,
    plot_rollout_qpepre_panel,
    plot_rollout_rmse,
    write_markdown_summary,
)

PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
DEFAULT_RUNS_ROOT = os.path.join(PROJECT_ROOT, "runs")

# Path templates relative to a runs-root. Same layout as the local sync of
# /data/exp_3_train_2_5_yrs_val_1yr_tp1/ from zettabyte. The shell launcher
# overrides --runs-root or each individual --reg/--diff/--flow-* path so the
# same script runs locally and on the cloud worker.
REL_REG_LOG1P = (
    "regression_zettabyte_v1_cleaned_4_27_2026/regression_zettabyte_cleaned_4_27_2026/run_0/"
    "checkpoints_regression/StormCastUNet.0.8000.mdlus"
)
REL_REG_NOLOG1P = (
    "regression_zettabyte_v1_cleaned_4_27_2026_NO_log1p/regression_cleaned_NO_log1p/run_0/"
    "checkpoints_regression/StormCastUNet.0.10000.mdlus"
)
REL_DIFF_LOG1P_DIR = (
    "diffusion_zettabyte_v1_cleaned_4_27_2026/diffusion_zettabyte_cleaned_4_27_2026/run_0"
)
REL_DIFF_NOLOG1P_DIR = (
    "diffusion_zettabyte_v1_cleaned_4_27_2026_NO_log1p/edm_cleaned_NO_log1p/run_0"
)
REL_FLOW_LOG1P_DIR = (
    "flowcast_zettabyte_v1_cleaned_4_27_2026/flowcast_zettabyte_cleaned_4_27_2026/run_0"
)
REL_FLOW_NOLOG1P_DIR = (
    "flowcast_zettabyte_v1_cleaned_4_27_2026_NO_log1p/flowcast_cleaned_NO_log1p/run_0"
)

# EDM teacher sampler kwargs at inference time. Match the cfg/sampler/edm_deterministic
# that the diffusion training expects, but use sigma_max=80 to match the EDM
# preconditioner the model was trained with.
EDM_SAMPLER_KWARGS = dict(
    num_steps=18,
    sigma_min=0.002,
    sigma_max=80.0,
    rho=7.0,
    S_churn=0.0,
    S_min=0.0,
    S_max=float("inf"),
    S_noise=1.0,
)

# FlowCast sampler kwargs (training used Euler, num_steps=10, sigma_data=0.5).
FLOWCAST_SAMPLER_KWARGS = dict(num_steps=10, sigma_data=0.5, solver="euler")

# Architecture kwargs needed to materialise FlowCastPrecond from the EMA shadow.
FLOWCAST_ARCH = dict(
    img_resolution=(192, 96),
    target_channels=4,
    conditional_channels=10,  # 4 state + 4 regression + 2 invariant
    spatial_pos_embed=True,
    attn_resolutions=[],
    sigma_data=0.5,
    time_scale=1000.0,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out-dir",
        default=os.path.join(THIS_DIR, "results", "log1p_ablation"),
        help="Where to write CSVs/PNGs/summary.md.",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap number of 1-step validation samples (default: full year ~8759).",
    )
    p.add_argument(
        "--spectrum-stride",
        type=int,
        default=24,
        help="Stride for accumulating PS1D (every Nth sample). Default 24 = 1/day.",
    )
    p.add_argument(
        "--rollout-steps",
        type=int,
        default=12,
        help="Rollout length in hours per case study.",
    )
    p.add_argument(
        "--cases",
        nargs="*",
        default=list(DEFAULT_CASE_STUDIES),
        help="ISO-format initial timestamps for case-study rollouts.",
    )
    p.add_argument(
        "--no-rollouts", action="store_true", help="Skip the autoregressive rollouts."
    )
    p.add_argument(
        "--include-regression",
        action="store_true",
        help="Also evaluate the deterministic regression baseline for both variants.",
    )
    p.add_argument(
        "--device", default=None, help="Override torch device, e.g. cuda:0."
    )

    # ------------- path overrides (zettabyte / cluster use) -----------------
    p.add_argument(
        "--runs-root", default=DEFAULT_RUNS_ROOT,
        help="Root directory containing the *_zettabyte_v1_* run subdirs. "
             "On zettabyte: /data/exp_3_train_2_5_yrs_val_1yr_tp1.",
    )
    p.add_argument(
        "--data-log1p", default=DATA_LOG1P,
        help="Root of the cleaned (log1p-on-qpepre) zarr dataset.",
    )
    p.add_argument(
        "--data-raw", default=DATA_RAW,
        help="Root of the raw (NO_log1p) zarr dataset.",
    )
    p.add_argument(
        "--checkpoint-step", type=int, default=20000,
        help="Step number of the EDMPrecond/FlowCastPrecond checkpoint (used "
             "only to locate the .mdlus file under checkpoints_*; ignored when "
             "--diff-* / --flow-*-dir are passed in full).",
    )
    p.add_argument("--reg-log1p", default=None, help="Override regression-log1p .mdlus.")
    p.add_argument("--reg-no-log1p", default=None, help="Override regression-NO_log1p .mdlus.")
    p.add_argument("--diff-log1p-dir", default=None,
                   help="Run dir containing checkpoints_diffusion/ for the log1p EDM run.")
    p.add_argument("--diff-no-log1p-dir", default=None,
                   help="Same, for the NO_log1p EDM run.")
    p.add_argument("--flow-log1p-dir", default=None,
                   help="Run dir containing ema_state.pt for the log1p FlowCast run.")
    p.add_argument("--flow-no-log1p-dir", default=None,
                   help="Same, for the NO_log1p FlowCast run.")
    return p.parse_args()


def _resolve_paths(args):
    """Resolve the four checkpoint / two ema-dir paths from CLI overrides
    or, failing that, from --runs-root + the canonical layout."""
    runs = args.runs_root
    step = args.checkpoint_step
    reg_log1p = args.reg_log1p or os.path.join(runs, REL_REG_LOG1P)
    reg_nolog1p = args.reg_no_log1p or os.path.join(runs, REL_REG_NOLOG1P)
    diff_log1p_dir = args.diff_log1p_dir or os.path.join(runs, REL_DIFF_LOG1P_DIR)
    diff_nolog1p_dir = args.diff_no_log1p_dir or os.path.join(runs, REL_DIFF_NOLOG1P_DIR)
    flow_log1p_dir = args.flow_log1p_dir or os.path.join(runs, REL_FLOW_LOG1P_DIR)
    flow_nolog1p_dir = args.flow_no_log1p_dir or os.path.join(runs, REL_FLOW_NOLOG1P_DIR)

    diff_log1p = os.path.join(
        diff_log1p_dir, "checkpoints_diffusion", f"EDMPrecond.0.{step}.mdlus"
    )
    diff_nolog1p = os.path.join(
        diff_nolog1p_dir, "checkpoints_diffusion", f"EDMPrecond.0.{step}.mdlus"
    )
    flow_log1p_ema = os.path.join(flow_log1p_dir, "ema_state.pt")
    flow_nolog1p_ema = os.path.join(flow_nolog1p_dir, "ema_state.pt")
    return {
        "reg_log1p": reg_log1p,
        "reg_nolog1p": reg_nolog1p,
        "diff_log1p": diff_log1p,
        "diff_nolog1p": diff_nolog1p,
        "flow_log1p_ema": flow_log1p_ema,
        "flow_nolog1p_ema": flow_nolog1p_ema,
    }


def make_specs(args, paths, device):
    """Yield (run_label, dataset_loc, qpepre_log1p, regression_path, StudentSpec)."""
    data_log1p = args.data_log1p
    data_raw = args.data_raw

    # --- log1p variant -------------------------------------------------------
    diff_log1p = load_diffusion(paths["diff_log1p"], device)
    flow_log1p = load_flowcast(
        ema_path=paths["flow_log1p_ema"],
        **FLOWCAST_ARCH,
        device=device,
    )
    yield (
        "edm_log1p",
        data_log1p, True, paths["reg_log1p"],
        StudentSpec("edm_log1p", "edm", diff_log1p, dict(EDM_SAMPLER_KWARGS)),
    )
    yield (
        "flowcast_log1p",
        data_log1p, True, paths["reg_log1p"],
        StudentSpec("flowcast_log1p", "flowcast", flow_log1p, dict(FLOWCAST_SAMPLER_KWARGS)),
    )
    if args.include_regression:
        yield (
            "regression_log1p",
            data_log1p, True, paths["reg_log1p"],
            StudentSpec("regression_log1p", "regression", None, {}),
        )

    # --- NO_log1p variant ----------------------------------------------------
    diff_nolog1p = load_diffusion(paths["diff_nolog1p"], device)
    flow_nolog1p = load_flowcast(
        ema_path=paths["flow_nolog1p_ema"],
        **FLOWCAST_ARCH,
        device=device,
    )
    yield (
        "edm_NO_log1p",
        data_raw, False, paths["reg_nolog1p"],
        StudentSpec("edm_NO_log1p", "edm", diff_nolog1p, dict(EDM_SAMPLER_KWARGS)),
    )
    yield (
        "flowcast_NO_log1p",
        data_raw, False, paths["reg_nolog1p"],
        StudentSpec("flowcast_NO_log1p", "flowcast", flow_nolog1p, dict(FLOWCAST_SAMPLER_KWARGS)),
    )
    if args.include_regression:
        yield (
            "regression_NO_log1p",
            data_raw, False, paths["reg_nolog1p"],
            StudentSpec("regression_NO_log1p", "regression", None, {}),
        )


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[log1p_ablation] device={device}")

    paths = _resolve_paths(args)
    for k, v in paths.items():
        print(f"  {k}: {v}")
    missing = [k for k, v in paths.items() if not os.path.exists(v)]
    if missing:
        raise SystemExit(
            "[log1p_ablation] missing checkpoint paths: " + ", ".join(missing)
        )
    for d in (args.data_log1p, args.data_raw):
        if not os.path.isdir(d):
            raise SystemExit(f"[log1p_ablation] dataset dir does not exist: {d}")

    results: dict[str, dict] = {}
    rollouts: dict[str, list[dict]] = {}

    # Cache datasets and regression models so we don't reload twice per variant.
    dataset_cache: dict[tuple[str, bool], object] = {}
    regression_cache: dict[str, object] = {}

    # We construct specs lazily — students live in GPU memory, so iterate
    # one-at-a-time and free in between to keep memory bounded.
    for label, data_loc, qpepre_log1p, reg_path, spec in make_specs(args, paths, device):
        print(f"\n=== {label} ===")
        ds_key = (data_loc, qpepre_log1p)
        if ds_key not in dataset_cache:
            dataset_cache[ds_key] = open_validation_dataset(
                data_loc, qpepre_log1p, img_size=(192, 96)
            )
        dataset = dataset_cache[ds_key]
        if reg_path not in regression_cache:
            regression_cache[reg_path] = load_regression(reg_path, device)
        regression_model = regression_cache[reg_path]

        # 1-step aggregated metrics over the validation year.
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

        # Case studies.
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

        # Free the heavy student model now that we're done with it.
        if spec.model is not None:
            del spec.model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ------------------------------------------------------------------ outputs
    state_channels = ("u10", "v10", "t2m", "qpepre")

    # metrics.csv
    rows = []
    for name, s in results.items():
        row = {"run": name, "n_samples": s["n_samples"]}
        for c in state_channels:
            row[f"rmse_{c}"] = s["rmse"].get(c, float("nan"))
            row[f"mae_{c}"] = s["mae"].get(c, float("nan"))
        rows.append(row)
    fieldnames = ["run", "n_samples"] + [
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
        hdr = ["run"]
        for thr in QPEPRE_THRESHOLDS:
            hdr += [f"csi@{thr}", f"pod@{thr}", f"far@{thr}", f"bias@{thr}"]
        hdr += ["wet_frac_pred", "wet_frac_true", "p99_pred", "p99_true"]
        w = csv.writer(f)
        w.writerow(hdr)
        for name, s in results.items():
            if "qpepre_csi" not in s:
                continue
            row = [name]
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

    # rollout_rmse.csv
    if any(rollouts.values()):
        roll_path = os.path.join(args.out_dir, "rollout_rmse.csv")
        with open(roll_path, "w", newline="") as f:
            hdr = ["run", "case_init", "lead_h"] + [f"rmse_{c}" for c in state_channels]
            w = csv.writer(f)
            w.writerow(hdr)
            for name, cases in rollouts.items():
                for case in cases:
                    chans = case["state_channels"]
                    rmse = case["rmse_per_step"]
                    for h in range(rmse.shape[0]):
                        row = [name, case["initial_time"], h + 1]
                        for c in state_channels:
                            cidx = chans.index(c)
                            row.append(f"{rmse[h, cidx]:.4f}")
                        w.writerow(row)

    # PSD plots per channel.
    for c in state_channels:
        plot_psd_comparison(
            os.path.join(args.out_dir, f"psd_{c}.png"),
            results,
            c,
        )

    # Rollout RMSE-vs-leadtime plots per channel.
    if any(rollouts.values()):
        for c in state_channels:
            plot_rollout_rmse(
                os.path.join(args.out_dir, f"rollout_rmse_{c}.png"),
                rollouts,
                c,
            )

        # qpepre side-by-side rollout panels for each case.
        for ci, _ts in enumerate(args.cases):
            plot_rollout_qpepre_panel(
                os.path.join(args.out_dir, f"case_{ci}_qpepre_panel.png"),
                rollouts,
                case_idx=ci,
            )

    # Markdown summary
    write_markdown_summary(
        os.path.join(args.out_dir, "summary.md"),
        "log1p ablation — 1h aggregated metrics over 2022 valid year",
        results,
    )

    # Persist raw results dict for later reuse.
    cleaned = {}
    for name, s in results.items():
        d = dict(s)
        for k in ("ps_k", "ps_pred", "ps_true"):
            if k in d:
                d[k] = d[k].tolist()
        if "qpepre_csi" in d:
            for k2 in ("qpepre_csi", "qpepre_pod", "qpepre_far", "qpepre_bias"):
                d[k2] = {str(thr): float(v) for thr, v in d[k2].items()}
        cleaned[name] = d
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(cleaned, f, indent=2)

    print(f"\n[log1p_ablation] wrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
