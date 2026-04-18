"""Failure-mode diagnostics: precipitation mode collapse, tail-quantile
regression, and per-phase validation-loss dynamics.

Outputs (``results/failure_modes/``):

    wet_fraction_vs_threshold.pdf/.png
    qpepre_tail_quantiles.pdf/.png
    qpepre_histogram.pdf/.png
    training_loss_per_phase.pdf/.png
    per_channel_rmse_per_phase.pdf/.png
    failure_modes.csv            per-model wet-fraction + P95 / P99 / P99.9

These correspond to Section 4.7 "Failure Modes" and Section 4.3 "Training
dynamics" of ``NTU-Thesis-LaTeX-Template/contents/chapter04.tex``. The
training-dynamics plots are read from the per-phase CSVs shipped alongside
the progressive checkpoints, so they run without any forward passes.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch

from common import (
    CHANNELS, DEFAULT_DATA_ROOT, DEFAULT_PHASES, DEFAULT_REGRESSION,
    DEFAULT_RESULTS, DEFAULT_TEACHER, PRECIP_THRESHOLDS, TARGET_STEPS,
    TEACHER_STEPS,
    banner, build_model_plan, choose_sample_indices, discover_phases,
    init_device, label_colour, label_marker, load_dataset, load_invariants,
    load_model, prepare_sample_inputs, run_model, save_figure, set_thesis_style,
    timeit,
)

TAIL_QUANTILES = (0.90, 0.95, 0.99, 0.999)


def _predict_stack(entry, sample_inputs, invariant, regression, dataset, device, seed):
    model = load_model(entry.ckpt, device)
    preds = []
    for sidx, (bg, st_in, _) in enumerate(sample_inputs):
        torch.manual_seed(seed + sidx)
        pred = run_model(model, bg, st_in, invariant, regression,
                         mode=entry.mode, num_steps=entry.num_steps)
        preds.append(dataset.denormalize_state(pred.cpu().numpy()[0].copy()))
    del model
    torch.cuda.empty_cache()
    return np.stack(preds, axis=0)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_wet_fraction(data: Dict[str, Dict[float, float]], truth_line: Dict[float, float],
                      out_dir: Path):
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.plot(list(PRECIP_THRESHOLDS), [truth_line[t] for t in PRECIP_THRESHOLDS],
            linestyle="--", color="#2E7D32", marker="o",
            markeredgecolor="#222", markeredgewidth=0.5, label="truth")
    for lab, wf in data.items():
        ax.plot(list(PRECIP_THRESHOLDS), [wf[t] for t in PRECIP_THRESHOLDS],
                marker=label_marker(lab), color=label_colour(lab),
                markeredgecolor="#222", markeredgewidth=0.5, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"threshold $\tau$  [mm h$^{-1}$]")
    ax.set_ylabel(r"wet fraction  $\Pr[\text{qpepre} \geq \tau]$")
    ax.set_title("Wet-fraction drift (mode-collapse diagnostic)")
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    save_figure(fig, out_dir, "wet_fraction_vs_threshold")


def plot_tail_quantiles(quantiles: Dict[str, Dict[float, float]],
                        truth_q: Dict[float, float], out_dir: Path):
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    xs = list(TAIL_QUANTILES)
    ax.plot(xs, [truth_q[q] for q in xs],
            linestyle="--", color="#2E7D32", marker="o",
            markeredgecolor="#222", markeredgewidth=0.5, label="truth")
    for lab, per_q in quantiles.items():
        ax.plot(xs, [per_q[q] for q in xs],
                marker=label_marker(lab), color=label_colour(lab),
                markeredgecolor="#222", markeredgewidth=0.5, label=lab)
    ax.set_xscale("logit")
    ax.set_yscale("log")
    ax.set_xlabel("quantile")
    ax.set_ylabel(r"qpepre  [mm h$^{-1}$]")
    ax.set_title("Tail-quantile regression (extreme-precipitation diagnostic)")
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    save_figure(fig, out_dir, "qpepre_tail_quantiles")


def plot_histogram(preds: Dict[str, np.ndarray], truth: np.ndarray, out_dir: Path):
    qi = CHANNELS.index("qpepre")
    bins = np.logspace(-3, np.log10(max(5.0, float(truth[:, qi].max()) + 1e-3)), 40)
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.hist(truth[:, qi].ravel(), bins=bins, histtype="step",
            linestyle="--", color="#2E7D32", label="truth", linewidth=1.3)
    for lab, arr in preds.items():
        ax.hist(arr[:, qi].ravel(), bins=bins, histtype="step",
                color=label_colour(lab), label=lab, linewidth=1.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"qpepre  [mm h$^{-1}$]")
    ax.set_ylabel("pixel count")
    ax.set_title("qpepre intensity distribution")
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    save_figure(fig, out_dir, "qpepre_histogram")


def plot_training_loss(phases_dir: Path, phases: list[int], out_dir: Path):
    import csv as _csv
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    plotted = False
    for p in phases:
        csv_path = phases_dir / f"phase_{p}" / "valid_loss.csv"
        if not csv_path.exists():
            continue
        steps, losses = [], []
        with open(csv_path) as f:
            for row in _csv.DictReader(f):
                try:
                    steps.append(int(row["step"]))
                    losses.append(float(row["loss"]))
                except (KeyError, ValueError):
                    continue
        if steps:
            ax.plot(steps, losses, color=label_colour(f"phase_{p}"),
                    label=f"phase {p}", linewidth=1.2)
            plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("global step")
    ax.set_ylabel("validation MSE")
    ax.set_yscale("log")
    ax.set_title("Progressive distillation — per-phase validation loss")
    ax.legend(loc="best")
    fig.tight_layout()
    save_figure(fig, out_dir, "training_loss_per_phase")


def plot_per_channel_rmse(phases_dir: Path, phases: list[int], out_dir: Path):
    import csv as _csv
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True)
    for ci, ch in enumerate(CHANNELS):
        ax = axes[ci // 2, ci % 2]
        any_plotted = False
        for p in phases:
            csv_path = phases_dir / f"phase_{p}" / f"rmse_{ch}.csv"
            if not csv_path.exists():
                continue
            steps, vals = [], []
            with open(csv_path) as f:
                for row in _csv.DictReader(f):
                    try:
                        steps.append(int(row["step"]))
                        vals.append(float(row["rmse"]))
                    except (KeyError, ValueError):
                        continue
            if steps:
                ax.plot(steps, vals, color=label_colour(f"phase_{p}"),
                        label=f"phase {p}", linewidth=1.1)
                any_plotted = True
        ax.set_title(ch)
        if ci // 2 == 1:
            ax.set_xlabel("global step")
        if ci % 2 == 0:
            ax.set_ylabel("RMSE (normalised)")
        if any_plotted and ci == 0:
            ax.legend(loc="best", ncol=len(phases))
    fig.suptitle("Per-channel validation RMSE over training")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, out_dir, "per_channel_rmse_per_phase")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases-dir", type=Path, default=DEFAULT_PHASES)
    ap.add_argument("--data-location", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--regression-checkpoint", type=Path, default=DEFAULT_REGRESSION)
    ap.add_argument("--teacher-checkpoint", type=Path, default=DEFAULT_TEACHER)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS / "failure_modes")
    ap.add_argument("--n-samples", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--valid-dates", nargs=2, default=["2022/01/01", "2022/12/31"])
    ap.add_argument("--initial-num-steps", type=int, default=TEACHER_STEPS)
    ap.add_argument("--target-num-steps", type=int, default=TARGET_STEPS)
    args = ap.parse_args()

    set_thesis_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ----- Training-dynamics plots (no GPU needed) -----
    banner("Training dynamics (per-phase CSVs)")
    phases = discover_phases(args.phases_dir)
    if phases:
        plot_training_loss(args.phases_dir, phases, args.output_dir)
        plot_per_channel_rmse(args.phases_dir, phases, args.output_dir)
    else:
        print("[train-dyn] no phase CSVs discovered — skipping")

    # ----- Inference-based diagnostics -----
    device = init_device()
    banner("Loading dataset")
    dataset = load_dataset(args.data_location, tuple(args.valid_dates))
    invariant = load_invariants(dataset, device)

    regression = load_model(args.regression_checkpoint, device)
    plan = build_model_plan(args.phases_dir,
                            teacher_ckpt=args.teacher_checkpoint,
                            initial_steps=args.initial_num_steps,
                            target_steps=args.target_num_steps)

    indices = choose_sample_indices(dataset, args.n_samples, seed=args.seed)
    sample_inputs, truth_arr, _ = prepare_sample_inputs(dataset, indices, device)
    print(f"[samples] {len(indices)}")

    preds: Dict[str, np.ndarray] = {}
    for entry in plan:
        banner(f"Inference: {entry.label}")
        with timeit(entry.label):
            preds[entry.label] = _predict_stack(
                entry, sample_inputs, invariant, regression, dataset, device, args.seed
            )

    qp = CHANNELS.index("qpepre")
    truth_qp = truth_arr[:, qp]

    # Wet fraction @ threshold
    wet_truth = {t: float(np.mean(truth_qp >= t)) for t in PRECIP_THRESHOLDS}
    wet_model: Dict[str, Dict[float, float]] = {
        lab: {t: float(np.mean(arr[:, qp] >= t)) for t in PRECIP_THRESHOLDS}
        for lab, arr in preds.items()
    }
    plot_wet_fraction(wet_model, wet_truth, args.output_dir)

    # Tail quantiles
    truth_q = {q: float(np.quantile(truth_qp, q)) for q in TAIL_QUANTILES}
    model_q: Dict[str, Dict[float, float]] = {
        lab: {q: float(np.quantile(arr[:, qp], q)) for q in TAIL_QUANTILES}
        for lab, arr in preds.items()
    }
    plot_tail_quantiles(model_q, truth_q, args.output_dir)

    plot_histogram(preds, truth_arr, args.output_dir)

    csv_path = args.output_dir / "failure_modes.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model",
                    *[f"wet_frac_{t}" for t in PRECIP_THRESHOLDS],
                    *[f"qpepre_q{q}" for q in TAIL_QUANTILES]])
        w.writerow(["truth",
                    *[f"{wet_truth[t]:.6e}" for t in PRECIP_THRESHOLDS],
                    *[f"{truth_q[q]:.6f}" for q in TAIL_QUANTILES]])
        for lab in preds:
            w.writerow([lab,
                        *[f"{wet_model[lab][t]:.6e}" for t in PRECIP_THRESHOLDS],
                        *[f"{model_q[lab][q]:.6f}" for q in TAIL_QUANTILES]])
    print(f"[failure] {csv_path}")

    print(f"\n✓ failure-mode diagnostics complete → {args.output_dir}")


if __name__ == "__main__":
    main()
