"""Deterministic, precipitation-skill and spectral metrics for the progressive-
distillation students against the EDM teacher baseline.

Outputs (all written to ``results/metrics/``):

    metrics_deterministic.csv      per-model × per-channel RMSE / MAE
    metrics_precip.csv             per-model × threshold  CSI / FB / FSS
    spectra.csv                    radial log-PSD per model per channel
    rmse_vs_truth.pdf/.png         per-channel RMSE bar chart
    mae_vs_truth.pdf/.png          per-channel MAE  bar chart
    nfe_rmse_pareto.pdf/.png       NFE vs RMSE (mean over channels)
    csi_vs_threshold.pdf/.png      CSI(τ) for qpepre, one line per model
    fss_qpepre.pdf/.png            FSS(τ) at neighbourhood radii {5, 11, 21}
    spectra_qpepre.pdf/.png        log-P(k) overlay for qpepre
    spectra_panel.pdf/.png         log-P(k) overlay for every channel

This script realises Sections 4.2 (Metrics), 4.3 (Progressive-Distillation
Results) and 4.5 (PD vs teacher at matched NFE) of
``NTU-Thesis-LaTeX-Template/contents/chapter04.tex``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from common import (
    CHANNELS, FSS_NEIGHBOURHOODS, PRECIP_THRESHOLDS,
    DEFAULT_DATA_ROOT, DEFAULT_PHASES, DEFAULT_REGRESSION,
    DEFAULT_RESULTS, DEFAULT_TEACHER, RHO, SIGMA_MAX, SIGMA_MIN, TARGET_STEPS,
    TEACHER_STEPS,
    banner, build_model_plan, choose_sample_indices, contingency, csi_from_contingency,
    fb_from_contingency, fss_batch, init_device, label_colour, label_marker,
    load_dataset, load_invariants, load_model, mean_radial_log_psd,
    per_channel_mae, per_channel_rmse, prepare_sample_inputs, run_model,
    save_figure, set_thesis_style, timeit,
)


def _predict_stack(plan_entry, sample_inputs, invariant, regression, dataset, device, seed):
    model = load_model(plan_entry.ckpt, device)
    use_cuda = device.type == "cuda"
    preds = []
    for sidx, (bg, st_in, _) in enumerate(sample_inputs):
        torch.manual_seed(seed + sidx)
        pred = run_model(model, bg, st_in, invariant, regression,
                         mode=plan_entry.mode, num_steps=plan_entry.num_steps)
        preds.append(dataset.denormalize_state(pred.cpu().numpy()[0].copy()))
    if use_cuda:
        torch.cuda.synchronize()
    del model
    torch.cuda.empty_cache()
    return np.stack(preds, axis=0)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_channel_bars(values: dict[str, np.ndarray], ylabel: str, title: str,
                      out_dir: Path, stem: str):
    labels = list(values.keys())
    n = len(labels)
    x = np.arange(len(CHANNELS))
    width = 0.8 / max(n, 1)
    fig, ax = plt.subplots(figsize=(1.55 * len(CHANNELS) + 2.0, 3.6))
    for i, lab in enumerate(labels):
        ax.bar(x + (i - (n - 1) / 2) * width, values[lab], width,
               label=lab, color=label_colour(lab), edgecolor="#222", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(CHANNELS)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(ncol=min(4, n), loc="upper left")
    ax.margins(x=0.02)
    fig.tight_layout()
    save_figure(fig, out_dir, stem)


def plot_pareto(plan_entries, rmse_mean, out_dir: Path):
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    for entry in plan_entries:
        ax.plot([entry.num_steps], [rmse_mean[entry.label]],
                marker=label_marker(entry.label), color=label_colour(entry.label),
                linestyle="none", markersize=7, markeredgecolor="#222",
                markeredgewidth=0.6, label=entry.label)
    ax.set_xscale("log")
    ax.set_xlabel("sampling steps (NFE)")
    ax.set_ylabel("RMSE vs. ground truth  (mean over channels)")
    ax.set_title("NFE–RMSE Pareto (progressive students vs. teacher)")
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    save_figure(fig, out_dir, "nfe_rmse_pareto")


def plot_csi(values: dict[str, dict], out_dir: Path):
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    for lab, per_tau in values.items():
        y = [per_tau[t] for t in PRECIP_THRESHOLDS]
        ax.plot(list(PRECIP_THRESHOLDS), y,
                marker=label_marker(lab), color=label_colour(lab),
                markeredgecolor="#222", markeredgewidth=0.5, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel(r"threshold $\tau$  [mm h$^{-1}$]")
    ax.set_ylabel("Critical Success Index")
    ax.set_title("Precipitation skill on qpepre")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    save_figure(fig, out_dir, "csi_vs_threshold")


def plot_fss(fss_results: dict, out_dir: Path):
    """One subplot per neighbourhood radius."""
    n = len(FSS_NEIGHBOURHOODS)
    fig, axes = plt.subplots(1, n, figsize=(3.3 * n + 0.3, 3.3), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, radius in zip(axes, FSS_NEIGHBOURHOODS):
        for lab, by_nr in fss_results.items():
            y = [by_nr[(radius, t)] for t in PRECIP_THRESHOLDS]
            ax.plot(list(PRECIP_THRESHOLDS), y,
                    marker=label_marker(lab), color=label_colour(lab),
                    markeredgecolor="#222", markeredgewidth=0.5, label=lab)
        ax.set_xscale("log")
        ax.set_xlabel(r"$\tau$  [mm h$^{-1}$]")
        ax.set_title(f"$n = {radius}$")
        ax.set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Fractions Skill Score")
    axes[-1].legend(loc="lower left", ncol=1)
    fig.suptitle("FSS on qpepre by neighbourhood radius", y=1.02)
    fig.tight_layout()
    save_figure(fig, out_dir, "fss_qpepre")


def plot_spectra(k: np.ndarray, spectra: dict[str, np.ndarray], out_dir: Path):
    # qpepre dedicated plot
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    qi = CHANNELS.index("qpepre")
    for lab, Pk in spectra.items():
        ax.loglog(k[1:], Pk[qi, 1:], color=label_colour(lab),
                  linestyle="--" if lab == "truth" else "-",
                  label=lab, linewidth=1.2)
    ax.set_xlabel("wavenumber  $k$")
    ax.set_ylabel(r"$P(k)$")
    ax.set_title("Radial log-PSD of qpepre (2022 mean)")
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    save_figure(fig, out_dir, "spectra_qpepre")

    # 2x2 panel for all channels
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True)
    for ci, ch in enumerate(CHANNELS):
        ax = axes[ci // 2, ci % 2]
        for lab, Pk in spectra.items():
            ax.loglog(k[1:], Pk[ci, 1:], color=label_colour(lab),
                      linestyle="--" if lab == "truth" else "-",
                      label=lab, linewidth=1.2)
        ax.set_title(ch)
        if ci // 2 == 1:
            ax.set_xlabel("$k$")
        if ci % 2 == 0:
            ax.set_ylabel("$P(k)$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Radial log-PSD per channel (2022 mean)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    save_figure(fig, out_dir, "spectra_panel")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases-dir", type=Path, default=DEFAULT_PHASES)
    ap.add_argument("--data-location", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--regression-checkpoint", type=Path, default=DEFAULT_REGRESSION)
    ap.add_argument("--teacher-checkpoint", type=Path, default=DEFAULT_TEACHER)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS / "metrics")
    ap.add_argument("--n-samples", type=int, default=256,
                    help="number of validation samples to draw (default: 256 → "
                         "≈3%% of the 2022 year, enough to stabilise CSI/FSS).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--valid-dates", nargs=2, default=["2022/01/01", "2022/12/31"])
    ap.add_argument("--initial-num-steps", type=int, default=TEACHER_STEPS)
    ap.add_argument("--target-num-steps", type=int, default=TARGET_STEPS)
    ap.add_argument("--skip-teacher", action="store_true")
    args = ap.parse_args()

    set_thesis_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = init_device()
    banner("Loading dataset")
    dataset = load_dataset(args.data_location, tuple(args.valid_dates))
    print(f"[dataset] {len(dataset)} validation samples, channels = {dataset.state_channels()}")
    invariant = load_invariants(dataset, device)

    regression = load_model(args.regression_checkpoint, device)
    plan = build_model_plan(args.phases_dir,
                            teacher_ckpt=args.teacher_checkpoint,
                            initial_steps=args.initial_num_steps,
                            target_steps=args.target_num_steps,
                            skip_teacher=args.skip_teacher)
    print("[plan]")
    for e in plan:
        print(f"  - {e.label:<20s}  mode={e.mode:<8s}  N={e.num_steps}")

    indices = choose_sample_indices(dataset, args.n_samples, seed=args.seed)
    sample_inputs, truth_arr, _ = prepare_sample_inputs(dataset, indices, device)
    print(f"[samples] {len(indices)} indices drawn")

    # -------- run each model over the sample batch --------
    preds: dict[str, np.ndarray] = {}
    for entry in plan:
        banner(f"Inference: {entry.label}  (N={entry.num_steps})")
        with timeit(entry.label):
            preds[entry.label] = _predict_stack(
                entry, sample_inputs, invariant, regression, dataset, device, args.seed
            )

    # =========================================================
    # 1. Deterministic RMSE / MAE
    # =========================================================
    banner("Deterministic RMSE / MAE")
    rmse_vs_truth = {lab: per_channel_rmse(arr, truth_arr).mean(axis=0)
                     for lab, arr in preds.items()}
    mae_vs_truth = {lab: per_channel_mae(arr, truth_arr).mean(axis=0)
                    for lab, arr in preds.items()}

    det_csv = args.output_dir / "metrics_deterministic.csv"
    with open(det_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "nfe", "metric", *CHANNELS, "mean"])
        n_by_lab = {e.label: e.num_steps for e in plan}
        for lab, v in rmse_vs_truth.items():
            w.writerow([lab, n_by_lab[lab], "rmse",
                        *[f"{x:.6f}" for x in v], f"{float(v.mean()):.6f}"])
        for lab, v in mae_vs_truth.items():
            w.writerow([lab, n_by_lab[lab], "mae",
                        *[f"{x:.6f}" for x in v], f"{float(v.mean()):.6f}"])
    print(f"[det] {det_csv}")

    plot_channel_bars(rmse_vs_truth,
                      ylabel="RMSE (physical units)",
                      title="Per-channel RMSE against RWRF target",
                      out_dir=args.output_dir, stem="rmse_vs_truth")
    plot_channel_bars(mae_vs_truth,
                      ylabel="MAE (physical units)",
                      title="Per-channel MAE against RWRF target",
                      out_dir=args.output_dir, stem="mae_vs_truth")

    rmse_mean = {lab: float(v.mean()) for lab, v in rmse_vs_truth.items()}
    plot_pareto(plan, rmse_mean, args.output_dir)

    # =========================================================
    # 2. Precipitation skill (qpepre only)
    # =========================================================
    banner("Precipitation skill")
    qp = CHANNELS.index("qpepre")
    truth_qp = truth_arr[:, qp]

    csi: dict[str, dict[float, float]] = {}
    fb: dict[str, dict[float, float]] = {}
    fss: dict[str, dict[tuple, float]] = {}
    for lab, arr in preds.items():
        pred_qp = arr[:, qp]
        csi[lab] = {}
        fb[lab] = {}
        fss[lab] = {}
        for tau in PRECIP_THRESHOLDS:
            c = contingency(pred_qp, truth_qp, tau)
            csi[lab][tau] = csi_from_contingency(c)
            fb[lab][tau] = fb_from_contingency(c)
            for n in FSS_NEIGHBOURHOODS:
                fss[lab][(n, tau)] = fss_batch(pred_qp, truth_qp, tau, n)
            print(f"  {lab:<20s} τ={tau:<5.1f}  "
                  f"CSI={csi[lab][tau]:.3f}  FB={fb[lab][tau]:.3f}  "
                  f"FSS(11)={fss[lab][(11, tau)]:.3f}")

    precip_csv = args.output_dir / "metrics_precip.csv"
    with open(precip_csv, "w", newline="") as f:
        w = csv.writer(f)
        header = ["model", "threshold_mmph", "csi", "freq_bias",
                  *[f"fss_n{n}" for n in FSS_NEIGHBOURHOODS]]
        w.writerow(header)
        for lab in preds:
            for tau in PRECIP_THRESHOLDS:
                w.writerow([lab, tau,
                            f"{csi[lab][tau]:.6f}", f"{fb[lab][tau]:.6f}",
                            *[f"{fss[lab][(n, tau)]:.6f}" for n in FSS_NEIGHBOURHOODS]])
    print(f"[precip] {precip_csv}")

    plot_csi(csi, args.output_dir)
    plot_fss(fss, args.output_dir)

    # =========================================================
    # 3. Spectral diagnostics
    # =========================================================
    banner("Spectral diagnostics")
    spectra = {}
    k, Pk_truth = mean_radial_log_psd(truth_arr)
    spectra["truth"] = Pk_truth
    for lab, arr in preds.items():
        _, Pk = mean_radial_log_psd(arr)
        spectra[lab] = Pk

    sp_csv = args.output_dir / "spectra.csv"
    with open(sp_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "channel", "k", "P_k"])
        for lab, Pk in spectra.items():
            for ci, ch in enumerate(CHANNELS):
                for ki, kv in enumerate(k):
                    w.writerow([lab, ch, int(kv), f"{float(Pk[ci, ki]):.6e}"])
    print(f"[spectra] {sp_csv}")

    plot_spectra(k, spectra, args.output_dir)

    print(f"\n✓ metrics complete → {args.output_dir}")


if __name__ == "__main__":
    main()
