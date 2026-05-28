#!/usr/bin/env python3
"""FlowCast NFE Pareto sweep — quality vs. number of Euler steps.

Replicates Fig. 5 / Table 9 of the FlowCast paper (Ribeiro & Pucer 2025) on
the Taiwan RWRF domain: hold the FlowCast checkpoint fixed and sweep the
inference step count S in {1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 40, 50},
recording per-channel RMSE, kernel CRPS, qpepre categorical scores, and wall
clock per sequence. The paper claims CFM saturates at S=3-10 — this script
tests that on a regional convective dataset for the first time.

Builds on the reusable parts of ``compare_diffusion_vs_flowcast.py``:

    build_dataset(args, device)
    run_method_ensemble(... sampler_kwargs={"num_steps": S, ...})
    aggregate(preds_ens, truth, channels)

The diffusion teacher is NOT swept here -- a separate run would be needed to
do the same curve for the EDM Heun sampler. Add ``--also-diffusion`` later
if you want both curves on one plot.

Output layout::

    <output-dir>/
        nfe_sweep.csv        # one row per NFE; column 1 = NFE, then RMSE/CRPS/...
        nfe_sweep.md         # markdown twin
        plot_quality_vs_nfe.png   # CRPS-qpepre + CSI-M vs NFE  (log-x)
        plot_time_vs_nfe.png      # wall-clock per sequence vs NFE (log-x, log-y)
        plot_pareto.png           # quality vs time (Pareto curve, log-x)
        nfe_<N>/scoreboard.{md,csv}   # per-NFE detail dump
        nfe_<N>/per_threshold.csv     # per-threshold CSI/FAR/HSS at that NFE
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

# --- repo path bootstrap (must run before any stormcast/ imports) ----------
REPO_ROOT = Path(__file__).resolve().parents[1]
STORMCAST_ROOT = REPO_ROOT / "stormcast"
sys.path.insert(0, str(STORMCAST_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse the heavy lifting from compare_diffusion_vs_flowcast.py.
from compare_diffusion_vs_flowcast import (  # noqa: E402
    FSS_WINDOWS_PIX,
    KEPT_HIGHRES,
    P16_THRESHOLD,
    PRECIP_THRESHOLDS,
    aggregate,
    build_dataset,
    run_method_ensemble,
    write_scoreboard,
)
from physicsnemo.distributed import DistributedManager  # noqa: E402
from physicsnemo.models import Module  # noqa: E402


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
# Default to the latest cleaned FlowCast checkpoint (~13 M samples seen). For
# the matched-2M comparison with the EDM teacher, override to
# FlowCastPrecond.0.20000.mdlus.
DEFAULT_FLOWCAST = (
    REPO_ROOT
    / "runs/flowcast_zettabyte_v1_cleaned_4_27_2026"
    / "flowcast_zettabyte_cleaned_4_27_2026/run_0"
    / "checkpoints_flowcast/FlowCastPrecond.0.140000.mdlus"
)

# FlowCast paper's Fig. 5 NFE list, extended to cover the diffusion-comparable
# regime (~30-50 NFE) we want to call out as "you can stop here". The set is
# small enough that the whole sweep fits in one ~20-min run on a single A6000
# at S=12 sequences x T=1 step x K=4 members.
DEFAULT_NFES = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 40, 50]


def _ts() -> str:
    """Compact wall-clock timestamp for log lines."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{_ts()}] [nfe_sweep] {msg}", flush=True)


def banner(msg: str) -> None:
    line = "=" * 64
    print(f"\n{line}\n[{_ts()}] {msg}\n{line}", flush=True)


def write_sweep_table(rows: list[dict], channels: list[str], out_csv: Path, out_md: Path) -> None:
    """Persist the sweep table both as CSV (for plotting) and Markdown (for review)."""
    # Stable column order: identification, then per-channel metrics, then aggregates.
    cols = ["nfe", "time_per_seq_s"]
    for ch in channels:
        cols.append(f"rmse_{ch}")
    for ch in channels:
        cols.append(f"crps_{ch}")
    cols.extend(["csi_m", "csi_p16", "fss_p16_m", "hss_m", "far_m"])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([f"{r[c]:.6f}" if isinstance(r[c], float) else r[c] for c in cols])

    with open(out_md, "w") as f:
        f.write("# FlowCast NFE Pareto sweep\n\n")
        f.write("Per-NFE single-step validation skill, qpepre in mm/h after `denormalize_state`.\n\n")
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join("---:" for _ in cols) + "|\n")
        for r in rows:
            cells = []
            for c in cols:
                v = r[c]
                if c == "nfe":
                    cells.append(str(int(v)))
                elif isinstance(v, float):
                    cells.append(f"{v:.4f}")
                else:
                    cells.append(str(v))
            f.write("| " + " | ".join(cells) + " |\n")


def _plot_quality_vs_nfe(rows: list[dict], channels: list[str], out_path: Path) -> None:
    nfes = np.array([r["nfe"] for r in rows], dtype=int)
    qp_idx = channels.index("qpepre")
    crps_qp = np.array([r[f"crps_{channels[qp_idx]}"] for r in rows])
    csi_m = np.array([r["csi_m"] for r in rows])

    fig, ax1 = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    color_l, color_r = "tab:blue", "tab:red"
    ax1.set_xscale("log")
    ax1.set_xlabel("NFE (number of Euler steps)")
    ax1.set_ylabel("CRPS qpepre  (mm/h, lower = better)", color=color_l)
    ax1.plot(nfes, crps_qp, "o-", color=color_l, label="CRPS qpepre")
    ax1.tick_params(axis="y", labelcolor=color_l)
    ax1.grid(True, which="both", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.set_ylabel("CSI-M (qpepre, higher = better)", color=color_r)
    ax2.plot(nfes, csi_m, "s-", color=color_r, label="CSI-M")
    ax2.tick_params(axis="y", labelcolor=color_r)

    ax1.set_title("FlowCast quality vs. NFE")
    ax1.set_xticks(nfes)
    ax1.set_xticklabels([str(n) for n in nfes], fontsize=8)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_time_vs_nfe(rows: list[dict], out_path: Path) -> None:
    nfes = np.array([r["nfe"] for r in rows], dtype=int)
    times = np.array([r["time_per_seq_s"] for r in rows])
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(nfes, times, "o-", color="tab:green")
    ax.set_xlabel("NFE (number of Euler steps)")
    ax.set_ylabel("Wall-clock per sequence (s)")
    ax.set_title("FlowCast latency vs. NFE")
    ax.set_xticks(nfes)
    ax.set_xticklabels([str(n) for n in nfes], fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    # Linear-time-vs-NFE reference line through the smallest NFE point.
    if len(nfes) > 1:
        anchor_n, anchor_t = nfes[0], times[0]
        ref = anchor_t * (nfes / anchor_n)
        ax.plot(nfes, ref, ":", color="gray", label="linear in NFE")
        ax.legend()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_pareto(rows: list[dict], channels: list[str], out_path: Path) -> None:
    qp_idx = channels.index("qpepre")
    times = np.array([r["time_per_seq_s"] for r in rows])
    crps = np.array([r[f"crps_{channels[qp_idx]}"] for r in rows])
    nfes = np.array([r["nfe"] for r in rows], dtype=int)
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot(times, crps, "o-", color="tab:purple")
    for x, y, n in zip(times, crps, nfes):
        ax.annotate(str(n), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Wall-clock per sequence (s)  --  lower-left is better")
    ax.set_ylabel("CRPS qpepre (mm/h)")
    ax.set_title("FlowCast NFE Pareto: quality vs. latency")
    ax.grid(True, which="both", alpha=0.25)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


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
    )
    ap.add_argument("--kept-channels", nargs=4, default=KEPT_HIGHRES)

    ap.add_argument("--regression-checkpoint", type=Path, default=DEFAULT_REGRESSION)
    ap.add_argument("--flowcast-checkpoint", type=Path, default=DEFAULT_FLOWCAST)

    ap.add_argument(
        "--nfes",
        nargs="+",
        type=int,
        default=DEFAULT_NFES,
        help="Euler step counts to sweep.",
    )
    ap.add_argument("--solver", choices=("euler", "midpoint"), default="euler",
                    help="Midpoint doubles the NFE per integration step; the x-axis "
                         "will reflect num_steps, not effective NFE.")
    ap.add_argument("--sigma-data", type=float, default=0.5)

    ap.add_argument("--n-sequences", type=int, default=12,
                    help="Number of evenly-spaced initial times. Smaller than the "
                         "main experiment because we are running ~15 sub-runs.")
    ap.add_argument("--n-steps", type=int, default=1,
                    help="Autoregressive forecast horizon. Default 1 = single-step "
                         "skill, which is what the FlowCast Fig. 5 curve isolates.")
    ap.add_argument("--ensemble", type=int, default=4,
                    help="Members per (sequence, NFE). Must be >= 2 for kernel CRPS.")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--output-dir", type=Path,
                    default=REPO_ROOT / "experiment_scripts" / "results"
                    / "flowcast_nfe_sweep")

    ap.add_argument("--keep-per-nfe-detail",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="Also write nfe_<N>/scoreboard.{md,csv} + per-threshold CSV.")

    args = ap.parse_args()

    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device
    if device.type == "cuda":
        torch.cuda.empty_cache()

    banner("FlowCast NFE Pareto sweep")
    log(f"device     = {device}")
    log(f"data       = {args.data_location}")
    log(f"regression = {args.regression_checkpoint}")
    log(f"flowcast   = {args.flowcast_checkpoint}")
    log(f"NFE list   = {args.nfes}  ({len(args.nfes)} values)")
    log(f"solver     = {args.solver}  sigma_data = {args.sigma_data}")
    log(f"S={args.n_sequences} sequences x T={args.n_steps} steps x K={args.ensemble} members")
    log(f"output     = {args.output_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build dataset + invariant tensor (cheap) ONCE.
    log("loading dataset ...")
    dataset, invariant_tensor = build_dataset(args, device)
    channels = list(dataset.state_channels())
    log(f"dataset    : usable pairs = {len(dataset)}  channels = {channels}")

    if len(dataset) < args.n_sequences * args.n_steps + 1:
        raise SystemExit(
            f"Not enough validation samples ({len(dataset)}) for "
            f"{args.n_sequences} sequences of length {args.n_steps}."
        )

    t0_indices = (
        np.linspace(0, len(dataset) - args.n_steps - 1, args.n_sequences)
        .astype(int)
        .tolist()
    )
    log(f"sequences  : {len(t0_indices)} t0 indices ({t0_indices[0]} ... {t0_indices[-1]})")

    # Load the regression mean + flowcast student ONCE; reuse across NFEs.
    log(f"loading regression {args.regression_checkpoint}")
    regression = Module.from_checkpoint(str(args.regression_checkpoint)).to(device).eval()
    log(f"loading flowcast   {args.flowcast_checkpoint}")
    flow_model = Module.from_checkpoint(str(args.flowcast_checkpoint)).to(device).eval()

    rows: list[dict] = []
    sweep_t0 = time.perf_counter()
    for idx, nfe in enumerate(args.nfes, start=1):
        banner(f"NFE = {nfe}   ({idx}/{len(args.nfes)})")
        sampler_kwargs = dict(
            num_steps=int(nfe),
            sigma_data=args.sigma_data,
            solver=args.solver,
        )
        nfe_t0 = time.perf_counter()
        preds_ens, truth_seq, times = run_method_ensemble(
            model=flow_model,
            method="flowcast",
            regression=regression,
            invariant=invariant_tensor,
            dataset=dataset,
            t0_indices=t0_indices,
            n_steps=args.n_steps,
            n_ensemble=args.ensemble,
            sampler_kwargs=sampler_kwargs,
            device=device,
            seed=args.seed,
        )
        metrics = aggregate(preds_ens, truth_seq, channels)
        time_per_seq = float(times.mean()) if len(times) else float("nan")
        nfe_elapsed = time.perf_counter() - nfe_t0
        log(
            f"NFE={nfe:3d}  t/seq={time_per_seq:5.2f}s  "
            f"CRPS-qpepre={metrics['crps_per_channel'][channels.index('qpepre')]:.4f}  "
            f"CSI-M={metrics['csi_mean']:.4f}  "
            f"RMSE-qpepre={metrics['rmse_per_channel'][channels.index('qpepre')]:.4f}  "
            f"NFE-elapsed={nfe_elapsed/60:.1f}m"
        )

        row = {"nfe": int(nfe), "time_per_seq_s": time_per_seq}
        for c, ch in enumerate(channels):
            row[f"rmse_{ch}"] = float(metrics["rmse_per_channel"][c])
            row[f"crps_{ch}"] = float(metrics["crps_per_channel"][c])
        row["csi_m"] = metrics["csi_mean"]
        row["csi_p16"] = metrics["csi_p16"]
        row["fss_p16_m"] = metrics["fss_p16_mean"]
        row["hss_m"] = metrics["hss_mean"]
        row["far_m"] = metrics["far_mean"]
        rows.append(row)

        if args.keep_per_nfe_detail:
            sub = args.output_dir / f"nfe_{nfe:03d}"
            sub.mkdir(parents=True, exist_ok=True)
            write_scoreboard(
                {"flowcast": metrics},
                channels,
                {"flowcast": time_per_seq},
                sub / "scoreboard.md",
            )
            with open(sub / "per_threshold.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["metric", *PRECIP_THRESHOLDS])
                w.writerow(["csi", *[f"{v:.6f}" for v in metrics["csi_per"]]])
                w.writerow(["far", *[f"{v:.6f}" for v in metrics["far_per"]]])
                w.writerow(["hss", *[f"{v:.6f}" for v in metrics["hss_per"]]])
            with open(sub / "rmse_per_channel.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["channel", "rmse", "crps"])
                for c, ch in enumerate(channels):
                    w.writerow([
                        ch,
                        f"{metrics['rmse_per_channel'][c]:.6f}",
                        f"{metrics['crps_per_channel'][c]:.6f}",
                    ])

        # Write the master sweep table incrementally so a long sweep is
        # always partially observable -- handy if you Ctrl-C halfway.
        write_sweep_table(
            rows, channels,
            args.output_dir / "nfe_sweep.csv",
            args.output_dir / "nfe_sweep.md",
        )

    banner(f"sweep finished in {(time.perf_counter() - sweep_t0)/60:.1f} min")
    log("rendering plots ...")
    _plot_quality_vs_nfe(rows, channels, args.output_dir / "plot_quality_vs_nfe.png")
    _plot_time_vs_nfe(rows, args.output_dir / "plot_time_vs_nfe.png")
    _plot_pareto(rows, channels, args.output_dir / "plot_pareto.png")

    log(f"done -- outputs under {args.output_dir}")
    log("  table : nfe_sweep.{csv,md}")
    log("  plots : plot_quality_vs_nfe.png  plot_time_vs_nfe.png  plot_pareto.png")
    if args.keep_per_nfe_detail:
        log("  detail: nfe_<NFE>/scoreboard.{md,csv} + per_threshold.csv + rmse_per_channel.csv")


if __name__ == "__main__":
    main()
