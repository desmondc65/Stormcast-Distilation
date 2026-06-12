#!/usr/bin/env python3
"""Regenerate the data-driven thesis result figures from the canonical CSVs.

This rebuilds the six figures referenced in Chapter~4 of the thesis directly
into ``NTU-Thesis-LaTeX-Template/figures/results/`` (overwrite in place), using
the project-wide viridis palette defined in :mod:`thesis_style` so they match
``plot_weight_comparison_grid.py``:

    figures/results/scoreboard_3way.png     <- main_experiment/scoreboard_3way.csv
    figures/results/rmse_per_channel.png    <- main_experiment/cleaned_2M/rmse_per_channel.csv
    figures/results/crps_per_channel.png    <- main_experiment/cleaned_2M/crps_per_channel.csv
    figures/results/csi_per_threshold.png   <- main_experiment/cleaned_2M/per_threshold.csv
    figures/results/fss_p16.png             <- main_experiment/cleaned_2M/fss_p16.csv
    figures/results/rollout_rmse.png        <- main_experiment/rmse_per_step_3way.csv

No model inference and no GPU: every number here already exists in the
experiment CSVs (the same numbers tabulated in Chapter~4). To re-run those
underlying experiments, see ``run_main_experiment.sh``.

Usage::

    python make_thesis_figures.py                 # write into the thesis tree
    python make_thesis_figures.py --dry-run        # write into results/ instead
    python make_thesis_figures.py --results-root <dir> --fig-out <dir>
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import thesis_style as ts

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO_ROOT / "experiment_scripts" / "results" / "main_experiment"
DEFAULT_FIGOUT = REPO_ROOT / "NTU-Thesis-LaTeX-Template" / "figures" / "results"

# Metric direction: True = higher is better. Drives winner highlighting + arrows.
HIGHER_BETTER = {
    "Time/Seq.(s)": False, "CRPS": False, "CSI-M": True, "CSI-P16": True,
    "FSS-P16-M": True, "HSS-M": True, "FAR-M": False,
    "RMSE_u10": False, "RMSE_v10": False, "RMSE_t2m": False, "RMSE_qpepre": False,
    "csi": True, "far": False, "hss": True,
}

# Scoreboard metrics to draw (CSI-P16 dropped: ~0 and noisy at this budget).
SCOREBOARD_METRICS = [
    "Time/Seq.(s)", "CRPS", "CSI-M", "FSS-P16-M", "HSS-M",
    "FAR-M", "RMSE_u10", "RMSE_v10", "RMSE_t2m", "RMSE_qpepre",
]
# Method draw order (rollout adds legacy in front).
CLEANED_ORDER = ["diffusion", "flowcast_nfe10", "flowcast_nfe15", "flowcast_nfe20",
                 "meanflow_nfe1", "meanflow_nfe2"]
ROLLOUT_ORDER = ["legacy_edm", "cleaned_edm", "cleaned_flow_nfe10",
                 "cleaned_flow_nfe15", "cleaned_flow_nfe20",
                 "cleaned_meanflow_nfe1", "cleaned_meanflow_nfe2"]


def arrow(metric: str) -> str:
    if metric not in HIGHER_BETTER:
        return ""
    return r"$\uparrow$" if HIGHER_BETTER[metric] else r"$\downarrow$"


def read_csv(path: Path):
    with path.open() as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


def _f(v):
    try:
        x = float(v)
        return x
    except (TypeError, ValueError):
        return np.nan


def clean_metric(name: str) -> str:
    return name.replace("↓", "").replace("↑", "").strip()


# ---------------------------------------------------------------------------
# 1. Scoreboard (per-metric bar grid; EDM diffusion + FlowCast K in {10,15,20})
# ---------------------------------------------------------------------------
def fig_scoreboard(results_root: Path, out: Path):
    header, data = read_csv(results_root / "scoreboard_3way.csv")
    metrics_all = [clean_metric(m) for m in header[1:]]
    # Drop the legacy row; this figure is the cleaned EDM-vs-FlowCast comparison.
    rows = [r for r in data if ts.canonical_method(r[0]) != "legacy"]
    methods = [r[0] for r in rows]
    labels = [ts.method_label(m) for m in methods]
    colors = [ts.method_color(m) for m in methods]

    metrics = [m for m in SCOREBOARD_METRICS if m in metrics_all]
    col_of = {m: metrics_all.index(m) for m in metrics}

    ncol = 5
    nrow = int(np.ceil(len(metrics) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.95 * ncol, 3.7 * nrow))
    axes = np.atleast_1d(axes).flatten()
    for ax, m in zip(axes, metrics):
        vals = np.array([_f(r[col_of[m] + 1]) for r in rows])
        bars = ax.bar(range(len(methods)), vals, color=colors,
                      edgecolor="black", linewidth=0.6)
        higher = HIGHER_BETTER.get(m, False)
        win = int(np.nanargmax(vals) if higher else np.nanargmin(vals))
        bars[win].set_edgecolor("#222222")
        bars[win].set_linewidth(2.2)
        bars[win].set_hatch("//")
        ax.set_title(f"{clean_metric(m)} {arrow(m)}")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        for k, v in enumerate(vals):
            if not np.isnan(v):
                ax.text(k, v, f"{v:.3g}", ha="center", va="bottom", fontsize=8)
        ax.margins(y=0.20)
        ax.grid(True, axis="y", alpha=0.25)
    for ax in axes[len(metrics):]:
        ax.axis("off")
    fig.suptitle("Three-way scoreboard: EDM diffusion vs. FlowCast vs. MeanFlow "
                 "(2022 validation; best per metric hatched)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 2/3. Per-channel grouped bars (RMSE, CRPS)
# ---------------------------------------------------------------------------
def fig_per_channel(results_root: Path, out: Path, fname: str, kind: str):
    header, data = read_csv(results_root / "cleaned_2M" / fname)
    channels = header[1:]
    order = {ts.canonical_method(m): i for i, m in enumerate(CLEANED_ORDER)}
    data = sorted(data, key=lambda r: order.get(ts.canonical_method(r[0]), 99))
    methods = [r[0] for r in data]
    values = np.array([[_f(v) for v in r[1:]] for r in data])

    fig, ax = plt.subplots(figsize=(1.55 * len(channels) + 2.5, 3.5))
    x = np.arange(len(channels))
    w = 0.8 / max(len(methods), 1)
    for i, m in enumerate(methods):
        ax.bar(x + (i - (len(methods) - 1) / 2) * w, values[i], width=w,
               label=ts.method_label(m), color=ts.method_color(m),
               edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(list(channels))
    ax.set_ylabel(f"{kind.upper()} $\\downarrow$")
    ax.set_title(f"Per-channel {kind.upper()} (physical units) $\\downarrow$")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 4. CSI / FAR / HSS vs threshold
# ---------------------------------------------------------------------------
def fig_csi_per_threshold(results_root: Path, out: Path):
    header, data = read_csv(results_root / "cleaned_2M" / "per_threshold.csv")
    thresholds = [float(t) for t in header[2:]]
    by_metric: dict[str, dict[str, list[float]]] = {}
    for r in data:
        by_metric.setdefault(r[1], {})[r[0]] = [_f(v) for v in r[2:]]

    order = {ts.canonical_method(m): i for i, m in enumerate(CLEANED_ORDER)}
    metrics = ["csi", "far", "hss"]
    metrics = [m for m in metrics if m in by_metric]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.3 * len(metrics), 3.8))
    axes = np.atleast_1d(axes)
    for ax, metric in zip(axes, metrics):
        series = sorted(by_metric[metric].items(),
                        key=lambda kv: order.get(ts.canonical_method(kv[0]), 99))
        for method, vals in series:
            ax.plot(thresholds, vals, marker="o", lw=1.7, ms=5,
                    label=ts.method_label(method), color=ts.method_color(method))
        ax.set_xscale("log")
        ax.set_xlabel(r"rainfall threshold $\tau$ (mm/h)")
        ax.set_ylabel(f"{metric.upper()} {arrow(metric)}")
        ax.set_title(f"{metric.upper()} vs. threshold {arrow(metric)}")
        ax.grid(True, alpha=0.3)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 5. FSS at >=16 mm/h vs neighbourhood size
# ---------------------------------------------------------------------------
def fig_fss(results_root: Path, out: Path):
    header, data = read_csv(results_root / "cleaned_2M" / "fss_p16.csv")
    scales = [int(s) for s in header[2:]]
    order = {ts.canonical_method(m): i for i, m in enumerate(CLEANED_ORDER)}
    data = sorted(data, key=lambda r: order.get(ts.canonical_method(r[0]), 99))
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    for r in data:
        vals = [_f(v) for v in r[2:]]
        ax.plot(scales, vals, marker="o", lw=1.7, ms=6,
                label=ts.method_label(r[0]), color=ts.method_color(r[0]))
    ax.set_xticks(scales)
    ax.set_xlabel(r"neighbourhood size $n$ (pixels)")
    ax.set_ylabel(r"FSS at $\geq 16$ mm/h $\uparrow$")
    ax.set_title(r"Heavy-rain FSS ($\tau = 16$ mm/h) $\uparrow$")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 6. Rollout RMSE vs lead time (per channel; legacy + EDM + FlowCast)
# ---------------------------------------------------------------------------
def fig_rollout(results_root: Path, out: Path):
    src = results_root / "rmse_per_step_3way.csv"
    header, data = read_csv(src)
    channels = header[2:]  # u10, v10, t2m, qpepre
    by_method: dict[str, dict[str, list[tuple[int, float]]]] = {}
    for r in data:
        m, s = r[0], int(r[1])
        for j, ch in enumerate(channels):
            by_method.setdefault(m, {}).setdefault(ch, []).append((s, _f(r[2 + j])))

    order = {ts.canonical_method(m): i for i, m in enumerate(ROLLOUT_ORDER)}
    methods = sorted(by_method, key=lambda m: order.get(ts.canonical_method(m), 99))

    fig, axes = plt.subplots(1, len(channels), figsize=(4.0 * len(channels), 3.7))
    axes = np.atleast_1d(axes)
    for ax, ch in zip(axes, channels):
        all_vals = []
        for m in methods:
            sv = sorted(by_method[m][ch])
            xs = [s for s, _ in sv]
            ys = [v for _, v in sv]
            all_vals.extend(v for v in ys if not np.isnan(v))
            ax.plot(xs, ys, marker="o", lw=1.7, ms=4,
                    label=ts.method_label(m), color=ts.method_color(m))
        ax.set_xlabel("lead time (h)")
        ax.set_ylabel(r"RMSE $\downarrow$")
        ax.set_title(f"{ch} $\\downarrow$")
        ax.grid(True, alpha=0.3)
        ax.margins(x=0.02)
        # Clip transient blow-ups (e.g. the legacy qpepre spike at step 9) so the
        # readable region is usable; annotate the clipped maximum.
        if all_vals:
            arr = np.asarray(all_vals)
            p95 = float(np.percentile(arr, 95))
            vmax = float(arr.max())
            if vmax > 2.5 * p95 and p95 > 0:
                ax.set_ylim(0, 1.25 * p95)
                ax.text(0.98, 0.96, f"max {vmax:.2f} clipped", transform=ax.transAxes,
                        ha="right", va="top", fontsize=8, color="0.4",
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                                  alpha=0.7, edgecolor="0.7"))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(methods), 5),
               frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(r"Per-channel RMSE vs. autoregressive lead time $\downarrow$",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 7. NFE Pareto sweep (FlowCast: quality + latency vs Euler step count)
# ---------------------------------------------------------------------------
def fig_nfe(results_root: Path, out: Path):
    """Quality-saturation + latency-linearity figure from nfe_sweep.csv.

    Returns the path, or None if the sweep CSV has not been produced yet
    (run flowcast_nfe_sweep.py / run_flowcast_nfe_sweep.sh first).
    """
    src = results_root.parent / "flowcast_nfe_sweep" / "nfe_sweep.csv"
    if not src.exists():
        print(f"[make_thesis_figures] skip nfe_pareto (no {src})")
        return None

    def _load_sweep(path):
        header, data = read_csv(path)
        col = {h: i for i, h in enumerate(header)}
        return (
            np.array([float(r[col["nfe"]]) for r in data]),
            np.array([_f(r[col["crps_qpepre"]]) for r in data]),
            np.array([_f(r[col["time_per_seq_s"]]) for r in data]),
        )

    nfes, crps_qp, times = _load_sweep(src)

    # MeanFlow sweep (average-velocity sampler) overlaid when present — the
    # headline novelty: it reaches FlowCast's CRPS at 1–2 NFE instead of ~10.
    mf_src = results_root.parent / "meanflow_nfe_sweep" / "nfe_sweep.csv"
    mf = _load_sweep(mf_src) if mf_src.exists() else None

    fc_color = ts.method_color("flowcast")
    mf_color = ts.method_color("meanflow")
    # The K used by Table 4.1's matched-budget rows; shade it for context. The
    # absolute EDM number is on a different (24-seq) sample set than this sweep,
    # so we do NOT draw an EDM line here — the head-to-head lives in Table 4.1.
    band_color = ts.method_color("diffusion")

    fig, (axq, axt) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    # (a) quality vs NFE
    axq.axvspan(10, 20, color=band_color, alpha=0.12, label="Table 4.1 range ($K{=}10$–$20$)")
    axq.plot(nfes, crps_qp, "o-", lw=1.8, ms=5, color=fc_color, label="FlowCast")
    if mf is not None:
        axq.plot(mf[0], mf[1], "s--", lw=1.8, ms=5, color=mf_color, label="MeanFlow")
    axq.set_xscale("log")
    axq.set_xlabel("NFE (sampler steps $K$)")
    axq.set_ylabel(r"CRPS qpepre (mm/h) $\downarrow$")
    axq.set_title(r"Quality saturates in a few steps $\downarrow$")
    axq.set_xticks(nfes)
    axq.set_xticklabels([str(int(n)) for n in nfes], fontsize=7)
    axq.grid(True, which="both", alpha=0.3)
    axq.legend(frameon=False, fontsize=9)
    # (b) latency vs NFE (log-log) + linear reference
    axt.plot(nfes, times, "o-", lw=1.8, ms=5, color=fc_color, label="FlowCast")
    if mf is not None:
        axt.plot(mf[0], mf[2], "s--", lw=1.8, ms=5, color=mf_color, label="MeanFlow")
    # Anchor the linear-in-NFE reference at the highest-NFE point (per-step cost
    # is most reliable there; the lowest NFEs carry CUDA-warmup / fixed overhead).
    ref = times[-1] * nfes / nfes[-1]
    axt.plot(nfes, ref, ":", lw=1.4, color="0.5", label="linear in NFE")
    axt.set_xscale("log"); axt.set_yscale("log")
    axt.set_xlabel("NFE (sampler steps $K$)")
    axt.set_ylabel("wall-clock per sequence (s)")
    axt.set_title("Latency is linear in NFE")
    axt.set_xticks(nfes)
    axt.set_xticklabels([str(int(n)) for n in nfes], fontsize=7)
    axt.grid(True, which="both", alpha=0.3)
    axt.legend(frameon=False, fontsize=9)
    _title = ("FlowCast vs. MeanFlow NFE Pareto sweep" if mf is not None
              else "FlowCast NFE Pareto sweep")
    fig.suptitle(f"{_title} (matched ~2M checkpoint, 2022 validation)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS,
                    help="main_experiment results dir holding the CSVs.")
    ap.add_argument("--fig-out", type=Path, default=DEFAULT_FIGOUT,
                    help="Directory to write the thesis PNGs into.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Write into results/_thesis_figures_preview/ instead of the thesis tree.")
    args = ap.parse_args()

    figout = (args.results_root.parent / "_thesis_figures_preview") if args.dry_run else args.fig_out
    figout.mkdir(parents=True, exist_ok=True)
    rr = args.results_root

    produced = [
        fig_scoreboard(rr, figout / "scoreboard_3way.png"),
        fig_per_channel(rr, figout / "rmse_per_channel.png", "rmse_per_channel.csv", "rmse"),
        fig_per_channel(rr, figout / "crps_per_channel.png", "crps_per_channel.csv", "crps"),
        fig_csi_per_threshold(rr, figout / "csi_per_threshold.png"),
        fig_fss(rr, figout / "fss_p16.png"),
        fig_rollout(rr, figout / "rollout_rmse.png"),
    ]
    nfe = fig_nfe(rr, figout / "nfe_pareto.png")
    if nfe is not None:
        produced.append(nfe)
    print(f"[make_thesis_figures] wrote {len(produced)} figures to {figout}")
    for p in produced:
        print("  ", p.name)


if __name__ == "__main__":
    main()
