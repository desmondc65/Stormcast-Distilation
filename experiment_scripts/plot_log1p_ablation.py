"""Render LaTeX figures for the log1p vs NO_log1p qpepre encoding ablation.

Reads pre-computed CSVs under experiment_scripts/results/log1p_ablation and
writes PNGs back to the same directory. PNGs are produced via matplotlib's
PGF backend driving pdflatex, so axis labels and legends are typeset by
LaTeX itself (no dvipng required).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("pgf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import _nbhd_io as nbio  # noqa: E402

plt.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "pgf.rcfonts": False,
        "pgf.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)

COLOR_LOG1P = "#1f77b4"
COLOR_RAW = "#d62728"
LABEL_LOG1P = r"\textsc{log1p}"
LABEL_RAW = r"\textsc{no\_log1p} (raw mm/h)"

CHANNEL_TEX = {
    "u10": r"$u_{10}$ [m/s]",
    "v10": r"$v_{10}$ [m/s]",
    "t2m": r"$T_{2\mathrm{m}}$ [K]",
    "qpepre": r"qpepre [mm/h]",
}


def plot_rmse_per_step(root: Path) -> Path:
    df = pd.read_csv(root / "rmse_per_step_log1p.csv")
    channels = ["u10", "v10", "t2m", "qpepre"]
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.0), sharex=True)
    for ax, ch in zip(axes.flat, channels):
        for method, color, label in [
            ("log1p_flowcast", COLOR_LOG1P, LABEL_LOG1P),
            ("NO_log1p_flowcast", COLOR_RAW, LABEL_RAW),
        ]:
            sub = df[df["method"] == method].sort_values("step_h")
            ax.plot(
                sub["step_h"],
                sub[ch],
                marker="o",
                markersize=4,
                linewidth=1.6,
                color=color,
                label=label,
            )
        ax.set_title(CHANNEL_TEX[ch])
        ax.set_ylabel(r"RMSE")
    for ax in axes[-1, :]:
        ax.set_xlabel(r"lead time [h]")
    axes[0, 0].legend(loc="best", framealpha=0.9)
    fig.suptitle(
        r"Per-channel RMSE vs.\ lead time --- "
        r"\textsc{log1p} vs.\ raw \textsc{mm/h} qpepre encoding",
        y=1.02,
    )
    fig.tight_layout()
    out = root / "rmse_per_step.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_rmse_bar(root: Path) -> Path:
    log1p = pd.read_csv(root / "log1p" / "rmse_per_channel.csv").iloc[0]
    raw = pd.read_csv(root / "NO_log1p" / "rmse_per_channel.csv").iloc[0]
    channels = ["u10", "v10", "t2m", "qpepre"]
    x = np.arange(len(channels))
    w = 0.38

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    b1 = ax.bar(
        x - w / 2,
        [log1p[c] for c in channels],
        width=w,
        color=COLOR_LOG1P,
        label=LABEL_LOG1P,
        edgecolor="black",
        linewidth=0.4,
    )
    b2 = ax.bar(
        x + w / 2,
        [raw[c] for c in channels],
        width=w,
        color=COLOR_RAW,
        label=LABEL_RAW,
        edgecolor="black",
        linewidth=0.4,
    )
    for bars in (b1, b2):
        for r in bars:
            ax.text(
                r.get_x() + r.get_width() / 2,
                r.get_height(),
                f"{r.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([CHANNEL_TEX[c] for c in channels])
    ax.set_ylabel(r"RMSE (mean over 2022 validation)")
    ax.set_title(r"Per-channel RMSE --- qpepre encoding ablation")
    ax.legend()
    ax.set_ylim(0, max(ax.get_ylim()[1], 2.1))
    fig.tight_layout()
    out = root / "rmse_per_channel.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_crps_bar(root: Path) -> Path:
    log1p = pd.read_csv(root / "log1p" / "crps_per_channel.csv").iloc[0]
    raw = pd.read_csv(root / "NO_log1p" / "crps_per_channel.csv").iloc[0]
    channels = ["u10", "v10", "t2m", "qpepre"]
    x = np.arange(len(channels))
    w = 0.38

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    b1 = ax.bar(
        x - w / 2,
        [log1p[c] for c in channels],
        width=w,
        color=COLOR_LOG1P,
        label=LABEL_LOG1P,
        edgecolor="black",
        linewidth=0.4,
    )
    b2 = ax.bar(
        x + w / 2,
        [raw[c] for c in channels],
        width=w,
        color=COLOR_RAW,
        label=LABEL_RAW,
        edgecolor="black",
        linewidth=0.4,
    )
    for bars in (b1, b2):
        for r in bars:
            ax.text(
                r.get_x() + r.get_width() / 2,
                r.get_height(),
                f"{r.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([CHANNEL_TEX[c] for c in channels])
    ax.set_ylabel(r"CRPS$\downarrow$")
    ax.set_title(r"Per-channel CRPS --- qpepre encoding ablation")
    ax.legend()
    fig.tight_layout()
    out = root / "crps_per_channel.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def _pick_metric(df: pd.DataFrame, metric: str) -> pd.Series:
    """One row for ``metric``; if the file carries a neighbourhood ``kernel``
    column, take the headline (0.25 deg / ERA5) kernel."""
    sub = df[df["metric"] == metric]
    if "kernel" in df.columns:
        sub = sub[sub["kernel"] == nbio.HEADLINE_KERNEL]
    return sub.iloc[0]


def plot_per_threshold(root: Path) -> Path:
    log1p = pd.read_csv(root / "log1p" / "per_threshold.csv")
    raw = pd.read_csv(root / "NO_log1p" / "per_threshold.csv")
    thresholds = ["0.1", "1.0", "5.0", "10.0", "16.0", "32.0"]
    xnum = np.array([float(t) for t in thresholds])

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6))
    titles = {
        "csi": r"CSI $\uparrow$",
        "hss": r"HSS $\uparrow$",
        "far": r"FAR $\downarrow$",
    }
    for ax, metric in zip(axes, ["csi", "hss", "far"]):
        for df, color, label in [
            (log1p, COLOR_LOG1P, LABEL_LOG1P),
            (raw, COLOR_RAW, LABEL_RAW),
        ]:
            row = _pick_metric(df, metric)
            y = np.array([row[t] for t in thresholds], dtype=float)
            ax.plot(
                xnum,
                y,
                marker="o",
                markersize=5,
                linewidth=1.6,
                color=color,
                label=label,
            )
        ax.set_xscale("log")
        ax.set_xticks(xnum)
        ax.set_xticklabels([t for t in thresholds])
        ax.set_xlabel(r"qpepre threshold [mm/h]")
        ax.set_title(titles[metric])
    axes[0].set_ylabel(r"score")
    axes[0].legend(loc="best", framealpha=0.9)
    fig.suptitle(
        r"qpepre threshold metrics --- CSI / HSS / FAR vs.\ threshold",
        y=1.04,
    )
    fig.tight_layout()
    out = root / "per_threshold.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def _load_fss(legdir: Path):
    """Return ``(mode, series)`` for one leg.

    New schema ``fss_nbhd.csv``: ``mode='kernel'``, series = FSS by threshold at
    the headline kernel. Legacy ``fss_p16.csv``: ``mode='window'``, series = FSS
    at p16 by neighbourhood half-width.
    """
    new = legdir / "fss_nbhd.csv"
    old = legdir / "fss_p16.csv"
    if new.exists():
        df = pd.read_csv(new)
        if "kernel" in df.columns:
            df = df[df["kernel"] == nbio.HEADLINE_KERNEL]
        return "kernel", df.iloc[0]
    return "window", pd.read_csv(old).iloc[0]


def plot_fss(root: Path) -> Path:
    mode_l, log1p = _load_fss(root / "log1p")
    mode_r, raw = _load_fss(root / "NO_log1p")

    if mode_l == "kernel":
        cols = ["0.1", "1.0", "5.0", "10.0", "16.0", "32.0"]
        xlabels = [rf"$\geq{c}$" for c in cols]
        xlabel = r"qpepre threshold [mm/h]"
        ylabel = r"FSS $\uparrow$"
        title = (r"Fractions Skill Score (qpepre, "
                 + nbio.kernel_pretty(nbio.HEADLINE_KERNEL) + r" kernel)")
    else:
        cols = ["3", "7", "15"]
        xlabels = [rf"${s}\!\times\!{s}$" for s in cols]
        xlabel = r"neighborhood window (pixels)"
        ylabel = r"FSS at p16 threshold $\uparrow$"
        title = r"Fractions Skill Score (qpepre, p16)"

    x = np.arange(len(cols))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.bar(
        x - w / 2,
        [float(log1p[c]) for c in cols],
        width=w,
        color=COLOR_LOG1P,
        label=LABEL_LOG1P,
        edgecolor="black",
        linewidth=0.4,
    )
    ax.bar(
        x + w / 2,
        [float(raw[c]) for c in cols],
        width=w,
        color=COLOR_RAW,
        label=LABEL_RAW,
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    out = root / "fss_p16.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_scoreboard(root: Path) -> Path:
    df = pd.read_csv(root / "scoreboard_log1p.csv")
    df = df.set_index("method")
    cols = [c for c in df.columns if c != "Time/Seq.(s)"]
    log1p_row = df.loc["log1p_flowcast", cols].astype(float)
    raw_row = df.loc["NO_log1p_flowcast", cols].astype(float)
    raw_rel = (raw_row - log1p_row) / log1p_row * 100.0
    # Sign-convention: positive = NO_log1p is worse. Flip for ↑-is-better metrics.
    signs = np.array([-1.0 if "↑" in c else 1.0 for c in cols])
    rel = raw_rel.values * signs

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in rel]
    bars = ax.bar(np.arange(len(cols)), rel, color=colors, edgecolor="black", linewidth=0.4)
    for r, v in zip(bars, rel):
        ax.text(
            r.get_x() + r.get_width() / 2,
            r.get_height() + (1.5 if v >= 0 else -1.5),
            f"{v:+.1f}\\%",
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=8,
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(cols)))
    tex_cols = [
        c.replace("↓", r"$\downarrow$").replace("↑", r"$\uparrow$").replace("_", r"\_")
        for c in cols
    ]
    ax.set_xticklabels(tex_cols, rotation=30, ha="right")
    ax.set_ylabel(r"sign-aligned change: \textsc{no\_log1p} vs.\ \textsc{log1p} [\%]")
    ax.set_title(
        r"Encoding ablation --- bars are sign-flipped where $\uparrow$ is better"
        "\n"
        r"so red/up $=$ \textsc{no\_log1p} worse, green/down $=$ \textsc{no\_log1p} better"
    )
    fig.tight_layout()
    out = root / "scoreboard_relative.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/home3/davidlcs/Econ-Rag/Local_LLM/test_meeting/"
            "Stormcast-Distilation/experiment_scripts/results/log1p_ablation"
        ),
    )
    args = parser.parse_args()

    outputs = [
        plot_rmse_per_step(args.root),
        plot_rmse_bar(args.root),
        plot_crps_bar(args.root),
        plot_per_threshold(args.root),
        plot_fss(args.root),
        plot_scoreboard(args.root),
    ]
    for p in outputs:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
