"""Render results/main_experiment/ tables and metric plots to PNG via LaTeX.

Tables go through pdflatex (booktabs, math-mode arrows). Plots use matplotlib
mathtext for the same arrow glyphs in axis labels / titles.

Outputs land in results/main_experiment/figures/. Intermediates (.tex/.pdf/.aux/.log)
stay alongside the PNG so the LaTeX can be hand-tweaked later.
"""
from __future__ import annotations

import csv
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent / "results" / "main_experiment"
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Metric direction registry: True = higher is better (↑), False = lower (↓)
# ---------------------------------------------------------------------------
HIGHER_BETTER = {
    "Time/Seq.(s)": False,
    "CRPS": False,
    "CSI-M": True,
    "CSI-P16": True,
    "FSS-P16-M": True,
    "HSS-M": True,
    "FAR-M": False,
    "RMSE_u10": False,
    "RMSE_v10": False,
    "RMSE_t2m": False,
    "RMSE_qpepre": False,
    "csi": True,
    "far": False,
    "hss": True,
    "fss_p16": True,
    "u10": False,  # context: per-channel RMSE/CRPS — both lower-better
    "v10": False,
    "t2m": False,
    "qpepre": False,
}


def arrow_tex(metric: str) -> str:
    if metric not in HIGHER_BETTER:
        return ""
    return r"$\uparrow$" if HIGHER_BETTER[metric] else r"$\downarrow$"


def arrow_plain(metric: str) -> str:
    if metric not in HIGHER_BETTER:
        return ""
    return r"$\uparrow$" if HIGHER_BETTER[metric] else r"$\downarrow$"


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------
LATEX_DOC = r"""\documentclass[border=10pt]{standalone}
\usepackage{booktabs}
\usepackage{amsmath,amssymb}
\usepackage{array}
\renewcommand{\arraystretch}{1.15}
\begin{document}
%(body)s
\end{document}
"""


def compile_tex(name: str, body: str) -> Path:
    tex_path = OUT / f"{name}.tex"
    tex_path.write_text(LATEX_DOC % {"body": body})
    res = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=OUT,
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        raise RuntimeError(f"pdflatex failed for {name}:\n{res.stdout[-2000:]}")
    pdf_path = OUT / f"{name}.pdf"
    subprocess.run(
        ["pdftoppm", "-r", "300", "-png", "-singlefile", pdf_path.name, name],
        cwd=OUT,
        check=True,
    )
    # Clean up only the noisy aux/log; keep .tex and .pdf for re-tweaking.
    for ext in (".aux", ".log"):
        p = OUT / f"{name}{ext}"
        if p.exists():
            p.unlink()
    return OUT / f"{name}.png"


def fmt(v, decimals=4):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if np.isnan(f):
        return "--"
    return f"{f:.{decimals}f}"


def best_idx(values, higher_better):
    """Index of the winning row, ignoring NaN."""
    nums = []
    for v in values:
        try:
            f = float(v)
            if np.isnan(f):
                f = -np.inf if higher_better else np.inf
        except (TypeError, ValueError):
            f = -np.inf if higher_better else np.inf
        nums.append(f)
    return int(np.argmax(nums) if higher_better else np.argmin(nums))


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def read_csv(path: Path):
    with path.open() as f:
        return list(csv.reader(f))


def load_3way():
    rows = read_csv(ROOT / "scoreboard_3way.csv")
    header, *data = rows
    return header, data


# ---------------------------------------------------------------------------
# Table renderers
# ---------------------------------------------------------------------------

# Metric columns to drop from the headline 3-way table (kept in the underlying
# CSV, just not shown in this figure). Per user, FAR/HSS/FSS/CSI-P16 are noisy
# at the heavy-rain tail given the current sample size — keep CRPS, CSI-M, and
# the four RMSE columns for the headline read.
DROPPED_3WAY_METRICS = {"CSI-P16", "FSS-P16-M", "HSS-M", "FAR-M"}


def parse_run_config(log_path: Path) -> dict:
    """Extract (n_seqs, n_steps, ensemble) from a per-leg run log."""
    import re
    out = {"n_seqs": "?", "n_steps": "?", "ensemble": "?"}
    if not log_path.exists():
        return out
    text = log_path.read_text(errors="ignore")
    m = re.search(r"\[seqs\]\s*(\d+)\s*sequences\s*x\s*(\d+)\s*steps", text)
    if m:
        out["n_seqs"] = m.group(1)
        out["n_steps"] = m.group(2)
    m = re.search(r"K=(\d+)\s*x\s*S=\d+\s*x\s*T=\d+", text)
    if m:
        out["ensemble"] = m.group(1)
    return out


def render_3way_table():
    header, data = load_3way()
    metric_cols_raw = header[1:]

    clean_metrics_all = [m.replace("↓", "").replace("↑", "").strip()
                         for m in metric_cols_raw]
    keep_idx = [i for i, m in enumerate(clean_metrics_all)
                if m not in DROPPED_3WAY_METRICS]
    clean_metrics = [clean_metrics_all[i] for i in keep_idx]

    body_cols = ["l"] + ["c"] * len(clean_metrics)
    n_cols = len(clean_metrics) + 1

    head_cells = [r"\textbf{Method}"]
    for m in clean_metrics:
        a = arrow_tex(m)
        m_disp = m.replace("_", r"\_")
        head_cells.append(rf"\textbf{{{m_disp}}} {a}".strip())

    # Winner per kept column (using the original CSV index).
    winners = []
    for j, m in enumerate(clean_metrics):
        orig_idx = keep_idx[j]
        col_vals = [r[orig_idx + 1] for r in data]
        winners.append(best_idx(col_vals, HIGHER_BETTER.get(m, False)))

    lines = []
    for i, row in enumerate(data):
        cells = [display_method(row[0])]
        for j, orig_idx in enumerate(keep_idx):
            txt = fmt(row[orig_idx + 1], 4)
            if winners[j] == i:
                txt = rf"\textbf{{{txt}}}"
            cells.append(txt)
        lines.append(" & ".join(cells) + r" \\")

    cfg = parse_run_config(ROOT / "cleaned_2M.log")
    setup_line = (
        rf"\multicolumn{{{n_cols}}}{{l}}{{\footnotesize "
        rf"\textbf{{Setup:}} {cfg['n_seqs']} sequences $\times$ "
        rf"{cfg['n_steps']}\,h rollout $\times$ {cfg['ensemble']}-member ensemble"
        r"} \\"
    )
    training_line = (
        rf"\multicolumn{{{n_cols}}}{{l}}{{\footnotesize "
        r"\textbf{Training:} cleaned EDM and I-CFM at $\sim$2\,M samples "
        r"(matched budget); legacy from upstream NVIDIA checkpoint"
        r"} \\"
    )
    nfe_line = (
        rf"\multicolumn{{{n_cols}}}{{l}}{{\footnotesize "
        r"\textbf{NFE:} EDM = 18 Heun steps (NFE\,=\,36) "
        r"\quad I-CFM = Euler (NFE per row label)"
        r"} \\"
    )

    body = (
        rf"\begin{{tabular}}{{{''.join(body_cols)}}}" + "\n"
        r"\toprule" + "\n"
        + " & ".join(head_cells) + r" \\" + "\n"
        + r"\midrule" + "\n"
        + "\n".join(lines) + "\n"
        + r"\bottomrule" + "\n"
        + r"\addlinespace[3pt]" + "\n"
        + setup_line + "\n"
        + training_line + "\n"
        + nfe_line + "\n"
        + r"\end{tabular}"
    )
    return compile_tex("scoreboard_3way", body)


def render_per_threshold(leg: str):
    rows = read_csv(ROOT / leg / "per_threshold.csv")
    header, *data = rows
    thresholds = header[2:]
    n_cols = len(thresholds)

    head_cells = [r"\textbf{Method}", r"\textbf{Metric}"]
    for t in thresholds:
        head_cells.append(rf"$\geq {t}$ mm/h")

    # For best-row markup: group by metric, bold winner across method rows for each threshold col
    # data rows: [method, metric, v1, v2, ...]
    # group by metric
    by_metric = {}
    for r in data:
        by_metric.setdefault(r[1], []).append(r)

    lines = []
    for metric, rs in by_metric.items():
        higher = HIGHER_BETTER.get(metric, True)
        # winners per threshold column
        winners = []
        for k in range(n_cols):
            col = [r[2 + k] for r in rs]
            winners.append(best_idx(col, higher))
        for i, r in enumerate(rs):
            cells = [display_method(r[0], leg=leg), rf"{metric} {arrow_tex(metric)}".strip()]
            for k in range(n_cols):
                txt = fmt(r[2 + k], 4)
                if winners[k] == i:
                    txt = rf"\textbf{{{txt}}}"
                cells.append(txt)
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    if lines and lines[-1] == r"\midrule":
        lines.pop()  # don't end with a midrule

    body = (
        rf"\begin{{tabular}}{{ll{'c' * n_cols}}}" + "\n"
        r"\toprule" + "\n"
        + " & ".join(head_cells) + r" \\" + "\n"
        + r"\midrule" + "\n"
        + "\n".join(lines) + "\n"
        + r"\bottomrule" + "\n"
        + r"\end{tabular}"
    )
    return compile_tex(f"{leg}_per_threshold_table", body)


def render_per_channel_csv(leg: str, fname: str, kind: str):
    """RMSE / CRPS per channel — rows are methods, columns are channels."""
    rows = read_csv(ROOT / leg / fname)
    header, *data = rows
    channels = header[1:]
    n = len(channels)
    head_cells = [r"\textbf{Method}"]
    metric_label = kind.upper()
    for ch in channels:
        head_cells.append(rf"\textbf{{{ch}}} $\downarrow$")

    # winners per column (kind=rmse/crps both lower-better)
    winners = []
    for k in range(n):
        winners.append(best_idx([r[1 + k] for r in data], higher_better=False))

    lines = []
    for i, r in enumerate(data):
        cells = [display_method(r[0], leg=leg)]
        for k in range(n):
            txt = fmt(r[1 + k], 4)
            if winners[k] == i:
                txt = rf"\textbf{{{txt}}}"
            cells.append(txt)
        lines.append(" & ".join(cells) + r" \\")

    body = (
        rf"\begin{{tabular}}{{l{'c' * n}}}" + "\n"
        r"\toprule" + "\n"
        + rf"\multicolumn{{{n + 1}}}{{c}}{{\textbf{{{metric_label} per channel}} $\downarrow$}} \\" + "\n"
        + r"\midrule" + "\n"
        + " & ".join(head_cells) + r" \\" + "\n"
        + r"\midrule" + "\n"
        + "\n".join(lines) + "\n"
        + r"\bottomrule" + "\n"
        + r"\end{tabular}"
    )
    return compile_tex(f"{leg}_{kind}_per_channel_table", body)


def render_fss(leg: str):
    rows = read_csv(ROOT / leg / "fss_p16.csv")
    header, *data = rows
    scales = header[2:]
    n = len(scales)
    head_cells = [r"\textbf{Method}", r"\textbf{Metric}"]
    for s in scales:
        head_cells.append(rf"$n={s}$")

    winners = []
    for k in range(n):
        winners.append(best_idx([r[2 + k] for r in data], higher_better=True))

    lines = []
    for i, r in enumerate(data):
        metric_label = r[1].replace("_", r"\_")
        cells = [display_method(r[0], leg=leg), rf"{metric_label} $\uparrow$"]
        for k in range(n):
            txt = fmt(r[2 + k], 4)
            if winners[k] == i:
                txt = rf"\textbf{{{txt}}}"
            cells.append(txt)
        lines.append(" & ".join(cells) + r" \\")

    body = (
        rf"\begin{{tabular}}{{ll{'c' * n}}}" + "\n"
        r"\toprule" + "\n"
        + rf"\multicolumn{{{n + 2}}}{{c}}{{\textbf{{FSS at $\geq 16$ mm/h, neighborhood size $n$}} $\uparrow$}} \\" + "\n"
        + r"\midrule" + "\n"
        + " & ".join(head_cells) + r" \\" + "\n"
        + r"\midrule" + "\n"
        + "\n".join(lines) + "\n"
        + r"\bottomrule" + "\n"
        + r"\end{tabular}"
    )
    return compile_tex(f"{leg}_fss_p16_table", body)


# ---------------------------------------------------------------------------
# Plots (matplotlib mathtext for arrow glyphs)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "mathtext.fontset": "cm",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

# Display labels: 3-way (now 5-way) scoreboard rows + per-leg rows. Keys are
# the raw CSV strings; values are what gets shown in tables/plots.
METHOD_LABELS = {
    "legacy_edm": r"stormcast (224$\times$128)",
    "cleaned_edm": r"stormcast (192$\times$96)",
    # Single-NFE legacy name
    "cleaned_flow": "I-CFM",
    # Multi-NFE rows produced by run_main_experiment.sh @ FLOWCAST_NFES="10 15 20"
    "cleaned_flow_nfe10": "I-CFM (10 NFE)",
    "cleaned_flow_nfe15": "I-CFM (15 NFE)",
    "cleaned_flow_nfe20": "I-CFM (20 NFE)",
    # Single-NFE legacy name
    "cleaned_meanflow": "MeanFlow",
    # Multi-NFE rows produced by run_main_experiment.sh @ MEANFLOW_NFES="1 2"
    "cleaned_meanflow_nfe1": "MeanFlow (1 NFE)",
    "cleaned_meanflow_nfe2": "MeanFlow (2 NFE)",
}
# Per-leg CSVs use generic "diffusion" / "flowcast" / "flowcast_nfe<N>" — map by leg.
PER_LEG_LABELS = {
    "legacy": {
        "diffusion": r"stormcast (224$\times$128)",
        "flowcast": "I-CFM",
    },
    "cleaned_2M": {
        "diffusion": r"stormcast (192$\times$96)",
        "flowcast": "I-CFM",
        "flowcast_nfe10": "I-CFM (10 NFE)",
        "flowcast_nfe15": "I-CFM (15 NFE)",
        "flowcast_nfe20": "I-CFM (20 NFE)",
        "meanflow": "MeanFlow",
        "meanflow_nfe1": "MeanFlow (1 NFE)",
        "meanflow_nfe2": "MeanFlow (2 NFE)",
    },
}

METHOD_COLORS = {
    r"stormcast (224$\times$128)": "#7f7f7f",
    r"stormcast (192$\times$96)": "#1f77b4",
    "I-CFM": "#d62728",
    # Three reds increasing in saturation with NFE.
    "I-CFM (10 NFE)": "#fc9272",
    "I-CFM (15 NFE)": "#de2d26",
    "I-CFM (20 NFE)": "#a50f15",
    "MeanFlow": "#6a51a3",
    # Two purples increasing in saturation with NFE.
    "MeanFlow (1 NFE)": "#9e9ac8",
    "MeanFlow (2 NFE)": "#54278f",
}


def display_method(name: str, leg: str | None = None) -> str:
    if leg is not None and name in PER_LEG_LABELS.get(leg, {}):
        return PER_LEG_LABELS[leg][name]
    return METHOD_LABELS.get(name, name)


def plot_3way_bars():
    header, data = load_3way()
    metric_cols = [m.replace("↓", "").replace("↑", "").strip() for m in header[1:]]
    methods = [display_method(r[0]) for r in data]
    values = np.array([[float(v) for v in r[1:]] for r in data])

    n_metrics = len(metric_cols)
    n_methods = len(methods)
    n_cols = (n_metrics + 1) // 2
    fig, axes = plt.subplots(2, n_cols, figsize=(2.9 * n_cols, 7.4))
    axes = axes.flatten()
    for j, m in enumerate(metric_cols):
        ax = axes[j]
        higher = HIGHER_BETTER.get(m, False)
        bars = ax.bar(
            methods,
            values[:, j],
            color=[METHOD_COLORS.get(mt, "#444") for mt in methods],
            edgecolor="black",
            linewidth=0.6,
        )
        winner = int(np.argmax(values[:, j]) if higher else np.argmin(values[:, j]))
        bars[winner].set_edgecolor("gold")
        bars[winner].set_linewidth(2.0)
        arrow = r"$\uparrow$" if higher else r"$\downarrow$"
        ax.set_title(f"{m} {arrow}")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=8)
        for k, v in enumerate(values[:, j]):
            ax.text(k, v, f"{v:.3g}", ha="center", va="bottom", fontsize=8)
        ax.margins(y=0.18)
    for k in range(n_metrics, len(axes)):
        axes[k].axis("off")
    fig.suptitle("Main experiment — 3-way scoreboard (winner outlined in gold)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT / "scoreboard_3way_plot.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_per_threshold_curves(leg: str):
    rows = read_csv(ROOT / leg / "per_threshold.csv")
    header, *data = rows
    thresholds = [float(t) for t in header[2:]]
    by_metric = {}
    for r in data:
        by_metric.setdefault(r[1], {})[display_method(r[0], leg=leg)] = [
            float(v) if v not in ("nan", "") else np.nan for v in r[2:]
        ]

    metrics = list(by_metric)
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.8), sharex=True)
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        higher = HIGHER_BETTER.get(metric, True)
        arrow = r"$\uparrow$" if higher else r"$\downarrow$"
        for method, vals in by_metric[metric].items():
            ax.plot(
                thresholds, vals, marker="o", lw=1.6,
                label=method, color=METHOD_COLORS.get(method, None),
            )
        ax.set_xscale("log")
        ax.set_xlabel("threshold (mm/h)")
        ax.set_ylabel(f"{metric} {arrow}")
        ax.set_title(f"{metric.upper()} vs threshold {arrow}")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
    fig.suptitle(f"Per-threshold precip skill ({leg})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = OUT / f"{leg}_per_threshold_plot.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_per_channel_grouped(leg: str, fname: str, kind: str):
    rows = read_csv(ROOT / leg / fname)
    header, *data = rows
    channels = header[1:]
    methods = [display_method(r[0], leg=leg) for r in data]
    values = np.array([[float(v) for v in r[1:]] for r in data])

    fig, ax = plt.subplots(figsize=(1.4 * len(channels) + 2, 3.4))
    x = np.arange(len(channels))
    w = 0.8 / max(len(methods), 1)
    for i, m in enumerate(methods):
        ax.bar(
            x + (i - (len(methods) - 1) / 2) * w,
            values[i],
            width=w,
            label=m,
            color=METHOD_COLORS.get(m, None),
            edgecolor="black",
            linewidth=0.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(channels)
    ax.set_ylabel(f"{kind.upper()} $\\downarrow$")
    ax.set_title(f"{kind.upper()} per channel ({leg}) $\\downarrow$")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = OUT / f"{leg}_{kind}_per_channel_plot.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_rmse_per_step_3way():
    """Lead-time RMSE per channel — one line per method, one subplot per channel.

    Reads ``rmse_per_step_3way.csv`` (cols: method, step_h, u10, v10, t2m, qpepre).
    Returns the output PNG path, or None if the file is missing.
    """
    src = ROOT / "rmse_per_step_3way.csv"
    if not src.exists():
        return None
    rows = read_csv(src)
    header, *data = rows
    channels = header[2:]  # u10, v10, t2m, qpepre

    # method -> {channel -> list of (step_h, value)}
    by_method: dict[str, dict[str, list[tuple[int, float]]]] = {}
    for r in data:
        m = r[0]
        s = int(r[1])
        for j, ch in enumerate(channels):
            try:
                v = float(r[2 + j])
            except (TypeError, ValueError):
                v = np.nan
            by_method.setdefault(m, {}).setdefault(ch, []).append((s, v))

    methods = list(by_method)

    fig, axes = plt.subplots(1, len(channels), figsize=(4.0 * len(channels), 3.6))
    if len(channels) == 1:
        axes = [axes]
    for ax, ch in zip(axes, channels):
        all_vals: list[float] = []
        for m in methods:
            label = display_method(m)
            steps_vals = sorted(by_method[m][ch])
            xs = [s for s, _ in steps_vals]
            ys = [v for _, v in steps_vals]
            all_vals.extend(v for v in ys if not np.isnan(v))
            ax.plot(xs, ys, marker="o", lw=1.6, ms=4,
                    label=label, color=METHOD_COLORS.get(label, None))
        ax.set_xlabel("lead time (h)")
        ax.set_ylabel(r"RMSE $\downarrow$")
        ax.set_title(f"{ch} $\\downarrow$")
        ax.grid(True, alpha=0.3)
        ax.margins(x=0.02)
        # Robust y-cap: if the max blows past 2.5x the 95th percentile, clip
        # to that bound so transient blow-ups (e.g. one-sequence numerical
        # instabilities) don't make the readable region unusable. Outlier
        # lines simply leave the visible area, marked with a small annotation.
        if all_vals:
            arr = np.asarray(all_vals)
            p95 = float(np.percentile(arr, 95))
            vmax = float(arr.max())
            if vmax > 2.5 * p95 and p95 > 0:
                ax.set_ylim(0, 1.25 * p95)
                ax.text(0.98, 0.96, f"max {vmax:.2f} clipped",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=8, color="0.4",
                        bbox=dict(boxstyle="round,pad=0.25",
                                  facecolor="white", alpha=0.7,
                                  edgecolor="0.7"))
    # One legend below the row, to keep panels uncluttered.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(methods), 5),
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(r"Per-channel RMSE vs lead time (main experiment) $\downarrow$", fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    out = OUT / "rmse_per_step_3way.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_fss(leg: str):
    rows = read_csv(ROOT / leg / "fss_p16.csv")
    header, *data = rows
    scales = [int(s) for s in header[2:]]
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for r in data:
        method = display_method(r[0], leg=leg)
        vals = [float(v) for v in r[2:]]
        ax.plot(scales, vals, marker="o", lw=1.6, label=method,
                color=METHOD_COLORS.get(method, None))
    ax.set_xlabel("neighborhood size $n$ (pixels)")
    ax.set_ylabel(r"FSS at $\geq 16$ mm/h $\uparrow$")
    ax.set_title(r"FSS sweep ($\geq 16$ mm/h) " + f"({leg}) $\\uparrow$")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = OUT / f"{leg}_fss_p16_plot.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    produced = []
    produced.append(render_3way_table())
    produced.append(plot_3way_bars())
    p = plot_rmse_per_step_3way()
    if p is not None:
        produced.append(p)

    for leg in ("legacy", "cleaned_2M"):
        produced.append(render_per_threshold(leg))
        produced.append(render_per_channel_csv(leg, "rmse_per_channel.csv", "rmse"))
        produced.append(render_per_channel_csv(leg, "crps_per_channel.csv", "crps"))
        produced.append(render_fss(leg))
        produced.append(plot_per_threshold_curves(leg))
        produced.append(plot_per_channel_grouped(leg, "rmse_per_channel.csv", "rmse"))
        produced.append(plot_per_channel_grouped(leg, "crps_per_channel.csv", "crps"))
        produced.append(plot_fss(leg))

    print(f"Wrote {len(produced)} figures to {OUT}")
    for p in produced:
        print(" ", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
