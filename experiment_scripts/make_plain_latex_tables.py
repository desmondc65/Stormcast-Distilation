#!/usr/bin/env python3
"""Render the main-experiment result tables as plain, standalone LaTeX + PNG.

Reads the canonical harness CSVs under ``results/main_experiment/`` and emits one
self-contained booktabs table per ``results.md`` section into
``results/plain_latex_tables/`` as both a compilable ``<name>.tex`` (``standalone``
document class -- copy the ``tabular`` straight into the thesis and add your own
``\\caption``) and a 300-dpi ``<name>.png`` render (``pdflatex`` -> ``pdftoppm``).

Tables produced:
    metrics_glossary      -- definitions / equations / direction for every metric
    scoreboard            -- headline: time, CRPS, per-channel RMSE (all methods)
    rmse_per_channel      -- cleaned per-channel RMSE
    crps_per_channel      -- cleaned per-channel CRPS
    precip_skill_window   -- CSI-M / HSS-M / FAR-M / FSS-P16 per pooling window
    csi_by_threshold      -- neighbourhood CSI vs rainfall threshold (cleaned)
    fss_by_threshold      -- FSS vs rainfall threshold (cleaned)
    rollout_rmse          -- per-channel RMSE vs autoregressive lead time

Best value per metric column is bolded in the wide tables (scoreboard / per
channel). No GPU; pure file reads + pdflatex.

Usage::

    python make_plain_latex_tables.py
"""
from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import numpy as np

import _nbhd_io as nbio

ROOT = Path(__file__).resolve().parent / "results" / "main_experiment"
OUT = Path(__file__).resolve().parent / "results" / "plain_latex_tables"
OUT.mkdir(parents=True, exist_ok=True)

# Direction registry: True = higher-is-better. Keyed by BASE metric name.
HIGHER_BETTER = {
    "Time/Seq.(s)": False, "CRPS": False, "CSI-M": True, "HSS-M": True,
    "FAR-M": False, "FSS-P16": True, "RMSE_u10": False, "RMSE_v10": False,
    "RMSE_t2m": False, "RMSE_qpepre": False, "csi": True, "far": False,
    "hss": True, "pod": True, "fss": True, "u10": False, "v10": False,
    "t2m": False, "qpepre": False,
}

# Display labels (LaTeX math for K subscripts / x). Handles both the stitched
# scoreboard_3way names and the per-leg cleaned_2M names.
LABELS = {
    "legacy_edm": r"Legacy EDM ($224{\times}128$)",
    "cleaned_edm": "EDM diffusion", "diffusion": "EDM diffusion",
    "cleaned_flow_nfe10": r"FlowCast ($K{=}10$)", "flowcast": r"FlowCast ($K{=}10$)",
    "cleaned_meanflow_nfe1": r"MeanFlow ($K{=}1$)", "meanflow_nfe1": r"MeanFlow ($K{=}1$)",
    "cleaned_meanflow_nfe2": r"MeanFlow ($K{=}2$)", "meanflow_nfe2": r"MeanFlow ($K{=}2$)",
}


def label(method: str) -> str:
    return LABELS.get(method, method.replace("_", r"\_"))


LATEX_DOC = r"""\documentclass[border=10pt]{standalone}
\usepackage{booktabs}
\usepackage{amsmath,amssymb}
\usepackage{array}
\renewcommand{\arraystretch}{1.2}
\begin{document}
%(body)s
\end{document}
"""


def compile_tex(name: str, body: str) -> Path:
    tex = OUT / f"{name}.tex"
    tex.write_text(LATEX_DOC % {"body": body})
    res = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name],
        cwd=OUT, capture_output=True, text=True,
    )
    if res.returncode != 0:
        raise RuntimeError(f"pdflatex failed for {name}:\n{res.stdout[-1500:]}")
    subprocess.run(
        ["pdftoppm", "-r", "300", "-png", "-singlefile", f"{name}.pdf", name],
        cwd=OUT, check=True,
    )
    for ext in (".aux", ".log"):
        p = OUT / f"{name}{ext}"
        if p.exists():
            p.unlink()
    print(f"  wrote {name}.tex + {name}.png")
    return OUT / f"{name}.png"


def read_csv(path: Path):
    with path.open() as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


def fnum(v, dec=4):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "--" if np.isnan(f) else f"{f:.{dec}f}"


def fval(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def bold_best(cells, raws, higher):
    """Bold the best (max if higher else min) entry of one column."""
    a = np.array(raws, float)
    if np.all(np.isnan(a)):
        return cells
    win = int(np.nanargmax(a) if higher else np.nanargmin(a))
    cells = list(cells)
    cells[win] = rf"\textbf{{{cells[win]}}}"
    return cells


def tabular(colspec, header_cells, body_rows):
    lines = [rf"\begin{{tabular}}{{{colspec}}}", r"\toprule",
             " & ".join(header_cells) + r" \\", r"\midrule"]
    # A body entry that is a bare string (e.g. r"\midrule") is emitted raw so
    # callers can insert inter-group rules; list entries are normal table rows.
    for r in body_rows:
        lines.append(r if isinstance(r, str) else " & ".join(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def base(name):
    return nbio.base_metric(name)


def arr(metric):
    b = base(metric)
    return "" if b not in HIGHER_BETTER else (
        r"$\uparrow$" if HIGHER_BETTER[b] else r"$\downarrow$")


# ---------------------------------------------------------------------------
def t_scoreboard():
    header, data = read_csv(ROOT / "scoreboard_3way.csv")
    want = ["Time/Seq.(s)", "CRPS", "RMSE_u10", "RMSE_v10", "RMSE_t2m", "RMSE_qpepre"]
    col_of = {base(h): i for i, h in enumerate(header)}
    disp = {"Time/Seq.(s)": "Time/seq (s)", "CRPS": "CRPS", "RMSE_u10": "RMSE u10",
            "RMSE_v10": "RMSE v10", "RMSE_t2m": "RMSE t2m", "RMSE_qpepre": "RMSE qpepre"}

    # Speedup vs the cleaned EDM-diffusion baseline (Time/seq ratio). The legacy
    # EDM runs on a larger 224x128 grid, so its wall-clock isn't comparable on the
    # same footing -- dash it rather than print a misleading ratio.
    methods = [r[0] for r in data]
    times = [fval(r[col_of["Time/Seq.(s)"]]) for r in data]
    base_method = next((m for m in ("cleaned_edm", "diffusion") if m in methods), None)
    base_time = times[methods.index(base_method)] if base_method else np.nan
    DIFF_DOMAIN = {"legacy_edm"}  # different grid -> speedup not comparable
    sp_raw, sp_cell = [], []
    for m, t in zip(methods, times):
        if m in DIFF_DOMAIN or np.isnan(base_time) or np.isnan(t) or t == 0:
            sp_raw.append(np.nan)
            sp_cell.append("--")
        else:
            s = base_time / t
            sp_raw.append(s)
            sp_cell.append(rf"{s:.1f}$\times$")
    sp_cell = bold_best(sp_cell, sp_raw, True)

    head = ["Method", f"{disp['Time/Seq.(s)']} {arr('Time/Seq.(s)')}",
            r"Speedup vs EDM $\uparrow$"] + [f"{disp[w]} {arr(w)}" for w in want[1:]]
    cols, raws = {}, {}
    for w in want:
        cols[w] = [fnum(r[col_of[w]]) for r in data]
        raws[w] = [fval(r[col_of[w]]) for r in data]
        cols[w] = bold_best(cols[w], raws[w], HIGHER_BETTER[base(w)])
    body = [[label(data[i][0]), cols["Time/Seq.(s)"][i], sp_cell[i]]
            + [cols[w][i] for w in want[1:]] for i in range(len(data))]
    compile_tex("scoreboard", tabular("l" + "c" * (len(want) + 1), head, body))


def t_per_channel(fname, name):
    header, data = read_csv(ROOT / "cleaned_2M" / fname)
    chans = header[1:]
    cols, raws = {}, {}
    for j, c in enumerate(chans):
        cols[c] = [fnum(r[1 + j]) for r in data]
        raws[c] = [fval(r[1 + j]) for r in data]
        cols[c] = bold_best(cols[c], raws[c], False)
    head = ["Method"] + [f"{c} $\\downarrow$" for c in chans]
    body = [[label(data[i][0])] + [cols[c][i] for c in chans] for i in range(len(data))]
    compile_tex(name, tabular("l" + "c" * len(chans), head, body))


def t_precip_skill_window():
    header, data = read_csv(ROOT / "scoreboard_3way.csv")
    # Key on the full "<metric>@<window>" (arrows stripped) so per-window columns
    # don't collapse — base() would strip the @window suffix and alias them.
    col = {h.replace("↑", "").replace("↓", "").strip(): i
           for i, h in enumerate(header)}
    metrics = ["CSI-M", "HSS-M", "FAR-M", "FSS-P16"]
    windows = nbio.KERNELS
    head = ["Method", "Window"] + [f"{m} {arr(m)}" for m in metrics]
    body = []
    for r in data:
        for wi, w in enumerate(windows):
            cells = [label(r[0]) if wi == 0 else "", nbio.kernel_pretty(w)]
            for m in metrics:
                idx = col.get(f"{m}@{w}")
                cells.append(fnum(r[idx]) if idx is not None else "--")
            body.append(cells)
    compile_tex("precip_skill_window", tabular("ll" + "c" * len(metrics), head, body))


def t_by_threshold(metric, name):
    thresholds, recs = nbio.read_per_threshold(ROOT / "cleaned_2M" / "per_threshold.csv")
    windows = nbio.kernels_present(recs)
    head = ["Method", "Window"] + [f"$\\geq${t} mm/h" for t in thresholds]
    rows = [r for r in recs if r["metric"] == metric]
    methods = []
    for r in rows:
        if r["method"] not in methods:
            methods.append(r["method"])
    body = []
    for m in methods:
        for wi, w in enumerate(windows):
            rec = next((r for r in rows if r["method"] == m and r["kernel"] == w), None)
            if rec is None:
                continue
            body.append([label(m) if wi == 0 else "", nbio.kernel_pretty(w)]
                        + [fnum(v) for v in rec["values"]])
    compile_tex(name, tabular("ll" + "c" * len(thresholds), head, body))


# Method labels for the timing CSV (its own method names: stormcast_edm,
# flowcast_nfe10/15/20, meanflow_nfe1/2). Kept separate from LABELS so the
# FlowCast K is read from the timing row, not pinned to the main-experiment K.
TIMING_LABELS = {
    "stormcast_edm": "EDM diffusion",
    "flowcast_nfe10": r"FlowCast ($K{=}10$)", "flowcast_nfe15": r"FlowCast ($K{=}15$)",
    "flowcast_nfe20": r"FlowCast ($K{=}20$)",
    "meanflow_nfe1": r"MeanFlow ($K{=}1$)", "meanflow_nfe2": r"MeanFlow ($K{=}2$)",
}
TIMING_ORDER = ["stormcast_edm", "flowcast_nfe10", "flowcast_nfe15",
                "flowcast_nfe20", "meanflow_nfe1", "meanflow_nfe2"]


def _timing_devices():
    """Discover per-device timing CSVs under results/timing/.

    A6000 := a6000.csv if present else the generic timing.csv (the harness writes
    timing.csv and run_timing.sh defaults to one A6000). H100 := h100.csv.
    Returns an ordered list of (device_label, path) for files that exist.
    """
    tdir = ROOT.parent / "timing"
    devs = []
    a = tdir / "a6000.csv"
    a = a if a.exists() else tdir / "timing.csv"
    if a.exists():
        devs.append(("A6000", a))
    if (tdir / "h100.csv").exists():
        devs.append(("H100", tdir / "h100.csv"))
    return devs


def t_timing():
    devs = _timing_devices()
    if not devs:
        print("  skip timing (no results/timing/*.csv)")
        return
    # Load each device CSV: {method: row dict}.
    perdev = {}
    nfe = {}
    for dev, path in devs:
        with path.open() as f:
            rows = {r["method"]: r for r in csv.DictReader(f)}
        perdev[dev] = rows
        for m, r in rows.items():
            nfe[m] = r.get("nfe_per_step", "")
    methods = [m for m in TIMING_ORDER if any(m in perdev[d] for d, _ in devs)]
    methods += [m for d, _ in devs for m in perdev[d] if m not in methods]

    def tl(m):
        return TIMING_LABELS.get(m, m.replace("_", r"\_"))

    if len(devs) == 1:
        dev = devs[0][0]
        head = ["Method", "NFE/step",
                rf"{dev}: s/forecast $\downarrow$", r"Speedup vs EDM $\uparrow$",
                "s/member $\\downarrow$", "s/fcst-hr $\\downarrow$", "s/NFE $\\downarrow$"]
        rows = perdev[dev]
        present = [m for m in methods if m in rows]
        f_fc = [fval(rows[m]["sec_per_ensemble_forecast"]) for m in present]
        # Speedup of every method vs the EDM diffusion baseline (s/forecast).
        base_fc = fval(rows.get("stormcast_edm", {}).get(
            "sec_per_ensemble_forecast", np.nanmax(f_fc)))
        speedups = [base_fc / fval(rows[m]["sec_per_ensemble_forecast"]) for m in present]
        body = []
        for m, sp in zip(present, speedups):
            r = rows[m]
            body.append([tl(m), nfe.get(m, ""),
                         f'{float(r["sec_per_ensemble_forecast"]):.1f}',
                         rf'{sp:.1f}$\times$',
                         f'{float(r["sec_per_member_rollout"]):.2f}',
                         f'{float(r["sec_per_forecast_hour"]):.3f}',
                         f'{float(r["sec_per_nfe"]):.4f}'])
        # bold the fastest (min s/forecast) and the largest speedup (same row).
        win = int(np.nanargmin(np.array(f_fc, float)))
        body[win][2] = rf"\textbf{{{body[win][2]}}}"
        body[win][3] = rf"\textbf{{{body[win][3]}}}"
        compile_tex("timing", tabular("llccccc", head, body))
    else:
        # Two (or more) devices: per-device s/forecast + s/member, plus speedup
        # of the last device vs the first on s/forecast.
        d0, dN = devs[0][0], devs[-1][0]
        # EDM baseline s/forecast on the deployment (last) device, for the
        # within-device "improvement vs baseline" column.
        base_dN = fval(perdev[dN].get("stormcast_edm", {}).get(
            "sec_per_ensemble_forecast", np.nan))
        head = ["Method", "NFE/step"]
        for dev, _ in devs:
            head += [rf"{dev} s/fcst $\downarrow$", rf"{dev} s/mem $\downarrow$"]
        head += [rf"{dN} vs EDM $\uparrow$", rf"{d0}/{dN} $\uparrow$"]
        body = []
        for m in methods:
            cells = [tl(m), nfe.get(m, "")]
            f0 = fN = np.nan
            for di, (dev, _) in enumerate(devs):
                r = perdev[dev].get(m)
                if r is None:
                    cells += ["--", "--"]
                    continue
                fc = float(r["sec_per_ensemble_forecast"])
                cells += [f"{fc:.1f}", f'{float(r["sec_per_member_rollout"]):.2f}']
                if di == 0:
                    f0 = fc
                if di == len(devs) - 1:
                    fN = fc
            cells.append(rf"{base_dN / fN:.1f}$\times$"
                         if (fN and not np.isnan(base_dN) and not np.isnan(fN)) else "--")
            cells.append(f"{f0 / fN:.2f}x" if (fN and not np.isnan(f0) and not np.isnan(fN)) else "--")
            body.append(cells)
        colspec = "ll" + "cc" * len(devs) + "cc"
        compile_tex("timing", tabular(colspec, head, body))
    print(f"  timing devices: {', '.join(d for d, _ in devs)}")


# qpepre RMSE (mm/h) above this is a corrupt rollout sample: flag it, and keep it
# out of the per-hour best-value bolding so it can never count as the winner.
ROLLOUT_QP_OUTLIER = 5.0


def t_rollout():
    header, data = read_csv(ROOT / "rmse_per_step_3way.csv")
    chans = header[2:]
    leads = sorted({int(r[1]) for r in data})
    methods = []
    for r in data:
        if r[0] not in methods:
            methods.append(r[0])
    head = ["Lead (h)", "Method"] + [f"{c} $\\downarrow$" for c in chans]
    body, note_dagger, note_ddagger = [], False, False
    # Group by lead time and bold the best (min) model per channel within each
    # hour, so the table reads "who wins at each lead" down the columns.
    for li, lead in enumerate(leads):
        lrows = sorted((r for r in data if int(r[1]) == lead),
                       key=lambda r: methods.index(r[0]))
        cols = {}
        for j, c in enumerate(chans):
            cells = [fnum(r[2 + j]) for r in lrows]
            raws = [fval(r[2 + j]) for r in lrows]
            if c == "qpepre":
                for i, v in enumerate(raws):
                    if v > ROLLOUT_QP_OUTLIER:
                        raws[i] = np.nan            # never the per-hour winner
                        cells[i] += r"$^\ddagger$"
                        note_ddagger = True
            cells = bold_best(cells, raws, False)
            if c == "qpepre":
                # Legacy EDM precip is on a different grid/encoding -> mark its
                # qpepre as not-directly-comparable even when it's the min.
                for i, r in enumerate(lrows):
                    if r[0] == "legacy_edm" and r"\ddagger" not in cells[i]:
                        cells[i] += r"$^\dagger$"
                        note_dagger = True
            cols[c] = cells
        for i, r in enumerate(lrows):
            body.append([str(lead) if i == 0 else "", label(r[0])]
                        + [cols[c][i] for c in chans])
        if li < len(leads) - 1:
            body.append(r"\midrule")
    tbl = tabular("ll" + "c" * len(chans), head, body)
    notes = []
    if note_dagger:
        notes.append(r"$\dagger$~Legacy EDM qpepre is scored on the legacy "
                     r"$224{\times}128$ grid (raw mm\,h$^{-1}$ encoding) and is "
                     r"\emph{not} directly comparable to the cleaned "
                     r"$192{\times}96$ runs.")
    if note_ddagger:
        notes.append(r"$\ddagger$~Corrupt sample; excluded from the best-value "
                     r"bolding.")
    if notes:
        # standalone typesets the body as a single horizontal box, so a trailing
        # paragraph would land beside the table. Stack table + note as two rows
        # of a one-column outer tabular to force the note underneath.
        tbl = ("\\begin{tabular}{@{}l@{}}\n"
               + tbl + " \\\\[6pt]\n"
               + r"\begin{minipage}{11cm}\footnotesize " + " ".join(notes)
               + "\\end{minipage} \\\\\n\\end{tabular}")
    compile_tex("rollout_rmse", tbl)


# ---------------------------------------------------------------------------
# Static reference table: terse, presentation-style glossary -- one short line
# per metric / evaluation setting (formula + plain meaning + better-direction
# arrow), sized landscape via a wide p{} meaning column. Formulas mirror the
# implementations so the thesis and code can't drift: RMSE -> _eval_utils
# .MetricAccumulator.summary; CSI/FAR/HSS contingency table + Roberts--Lean FSS
# + fair kernel CRPS -> compare_diffusion_vs_flowcast.{_categorical_scores,fss,
# crps_field}; rollout / ensemble -> compare_diffusion_vs_flowcast.rollout.
# Pure text -- no CSV reads. TP/FP/FN/TN = hits / false alarms / misses / correct
# negatives (standard contingency cells); kept terse, no symbol footnote.
GLOSSARY_ROWS = [
    (r"RMSE~$\downarrow$",
     r"$\sqrt{\dfrac{1}{N}\sum_{i=1}^{N}\bigl(\hat{x}_i-x_i\bigr)^2}$",
     r"Mean per-pixel error in physical units; reported per channel."),
    (r"CRPS~$\downarrow$",
     r"$\dfrac{1}{K}\sum_{k}\lvert x_k-y\rvert"
     r"-\dfrac{1}{2K(K-1)}\sum_{j,k}\lvert x_j-x_k\rvert$",
     r"Probabilistic MAE: rewards accuracy and a calibrated ensemble spread."),
    (r"CSI~$\uparrow$",
     r"$\dfrac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}+\mathrm{FN}}$",
     r"Hit fraction among forecast-or-observed rain pixels; ignores dry pixels."),
    (r"HSS~$\uparrow$",
     r"$\dfrac{\mathrm{TP}+\mathrm{TN}-E}{N-E}$",
     r"Categorical skill above random chance $E$ ($1$ = perfect)."),
    (r"FAR~$\downarrow$",
     r"$\dfrac{\mathrm{FP}}{\mathrm{TP}+\mathrm{FP}}$",
     r"Fraction of forecast rain pixels that stayed dry."),
    (r"FSS~$\uparrow$",
     r"$1-\dfrac{\overline{(P_f-P_o)^2}}"
     r"{\overline{P_f^{\,2}}+\overline{P_o^{\,2}}}$",
     r"Neighbourhood rain-fraction overlap; tolerant to small displacement."),
]

# Evaluation protocols (how forecasts are produced/scored), not metrics with a
# direction. Rendered below the metrics under their own sub-heading.
GLOSSARY_PROTOCOL_ROWS = [
    ("Rollout",
     r"$\hat{x}_{t+k}=\mathcal{M}\bigl(\hat{x}_{t+k-1}\bigr)$",
     r"Autoregressive feedback; tracks error growth vs lead time ($1$--$6$\,h)."),
    ("Ensemble",
     r"$\{x_k\}_{k=1}^{K},\ x_k=\mu+r_\theta(c,\epsilon_k)$",
     r"$K$ stochastic samples: mean is the forecast, spread the uncertainty."),
]

def t_metrics_glossary():
    colspec = r"@{}l l >{\raggedright\arraybackslash}p{9cm}@{}"
    lines = [rf"\begin{{tabular}}{{{colspec}}}", r"\toprule",
             r"Metric & Formula & Meaning \\", r"\midrule"]
    lines += [" & ".join(row) + r" \\[5pt]" for row in GLOSSARY_ROWS]
    lines += [
        r"\midrule",
        r"\multicolumn{3}{@{}l@{}}{\textit{Evaluation protocols}} \\",
        r"\addlinespace[1pt]",
    ]
    lines += [" & ".join(row) + r" \\[5pt]" for row in GLOSSARY_PROTOCOL_ROWS]
    lines += [r"\bottomrule", r"\end{tabular}"]
    compile_tex("metrics_glossary", "\n".join(lines))


def main():
    print(f"[plain_latex_tables] writing to {OUT}")
    t_metrics_glossary()
    t_scoreboard()
    t_per_channel("rmse_per_channel.csv", "rmse_per_channel")
    t_per_channel("crps_per_channel.csv", "crps_per_channel")
    t_precip_skill_window()
    t_by_threshold("csi", "csi_by_threshold")
    t_by_threshold("fss", "fss_by_threshold")
    t_rollout()
    t_timing()
    pngs = sorted(OUT.glob("*.png"))
    print(f"[plain_latex_tables] done: {len(pngs)} tables (.tex + .png) in {OUT}")


if __name__ == "__main__":
    main()
