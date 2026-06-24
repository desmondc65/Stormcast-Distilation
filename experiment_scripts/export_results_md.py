#!/usr/bin/env python3
"""Consolidate every computed experiment scoreboard into a single results.md.

Reads the harness CSV/MD outputs under ``experiment_scripts/results/`` and emits
one master ``results.md`` (default: ``experiment_scripts/results.md``) with the
headline numbers, per-channel / per-threshold breakdowns, and the encoding and
NFE/qpw ablations. Every section states its provenance (which CSV, which
checkpoints) so the document is traceable and regenerable.

Pure file reads; no GPU, no inference. Run after ``run_main_experiment.sh`` has
populated ``results/main_experiment/``. Missing inputs are skipped with a
visible note rather than failing.

Usage::

    python export_results_md.py                 # -> experiment_scripts/results.md
    python export_results_md.py --out <path>
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import thesis_style as ts
import _nbhd_io as nbio

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "experiment_scripts" / "results"


def read_csv(path: Path):
    if not path.exists():
        return None, None
    with path.open() as f:
        rows = list(csv.reader(f))
    return (rows[0], rows[1:]) if rows else (None, None)


def fnum(v, dec=4):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if x != x:  # NaN
        return "--"
    return f"{x:.{dec}f}"


def md_table(header, rows) -> str:
    head = "| " + " | ".join(header) + " |"
    sep = "| " + " | ".join("---" for _ in header) + " |"
    body = "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows)
    return "\n".join([head, sep, body])


def label(method: str) -> str:
    """Plain-text method label (strip the LaTeX mathtext from thesis_style)."""
    s = ts.method_label(method)
    return (s.replace(r"$\times$", "x").replace(r"$K{=}10$", "K=10")
            .replace(r"$K{=}15$", "K=15").replace(r"$K{=}20$", "K=20")
            .replace("$", ""))


def missing(name: str, hint: str) -> str:
    return (f"_**{name}** not found — run `{hint}` to populate it. "
            f"(Skipped, not an error.)_\n")


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------
def sec_main_scoreboard() -> str:
    header, data = read_csv(RESULTS / "main_experiment" / "scoreboard_3way.csv")
    if header is None:
        return missing("main_experiment/scoreboard_3way.csv", "run_main_experiment.sh")
    out_rows = [[label(r[0])] + [fnum(v) for v in r[1:]] for r in data]
    head = ["Method"] + [h for h in header[1:]]
    return md_table(head, out_rows)


def sec_per_channel(leg="cleaned_2M") -> str:
    parts = []
    for fname, kind in (("rmse_per_channel.csv", "RMSE"), ("crps_per_channel.csv", "CRPS")):
        header, data = read_csv(RESULTS / "main_experiment" / leg / fname)
        if header is None:
            parts.append(missing(f"main_experiment/{leg}/{fname}", "run_main_experiment.sh"))
            continue
        rows = [[label(r[0])] + [fnum(v) for v in r[1:]] for r in data]
        parts.append(f"**{kind} per channel (physical units, lower better)**\n\n"
                     + md_table(["Method"] + list(header[1:]), rows))
    return "\n\n".join(parts)


def sec_per_threshold(leg="cleaned_2M") -> str:
    path = RESULTS / "main_experiment" / leg / "per_threshold.csv"
    if not path.exists():
        return missing(f"main_experiment/{leg}/per_threshold.csv", "run_main_experiment.sh")
    thresholds, records = nbio.read_per_threshold(path)
    kerneled = any(r["kernel"] is not None for r in records)
    if kerneled:
        head = ["Method", "Kernel", "Metric"] + [f">={t} mm/h" for t in thresholds]
        rows = [[label(r["method"]), nbio.kernel_pretty(r["kernel"]), r["metric"]]
                + [fnum(v) for v in r["values"]] for r in records]
    else:
        head = ["Method", "Metric"] + [f">={t} mm/h" for t in thresholds]
        rows = [[label(r["method"]), r["metric"]] + [fnum(v) for v in r["values"]]
                for r in records]
    return md_table(head, rows)


def sec_fss(leg="cleaned_2M") -> str:
    legdir = RESULTS / "main_experiment" / leg
    mode, xs, records = nbio.read_fss(legdir)
    if mode is None:
        return missing(f"main_experiment/{leg}/fss_nbhd.csv", "run_main_experiment.sh")
    if mode == "kernel":
        head = ["Method", "Kernel"] + [f">={t} mm/h" for t in xs]
        rows = [[label(r["method"]), nbio.kernel_pretty(r["kernel"])]
                + [fnum(v) for v in r["values"]] for r in records]
    else:  # legacy window schema (neighbourhood half-widths)
        head = ["Method", "Metric"] + [f"n={s}" for s in xs]
        rows = [[label(r["method"]), r["metric"]] + [fnum(v) for v in r["values"]]
                for r in records]
    return md_table(head, rows)


def sec_rollout() -> str:
    header, data = read_csv(RESULTS / "main_experiment" / "rmse_per_step_3way.csv")
    if header is None:
        return missing("main_experiment/rmse_per_step_3way.csv", "run_main_experiment.sh")
    channels = header[2:]
    # 6-hour rollout horizon (StormCast 1-6 h skill window): report every lead.
    want_steps = [1, 2, 3, 4, 5, 6]
    by = {}
    for r in data:
        by.setdefault(r[0], {})[int(r[1])] = r[2:]
    head = ["Method", "lead (h)"] + list(channels)
    rows = []
    for m, steps in by.items():
        for s in want_steps:
            if s in steps:
                rows.append([label(m), s] + [fnum(v) for v in steps[s]])
    return ("RMSE (physical units) at selected lead times:\n\n"
            + md_table(head, rows))


# ---------------------------------------------------------------------------
def build(out: Path):
    chunks = []
    chunks.append("# StormCast / FlowCast — Consolidated Results\n")
    chunks.append(
        "_Auto-generated by `experiment_scripts/export_results_md.py` from the "
        "harness CSV outputs under `experiment_scripts/results/`. Regenerate with "
        "`python export_results_md.py`. Figures for these tables are rebuilt by "
        "`make_thesis_figures.py` (data-driven) and `make_thesis_qualitative.sh` "
        "(field panels), both in the viridis palette of `thesis_style.py`._\n"
    )
    chunks.append(
        "**Scope.** All evaluation is on the 2022 validation year with "
        "**five-member ensembles** (the StormCast ensemble design, Pathak et "
        "al. 2024 §2.3: a 5-member ensemble propagated autoregressively each "
        "hour). Forecasts are rolled out autoregressively over a **6-hour "
        "horizon** (+1 h … +6 h), matching StormCast's headline 1-6 h skill "
        "window. The cleaned pipeline is the 192x96, log1p-qpepre store; the "
        "regression mean (`StormCastUNet.0.8000`) is shared by every cleaned "
        "head. Matched budget = ~2M training samples: cleaned EDM at step "
        "31000 (batch 64), FlowCast at step 20000 (batch 96). Precipitation "
        "scores are in mm/h after denormalisation; per-channel RMSE/CRPS are "
        "in physical units. The EDM Heun sampler at N=18 steps costs "
        "**2N-1 = 35 NFE** per hourly step; FlowCast costs K NFE.\n"
    )

    def add(title, body, intro=""):
        chunks.append(f"\n## {title}\n")
        if intro:
            chunks.append(intro + "\n")
        chunks.append(body + "\n")

    add("1. Headline scoreboard — EDM diffusion vs. FlowCast vs. MeanFlow (+ legacy)",
        sec_main_scoreboard(),
        "Source: `results/main_experiment/scoreboard_3way.csv`. `legacy_edm` is "
        "the upstream NVIDIA 224x128 raw-mm/h baseline (different grid/encoding — "
        "compare in magnitude only). Thesis Table 4.1 / Figure 4.x use the four "
        "cleaned rows.")
    add("2. Per-channel RMSE and CRPS (cleaned, ~2M)", sec_per_channel(),
        "Source: `results/main_experiment/cleaned_2M/{rmse,crps}_per_channel.csv`.")
    add("3. Precipitation skill by threshold (cleaned, ~2M)", sec_per_threshold(),
        "Source: `results/main_experiment/cleaned_2M/per_threshold.csv`. "
        "CSI/POD/HSS higher better; FAR lower better. Scored at the four "
        "StormCast (Pathak et al. 2024, Fig. 3) pooling windows on the ~2 km "
        "grid: **3 km** (1x1 px, grid scale), **15 km** (7x7 px), **27 km** "
        "(13x13 px) and **45 km** (23x23 px). CSI/POD/FAR/HSS use "
        "neighbourhood-max pooling (an event counts if any pixel within the "
        "window exceeds the threshold); FSS uses fractional coverage. The 3 km "
        "(grid-scale) categorical scores collapse toward 0 at heavy-rain "
        "thresholds — the single-pixel double-penalty the larger windows relax.")
    add("4. Heavy-rain FSS at >=16 mm/h (cleaned, ~2M)", sec_fss(),
        "Source: `results/main_experiment/cleaned_2M/fss_nbhd.csv`. Fractions "
        "Skill Score (Roberts & Lean 2008) per threshold at each StormCast "
        "pooling window (3 km = 1 px, 15 km = 7 px, 27 km = 13 px, 45 km = 23 px).")
    add("5. Autoregressive rollout RMSE vs. lead time", sec_rollout(),
        "Source: `results/main_experiment/rmse_per_step_3way.csv`.")
    out.write_text("\n".join(chunks).rstrip() + "\n")
    print(f"[export_results_md] wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "experiment_scripts" / "results.md")
    args = ap.parse_args()
    build(args.out)


if __name__ == "__main__":
    main()
