#!/usr/bin/env python3
"""Consolidate every computed experiment scoreboard into a single results.md.

Reads the harness CSV/MD outputs under ``experiment_scripts/results/`` and emits
one master ``results.md`` (default: ``experiment_scripts/results.md``) with the
headline numbers, per-channel / per-threshold breakdowns, and the encoding and
NFE/qpw ablations. Every section states its provenance (which CSV, which
checkpoints) so the document is traceable and regenerable.

Pure file reads; no GPU, no inference. Run after the experiments have populated
``results/`` (see ``run_main_experiment.sh``, ``run_log1p_ablation.sh``,
``run_flowcast_nfe_sweep.sh``). Missing inputs are skipped with a visible note
rather than failing.

Usage::

    python export_results_md.py                 # -> experiment_scripts/results.md
    python export_results_md.py --out <path>
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import thesis_style as ts

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
    header, data = read_csv(RESULTS / "main_experiment" / leg / "per_threshold.csv")
    if header is None:
        return missing(f"main_experiment/{leg}/per_threshold.csv", "run_main_experiment.sh")
    head = ["Method", "Metric"] + [f">={t} mm/h" for t in header[2:]]
    rows = [[label(r[0]), r[1]] + [fnum(v) for v in r[2:]] for r in data]
    return md_table(head, rows)


def sec_fss(leg="cleaned_2M") -> str:
    header, data = read_csv(RESULTS / "main_experiment" / leg / "fss_p16.csv")
    if header is None:
        return missing(f"main_experiment/{leg}/fss_p16.csv", "run_main_experiment.sh")
    head = ["Method", "Metric"] + [f"n={s}" for s in header[2:]]
    rows = [[label(r[0]), r[1]] + [fnum(v) for v in r[2:]] for r in data]
    return md_table(head, rows)


def sec_rollout() -> str:
    header, data = read_csv(RESULTS / "main_experiment" / "rmse_per_step_3way.csv")
    if header is None:
        return missing("main_experiment/rmse_per_step_3way.csv", "run_main_experiment.sh")
    channels = header[2:]
    want_steps = [1, 3, 6, 12]
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


def sec_log1p() -> str:
    header, data = read_csv(RESULTS / "log1p_ablation" / "scoreboard_log1p.csv")
    if header is None:
        return missing("log1p_ablation/scoreboard_log1p.csv", "run_log1p_ablation.sh")
    rows = [[r[0]] + [fnum(v) for v in r[1:]] for r in data]
    return md_table(["Run"] + list(header[1:]), rows)


def sec_nfe() -> str:
    header, data = read_csv(RESULTS / "flowcast_nfe_sweep" / "nfe_sweep.csv")
    if header is None:
        return missing("flowcast_nfe_sweep/nfe_sweep.csv", "run_flowcast_nfe_sweep.sh")
    rows = [[fnum(v, 3) for v in r] for r in data]
    return md_table(list(header), rows)


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
        "**ten-member ensembles** (the project standard; HREF-class EPS "
        "bracket). The cleaned pipeline is the 192x96, log1p-qpepre store; the "
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

    add("1. Headline scoreboard — FlowCast vs. EDM diffusion (+ legacy)",
        sec_main_scoreboard(),
        "Source: `results/main_experiment/scoreboard_3way.csv`. `legacy_edm` is "
        "the upstream NVIDIA 224x128 raw-mm/h baseline (different grid/encoding — "
        "compare in magnitude only). Thesis Table 4.1 / Figure 4.x use the four "
        "cleaned rows.")
    add("2. Per-channel RMSE and CRPS (cleaned, ~2M)", sec_per_channel(),
        "Source: `results/main_experiment/cleaned_2M/{rmse,crps}_per_channel.csv`.")
    add("3. Precipitation skill by threshold (cleaned, ~2M)", sec_per_threshold(),
        "Source: `results/main_experiment/cleaned_2M/per_threshold.csv`. "
        "CSI/HSS higher better; FAR lower better.")
    add("4. Heavy-rain FSS at >=16 mm/h (cleaned, ~2M)", sec_fss(),
        "Source: `results/main_experiment/cleaned_2M/fss_p16.csv`. Neighbourhood "
        "half-width n in pixels (~2 km/cell).")
    add("5. Autoregressive rollout RMSE vs. lead time", sec_rollout(),
        "Source: `results/main_experiment/rmse_per_step_3way.csv`.")
    add("6. Encoding ablation — log1p vs. raw mm/h (FlowCast)", sec_log1p(),
        "Source: `results/log1p_ablation/scoreboard_log1p.csv`. qpepre reported "
        "in mm/h on both legs, so columns are directly comparable.")
    add("7. NFE Pareto sweep (FlowCast)", sec_nfe(),
        "Source: `results/flowcast_nfe_sweep/nfe_sweep.csv`. One FlowCast "
        "checkpoint, Euler step count K swept; quality vs. wall-clock.")
    add("8. qpepre channel-weight (qpw) sweep",
        "Full 8-row sweep (standardised-unit single-step validation RMSE) lives "
        "in `result_table.md` section C and thesis Table `tab:qpw`. Headline: "
        "qpw=2.0 is the default (best t2m & qpepre); qpw=1.4 is the wind sweet "
        "spot. Above qpw=2.0 skill degrades across the board. The per-run "
        "validation CSVs are under `runs/flowcast_qpw_ablation/qpw*/.../run_0/`.",
        "Source: training-loop validation CSVs (not a single harness scoreboard).")

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
