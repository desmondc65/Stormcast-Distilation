"""Qualitative case studies for Chapter 4.

For each requested case (given as an ISO8601 timestamp), this script locates
the matching validation sample, runs the teacher and every PD student, and
renders a single panel figure with one row per channel and one column per
model (truth included). A second panel focuses on qpepre only and uses a
shared colour scale so the reader can compare precipitation intensities.

Default cases cover the archetypes listed in chapter04.tex §4.6:

    - typhoon landfall
    - Mei-yu frontal convection
    - orographic rainfall over the Central Mountain Range
    - dry quiescent day

Override with --cases 'label=YYYY-MM-DDTHH' on the CLI.

Outputs (``results/qualitative/<label>/``):

    fields_all_channels.pdf/.png   4×(1+K) panel, every channel
    fields_qpepre.pdf/.png         1×(1+K) qpepre-only panel
    summary.csv                    per-case RMSE / CSI(qpepre@1) for each model
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize, TwoSlopeNorm

from common import (
    CHANNELS, CHANNEL_CMAP, CHANNEL_UNITS, DEFAULT_DATA_ROOT, DEFAULT_PHASES,
    DEFAULT_REGRESSION, DEFAULT_RESULTS, DEFAULT_TEACHER, TARGET_STEPS,
    TEACHER_STEPS,
    banner, build_model_plan, contingency, csi_from_contingency, init_device,
    load_dataset, load_invariants, load_model, per_channel_rmse,
    prepare_sample_inputs, run_model, save_figure, set_thesis_style,
)

DEFAULT_CASES = [
    ("typhoon_landfall",     "2022-09-03T12"),   # Hinnamnor brush of Taiwan
    ("meiyu_front",          "2022-05-24T06"),   # Mei-yu season convection
    ("orographic_rain",      "2022-06-09T00"),   # CMR upslope rainfall
    ("dry_day",              "2022-01-15T00"),   # quiescent winter day
]


def parse_case_spec(spec: str) -> Tuple[str, datetime]:
    label, ts = spec.split("=", 1)
    dt = datetime.fromisoformat(ts.replace(" ", "T"))
    return label, dt


def find_index(dataset, target: datetime) -> int:
    """Return the index of the validation sample whose input-time matches
    ``target`` exactly. Falls back to the closest hour."""
    times = dataset.valid_samples
    for i, ts in enumerate(times):
        if ts == target:
            return i
    # fall back to closest
    deltas = np.array([abs((ts - target).total_seconds()) for ts in times])
    idx = int(deltas.argmin())
    print(f"[cases] exact match for {target} not found — using {times[idx]} (Δ={deltas[idx]/3600:.1f}h)")
    return idx


def _symmetric_range(arrs: List[np.ndarray]) -> Tuple[float, float]:
    hi = max(float(np.max(a)) for a in arrs)
    lo = min(float(np.min(a)) for a in arrs)
    r = max(abs(hi), abs(lo))
    return -r, r


def _plot_case_full(panels: Dict[str, np.ndarray], truth: np.ndarray,
                    out_dir: Path, title: str):
    labels = ["truth"] + [l for l in panels.keys()]
    arrays = {"truth": truth, **panels}
    n_rows = len(CHANNELS)
    n_cols = len(labels)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(1.9 * n_cols + 0.8, 2.2 * n_rows + 0.4),
        squeeze=False,
    )
    for r, ch in enumerate(CHANNELS):
        ch_arrays = [a[r] for a in arrays.values()]
        if ch in ("u10", "v10"):
            vmin, vmax = _symmetric_range(ch_arrays)
            if vmin >= 0 or vmax <= 0 or vmin == vmax:
                # Degenerate range — fall back to a plain diverging norm.
                span = max(abs(vmin), abs(vmax), 1e-3)
                norm = Normalize(vmin=-span, vmax=span)
            else:
                norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
            cmap = CHANNEL_CMAP[ch]
        elif ch == "qpepre":
            vmin = 0.0
            vmax = float(np.quantile(np.stack(ch_arrays), 0.995))
            if vmax <= 0:
                vmax = 1e-3
            norm = Normalize(vmin=vmin, vmax=vmax)
            cmap = CHANNEL_CMAP[ch]
        else:
            vmin = float(np.min(np.stack(ch_arrays)))
            vmax = float(np.max(np.stack(ch_arrays)))
            norm = Normalize(vmin=vmin, vmax=vmax)
            cmap = CHANNEL_CMAP[ch]

        for c, lab in enumerate(labels):
            ax = axes[r, c]
            im = ax.imshow(arrays[lab][r], origin="lower", norm=norm,
                           cmap=cmap, aspect="auto", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
                spine.set_edgecolor("#333")
            if r == 0:
                ax.set_title(lab, fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{ch}\n[{CHANNEL_UNITS[ch]}]", fontsize=9)

        cbar = fig.colorbar(im, ax=list(axes[r, :]),
                            fraction=0.018, pad=0.012, shrink=0.90)
        cbar.ax.tick_params(labelsize=8)
        cbar.outline.set_linewidth(0.4)

    fig.suptitle(title, fontsize=11)
    save_figure(fig, out_dir, "fields_all_channels")


def _plot_case_qpepre(panels: Dict[str, np.ndarray], truth: np.ndarray,
                      out_dir: Path, title: str):
    labels = ["truth"] + list(panels.keys())
    arrays = {"truth": truth, **panels}
    qi = CHANNELS.index("qpepre")

    stacked = np.stack([a[qi] for a in arrays.values()])
    vmax = float(np.quantile(stacked, 0.995))
    if vmax <= 0:
        vmax = 1e-3
    norm = Normalize(vmin=0.0, vmax=vmax)

    fig, axes = plt.subplots(
        1, len(labels), figsize=(1.9 * len(labels) + 0.8, 2.6),
        squeeze=False,
    )
    for c, lab in enumerate(labels):
        ax = axes[0, c]
        im = ax.imshow(arrays[lab][qi], origin="lower", norm=norm,
                       cmap=CHANNEL_CMAP["qpepre"], aspect="auto", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
            spine.set_edgecolor("#333")
        ax.set_title(lab, fontsize=9)
    cbar = fig.colorbar(im, ax=list(axes[0, :]),
                        fraction=0.018, pad=0.012, shrink=0.95)
    cbar.set_label(r"qpepre  [mm h$^{-1}$]", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_linewidth(0.4)
    fig.suptitle(title, fontsize=11)
    save_figure(fig, out_dir, "fields_qpepre")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases-dir", type=Path, default=DEFAULT_PHASES)
    ap.add_argument("--data-location", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--regression-checkpoint", type=Path, default=DEFAULT_REGRESSION)
    ap.add_argument("--teacher-checkpoint", type=Path, default=DEFAULT_TEACHER)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS / "qualitative")
    ap.add_argument("--cases", nargs="*", default=None,
                    help="label=YYYY-MM-DDTHH case specifications. Defaults: "
                         "typhoon landfall, mei-yu, orographic, dry day.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--valid-dates", nargs=2, default=["2022/01/01", "2022/12/31"])
    ap.add_argument("--initial-num-steps", type=int, default=TEACHER_STEPS)
    ap.add_argument("--target-num-steps", type=int, default=TARGET_STEPS)
    args = ap.parse_args()

    set_thesis_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.cases:
        case_list = [parse_case_spec(s) for s in args.cases]
    else:
        case_list = [(lab, datetime.fromisoformat(ts)) for lab, ts in DEFAULT_CASES]

    device = init_device()
    banner("Loading dataset")
    dataset = load_dataset(args.data_location, tuple(args.valid_dates))
    invariant = load_invariants(dataset, device)

    regression = load_model(args.regression_checkpoint, device)
    plan = build_model_plan(args.phases_dir,
                            teacher_ckpt=args.teacher_checkpoint,
                            initial_steps=args.initial_num_steps,
                            target_steps=args.target_num_steps)

    # Resolve indices once
    indices = []
    case_titles = []
    for label, dt in case_list:
        idx = find_index(dataset, dt)
        indices.append(idx)
        case_titles.append(f"{label.replace('_', ' ').title()} — {dataset.valid_samples[idx]}")

    sample_inputs, truth_arr, timestamps = prepare_sample_inputs(dataset, indices, device)

    # Run each model once over every case
    all_preds: Dict[str, np.ndarray] = {}
    for entry in plan:
        banner(f"Inference: {entry.label}")
        model = load_model(entry.ckpt, device)
        preds = []
        for sidx, (bg, st_in, _) in enumerate(sample_inputs):
            torch.manual_seed(args.seed + sidx)
            pred = run_model(model, bg, st_in, invariant, regression,
                             mode=entry.mode, num_steps=entry.num_steps)
            preds.append(dataset.denormalize_state(pred.cpu().numpy()[0].copy()))
        all_preds[entry.label] = np.stack(preds, axis=0)
        del model
        torch.cuda.empty_cache()

    # Plot + summarise each case
    qp = CHANNELS.index("qpepre")
    summary_rows = []
    for sidx, (label, dt) in enumerate(case_list):
        case_dir = args.output_dir / label
        case_dir.mkdir(parents=True, exist_ok=True)
        panels = {lab: all_preds[lab][sidx] for lab in all_preds}
        _plot_case_full(panels, truth_arr[sidx], case_dir, case_titles[sidx])
        _plot_case_qpepre(panels, truth_arr[sidx], case_dir, case_titles[sidx])

        for lab, arr in panels.items():
            rmse = per_channel_rmse(arr[None, ...], truth_arr[sidx:sidx + 1])[0]
            c = contingency(arr[qp:qp + 1], truth_arr[sidx:sidx + 1, qp], 1.0)
            csi = csi_from_contingency(c)
            summary_rows.append([label, str(timestamps[sidx]), lab,
                                 *[f"{x:.6f}" for x in rmse],
                                 f"{float(rmse.mean()):.6f}", f"{csi:.6f}"])
        print(f"[case] {label:<20s}  saved → {case_dir}")

    csv_path = args.output_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "timestamp", "model",
                    *[f"rmse_{c}" for c in CHANNELS], "rmse_mean", "csi_qpepre_1"])
        w.writerows(summary_rows)
    print(f"[cases] {csv_path}")

    print(f"\n✓ qualitative cases complete → {args.output_dir}")


if __name__ == "__main__":
    main()
