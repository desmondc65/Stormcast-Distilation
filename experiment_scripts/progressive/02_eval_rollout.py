"""Autoregressive 12-hour rollout evaluation.

For each initial condition we integrate the regression + diffusion (or
distilled) model forward for 12 hourly steps, feeding the predicted HighRes
state back in as ``state_in`` at the next step while drawing fresh LowRes
background and invariants from the validation zarr. At each requested lead
time the prediction is compared against the RWRF ground truth from the same
zarr.

Outputs (``results/rollout/``):

    rollout_metrics.csv            per-model × per-lead-time × per-channel RMSE + CSI(qpepre@1 mm h⁻¹)
    rollout_rmse.pdf/.png          channel-averaged RMSE vs lead time
    rollout_rmse_qpepre.pdf/.png   qpepre RMSE vs lead time
    rollout_csi_qpepre.pdf/.png    CSI(qpepre, τ=1 mm h⁻¹) vs lead time

This realises Section 4.3 "Rollout" / Failure-mode "Rollout divergence" of
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
    CHANNELS, DEFAULT_DATA_ROOT, DEFAULT_PHASES, DEFAULT_REGRESSION,
    DEFAULT_RESULTS, DEFAULT_TEACHER, TARGET_STEPS, TEACHER_STEPS,
    banner, build_model_plan, contingency, csi_from_contingency,
    init_device, label_colour, label_marker, load_dataset, load_invariants,
    load_model, per_channel_rmse, run_model, save_figure, set_thesis_style, timeit,
)

LEAD_TIMES = (1, 3, 6, 12)   # hours
CSI_TAU = 1.0                 # mm h^-1 — primary operational threshold


def _get_sample(dataset, idx, device):
    batch = dataset[idx]
    st_in = batch["state"][0].to(device=device, dtype=torch.float32).unsqueeze(0)
    st_tar = batch["state"][1].to(device=device, dtype=torch.float32).unsqueeze(0)
    bg = batch["background"].to(device=device, dtype=torch.float32).unsqueeze(0)
    return bg, st_in, st_tar


def _rollout_one(model, mode, num_steps, dataset, init_idx, invariant, regression,
                 device, horizon: int, seed: int):
    """Run an ``horizon``-hour autoregressive rollout.

    Returns:
        preds_norm: dict[lead → (C, H, W)] in normalised units.
        truths_norm: dict[lead → (C, H, W)] in normalised units.
    """
    bg, st_in, _ = _get_sample(dataset, init_idx, device)
    preds_norm: dict[int, np.ndarray] = {}
    truths_norm: dict[int, np.ndarray] = {}

    state = st_in
    for lead in range(1, horizon + 1):
        # background + truth for t+lead are taken from the zarr at (init_idx + lead - 1)
        bg_l, st_prev, st_tar = _get_sample(dataset, init_idx + lead - 1, device)
        # state that will be fed to the diffusion step = our current prediction
        torch.manual_seed(seed + lead)
        pred = run_model(model, bg_l, state, invariant, regression,
                         mode=mode, num_steps=num_steps)
        state = pred
        if lead in LEAD_TIMES:
            preds_norm[lead] = pred.cpu().numpy()[0].copy()
            truths_norm[lead] = st_tar.cpu().numpy()[0].copy()
    return preds_norm, truths_norm


def _choose_init_indices(dataset, n_inits: int, horizon: int, stride_hours: int):
    """Choose initial indices uniformly spaced by ``stride_hours`` that leave
    room for a ``horizon``-hour rollout within the validation split."""
    last_ok = len(dataset) - horizon - 1
    idxs = list(range(0, last_ok, stride_hours))[:n_inits]
    if not idxs:
        raise SystemExit("validation split too short for requested horizon")
    return idxs


def plot_rmse_vs_lead(data: dict[str, dict[int, float]], *, ylabel: str, title: str,
                      out_dir: Path, stem: str):
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    for lab, per_lead in data.items():
        leads = sorted(per_lead.keys())
        ax.plot(leads, [per_lead[l] for l in leads],
                marker=label_marker(lab), color=label_colour(lab),
                markeredgecolor="#222", markeredgewidth=0.5, label=lab)
    ax.set_xlabel("lead time  [h]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(list(LEAD_TIMES))
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    save_figure(fig, out_dir, stem)


def plot_csi_vs_lead(data: dict[str, dict[int, float]], out_dir: Path):
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    for lab, per_lead in data.items():
        leads = sorted(per_lead.keys())
        ax.plot(leads, [per_lead[l] for l in leads],
                marker=label_marker(lab), color=label_colour(lab),
                markeredgecolor="#222", markeredgewidth=0.5, label=lab)
    ax.set_xlabel("lead time  [h]")
    ax.set_ylabel(rf"CSI(qpepre, $\tau={CSI_TAU}$ mm h$^{{-1}}$)")
    ax.set_title("Autoregressive precipitation skill")
    ax.set_xticks(list(LEAD_TIMES))
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    save_figure(fig, out_dir, "rollout_csi_qpepre")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases-dir", type=Path, default=DEFAULT_PHASES)
    ap.add_argument("--data-location", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--regression-checkpoint", type=Path, default=DEFAULT_REGRESSION)
    ap.add_argument("--teacher-checkpoint", type=Path, default=DEFAULT_TEACHER)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS / "rollout")
    ap.add_argument("--n-inits", type=int, default=32,
                    help="number of initial conditions to roll out (default 32).")
    ap.add_argument("--stride-hours", type=int, default=96,
                    help="spacing between consecutive init times (default 4 days).")
    ap.add_argument("--horizon", type=int, default=12)
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
    print(f"[dataset] {len(dataset)} validation samples")
    invariant = load_invariants(dataset, device)

    regression = load_model(args.regression_checkpoint, device)
    plan = build_model_plan(args.phases_dir,
                            teacher_ckpt=args.teacher_checkpoint,
                            initial_steps=args.initial_num_steps,
                            target_steps=args.target_num_steps,
                            skip_teacher=args.skip_teacher)

    init_idxs = _choose_init_indices(dataset, args.n_inits, args.horizon, args.stride_hours)
    print(f"[rollouts] {len(init_idxs)} init indices, horizon={args.horizon}h, "
          f"stride={args.stride_hours}h")

    # Accumulators: per label → per lead → lists of (C, H, W) normalised arrays
    preds_acc: dict[str, dict[int, list]] = {e.label: {l: [] for l in LEAD_TIMES} for e in plan}
    truths_acc: dict[int, list] = {l: [] for l in LEAD_TIMES}

    for entry in plan:
        banner(f"Rollout: {entry.label}  (N={entry.num_steps})")
        model = load_model(entry.ckpt, device)
        with timeit(entry.label):
            for init_idx in init_idxs:
                preds_norm, truths_norm = _rollout_one(
                    model, entry.mode, entry.num_steps, dataset,
                    init_idx, invariant, regression, device,
                    horizon=args.horizon, seed=args.seed,
                )
                for lead in LEAD_TIMES:
                    preds_acc[entry.label][lead].append(preds_norm[lead])
                    # truths accumulate once per init, but we fill the structure
                    # on the first model only.
                    if entry is plan[0]:
                        truths_acc[lead].append(truths_norm[lead])
        del model
        torch.cuda.empty_cache()

    # Denormalise and stack
    truths_stack = {lead: np.stack([dataset.denormalize_state(x.copy())
                                    for x in truths_acc[lead]], axis=0)
                    for lead in LEAD_TIMES}
    preds_stack = {
        lab: {lead: np.stack([dataset.denormalize_state(x.copy()) for x in lst], axis=0)
              for lead, lst in by_lead.items()}
        for lab, by_lead in preds_acc.items()
    }

    # Metrics
    qp = CHANNELS.index("qpepre")
    rmse_mean: dict[str, dict[int, float]] = {}
    rmse_qp: dict[str, dict[int, float]] = {}
    csi_qp: dict[str, dict[int, float]] = {}
    csv_rows = []
    for lab, by_lead in preds_stack.items():
        rmse_mean[lab] = {}
        rmse_qp[lab] = {}
        csi_qp[lab] = {}
        for lead in LEAD_TIMES:
            pred = by_lead[lead]
            truth = truths_stack[lead]
            rmse = per_channel_rmse(pred, truth).mean(axis=0)       # (C,)
            rmse_mean[lab][lead] = float(rmse.mean())
            rmse_qp[lab][lead] = float(rmse[qp])
            c = contingency(pred[:, qp], truth[:, qp], CSI_TAU)
            csi_qp[lab][lead] = csi_from_contingency(c)
            print(f"  {lab:<20s} lead={lead:>2d}h  "
                  f"RMSE(mean)={rmse_mean[lab][lead]:.3f}  "
                  f"RMSE(qpepre)={rmse_qp[lab][lead]:.3f}  "
                  f"CSI(qpepre@1)={csi_qp[lab][lead]:.3f}")
            csv_rows.append([lab, lead,
                             *[f"{x:.6f}" for x in rmse],
                             f"{float(rmse.mean()):.6f}",
                             f"{csi_qp[lab][lead]:.6f}"])

    csv_path = args.output_dir / "rollout_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "lead_h", *[f"rmse_{c}" for c in CHANNELS],
                    "rmse_mean", f"csi_qpepre_{CSI_TAU}"])
        w.writerows(csv_rows)
    print(f"[rollout] {csv_path}")

    plot_rmse_vs_lead(rmse_mean,
                      ylabel="RMSE (mean over channels)",
                      title="Rollout RMSE vs lead time",
                      out_dir=args.output_dir, stem="rollout_rmse")
    plot_rmse_vs_lead(rmse_qp,
                      ylabel=r"RMSE of qpepre  [mm h$^{-1}$]",
                      title="Rollout RMSE on qpepre",
                      out_dir=args.output_dir, stem="rollout_rmse_qpepre")
    plot_csi_vs_lead(csi_qp, args.output_dir)

    print(f"\n✓ rollout complete → {args.output_dir}")


if __name__ == "__main__":
    main()
