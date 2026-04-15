"""Parse train_progressive.log and plot distill / train / val losses per phase.

Log line format (example):
  phase 0 step 50/50000 global 50 N_student 9 tot_time ... step_time ...
  cpumem ... gpumem ... distill_loss 0.002  train_loss 0.782  val_loss 0.942
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
LINE_RE = re.compile(
    r"phase\s+(?P<phase>\d+)\s+step\s+(?P<step>\d+)/\d+\s+global\s+(?P<gstep>\d+)"
    r".*?distill_loss\s+(?P<distill>[-\deE.+]+)"
    r"\s+train_loss\s+(?P<train>[-\deE.+]+)"
    r"\s+val_loss\s+(?P<val>[-\deE.+]+)"
)


def parse_log(path: Path, stride: int = 50):
    """Return dict[phase] -> dict of lists (step, distill, train, val)."""
    phases = defaultdict(lambda: {"step": [], "distill": [], "train": [], "val": []})
    with path.open("r", errors="replace") as f:
        for raw in f:
            line = ANSI_RE.sub("", raw)
            m = LINE_RE.search(line)
            if not m:
                continue
            step = int(m.group("step"))
            if step % stride != 0:
                continue
            phase = int(m.group("phase"))
            phases[phase]["step"].append(step)
            phases[phase]["distill"].append(float(m.group("distill")))
            phases[phase]["train"].append(float(m.group("train")))
            phases[phase]["val"].append(float(m.group("val")))
    return dict(sorted(phases.items()))


def plot_losses(phases: dict, out_path: Path, title_suffix: str = ""):
    loss_types = [
        ("distill", "Distillation loss (student vs teacher x̃, EDM-weighted)"),
        ("train", "Training loss (residual MSE, per-sample)"),
        ("val", "Validation loss (N-step sample vs ground truth MSE)"),
    ]
    n_phases = len(phases)
    fig, axes = plt.subplots(
        len(loss_types),
        1,
        figsize=(20, 4.5 * len(loss_types)),
        sharex=False,
    )
    if len(loss_types) == 1:
        axes = [axes]

    cmap = plt.get_cmap("tab10")
    for ax, (key, ylabel) in zip(axes, loss_types):
        for idx, (phase, data) in enumerate(phases.items()):
            steps = data["step"]
            values = data[key]
            if key == "val":
                # Drop sentinel -1.0 emitted before the first validation run.
                steps, values = zip(
                    *[(s, v) for s, v in zip(steps, values) if v >= 0]
                ) if any(v >= 0 for v in values) else ([], [])
            if not steps:
                continue
            ax.plot(
                steps,
                values,
                color=cmap(idx % 10),
                linewidth=0.9,
                alpha=0.85,
                label=f"phase {phase}",
            )
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=max(1, n_phases))
        if key == "distill":
            ax.set_yscale("log")
    axes[-1].set_xlabel("phase-local step")
    fig.suptitle(f"Progressive distillation losses{title_suffix}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"saved → {out_path}")


def plot_global(phases: dict, out_path: Path):
    """Same three panels, concatenated across phases on a global x-axis."""
    loss_types = [
        ("distill", "distill_loss"),
        ("train", "train_loss"),
        ("val", "val_loss"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(22, 12), sharex=True)
    cmap = plt.get_cmap("tab10")

    # global offset per phase (assumes 50k steps per phase)
    phase_keys = sorted(phases.keys())
    offsets = {}
    running = 0
    for p in phase_keys:
        offsets[p] = running
        running += max(phases[p]["step"]) if phases[p]["step"] else 0

    for ax, (key, label) in zip(axes, loss_types):
        for idx, p in enumerate(phase_keys):
            data = phases[p]
            steps = [s + offsets[p] for s in data["step"]]
            values = data[key]
            if key == "val":
                pairs = [(s, v) for s, v in zip(steps, values) if v >= 0]
                if not pairs:
                    continue
                steps, values = zip(*pairs)
            ax.plot(
                steps,
                values,
                color=cmap(idx % 10),
                linewidth=0.8,
                label=f"phase {p}",
            )
        for p in phase_keys[1:]:
            ax.axvline(offsets[p], color="k", linestyle="--", linewidth=0.6, alpha=0.5)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
        if key == "distill":
            ax.set_yscale("log")
    axes[-1].set_xlabel("cumulative step (phase boundaries dashed)")
    fig.suptitle("Progressive distillation losses — global view", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"saved → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--log",
        type=Path,
        default=Path(
            "/home/desmond/Documents/master_thesis/Stormcast-Distilation/"
            "zettabyte_results/progressive/train_progressive.log"
        ),
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=Path(
            "/home/desmond/Documents/master_thesis/Stormcast-Distilation/"
            "zettabyte_results/progressive"
        ),
    )
    ap.add_argument("--stride", type=int, default=50)
    args = ap.parse_args()

    phases = parse_log(args.log, stride=args.stride)
    if not phases:
        raise SystemExit(f"No loss lines parsed from {args.log}")
    for p, d in phases.items():
        print(f"phase {p}: {len(d['step'])} points, "
              f"step range {min(d['step'])}→{max(d['step'])}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    plot_losses(
        phases,
        args.outdir / "progressive_losses_per_phase.png",
        title_suffix=f"  (sampled every {args.stride} steps)",
    )
    plot_global(phases, args.outdir / "progressive_losses_global.png")


if __name__ == "__main__":
    main()
