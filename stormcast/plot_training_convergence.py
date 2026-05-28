#!/usr/bin/env python3
"""Simple training convergence visualizer.

Reads training and validation loss CSV files and saves a convergence plot.

Supported CSV formats:
- metrics_train.csv: columns [steps, train_loss]
- metrics_val.csv:   columns [steps, val_loss]
- train_loss.csv:    columns [step, loss]
- valid_loss.csv:    columns [step, loss]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


PointSeries = Tuple[List[int], List[float]]


def _read_series(csv_path: Path, step_key: str, loss_key: str) -> PointSeries:
    steps: List[int] = []
    losses: List[float] = []

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                steps.append(int(float(row[step_key])))
                losses.append(float(row[loss_key]))
            except (KeyError, TypeError, ValueError):
                continue

    if not steps:
        raise ValueError(f"No usable rows found in {csv_path}")

    order = np.argsort(np.asarray(steps))
    steps_sorted = [steps[i] for i in order]
    losses_sorted = [losses[i] for i in order]
    return steps_sorted, losses_sorted


def _moving_average(values: List[float], window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if window <= 1 or window > len(arr):
        return arr
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(arr, kernel, mode="same")


def _load_losses(run_dir: Path) -> Tuple[PointSeries, PointSeries]:
    train_candidates = [
        (run_dir / "metrics_train.csv", "steps", "train_loss"),
        (run_dir / "train_loss.csv", "step", "loss"),
    ]
    val_candidates = [
        (run_dir / "metrics_val.csv", "steps", "val_loss"),
        (run_dir / "valid_loss.csv", "step", "loss"),
    ]

    train_series = None
    val_series = None

    for path, step_key, loss_key in train_candidates:
        if path.exists():
            train_series = _read_series(path, step_key, loss_key)
            break

    for path, step_key, loss_key in val_candidates:
        if path.exists():
            val_series = _read_series(path, step_key, loss_key)
            break

    if train_series is None:
        raise FileNotFoundError(
            f"Could not find training CSV in {run_dir}. Expected one of: "
            "metrics_train.csv, train_loss.csv"
        )

    if val_series is None:
        raise FileNotFoundError(
            f"Could not find validation CSV in {run_dir}. Expected one of: "
            "metrics_val.csv, valid_loss.csv"
        )

    return train_series, val_series


def plot_convergence(run_dir: Path, output: Path, smooth_window: int) -> None:
    (train_steps, train_losses), (val_steps, val_losses) = _load_losses(run_dir)

    train_smooth = _moving_average(train_losses, smooth_window)
    val_smooth = _moving_average(val_losses, max(1, smooth_window // 2))

    best_val_idx = int(np.argmin(np.asarray(val_losses)))
    best_val_step = val_steps[best_val_idx]
    best_val_loss = val_losses[best_val_idx]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax = axes[0]
    ax.plot(train_steps, train_losses, color="#4C72B0", alpha=0.25, linewidth=1, label="train (raw)")
    ax.plot(train_steps, train_smooth, color="#1F4E8C", linewidth=2, label=f"train (MA w={smooth_window})")

    ax.plot(val_steps, val_losses, color="#DD8452", alpha=0.35, linewidth=1, marker="o", markersize=3, label="val (raw)")
    ax.plot(val_steps, val_smooth, color="#C44E1A", linewidth=2, label=f"val (MA w={max(1, smooth_window // 2)})")

    ax.scatter([best_val_step], [best_val_loss], color="#55A868", s=45, zorder=5, label="best val")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training Convergence: {run_dir.name}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    gap_steps = np.asarray(val_steps, dtype=float)
    train_interp = np.interp(gap_steps, np.asarray(train_steps, dtype=float), np.asarray(train_smooth, dtype=float))
    gap = np.asarray(val_losses, dtype=float) - train_interp

    ax2 = axes[1]
    ax2.axhline(0.0, color="gray", linewidth=1, alpha=0.7)
    ax2.plot(val_steps, gap, color="#8172B3", linewidth=1.8, marker="o", markersize=3)
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Val - Train")
    ax2.grid(True, alpha=0.25)

    summary = (
        f"best val={best_val_loss:.4f} @ step={best_val_step} | "
        f"final train={train_losses[-1]:.4f} | final val={val_losses[-1]:.4f}"
    )
    fig.text(0.02, 0.01, summary, fontsize=9)

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a simple training convergence chart")
    parser.add_argument("run_dir", type=Path, help="Run directory containing metrics CSV files")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: <run_dir>/convergence.png)",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=50,
        help="Moving average window for smoothing train curve (default: 50)",
    )
    args = parser.parse_args()

    output = args.output if args.output is not None else (args.run_dir / "convergence.png")
    plot_convergence(args.run_dir, output, max(1, args.smooth_window))
    print(f"Saved convergence plot to: {output}")


if __name__ == "__main__":
    main()
