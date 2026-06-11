#!/usr/bin/env python3
"""Stitch a qpepre-focused comparison across two inference legs on different grids.

Built for "legacy larger domain (224x128, raw mm/h) vs cleaned cropped domain
(192x96, log1p)": each leg is a directory produced by
``compare_diffusion_vs_flowcast.py`` (so it has ``scoreboard.csv``,
``crps_per_channel.csv``, ``per_threshold.csv`` and ``ps1d_qpepre.csv``). This
reads the **qpepre** numbers out of each and writes:

  qpepre_scoreboard.{md,csv}  -- one row per (leg, method): RMSE / CRPS / CSI at
                                 each threshold / CSI-P16 / FSS-P16, all in mm/h.
  qpepre_spectral_bias.png    -- PSD_pred / PSD_truth vs wavenumber for every
                                 method across both legs (the ratio is the only
                                 spectral quantity comparable across grids, since
                                 the two domains have different truth spectra).

Everything is qpepre in mm/h: the loader's ``denormalize_state`` undoes log1p on
the cleaned leg, so threshold metrics ARE directly comparable across legs (the
per-pixel RMSE/CRPS are comparable in magnitude but the fields of view differ by
~36% -- footnote when reporting). See CLAUDE.md §6.6.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# qpepre CSI thresholds to surface in the table (mm/h).
CSI_TAUS = [0.1, 1.0, 5.0, 10.0]


def _read_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _norm(key: str) -> str:
    """Strip the ↓/↑ sort arrows from a scoreboard column name."""
    return key.replace("↓", "").replace("↑", "").strip()


def _col(row: dict, name: str):
    for k, v in row.items():
        if _norm(k) == name:
            return v
    return None


def load_leg(dirpath: Path):
    sb = {r["method"]: r for r in _read_csv(dirpath / "scoreboard.csv")}
    crps = {r["method"]: r for r in _read_csv(dirpath / "crps_per_channel.csv")}
    csi = {}
    for r in _read_csv(dirpath / "per_threshold.csv"):
        if r.get("metric") == "csi":
            csi[r["method"]] = r
    return sb, crps, csi


def qpepre_row(method, label, grid, enc, sb, crps, csi):
    s, c, t = sb[method], crps[method], csi[method]

    def thr(tau):
        for k, v in t.items():
            if k in ("method", "metric"):
                continue
            try:
                if abs(float(k) - tau) < 1e-9:
                    return v
            except ValueError:
                pass
        return "nan"

    return {
        "leg": label,
        "grid": grid,
        "encoding": enc,
        "RMSE_qpepre": _col(s, "RMSE_qpepre"),
        "CRPS_qpepre": c.get("qpepre"),
        **{f"CSI@{tau}": thr(tau) for tau in CSI_TAUS},
        "CSI-P16": _col(s, "CSI-P16"),
        "FSS-P16": _col(s, "FSS-P16-M"),
    }


def write_scoreboard(rows, out_dir: Path):
    header = list(rows[0].keys())
    with open(out_dir / "qpepre_scoreboard.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)

    md = [
        "# qpepre inference comparison — legacy larger domain vs cleaned domain",
        "",
        "All values are **qpepre in mm/h** (the loader un-log1p's the cleaned leg). "
        "Threshold metrics (CSI/FSS) are directly comparable across legs; per-pixel "
        "RMSE/CRPS are comparable in magnitude but the two grids cover different "
        "fields of view (legacy 224×128 ≈ +36% area vs cleaned 192×96) — footnote "
        "when reporting. CSI/FSS: higher is better; RMSE/CRPS: lower is better.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for r in rows:
        md.append("| " + " | ".join(str(r[h]) for h in header) + " |")
    (out_dir / "qpepre_scoreboard.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def plot_spectral_bias(legacy_dir: Path, cleaned_dir: Path, out_dir: Path):
    """PSD_pred / PSD_truth vs k for qpepre, every method across both legs.

    The ratio (spectral bias) is the cross-grid-comparable quantity: each method
    is divided by its OWN leg's truth, so different domain spectra cancel out.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    styles = {"legacy": ("--", "224×128 raw"), "cleaned": ("-", "192×96 log1p")}
    drew = False
    for tag, (ls, dom) in styles.items():
        leg_dir = legacy_dir if tag == "legacy" else cleaned_dir
        psd_path = leg_dir / "ps1d_qpepre.csv"
        if not psd_path.exists():
            print(f"[stitch] no {psd_path} — skipping {tag} in spectral plot")
            continue
        rows = _read_csv(psd_path)
        k = np.array([float(r["k"]) for r in rows])
        truth = np.array([float(r["truth"]) for r in rows])
        methods = [c for c in rows[0] if c not in ("k", "truth")]
        for m in methods:
            p = np.array([float(r[m]) for r in rows])
            ratio = np.divide(p, truth, out=np.full_like(p, np.nan), where=truth > 0)
            ax.plot(k, ratio, ls, lw=1.1, label=f"{tag}:{m} ({dom})")
            drew = True
    if not drew:
        plt.close(fig)
        return
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("wavenumber k")
    ax.set_ylabel("PSD_pred / PSD_truth  (qpepre)")
    ax.set_title("qpepre spectral bias — legacy vs cleaned (ratio to each leg's own truth)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "qpepre_spectral_bias.png", dpi=130)
    plt.close(fig)
    print(f"[stitch] wrote {out_dir / 'qpepre_spectral_bias.png'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--legacy-dir", type=Path, required=True,
                    help="Leg dir for the legacy 224x128 raw-mm/h domain (EDM only).")
    ap.add_argument("--cleaned-dir", type=Path, required=True,
                    help="Leg dir for the cleaned 192x96 log1p domain (EDM+flow).")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--legacy-grid", default="224×128")
    ap.add_argument("--cleaned-grid", default="192×96")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    leg_sb, leg_crps, leg_csi = load_leg(args.legacy_dir)
    cln_sb, cln_crps, cln_csi = load_leg(args.cleaned_dir)

    rows = []
    # Legacy leg: EDM only.
    if "diffusion" in leg_sb:
        rows.append(qpepre_row("diffusion", "legacy_edm", args.legacy_grid, "raw mm/h",
                               leg_sb, leg_crps, leg_csi))
    # Cleaned leg: whichever methods are present.
    for method, label in [("diffusion", "cleaned_edm"),
                          ("flowcast", "cleaned_flowcast")]:
        if method in cln_sb:
            rows.append(qpepre_row(method, label, args.cleaned_grid, "log1p(mm/h)",
                                   cln_sb, cln_crps, cln_csi))

    if not rows:
        raise SystemExit("No methods found in either leg — nothing to stitch.")

    write_scoreboard(rows, args.out_dir)
    try:
        plot_spectral_bias(args.legacy_dir, args.cleaned_dir, args.out_dir)
    except Exception as e:  # plot is a bonus; never block the table.
        print(f"[stitch] WARN spectral-bias plot skipped: {e}")
    print(f"[stitch] done — {args.out_dir}")


if __name__ == "__main__":
    main()
