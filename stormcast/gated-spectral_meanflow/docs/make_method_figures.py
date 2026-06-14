#!/usr/bin/env python3
"""Render the GS-MeanFlow methodology figures (PNG) for docs/methodology.md.

Reproducible figure pipeline in the spirit of experiment_scripts/thesis_style.py:
everything samples the viridis ramp (the single thesis-wide colour convention)
and method colours reuse the canonical ramp positions from thesis_style
(diffusion 0.30, meanflow 0.42, flowcast 0.66). thesis_style itself is not
imported so this script stays runnable inside the stormcast_env venv without
path juggling; positions are mirrored here with a comment instead.

Usage (from anywhere):
    python stormcast/gated-spectral_meanflow/docs/make_method_figures.py

Outputs -> stormcast/gated-spectral_meanflow/docs/figures/*.png
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
DPI = 220

VIR = plt.get_cmap("viridis")

# Mirrors experiment_scripts/thesis_style.py::_METHOD_POS (do not change one
# without the other). gsmeanflow gets a fresh slot between meanflow and
# flowcast on the ramp.
C_DIFFUSION = VIR(0.30)
C_MEANFLOW = VIR(0.42)
C_GSMEANFLOW = VIR(0.52)
C_FLOWCAST = VIR(0.66)

# Component palette for the diagrams (all viridis-sampled).
C_DATA = (0.45, 0.45, 0.45, 1.0)   # neutral grey for raw tensors
C_FROZEN = VIR(0.08)               # frozen networks (cool end)
C_FLOWNET = VIR(0.42)              # trained average-velocity flow
C_GATE = VIR(0.72)                 # trained occurrence gate
C_SAMPLER = VIR(0.30)              # sampler / composition stages
C_LOSS = VIR(0.55)                 # loss nodes
C_OUT = VIR(0.62)                  # outputs

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "mathtext.fontset": "dejavusans",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def tint(color, f: float = 0.88):
    """Blend a colour toward white (light fill that keeps the hue legible)."""
    r, g, b, _ = to_rgba(color)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f, 1.0)


def box(
    ax,
    x,
    y,
    w,
    h,
    title,
    body=None,
    color=C_FLOWNET,
    lw=1.6,
    fs=10.0,
    bfs=8.2,
    ls="-",
    fill_f=0.88,
    title_color="0.12",
):
    """Rounded box with a bold title and an optional smaller body text."""
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.35,rounding_size=0.9",
        fc=tint(color, fill_f),
        ec=color,
        lw=lw,
        ls=ls,
        zorder=2,
    )
    ax.add_patch(p)
    cx = x + w / 2
    if body:
        ax.text(
            cx, y + h * 0.74, title, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=title_color, zorder=3,
        )
        ax.text(
            cx, y + h * 0.33, body, ha="center", va="center",
            fontsize=bfs, color="0.25", zorder=3, linespacing=1.35,
        )
    else:
        ax.text(
            cx, y + h / 2, title, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=title_color, zorder=3,
        )
    return p


def arrow(
    ax,
    p0,
    p1,
    color="0.35",
    lw=1.6,
    rad=0.0,
    ls="-",
    label=None,
    lxy=None,
    fs=8.0,
    lcolor=None,
    zorder=1,
    mutation=13,
):
    ar = FancyArrowPatch(
        p0,
        p1,
        arrowstyle="-|>",
        mutation_scale=mutation,
        color=color,
        lw=lw,
        linestyle=ls,
        connectionstyle=f"arc3,rad={rad}",
        zorder=zorder,
        shrinkA=2.5,
        shrinkB=2.5,
    )
    ax.add_patch(ar)
    if label:
        if lxy is None:
            lxy = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 + 1.2)
        ax.text(
            lxy[0], lxy[1], label, ha="center", va="center",
            fontsize=fs, color=lcolor or color, zorder=3,
        )
    return ar


def blank_canvas(figsize, xlim, ylim):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("auto")
    ax.axis("off")
    return fig, ax


def save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# ===========================================================================
# Figure 1 -- architecture & data flow of one autoregressive step
# ===========================================================================
def fig_architecture():
    fig, ax = blank_canvas((15, 9.0), (0, 104), (0, 67))

    ax.text(
        1, 65.3, "GS-MeanFlow — one autoregressive step",
        fontsize=14, fontweight="bold", color="0.1", va="center",
    )

    # --- inputs -----------------------------------------------------------
    box(ax, 2, 47, 12, 7, r"$M_t$", "HighRes state\n4 ch @ 192×96", color=C_DATA)
    box(ax, 2, 37, 12, 7, r"$S_t$", "ERA5 background\n24 ch", color=C_DATA)
    box(ax, 2, 27, 12, 7, r"$I$", "invariants\nlsm, orog", color=C_DATA)

    # --- frozen regression mean -------------------------------------------
    box(
        ax, 20, 42, 14, 11,
        r"regression $F_\theta$",
        "StormCastUNet\n(frozen)",
        color=C_FROZEN,
    )
    arrow(ax, (14, 50.5), (20, 49.5))
    arrow(ax, (14, 40.5), (20, 45.5), rad=0.15)
    arrow(ax, (14, 30.5), (20, 43.6), rad=0.25)

    box(ax, 38, 47, 10, 7, r"$\mu_{t+1}$", "regression\nmean", color=C_FROZEN)
    arrow(ax, (34, 49.0), (38, 50.0))

    # --- conditioning bundle -----------------------------------------------
    box(
        ax, 38, 22, 10, 10,
        r"condition $c$",
        "concat\n" + r"$[M_t,\,\mu_{t+1},\,I]$",
        color=C_DATA,
    )
    arrow(ax, (43, 47), (43, 32))
    arrow(ax, (14, 48.5), (38, 30), rad=-0.25)
    arrow(ax, (14, 29), (38, 25.5), rad=-0.08)

    # --- GSMeanFlowPrecond container ---------------------------------------
    cont = FancyBboxPatch(
        (53, 8), 26, 36,
        boxstyle="round,pad=0.4,rounding_size=1.2",
        fc=(0.97, 0.97, 0.99, 1.0), ec="0.6", lw=1.2, ls="--", zorder=1,
    )
    ax.add_patch(cont)
    ax.text(
        66, 42.0, "GSMeanFlowPrecond",
        ha="center", va="center", fontsize=10.5, color="0.35",
        fontweight="bold", style="italic",
    )

    box(
        ax, 55, 28, 22, 11,
        r"SongUNet flow $u_\theta(z_r, r, t, c)$",
        "average velocity over $[r,t]$\n~79 M params",
        color=C_FLOWNET,
    )
    box(
        ax, 55, 12, 22, 9,
        r"occurrence gate $g_\phi(c)$",
        "wet/dry logit · ~0.17 M params\n(~460× smaller)",
        color=C_GATE,
    )
    arrow(ax, (48, 29), (55, 33.5), rad=0.10)
    arrow(ax, (48, 24), (55, 16.5), rad=-0.10)

    # latent noise into the flow state
    box(ax, 22, 33, 12, 6.5, r"$z_0 \sim \mathcal{N}(0, I)$", "latent noise", color=C_DATA)
    arrow(ax, (34, 36), (55, 34.5), rad=0.0)

    # --- sampler + composition ---------------------------------------------
    box(
        ax, 84, 26, 14, 12,
        "1–2 NFE sampler",
        r"$z_{i+1} = z_i + \Delta t\; u_\theta$" + "\n" + r"$\hat r_{t+1} = \sigma_d\, z_K$",
        color=C_SAMPLER,
    )
    arrow(ax, (77, 33.5), (84, 32), label="K queries", lxy=(80.5, 35.2), fs=7.5)

    box(
        ax, 84, 46, 14, 10,
        "compose + hurdle",
        r"$\hat M_{t+1} = \mu_{t+1} + \hat r_{t+1}$"
        + "\n"
        + r"qpepre: $p_{wet}{<}\tau \Rightarrow 0$ mm/h",
        color=C_OUT,
    )
    arrow(ax, (48, 50.5), (84, 51), label="mean (skip)", lxy=(66, 52.6), fs=7.5)
    arrow(ax, (91, 38), (91, 46), label=r"$\hat r_{t+1}$", lxy=(93.4, 42), fs=8.5)
    arrow(
        ax, (77, 16.5), (97, 46), rad=-0.42,
        label=r"$p_{wet} = \sigma(g_\phi(c))$", lxy=(96.5, 24), fs=8.0,
        color=C_GATE, lcolor=VIR(0.55),
    )

    # output + autoregressive feedback
    arrow(ax, (98, 51), (103.4, 51), lw=2.4, color="0.15", mutation=16)
    ax.text(100.6, 53.4, r"$\hat M_{t+1}$", fontsize=11, fontweight="bold",
            ha="center", color="0.1")
    # autoregressive feedback: clean elbow along the free top lane
    fb_kw = dict(color="0.55", lw=1.6, ls=(0, (5, 4)), zorder=1)
    ax.plot([91, 91], [56.8, 61.5], **fb_kw)
    ax.plot([91, 8], [61.5, 61.5], **fb_kw)
    arrow(ax, (8, 61.5), (8, 54.8), ls=(0, (5, 4)), color="0.55")
    ax.text(
        49.5, 63.0, r"autoregressive rollout: $\hat M_{t+1} \to M_t$",
        ha="center", va="center", fontsize=8.5, color="0.45",
    )

    # --- legend -------------------------------------------------------------
    items = [
        (C_DATA, "data / tensors"),
        (C_FROZEN, "frozen"),
        (C_FLOWNET, "trained flow"),
        (C_GATE, "trained gate"),
        (C_SAMPLER, "sampling / composition"),
    ]
    x0 = 2
    for c, lab in items:
        ax.add_patch(
            Rectangle((x0, 2.2), 2.6, 2.6, fc=tint(c), ec=c, lw=1.4, zorder=2)
        )
        ax.text(x0 + 3.4, 3.5, lab, fontsize=8.2, va="center", color="0.25")
        x0 += 3.6 + len(lab) * 0.62 + 2.4

    save(fig, "gsmf_architecture.png")


# ===========================================================================
# Figure 2 -- training objective flow
# ===========================================================================
def fig_training():
    fig, ax = blank_canvas((12.5, 9.6), (0, 102), (0, 80))

    ax.text(
        1, 78.6, "GS-MeanFlow — training step",
        fontsize=14, fontweight="bold", color="0.1", va="center",
    )

    box(
        ax, 32, 70, 36, 6.5,
        "training batch",
        r"$(M_t, S_t, M_{t+1})$ from the cleaned zarr",
        color=C_DATA,
    )
    box(
        ax, 26, 59, 48, 8,
        "conditioning & residual target",
        r"$\mu = F_\theta(M_t, S_t, I)$ (frozen)  ·  $c = [M_t, \mu, I]$"
        + "\n"
        + r"$R_{t+1} = M_{t+1} - \mu$",
        color=C_FROZEN,
    )
    arrow(ax, (50, 70), (50, 67))

    box(
        ax, 26, 48, 48, 8,
        "flow-space setup",
        r"$x_1 = R/\sigma_d$, $x_0 \sim \mathcal{N}(0,I)$, $v = x_1 - x_0$"
        + "\n"
        + r"sample $r \leq t$;  $z_r = (1-r)\,x_0 + r\,x_1$",
        color=C_SAMPLER,
    )
    arrow(ax, (50, 59), (50, 56))

    # branch
    box(
        ax, 4, 35, 42, 9,
        r"MeanFlow identity  ($r < t$ · 25%)",
        r"$u_{tgt} = v + (t-r)\,\mathrm{d}u/\mathrm{d}r$"
        + "\nJVP forward-mode pass, stop-grad",
        color=C_FLOWNET,
    )
    box(
        ax, 54, 35, 42, 9,
        r"I-CFM  ($t = r$ · 75%)",
        r"$u_{tgt} = v$" + "\n(instantaneous straight-line target)",
        color=C_FLOWNET,
    )
    arrow(ax, (40, 48), (25, 44), rad=-0.1)
    arrow(ax, (60, 48), (75, 44), rad=0.1)

    box(
        ax, 26, 23, 48, 8,
        "single DDP forward",
        r"$u_{pred},\ \ell_{gate} = $ GSMeanFlowPrecond$(z_r, r, t, c)$"
        + "\nvelocity + gate logits in one pass",
        color=C_GSMEANFLOW,
    )
    arrow(ax, (25, 35), (40, 31), rad=-0.1)
    arrow(ax, (75, 35), (60, 31), rad=0.1)

    # losses
    box(
        ax, 2, 8, 32, 10,
        r"$L_{flow}$ — Pillar 1",
        "group-decoupled adaptive:\n"
        + r"$w_s\,\bar e_{smooth} + w_q\,\bar e_{qpepre}$"
        + "\n"
        + r"$w_g = (\bar e_g + \epsilon)^{-p}$ (stop-grad)",
        color=C_FLOWNET,
        bfs=7.8,
    )
    box(
        ax, 37, 8, 26, 10,
        r"$L_{spec}$",
        "radial log-PSD L1\non qpepre\n($t=r$ subset)",
        color=C_LOSS,
        bfs=7.8,
    )
    box(
        ax, 66, 8, 32, 10,
        r"$L_{gate}$ — Pillar 2",
        r"focal BCE$(\ell_{gate},\, \mathrm{wet})$"
        + "\n"
        + r"wet $= M_{t+1}^{qpepre} > 0.1$ mm/h"
        + "\n"
        + r"$\gamma{=}2$, $\alpha_{wet}{=}0.75$",
        color=C_GATE,
        bfs=7.8,
    )
    arrow(ax, (38, 23), (20, 18), rad=-0.08)
    arrow(ax, (50, 23), (50, 18))
    arrow(ax, (62, 23), (80, 18), rad=0.08)
    # wet mask side-channel from the batch: elbow down the free right margin
    wm_kw = dict(color=VIR(0.72), lw=1.6, ls=(0, (4, 3)), zorder=1)
    ax.plot([74, 100.5], [63, 63], **wm_kw)
    ax.plot([100.5, 100.5], [63, 20.5], **wm_kw)
    arrow(ax, (100.5, 20.5), (93, 18.3), color=VIR(0.72), ls=(0, (4, 3)))
    ax.text(
        87, 64.8, r"wet mask from $M_{t+1}$",
        ha="center", va="center", fontsize=8.0, color=VIR(0.55),
    )

    box(
        ax, 28, 0.5, 44, 5,
        r"$L = L_{flow} + \alpha_{spec} L_{spec} + \lambda_{gate} L_{gate}$",
        color="0.2",
        fs=10.5,
    )
    arrow(ax, (18, 8), (38, 4), rad=-0.08)
    arrow(ax, (50, 8), (50, 5.5))
    arrow(ax, (82, 8), (62, 4), rad=0.08)

    # EMA note
    box(
        ax, 78, 23, 21, 7.5,
        "EMA shadow",
        r"$\bar\theta \leftarrow 0.999\,\bar\theta$"
        + "\ninference weights",
        color=C_DATA,
        bfs=7.8,
    )
    arrow(ax, (74, 27), (78, 27), ls=(0, (3, 3)), color="0.55")

    save(fig, "gsmf_training_flow.png")


# ===========================================================================
# Figure 3 -- sampling: average velocity vs multi-step solvers + hurdle gate
# ===========================================================================
def fig_sampling():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.0))

    # ---- left: NFE / trajectory schematic ---------------------------------
    t = np.linspace(0, 1, 400)
    path = t + 0.33 * np.sin(np.pi * t) * (1 - t)  # curved ODE path, 0 -> 1

    def z_of(tq):
        return np.interp(tq, t, path)

    ax1.plot(t, path, color="0.55", lw=2.6, zorder=1,
             label="instantaneous-velocity ODE path")

    # EDM: dense Heun evaluations along the path
    t_edm = np.linspace(0, 1, 19)
    ax1.plot(t_edm, z_of(t_edm), "o", ms=3.4, color=C_DIFFUSION, zorder=3,
             label="EDM teacher · Heun, 18–36 NFE")

    # FlowCast: 10 Euler chords
    t_fc = np.linspace(0, 1, 11)
    ax1.plot(t_fc, z_of(t_fc), "-s", ms=4.5, lw=1.4, color=C_FLOWCAST,
             zorder=3, label="FlowCast · Euler, 10 NFE")

    # GS-MeanFlow: 1-NFE chord and 2-NFE chords
    ax1.annotate(
        "", xy=(1, z_of(1.0)), xytext=(0, z_of(0.0)),
        arrowprops=dict(arrowstyle="-|>", mutation_scale=18,
                        color=C_GSMEANFLOW, lw=3.0),
        zorder=4,
    )
    t2 = [0.0, 0.5, 1.0]
    ax1.plot(t2, z_of(np.array(t2)), "--D", ms=5.5, lw=1.8,
             color=C_MEANFLOW, zorder=4,
             label="GS-MeanFlow · 2 NFE")
    ax1.plot([], [], "-", color=C_GSMEANFLOW, lw=3.0,
             label=r"GS-MeanFlow · 1 NFE  $u_\theta(z_0, 0, 1)$")

    ax1.text(
        0.40, 0.16,
        r"$u(z_r, r, t) = \frac{1}{t-r}\int_r^t v(z_\tau, \tau)\,d\tau$"
        + "\naverage velocity = chord of the path",
        transform=ax1.transAxes, fontsize=9.5, color="0.25",
    )
    ax1.set_xlabel("flow time $t$  (noise $\\rightarrow$ residual)")
    ax1.set_ylabel("state $z$ (schematic)")
    ax1.set_title("One network call transports the whole interval", fontsize=11)
    ax1.legend(fontsize=8.0, loc="upper left", framealpha=0.95)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    # ---- right: hurdle gate on a precipitation transect --------------------
    rng = np.random.default_rng(7)
    x = np.arange(192)
    truth = 14.0 * np.exp(-0.5 * ((x - 58) / 7.0) ** 2) + 6.0 * np.exp(
        -0.5 * ((x - 132) / 5.0) ** 2
    )
    truth[truth < 0.05] = 0.0

    drizzle = np.abs(rng.normal(0.0, 0.16, size=x.size)) + 0.10
    ungated = (
        13.0 * np.exp(-0.5 * ((x - 58) / 8.5) ** 2)
        + 5.6 * np.exp(-0.5 * ((x - 132) / 6.2) ** 2)
        + drizzle
    )
    wet_mask = (truth > 0.1) | (np.roll(truth > 0.1, 3)) | (np.roll(truth > 0.1, -3))
    gated = np.where(wet_mask, ungated, 0.0)

    for lo, hi in [(40, 78), (117, 148)]:
        ax2.axvspan(lo, hi, color=tint(C_GATE, 0.8), zorder=0)

    ax2.plot(x, truth, color="0.15", lw=1.3, label="RWRF truth", zorder=3)
    ax2.plot(x, ungated, color=C_DIFFUSION, lw=1.6, ls="--", zorder=2,
             label="continuous flow only — spurious drizzle")
    ax2.plot(x, gated, color=VIR(0.6), lw=2.2, zorder=4,
             label="with occurrence gate — exact 0 mm/h dry")

    ax2.annotate(
        "gate forces exact zeros\n(point mass at 0 restored)",
        xy=(96, 0.05), xytext=(78, 5.2),
        fontsize=9, color="0.2",
        arrowprops=dict(arrowstyle="-|>", color="0.35", lw=1.3),
    )
    ax2.annotate(
        '"always slightly wet"',
        xy=(170, 0.35), xytext=(140, 3.0),
        fontsize=9, color=C_DIFFUSION,
        arrowprops=dict(arrowstyle="-|>", color=C_DIFFUSION, lw=1.3),
    )
    ax2.text(59, 14.6, "gate wet region", fontsize=8.2, ha="center",
             color=VIR(0.45))

    ax2.set_xlabel("pixel along transect (schematic)")
    ax2.set_ylabel("qpepre [mm/h]")
    ax2.set_title("Pillar 2 — hurdle composition at inference", fontsize=11)
    ax2.set_ylim(-0.6, 16.5)
    ax2.legend(fontsize=8.0, loc="upper right", framealpha=0.95)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    fig.suptitle("GS-MeanFlow sampling: 1–2 NFE average-velocity transport + occurrence gate",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "gsmf_sampling.png")


# ===========================================================================
# Figure 4 -- why the two pillars (evidence / distribution schematic)
# ===========================================================================
def fig_pillars():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.0))

    # ---- left: Pillar 1 evidence — qpw ablation conflict (real numbers) ----
    # Validation RMSE (standardized) @ ~2 M samples, CLAUDE.md §6.5.
    rmse = {
        "t2m": {1.4: 0.0890, 2.0: 0.0835},
        "u10": {1.4: 0.2733, 2.0: 0.2931},
        "v10": {1.4: 0.2050, 2.0: 0.2463},
        "qpepre": {1.4: 0.3996, 2.0: 0.3851},
    }
    chans = ["t2m", "u10", "v10", "qpepre"]
    pos = np.arange(len(chans))
    w = 0.36
    pen14, pen20 = [], []
    for ch in chans:
        best = min(rmse[ch].values())
        pen14.append(100.0 * (rmse[ch][1.4] / best - 1.0))
        pen20.append(100.0 * (rmse[ch][2.0] / best - 1.0))

    b1 = ax1.bar(pos - w / 2, pen14, w, color=VIR(0.55), label="qpw = 1.4")
    b2 = ax1.bar(pos + w / 2, pen20, w, color=VIR(0.80), label="qpw = 2.0")
    for bars, qpw in ((b1, 1.4), (b2, 2.0)):
        for rect, ch in zip(bars, chans):
            v = rmse[ch][qpw]
            ax1.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 0.45,
                f"{v:.4f}", ha="center", fontsize=7.6, color="0.3",
                rotation=0,
            )
    ax1.set_xticks(pos, chans)
    ax1.set_ylabel("RMSE penalty vs per-channel best [%]")
    ax1.set_title(
        "Pillar 1 evidence — no single channel weight wins\n"
        "(FlowCast qpw ablation, valid RMSE @ ~2M samples)",
        fontsize=10.5,
    )
    ax1.legend(fontsize=9)
    ax1.set_ylim(0, 28)
    ax1.text(
        0.5, 0.86,
        "winds want qpw=1.4 · t2m & qpepre want qpw=2.0\n"
        r"$\Rightarrow$ decouple the adaptive weight per group",
        transform=ax1.transAxes, ha="center", fontsize=9.2, color="0.2",
        bbox=dict(boxstyle="round,pad=0.4", fc=tint(C_FLOWNET, 0.92),
                  ec=C_FLOWNET, lw=1.0),
    )
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    # ---- right: Pillar 2 — the precip marginal is a hurdle ----------------
    xs = np.linspace(0, 8, 600)
    pi_dry = 0.78
    tail = np.exp(-xs / 1.6) / 1.6  # heavy-ish conditional intensity
    tail /= np.trapezoid(tail, xs)
    mix = (1 - pi_dry) * tail

    # continuous transport: atom smeared into a drizzle hump near zero
    smear = np.exp(-0.5 * ((xs - 0.5) / 0.45) ** 2)
    smear *= pi_dry / np.trapezoid(smear, xs)  # hump carries the dry mass
    cont = smear + mix

    ax2.bar([0], [pi_dry], width=0.10, color=VIR(0.6), zorder=3,
            label=r"true point mass: $\pi\,\delta_0$ (dry pixels)")
    ax2.plot(xs, mix, color=VIR(0.6), lw=2.2,
             label=r"true wet tail: $(1-\pi)\,f(x)$")
    ax2.plot(xs, cont, color=C_DIFFUSION, lw=1.8, ls="--",
             label="continuous transport — no atom,\ndrizzle hump instead")

    ax2.annotate(
        r"$p(x) = \pi\,\delta_0 + (1-\pi)\,f(x)$" + "\nhurdle factorization:"
        "\ngate handles $\\pi$, flow handles $f$",
        xy=(0.06, pi_dry * 0.96), xytext=(2.4, 0.62),
        fontsize=9.5, color="0.2",
        arrowprops=dict(arrowstyle="-|>", color="0.35", lw=1.3),
    )
    ax2.annotate(
        '"always slightly wet"',
        xy=(0.55, float(np.max(cont)) * 0.97), xytext=(1.9, 0.42),
        fontsize=9, color=C_DIFFUSION,
        arrowprops=dict(arrowstyle="-|>", color=C_DIFFUSION, lw=1.3),
    )

    ax2.set_xlabel("qpepre [mm/h] (schematic)")
    ax2.set_ylabel("probability density / mass")
    ax2.set_title(
        "Pillar 2 rationale — precipitation marginal is\npoint mass + heavy tail",
        fontsize=10.5,
    )
    ax2.set_xlim(-0.25, 8)
    ax2.set_ylim(0, 1.0)
    ax2.legend(fontsize=8.2, loc="upper right")
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    fig.suptitle("Why GS-MeanFlow: the two measured pathologies behind the pillars",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "gsmf_pillars.png")


if __name__ == "__main__":
    fig_architecture()
    fig_training()
    fig_sampling()
    fig_pillars()
    print("done.")
