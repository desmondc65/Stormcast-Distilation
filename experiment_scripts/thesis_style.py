#!/usr/bin/env python3
"""Shared plotting style for every thesis figure.

The single source of truth for colour in the thesis. The constraint (from the
project lead) is that *all* plotting must follow
``experiment_scripts/plot_weight_comparison_grid.py``, which renders 2-D fields
with the trainer-time ``validation_plot`` style: the ``viridis`` colormap with a
per-row shared (vmin, vmax) and ``origin="lower"``. This module promotes that
choice to a project-wide convention:

* **2-D field heatmaps** use :data:`FIELD_CMAP` = ``"viridis"`` (truth / model
  predictions, raw-data dumps, qualitative panels).
* **Signed difference maps** (``prediction - truth``) use :data:`DIFF_CMAP`, a
  *diverging* map centred on zero. ``plot_weight_comparison_grid.py`` has no
  difference panels, so viridis says nothing about them; we keep the
  conventional ``RdBu_r`` so positive/negative excursions read at a glance.
* **Categorical bar / line plots** (scoreboards, ablations, rollout curves) take
  their per-method colours from :func:`method_color`, which *samples the viridis
  colormap itself*. This keeps the whole thesis on one perceptually-uniform
  ramp instead of the ad-hoc ``tab:`` / hand-picked hex palettes the older
  ``export_main_experiment_figures.py`` used. The ramp is ordered so that the
  EDM diffusion baseline sits at the cool (blue/teal) end and FlowCast at the
  warm (green) end, with the three FlowCast NFE variants stepping toward
  brighter green as the step count grows.

Import this everywhere a thesis figure is produced; do not redefine colours
locally.
"""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# --- Field colormaps ---------------------------------------------------------
# Matches stormcast/utils/plots.py:validation_plot and
# plot_weight_comparison_grid.py (FIELD_CMAP = "viridis").
FIELD_CMAP = "viridis"
# Diverging map for signed (pred - truth) panels. Not dictated by viridis;
# kept conventional so the zero-crossing is obvious.
DIFF_CMAP = "RdBu_r"

_VIRIDIS = cm.get_cmap("viridis")

# --- Channel metadata (shared by every field plot) ---------------------------
CHANNEL_UNITS = {"t2m": "K", "u10": "m/s", "v10": "m/s", "qpepre": "mm/h"}
CHANNEL_LABELS = {ch: f"{ch}\n[{u}]" for ch, u in CHANNEL_UNITS.items()}
# Canonical row order used wherever the four HighRes channels are laid out.
ROW_VARS = ["t2m", "u10", "v10", "qpepre"]


def clim_for(arrs) -> tuple[float, float]:
    """Shared (vmin, vmax) = global min/max across a list of arrays.

    Identical to ``plot_weight_comparison_grid.clim_for`` / what
    ``validation_plot`` does across its generated/truth pair so a row of panels
    shares one colour scale.
    """
    stack = np.concatenate([np.asarray(a).ravel() for a in arrs])
    return float(np.min(stack)), float(np.max(stack))


def diverging_clim(arr) -> tuple[float, float]:
    """Symmetric (-m, m) clim for a difference field so 0 maps to white."""
    m = float(np.nanmax(np.abs(np.asarray(arr))))
    m = m if m > 0 else 1.0
    return -m, m


# --- Categorical (per-method) palette, sampled from viridis ------------------
# Position of each method on the viridis ramp in [0, 1]. Cool end = baselines,
# warm (green) end = FlowCast; brighter green with more NFE. Capped below the
# pure-yellow tail (~0.88) so bars/lines stay legible on white.
_METHOD_POS = {
    "legacy":          0.04,   # legacy old-StormCast EDM (224x128, raw)
    "diffusion":       0.30,   # cleaned EDM teacher
    "bridge":          0.46,   # mu->M bridge head (results.md only; not in thesis)
    "flowcast":        0.66,   # single-NFE FlowCast student
    "flowcast_nfe10":  0.55,
    "flowcast_nfe15":  0.68,
    "flowcast_nfe20":  0.82,
}

# Human-readable labels for legends/axes, keyed by the canonical method key.
_METHOD_LABELS = {
    "legacy":          r"Legacy EDM (224$\times$128)",
    "diffusion":       "EDM diffusion",
    "bridge":          "Bridge",
    "flowcast":        "FlowCast",
    "flowcast_nfe10":  "FlowCast ($K{=}10$)",
    "flowcast_nfe15":  "FlowCast ($K{=}15$)",
    "flowcast_nfe20":  "FlowCast ($K{=}20$)",
}

# Map the many raw CSV spellings onto the canonical keys above.
_METHOD_ALIASES = {
    "legacy_edm": "legacy",
    "cleaned_edm": "diffusion",
    "edm": "diffusion",
    "diffusion": "diffusion",
    "cleaned_bridge": "bridge",
    "bridge": "bridge",
    "cleaned_flow": "flowcast",
    "flowcast": "flowcast",
    "cleaned_flow_nfe10": "flowcast_nfe10",
    "cleaned_flow_nfe15": "flowcast_nfe15",
    "cleaned_flow_nfe20": "flowcast_nfe20",
    "flowcast_nfe10": "flowcast_nfe10",
    "flowcast_nfe15": "flowcast_nfe15",
    "flowcast_nfe20": "flowcast_nfe20",
}


def canonical_method(raw: str) -> str:
    """Normalise a raw CSV/scoreboard method string to a canonical key."""
    key = str(raw).strip().lower()
    return _METHOD_ALIASES.get(key, key)


def method_label(raw: str) -> str:
    """Display label (LaTeX mathtext) for a raw method string."""
    return _METHOD_LABELS.get(canonical_method(raw), str(raw))


def method_color(raw: str):
    """RGBA colour for a method, sampled from the viridis ramp.

    Unknown methods fall back to a mid-ramp teal so the figure still renders.
    """
    pos = _METHOD_POS.get(canonical_method(raw), 0.5)
    return _VIRIDIS(pos)


def categorical_palette(n: int):
    """``n`` evenly-spaced viridis colours for generic categorical series
    (e.g. an ablation sweep with no fixed method identity)."""
    if n <= 0:
        return []
    if n == 1:
        return [_VIRIDIS(0.5)]
    return [_VIRIDIS(x) for x in np.linspace(0.08, 0.85, n)]


# --- Matplotlib rcParams (uniform typography across all thesis figures) ------
def apply_rc():
    """Apply the thesis-wide matplotlib style. Call once at import time of any
    figure script (mirrors export_main_experiment_figures.py typography so old
    and new figures match)."""
    matplotlib.use("Agg")
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "mathtext.fontset": "cm",
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 110,
        "savefig.dpi": 200,
    })


# Apply on import so a bare ``import thesis_style`` already fixes the backend.
apply_rc()
