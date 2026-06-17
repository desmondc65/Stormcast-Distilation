from matplotlib import pyplot as plt
import numpy as np


color_limits = {
    "u10m": (-15, 15),
    "u10": (-15, 15),
    "v10": (-15, 15),
    "t2m": (270, 310),
    "tcwv": (0, 60),
    "msl": (0.1, 0.3),
    "refc": (-10, 30),
    "qpepre": (0, 30),
}

cmap_for_var = {
    "u10m": "RdBu_r",
    "u10": "RdBu_r",
    "v10": "RdBu_r",
    "t2m": "inferno",
    "tcwv": "magma",
    "msl": "magma",
    "refc": "magma",
    "qpepre": "viridis",
}

_units = {
    "t2m": "K",
    "u10": "m s$^{-1}$",
    "u10m": "m s$^{-1}$",
    "v10": "m s$^{-1}$",
    "qpepre": "mm h$^{-1}$",
    "tcwv": "kg m$^{-2}$",
    "msl": "hPa",
    "refc": "dBZ",
}


def _resolve_limits(variable, generated, truth):
    """Return (vmin, vmax, cmap) for a variable, falling back to shared
    data-driven limits for unknown fields."""
    if variable in color_limits:
        vmin, vmax = color_limits[variable]
    else:
        vmin = float(min(np.min(generated), np.min(truth)))
        vmax = float(max(np.max(generated), np.max(truth)))
    cmap = cmap_for_var.get(variable, "magma")
    return vmin, vmax, cmap


def validation_plot(generated, truth, variable, experiment_name=None, step=None):
    """Side-by-side prediction vs ground truth with a shared, variable-aware
    colour scale and a difference panel. Designed to read as a paper figure."""

    vmin, vmax, cmap = _resolve_limits(variable, generated, truth)
    diff = generated - truth
    dmax = float(np.max(np.abs(diff))) if diff.size else 1.0
    dmax = max(dmax, 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), constrained_layout=True)
    a, b, c = axes

    units = _units.get(variable, "")
    cbar_label = f"{variable} [{units}]" if units else variable

    im_a = a.imshow(generated, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    a.set_title("Prediction", fontsize=11)
    fig.colorbar(im_a, ax=a, fraction=0.046, pad=0.04, label=cbar_label)

    im_b = b.imshow(truth, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    b.set_title("Ground truth", fontsize=11)
    fig.colorbar(im_b, ax=b, fraction=0.046, pad=0.04, label=cbar_label)

    im_c = c.imshow(diff, origin="lower", cmap="RdBu_r", vmin=-dmax, vmax=dmax)
    c.set_title("Prediction $-$ truth", fontsize=11)
    fig.colorbar(im_c, ax=c, fraction=0.046, pad=0.04, label=cbar_label)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)

    sup_parts = [variable]
    if experiment_name is not None:
        sup_parts.append(str(experiment_name))
    if step is not None:
        sup_parts.append(f"step {int(step)}")
    fig.suptitle("  |  ".join(sup_parts), fontsize=11)

    return fig


def inference_plot(
    background,
    state_pred,
    state_true,
    plot_var_background,
    plot_var_state,
    initial_time,
    lead_time,
):
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))

    state_error = state_pred - state_true

    if plot_var_state in color_limits:
        im = ax[0].imshow(
            state_pred,
            origin="lower",  # fix orientation
            cmap="magma",
            clim=color_limits[plot_var_state],
        )
    else:
        im = ax[0].imshow(state_pred, origin="lower", cmap="magma")

    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
    ax[0].set_title(
        "Predicted, {}, \n initial time {} \n lead_time {} hours".format(
            plot_var_state, initial_time, lead_time
        )
    )
    if plot_var_state in color_limits:
        im = ax[1].imshow(
            state_true,
            origin="lower",  # fix orientation
            cmap="magma",
            clim=color_limits[plot_var_state],
        )
    else:
        im = ax[1].imshow(state_true, origin="lower", cmap="magma")
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    ax[1].set_title("Actual, {}".format(plot_var_state))

    if plot_var_background in color_limits:
        im = ax[2].imshow(
            background,
            origin="lower",  # fix orientation
            cmap="magma",
            clim=color_limits[plot_var_background],
        )
    else:
        im = ax[2].imshow(background, origin="lower", cmap="magma")
    fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
    ax[2].set_title("Background, {}".format(plot_var_background))

    maxerror = np.max(np.abs(state_error))
    im = ax[3].imshow(
        state_error,
        origin="lower",  # fix orientation
        cmap="RdBu_r",
        vmax=maxerror,
        vmin=-maxerror,
    )
    fig.colorbar(im, ax=ax[3], fraction=0.046, pad=0.04)
    ax[3].set_title("Error, {}".format(plot_var_state))

    return fig